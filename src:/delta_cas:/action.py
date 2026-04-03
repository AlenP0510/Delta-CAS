"""
delta_cas.action
================
Action layer — Outbox Pattern for idempotent external side effects.

Solves the "did this already happen?" problem for irreversible external
actions (API calls, emails, MCP tool calls, database writes, etc.)

Core principle
--------------
  Commit first. Execute only if commit succeeds.

  V_n = V_(n-1) + S_(n-1) + {A_i} + A_status(pending → success | failed)

Lifecycle
---------
  1. Agent writes state change + action declaration atomically via CAS.
     Status = "pending". expires_at is set. claimed_by = None.

  2. Executor claims the action by writing claimed_by = its agent_id.
     This is a lightweight CAS write — prevents two executors racing.

  3. Executor calls execute(action). On return it writes status = success/failed.

  4. If the executor dies before writing the result, the action stays claimed
     but claimed_at + ttl makes it eligible for takeover by another executor.

  5. On CAS conflict / rebase, agents call check_action_conflict() before
     redeclaring. A pending/claimed/succeeded action with the same
     idempotency_key triggers pivot or abort, preventing double-execution.

Action record (stored in world state under "actions.<action_id>")
-----------------------------------------------------------------
{
  "id":               "act_<12hex>",
  "type":             "send_email" | "mcp_call" | ...,
  "status":           "pending" | "claimed" | "success" | "failed",
  "payload":          {...},
  "idempotency_key":  "<uuid>",
  "agent_id":         "agent_A",         # declarer
  "declared_at":      "2026-...",
  "expires_at":       "2026-...",        # timeout deadline
  "claimed_by":       null | "exec_B",  # executor that claimed it
  "claimed_at":       null | "2026-...",
  "executed_at":      null | "2026-...",
  "error":            null | "...",
  "retry_count":      0
}

Status transitions
------------------
  pending  → claimed   (Executor.claim)
  claimed  → success   (Executor — execution succeeded)
  claimed  → failed    (Executor — execution failed or exception)
  claimed  → pending   (Executor.sweep_expired_claims — claimed too long ago)
  pending  → failed    (Executor.sweep_timed_out — past expires_at, not claimed)
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# ── Status constants ──────────────────────────────────────────

PENDING = "pending"
CLAIMED = "claimed"
SUCCESS = "success"
FAILED  = "failed"

# Default TTLs (seconds)
DEFAULT_EXPIRES_SECONDS = 300   # action must be picked up within 5 min
DEFAULT_CLAIM_TTL       = 60    # executor must finish within 1 min of claiming


# ════════════════════════════════════════════════════════════════
# Helpers for agents — building action declarations
# ════════════════════════════════════════════════════════════════

def make_action_id() -> str:
    return f"act_{uuid.uuid4().hex[:12]}"


def action_changes(
    action_type: str,
    payload: dict,
    agent_id: str,
    idempotency_key: str | None = None,
    action_id: str | None = None,
    expires_seconds: int = DEFAULT_EXPIRES_SECONDS,
) -> dict:
    """
    Build the delta changes dict for declaring a new action.
    Merge the returned dict into compute_changes() to atomically bind
    the action declaration to your state change.

    Parameters
    ----------
    action_type     : e.g. "send_email", "mcp_call", "api_request"
    payload         : parameters the executor needs to run the action
    agent_id        : the declaring agent's ID
    idempotency_key : dedup key — same key = same action. Auto-generated if omitted.
    action_id       : override action ID. Auto-generated if omitted.
    expires_seconds : seconds from now until the action is considered timed out.

    Example
    -------
    changes = {"score": 10}
    changes.update(action_changes(
        action_type="send_email",
        payload={"to": "user@example.com"},
        agent_id=self.agent_id,
        idempotency_key="email-2026-04-01",
        expires_seconds=120,
    ))
    return changes
    """
    aid        = action_id or make_action_id()
    ikey       = idempotency_key or uuid.uuid4().hex
    expires_at = _future(expires_seconds)
    prefix     = f"actions.{aid}"

    return {
        f"{prefix}.id":               aid,
        f"{prefix}.type":             action_type,
        f"{prefix}.status":           PENDING,
        f"{prefix}.payload":          payload,
        f"{prefix}.idempotency_key":  ikey,
        f"{prefix}.agent_id":         agent_id,
        f"{prefix}.declared_at":      _now(),
        f"{prefix}.expires_at":       expires_at,
        f"{prefix}.claimed_by":       None,
        f"{prefix}.claimed_at":       None,
        f"{prefix}.executed_at":      None,
        f"{prefix}.error":            None,
        f"{prefix}.retry_count":      0,
    }


def find_duplicate_action(
    state: dict,
    action_type: str,
    idempotency_key: str,
) -> dict | None:
    """
    Return an existing action with the same type + idempotency_key,
    or None if no duplicate found.
    """
    for action in state.get("actions", {}).values():
        if (action.get("type") == action_type and
                action.get("idempotency_key") == idempotency_key):
            return action
    return None


def list_actions(
    state: dict,
    status: str | None = None,
    action_type: str | None = None,
) -> list[dict]:
    """List actions, optionally filtered by status and/or type."""
    result = []
    for action in state.get("actions", {}).values():
        if status and action.get("status") != status:
            continue
        if action_type and action.get("type") != action_type:
            continue
        result.append(action)
    return sorted(result, key=lambda a: a.get("declared_at", ""))


def is_action_timed_out(action: dict) -> bool:
    """True if the action is past its expires_at deadline."""
    expires_at = action.get("expires_at")
    if not expires_at:
        return False
    try:
        return datetime.fromisoformat(expires_at) < datetime.now()
    except ValueError:
        return False


def is_claim_expired(action: dict, claim_ttl: int = DEFAULT_CLAIM_TTL) -> bool:
    """True if the executor claimed the action more than claim_ttl seconds ago."""
    claimed_at = action.get("claimed_at")
    if not claimed_at:
        return False
    try:
        deadline = datetime.fromisoformat(claimed_at) + timedelta(seconds=claim_ttl)
        return deadline < datetime.now()
    except ValueError:
        return False


# ════════════════════════════════════════════════════════════════
# Executor
# ════════════════════════════════════════════════════════════════

class Executor:
    """
    Base executor. Subclass and implement execute(action) -> bool.

    The executor:
      1. Scans for pending actions it can handle.
      2. Claims each action via a CAS write (claimed_by = self.agent_id).
      3. Executes the action.
      4. Writes the result (success / failed) via CAS.

    Timeout / takeover:
      sweep_timed_out()       — marks past-deadline pending actions as failed.
      sweep_expired_claims()  — releases stale claims so another executor can retry.

    Parameters
    ----------
    agent_id    : unique executor ID
    store       : the shared Delta-CAS store
    handles     : list of action_type strings to process (None = all types)
    max_retries : retry limit per action before giving up
    claim_ttl   : seconds before a claim is considered stale (default 60)
    """

    def __init__(
        self,
        agent_id: str,
        store: "Store",                  # noqa: F821
        handles: list[str] | None = None,
        max_retries: int = 3,
        claim_ttl: int = DEFAULT_CLAIM_TTL,
    ) -> None:
        self.agent_id    = agent_id
        self.store       = store
        self.handles     = handles
        self.max_retries = max_retries
        self.claim_ttl   = claim_ttl

    def execute(self, action: dict) -> bool:
        """
        Override this. Perform the actual external side effect.
        Return True = success, False = failed.
        Raise any exception = treated as False (failed).
        """
        raise NotImplementedError

    # ── Main entry point ──────────────────────────────────────

    def run_pending(self) -> dict[str, int]:
        """
        Full sweep: sweep timeouts → sweep stale claims → pick up pending.

        Returns {"succeeded": N, "failed": N, "skipped": N}
        """
        self.sweep_timed_out()
        self.sweep_expired_claims()

        state, ver = self.store.read()
        pending    = list_actions(state, status=PENDING)

        if self.handles:
            pending = [a for a in pending if a["type"] in self.handles]

        counts = {"succeeded": 0, "failed": 0, "skipped": 0}
        for action in pending:
            result = self._process_one(action)
            counts[result] += 1
            # Re-read after each write so base_version stays current
            state, ver = self.store.read()

        return counts

    # ── Timeout + claim sweep ─────────────────────────────────

    def sweep_timed_out(self) -> int:
        """
        Mark pending actions past their expires_at deadline as failed.
        Returns count of actions timed out.
        """
        state, _ = self.store.read()
        timed_out = [
            a for a in list_actions(state, status=PENDING)
            if is_action_timed_out(a)
            and (not self.handles or a["type"] in self.handles)
        ]
        count = 0
        for action in timed_out:
            aid = action["id"]
            logger.warning(
                f"[{self.agent_id}] Action {aid} timed out "
                f"(expires_at={action.get('expires_at')}) — marking failed"
            )
            ok = self._write_result(
                action_id=aid,
                new_status=FAILED,
                error="timeout: expires_at exceeded without execution",
            )
            if ok:
                count += 1
        return count

    def sweep_expired_claims(self) -> int:
        """
        Release stale claims (claimed but executor died before finishing).
        Resets status back to pending so another executor can pick up.
        Returns count of claims released.
        """
        state, _ = self.store.read()
        stale = [
            a for a in list_actions(state, status=CLAIMED)
            if is_claim_expired(a, self.claim_ttl)
            and (not self.handles or a["type"] in self.handles)
        ]
        count = 0
        for action in stale:
            aid = action["id"]
            logger.warning(
                f"[{self.agent_id}] Releasing stale claim on {aid} "
                f"(claimed_by={action.get('claimed_by')}, "
                f"claimed_at={action.get('claimed_at')})"
            )
            changes = {
                f"actions.{aid}.status":     PENDING,
                f"actions.{aid}.claimed_by": None,
                f"actions.{aid}.claimed_at": None,
            }
            state2, ver2 = self.store.read()
            ok, _ = self.store.write(
                agent_id=self.agent_id, base_version=ver2,
                changes=changes, trigger=f"release_claim_{aid}",
                intent=f"Release stale claim on {aid}",
            )
            if ok:
                count += 1
        return count

    # ── Internal ──────────────────────────────────────────────

    def _process_one(self, action: dict) -> str:
        """Claim → execute → write result. Returns 'succeeded'/'failed'/'skipped'."""
        aid         = action["id"]
        retry_count = action.get("retry_count", 0)

        if retry_count >= self.max_retries:
            logger.warning(
                f"[{self.agent_id}] Action {aid} exceeded "
                f"max_retries={self.max_retries} — skipping"
            )
            return "skipped"

        # Step 1: claim the action
        if not self._claim(action):
            logger.info(
                f"[{self.agent_id}] Could not claim {aid} "
                f"(another executor got it)"
            )
            return "skipped"

        # Step 2: execute
        logger.info(
            f"[{self.agent_id}] Executing {aid} type={action['type']}"
        )
        try:
            success   = self.execute(action)
            error_msg = None
        except Exception as exc:
            logger.error(f"[{self.agent_id}] Action {aid} raised: {exc}")
            success   = False
            error_msg = str(exc)

        # Step 3: write result
        ok = self._write_result(
            action_id=aid,
            new_status=SUCCESS if success else FAILED,
            error=error_msg,
            retry_delta=0 if success else 1,
        )
        if ok:
            return "succeeded" if success else "failed"
        return "failed"

    def _claim(self, action: dict) -> bool:
        """
        Atomically set status=claimed, claimed_by=self.agent_id.
        Returns True if we successfully claimed the action.
        Another executor claiming it first is not an error — just try next.
        """
        aid = action["id"]
        changes = {
            f"actions.{aid}.status":     CLAIMED,
            f"actions.{aid}.claimed_by": self.agent_id,
            f"actions.{aid}.claimed_at": _now(),
        }

        for attempt in range(5):
            state, ver = self.store.read()

            # Re-check status — another executor may have claimed/executed it
            current = state.get("actions", {}).get(aid, {})
            if current.get("status") not in (PENDING,):
                return False

            ok, _ = self.store.write(
                agent_id=self.agent_id, base_version=ver,
                changes=changes, trigger=f"claim_{aid}",
                intent=f"Claim action {aid}",
            )
            if ok:
                return True
            # CAS conflict — retry
        return False

    def _write_result(
        self,
        action_id: str,
        new_status: str,
        error: str | None = None,
        retry_delta: int = 0,
    ) -> bool:
        """Write execution result via CAS. Retries on conflict."""
        changes = {
            f"actions.{action_id}.status":      new_status,
            f"actions.{action_id}.executed_at": _now(),
            f"actions.{action_id}.error":       error,
        }

        for attempt in range(5):
            state, ver = self.store.read()

            # Get current retry count from state
            current = state.get("actions", {}).get(action_id, {})
            if retry_delta:
                changes[f"actions.{action_id}.retry_count"] = (
                    current.get("retry_count", 0) + retry_delta
                )

            ok, new_ver = self.store.write(
                agent_id=self.agent_id, base_version=ver,
                changes=changes,
                trigger=f"result_{action_id}_{new_status}",
                intent=f"Mark action {action_id} as {new_status}",
            )
            if ok:
                logger.info(
                    f"[{self.agent_id}] Action {action_id} → "
                    f"{new_status} at v{new_ver}"
                )
                return True

            # Check if another executor already handled it
            state2, _ = self.store.read()
            current2   = state2.get("actions", {}).get(action_id, {})
            if current2.get("status") in (SUCCESS, FAILED):
                logger.info(
                    f"[{self.agent_id}] Action {action_id} already "
                    f"resolved by another executor"
                )
                return True

        logger.error(
            f"[{self.agent_id}] Could not write result for {action_id}"
        )
        return False


# ════════════════════════════════════════════════════════════════
# ActionAwareMixin for Agent subclasses
# ════════════════════════════════════════════════════════════════

class ActionAwareMixin:
    """
    Mixin for Agent subclasses that declare external actions.
    Adds check_action_conflict() for semantic rebase guard.

    Example
    -------
    class MyAgent(ActionAwareMixin, Agent):
        def compute_changes(self, state, version):
            conflict = self.check_action_conflict(
                state, "send_email", self._ikey
            )
            if conflict == "abort":   return {}
            if conflict == "pivot":   return self._fallback_changes(state)
            if conflict == "retry":   pass   # re-declare same action
            # no conflict — proceed
            changes = {"score": 1}
            changes.update(action_changes("send_email", {...},
                           self.agent_id, self._ikey))
            return changes
    """

    def check_action_conflict(
        self,
        state: dict,
        action_type: str,
        idempotency_key: str,
    ) -> str:
        """
        Returns
        -------
        "none"   — no conflict, safe to declare
        "pivot"  — action is pending or claimed (in-flight); consider alternative
        "abort"  — action already succeeded; do not re-declare
        "retry"  — action failed; safe to re-declare with same idempotency_key
        """
        existing = find_duplicate_action(state, action_type, idempotency_key)
        if existing is None:
            return "none"

        status = existing.get("status")
        aid    = existing.get("id", "?")
        me     = getattr(self, "agent_id", "?")

        if status in (PENDING, CLAIMED):
            logger.info(
                f"[{me}] Action conflict: {action_type} key={idempotency_key} "
                f"is {status} ({aid}) — pivot"
            )
            return "pivot"
        if status == SUCCESS:
            logger.info(
                f"[{me}] Action conflict: {action_type} key={idempotency_key} "
                f"already succeeded ({aid}) — abort"
            )
            return "abort"
        if status == FAILED:
            logger.info(
                f"[{me}] Action conflict: {action_type} key={idempotency_key} "
                f"failed ({aid}) — retry allowed"
            )
            return "retry"
        return "none"


# ── helpers ───────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

def _future(seconds: int) -> str:
    return (datetime.now() + timedelta(seconds=seconds)).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
