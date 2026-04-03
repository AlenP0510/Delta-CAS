"""
delta_cas.agent
===============
Agent base class.

Subclass Agent and implement:
  compute_changes(state, version) -> dict
  describe_goal()                 -> str   (passed to intent_fn)

The full cycle per turn:
  1. WAL recovery (pre-flight)
  2. Wait for epoch alignment
  3. Read shared state
  4. Generate + validate intent via intent_fn
  5. Compute changes
  6. Write WAL locally (BEFORE CAS)
  7. CAS write to shared store
     * Success      → commit WAL, check epoch boundary
     * CAS conflict → stash pending delta, rebase on new state,
                      re-validate intent, retry
  8. Exponential backoff up to MAX_RETRIES
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Callable

from .core import Delta, Store, _apply_delta
from .epoch import EpochCoordinator
from .intent import IntentFn, intent_is_valid, noop_intent_fn
from .storage import local_agent_dir, ensure_dirs

logger = logging.getLogger(__name__)


class Agent:
    """
    Base agent with WAL crash safety, CAS conflict resolution,
    intent validation, and epoch awareness.

    Parameters
    ----------
    agent_id    : str        — unique identifier
    store       : Store      — the shared Delta-CAS store
    epoch       : EpochCoordinator | None
    intent_fn   : IntentFn   — callable(agent_id, state, goal) → str
                               Defaults to noop (always valid, no LLM call).
    local_dir   : str | None — override local WAL directory

    Example
    -------
    store = Store("./state")
    store.init({"score": 0})

    class ScoreAgent(Agent):
        def describe_goal(self):
            return "Increment the score by 1"
        def compute_changes(self, state, version):
            return {"score": state.get("score", 0) + 1}

    agent = ScoreAgent("scorer", store,
                       intent_fn=anthropic_intent_fn())
    success, version = agent.run()
    """

    MAX_RETRIES = 5
    RETRY_DELAY = 0.05  # seconds, multiplied by attempt number

    def __init__(
        self,
        agent_id: str,
        store: Store,
        epoch: EpochCoordinator | None = None,
        intent_fn: IntentFn | None = None,
        local_dir: str | None = None,
    ) -> None:
        self.agent_id  = agent_id
        self.store     = store
        self.epoch     = epoch
        self.intent_fn = intent_fn or noop_intent_fn

        self._local_dir = local_dir or local_agent_dir(
            store.archive_dir, agent_id
        )
        ensure_dirs(self._local_dir)

    # ── Override these ────────────────────────────────────────

    def compute_changes(self, state: dict, version: int) -> dict:
        """
        Return {key: new_value} for only the fields this agent changes.
        Called on initial read and after every rebase.
        Return {} to skip the write entirely.
        """
        raise NotImplementedError

    def describe_goal(self) -> str:
        """
        Plain-language description of what this agent is trying to do.
        Passed to intent_fn for semantic validation.
        """
        return f"{self.agent_id} goal"

    # ── WAL helpers ───────────────────────────────────────────

    def _wal_path(self, version: int) -> str:
        return os.path.join(self._local_dir, f"wal_{version:04d}.json")

    def _committed_marker(self, version: int) -> str:
        return os.path.join(self._local_dir, f"wal_{version:04d}.committed")

    def _stash_path(self, base_version: int) -> str:
        return os.path.join(self._local_dir, f"stash_{base_version:04d}.json")

    def _write_wal(self, delta: Delta) -> None:
        """Write delta locally BEFORE attempting CAS."""
        with open(self._wal_path(delta.version), "w") as f:
            json.dump(delta.to_dict(), f, indent=2)

    def _commit_wal(self, version: int) -> None:
        """Mark WAL entry as successfully committed."""
        with open(self._committed_marker(version), "w") as f:
            f.write(_now())

    def _stash_delta(self, delta: Delta, reason: str = "cas_conflict") -> None:
        """Archive a pending delta that could not be submitted as-is."""
        with open(self._stash_path(delta.base_version), "w") as f:
            json.dump({**delta.to_dict(), "stash_reason": reason}, f, indent=2)
        logger.info(
            f"[{self.agent_id}] Stashed delta base=v{delta.base_version} "
            f"reason={reason}"
        )

    def uncommitted_wal(self) -> list[Delta]:
        """Return WAL entries without a .committed marker."""
        pending = []
        for fname in sorted(os.listdir(self._local_dir)):
            if not (fname.startswith("wal_") and fname.endswith(".json")):
                continue
            try:
                version = int(fname[4:8])
            except ValueError:
                continue
            if not os.path.exists(self._committed_marker(version)):
                with open(os.path.join(self._local_dir, fname)) as f:
                    pending.append(Delta.from_dict(json.load(f)))
        return pending

    def restore_wal_to_store(self, delta: Delta) -> bool:
        """
        Transmission loss recovery:
        if delta is missing from shared store, restore from local WAL.
        Returns True if the delta is now present in shared store.
        """
        shared_path = self.store._delta_path(delta.version)
        wal_path    = self._wal_path(delta.version)

        if os.path.exists(shared_path):
            try:
                with open(shared_path) as f:
                    shared_data = json.load(f)
                if shared_data.get("checksum") != delta.checksum:
                    logger.error(
                        f"[{self.agent_id}] WAL v{delta.version} checksum "
                        f"mismatch with shared store"
                    )
            except Exception:
                pass
            self._commit_wal(delta.version)
            return True

        if not os.path.exists(wal_path):
            logger.error(f"[{self.agent_id}] WAL file missing for v{delta.version}")
            return False

        with open(wal_path, "r") as f:
            wal_data = json.load(f)
        with open(shared_path, "w") as f:
            json.dump(wal_data, f, indent=2)
        self._commit_wal(delta.version)
        logger.warning(
            f"[{self.agent_id}] Restored delta v{delta.version} from WAL"
        )
        return True

    # ── Main run loop ─────────────────────────────────────────

    def run(self, trigger: str = "") -> tuple[bool, int]:
        """
        Execute one full write cycle.

        Returns
        -------
        (success, version)
        success=False if intent was invalidated and no retry could fix it,
        or if MAX_RETRIES was exhausted.
        """
        # Pre-flight: recover uncommitted WAL entries
        uncommitted = self.uncommitted_wal()
        if uncommitted:
            logger.warning(
                f"[{self.agent_id}] {len(uncommitted)} uncommitted WAL "
                f"entries — attempting restoration"
            )
            for pending in uncommitted:
                self.restore_wal_to_store(pending)

        for attempt in range(self.MAX_RETRIES):

            # Wait if an epoch compaction is running
            if self.epoch:
                self.epoch.wait()

            # Step 1: read current state
            state, base_version = self.store.read()

            # Step 2: generate + validate intent
            intent_str = self.intent_fn(
                self.agent_id, state, self.describe_goal()
            )
            logger.info(f"[{self.agent_id}] Intent: {intent_str[:80]}")

            if not intent_is_valid(intent_str):
                logger.warning(
                    f"[{self.agent_id}] Intent invalid given current state "
                    f"— aborting. Reason: {intent_str}"
                )
                return False, base_version

            # Step 3: compute changes
            changes = self.compute_changes(state, base_version)
            if not changes:
                logger.info(f"[{self.agent_id}] No changes to write")
                return True, base_version

            # Step 4: WAL write BEFORE CAS
            optimistic = base_version + 1
            pending = Delta(
                version=optimistic, base_version=base_version,
                changes=changes, agent_id=self.agent_id,
                trigger=trigger, intent=intent_str,
            )
            self._write_wal(pending)

            # Step 5: CAS write
            success, result_version = self.store.write(
                agent_id=self.agent_id, base_version=base_version,
                changes=changes, trigger=trigger, intent=intent_str,
            )

            if success:
                self._commit_wal(result_version)
                if self.epoch:
                    self.epoch.check(self.store, result_version)
                logger.info(
                    f"[{self.agent_id}] Write succeeded: "
                    f"v{base_version} → v{result_version} (attempt {attempt+1})"
                )
                return True, result_version

            # ── CAS conflict ──────────────────────────────────
            # Another agent advanced the version.
            # Stash our pending delta, read new state, re-validate
            # intent on the new state, recompute changes, retry.
            logger.info(
                f"[{self.agent_id}] CAS conflict attempt {attempt+1} "
                f"(base=v{base_version}, current=v{result_version}) — rebasing"
            )
            self._stash_delta(pending, reason="cas_conflict")

            new_state, new_version = self.store.read()

            # Re-validate intent on the new state
            new_intent = self.intent_fn(
                self.agent_id, new_state, self.describe_goal()
            )
            logger.info(f"[{self.agent_id}] Rebased intent: {new_intent[:80]}")

            if not intent_is_valid(new_intent):
                logger.warning(
                    f"[{self.agent_id}] Intent invalid after rebase — aborting. "
                    f"Reason: {new_intent}"
                )
                return False, new_version

            # Recompute changes on the new state
            rebased_changes = self.compute_changes(new_state, new_version)
            if not rebased_changes:
                logger.info(
                    f"[{self.agent_id}] No changes after rebase on v{new_version}"
                )
                return True, new_version

            rebased = Delta(
                version=new_version + 1, base_version=new_version,
                changes=rebased_changes, agent_id=self.agent_id,
                trigger=f"{trigger}[rebased_v{new_version}]",
                intent=new_intent,
            )
            self._write_wal(rebased)

            success2, result2 = self.store.write(
                agent_id=self.agent_id, base_version=new_version,
                changes=rebased_changes, trigger=rebased.trigger,
                intent=new_intent,
            )

            if success2:
                self._commit_wal(result2)
                if self.epoch:
                    self.epoch.check(self.store, result2)
                logger.info(
                    f"[{self.agent_id}] Rebased write succeeded: "
                    f"v{new_version} → v{result2}"
                )
                return True, result2

            time.sleep(self.RETRY_DELAY * (attempt + 1))

        logger.error(f"[{self.agent_id}] Max retries reached — giving up")
        return False, -1


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
