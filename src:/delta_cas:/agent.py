"""
delta_cas.agent
===============
Agent base class with WAL + CAS + Intent + Epoch.

Full cycle per run() call:
  1. Pre-flight WAL recovery
  2. Wait for epoch alignment
  3. Read shared state
  4. Generate + validate intent (via intent_fn)
  5. Compute changes
  6. Write WAL locally (BEFORE CAS)
  7. CAS write to shared store
     * Success      → commit WAL, check epoch
     * CAS conflict → stash pending delta, read new state,
                      re-validate intent, rebase, retry
  8. Exponential backoff up to MAX_RETRIES
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime

from .core import Delta, Store
from .epoch import EpochCoordinator
from .intent import IntentFn, intent_is_valid, noop_intent_fn
from .storage import local_agent_dir, ensure_dirs

logger = logging.getLogger(__name__)


class Agent:
    """
    Base agent. Subclass and implement:
      compute_changes(state, version) -> dict
      describe_goal()                 -> str

    Parameters
    ----------
    agent_id  : unique identifier
    store     : the shared Delta-CAS Store
    epoch     : optional EpochCoordinator
    intent_fn : callable(agent_id, state, goal) -> str
                Defaults to noop (always valid, no LLM).
    local_dir : override local WAL directory
    """

    MAX_RETRIES = 5
    RETRY_DELAY = 0.05  # seconds × attempt number

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
        self._local_dir = local_dir or local_agent_dir(store.archive_dir, agent_id)
        ensure_dirs(self._local_dir)

    # ── Override these ────────────────────────────────────────

    def compute_changes(self, state: dict, version: int) -> dict:
        """Return {key: new_value} for fields to change. Return {} to skip."""
        raise NotImplementedError

    def describe_goal(self) -> str:
        """Plain-language description passed to intent_fn for validation."""
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
        with open(self._committed_marker(version), "w") as f:
            f.write(_now())

    def _stash_delta(self, delta: Delta, reason: str = "cas_conflict") -> None:
        with open(self._stash_path(delta.base_version), "w") as f:
            json.dump({**delta.to_dict(), "stash_reason": reason}, f, indent=2)
        logger.info(
            f"[{self.agent_id}] Stashed delta "
            f"base=v{delta.base_version} reason={reason}"
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
        """Copy WAL entry to shared store if missing. Returns True if now present."""
        shared_path = self.store._delta_path(delta.version)
        wal_path    = self._wal_path(delta.version)

        if os.path.exists(shared_path):
            try:
                with open(shared_path) as f:
                    data = json.load(f)
                if data.get("checksum") != delta.checksum:
                    logger.error(
                        f"[{self.agent_id}] WAL v{delta.version} "
                        f"checksum mismatch with shared store"
                    )
            except Exception:
                pass
            self._commit_wal(delta.version)
            return True

        if not os.path.exists(wal_path):
            logger.error(
                f"[{self.agent_id}] WAL file missing for v{delta.version}"
            )
            return False

        with open(wal_path) as f:
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

        Returns (success, version).
        success=False if intent was invalidated and could not be fixed,
        or MAX_RETRIES was exhausted.
        """
        # Pre-flight: restore uncommitted WAL entries
        uncommitted = self.uncommitted_wal()
        if uncommitted:
            logger.warning(
                f"[{self.agent_id}] {len(uncommitted)} uncommitted WAL "
                f"entries — restoring"
            )
            for pending in uncommitted:
                self.restore_wal_to_store(pending)

        for attempt in range(self.MAX_RETRIES):

            if self.epoch:
                self.epoch.wait()

            state, base_version = self.store.read()

            # Generate + validate intent
            intent_str = self.intent_fn(
                self.agent_id, state, self.describe_goal()
            )
            logger.info(f"[{self.agent_id}] Intent: {intent_str[:80]}")

            if not intent_is_valid(intent_str):
                logger.warning(
                    f"[{self.agent_id}] Intent invalid — aborting. "
                    f"Reason: {intent_str}"
                )
                return False, base_version

            changes = self.compute_changes(state, base_version)
            if not changes:
                logger.info(f"[{self.agent_id}] No changes to write")
                return True, base_version

            # WAL write before CAS
            pending = Delta(
                version=base_version + 1, base_version=base_version,
                changes=changes, agent_id=self.agent_id,
                trigger=trigger, intent=intent_str,
            )
            self._write_wal(pending)

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

            # CAS conflict — stash, rebase, retry
            logger.info(
                f"[{self.agent_id}] CAS conflict attempt {attempt+1} "
                f"(base=v{base_version}, current=v{result_version}) — rebasing"
            )
            self._stash_delta(pending)

            new_state, new_version = self.store.read()
            new_intent = self.intent_fn(
                self.agent_id, new_state, self.describe_goal()
            )
            logger.info(
                f"[{self.agent_id}] Rebased intent: {new_intent[:80]}"
            )

            if not intent_is_valid(new_intent):
                logger.warning(
                    f"[{self.agent_id}] Intent invalid after rebase — "
                    f"aborting. Reason: {new_intent}"
                )
                return False, new_version

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
