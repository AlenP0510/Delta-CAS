"""
Delta-based CAS (Compare-And-Swap) State Management
=====================================================

V_curr = V_base + sum(Deltas)

Multiple agents read from the same base state, apply their changes as Deltas,
and CAS-write back. Full snapshots are compacted every SNAPSHOT_INTERVAL versions
to keep the read chain short and I/O costs low.

Naming convention:
  S_0       → initial snapshot (base state)
  delta_001 → first delta after S_0
  delta_002 → second delta
  ...
  S_1       → compacted snapshot after SNAPSHOT_INTERVAL deltas
  delta_011 → first delta after S_1
  ...
"""

import json
import os
import time
import logging
import threading
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────

STATE_DIR          = "state"
LOCAL_ARCHIVE_DIR  = "local_archive"   # root for per-agent local backups
SNAPSHOT_INTERVAL  = 10                # compact every N versions


# ════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════

class Snapshot:
    """
    A full compacted state at a given version.
    Stored as S_0, S_1, S_2 ...
    """
    def __init__(self, version: int, state: dict, timestamp: str = None):
        self.version   = version
        self.state     = state
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    def to_dict(self) -> dict:
        return {
            "type":      "snapshot",
            "version":   self.version,
            "state":     self.state,
            "timestamp": self.timestamp
        }

    @staticmethod
    def from_dict(d: dict) -> "Snapshot":
        return Snapshot(
            version   = d["version"],
            state     = d["state"],
            timestamp = d.get("timestamp")
        )


class Delta:
    """
    An incremental change on top of a base version.
    Stores only what changed, not the full state.

    current_i = V_base + sum(delta_1 ... delta_i)
    """
    def __init__(self, version: int, base_version: int,
                 changes: dict, agent_id: str, trigger: str = ""):
        self.version      = version       # version this delta produces
        self.base_version = base_version  # version this delta was computed from
        self.changes      = changes       # {key: new_value} — only changed fields
        self.agent_id     = agent_id
        self.trigger      = trigger
        self.timestamp    = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    def to_dict(self) -> dict:
        return {
            "type":         "delta",
            "version":      self.version,
            "base_version": self.base_version,
            "changes":      self.changes,
            "agent_id":     self.agent_id,
            "trigger":      self.trigger,
            "timestamp":    self.timestamp
        }

    @staticmethod
    def from_dict(d: dict) -> "Delta":
        delta = Delta(
            version      = d["version"],
            base_version = d["base_version"],
            changes      = d["changes"],
            agent_id     = d["agent_id"],
            trigger      = d.get("trigger", "")
        )
        delta.timestamp = d.get("timestamp", "")
        return delta


# ════════════════════════════════════════════════════════════════
# Storage layer
# ════════════════════════════════════════════════════════════════

def _ensure_dirs():
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(LOCAL_ARCHIVE_DIR, exist_ok=True)


def _local_agent_dir(agent_id: str) -> str:
    """Each agent gets its own subdirectory under LOCAL_ARCHIVE_DIR."""
    path = os.path.join(LOCAL_ARCHIVE_DIR, agent_id)
    os.makedirs(path, exist_ok=True)
    return path


def _local_delta_path(agent_id: str, version: int) -> str:
    return os.path.join(_local_agent_dir(agent_id), f"delta_{version:04d}.json")

_ensure_dirs()


def _snapshot_path(snapshot_index: int) -> str:
    return os.path.join(STATE_DIR, f"S_{snapshot_index}.json")


def _delta_path(version: int) -> str:
    return os.path.join(STATE_DIR, f"delta_{version:04d}.json")


def _meta_path() -> str:
    return os.path.join(STATE_DIR, "meta.json")


def _load_meta() -> dict:
    path = _meta_path()
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "current_version":    0,
        "latest_snapshot_version": 0,
        "latest_snapshot_index":   0,
        "total_deltas_since_snapshot": 0
    }


def _save_meta(meta: dict):
    with open(_meta_path(), "w") as f:
        json.dump(meta, f, indent=2)


def _save_snapshot(snapshot: Snapshot, index: int):
    with open(_snapshot_path(index), "w") as f:
        json.dump(snapshot.to_dict(), f, indent=2)
    logger.info(f"Snapshot S_{index} saved at version {snapshot.version}")


def _load_snapshot(index: int) -> Snapshot:
    with open(_snapshot_path(index)) as f:
        return Snapshot.from_dict(json.load(f))


def _save_delta(delta: Delta):
    """Write delta to shared state dir. Also archive a local copy under the agent's dir."""
    data = delta.to_dict()
    # Shared store
    with open(_delta_path(delta.version), "w") as f:
        json.dump(data, f, indent=2)
    # Local agent archive — survives transmission failures
    local_path = _local_delta_path(delta.agent_id, delta.version)
    with open(local_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.debug(f"[{delta.agent_id}] Local archive written: {local_path}")


def _load_delta(version: int) -> Delta:
    with open(_delta_path(version)) as f:
        return Delta.from_dict(json.load(f))


# ════════════════════════════════════════════════════════════════
# Core: read current state
# V_curr = V_base + sum(Deltas since base)
# ════════════════════════════════════════════════════════════════

def read_current_state() -> tuple[dict, int]:
    """
    Reconstruct current state from latest snapshot + pending deltas.
    Returns (state, current_version).

    V_curr = V_base + sum(delta_i for i in base_version+1 ... current_version)
    """
    meta = _load_meta()
    base_version      = meta["latest_snapshot_version"]
    base_snap_index   = meta["latest_snapshot_index"]
    current_version   = meta["current_version"]

    # Load base snapshot
    if not os.path.exists(_snapshot_path(base_snap_index)):
        # No state yet — return empty
        return {}, 0

    snapshot = _load_snapshot(base_snap_index)
    state    = json.loads(json.dumps(snapshot.state))  # deep copy

    # Apply all deltas from base_version+1 to current_version
    for v in range(base_version + 1, current_version + 1):
        delta_file = _delta_path(v)
        if not os.path.exists(delta_file):
            logger.warning(f"Missing delta for version {v}, skipping")
            continue
        delta = _load_delta(v)
        _apply_delta(state, delta.changes)

    return state, current_version


def _apply_delta(state: dict, changes: dict):
    """
    Apply a delta's changes onto a state dict in-place.
    Supports nested keys via dot notation: "goals.goal_001.tension"
    """
    for key, value in changes.items():
        if "." in key:
            parts = key.split(".", 1)
            if parts[0] not in state:
                state[parts[0]] = {}
            _apply_delta(state[parts[0]], {parts[1]: value})
        else:
            state[key] = value


# ════════════════════════════════════════════════════════════════
# Core: CAS write
# ════════════════════════════════════════════════════════════════

# File-level lock — prevents concurrent writes within the same process
_write_lock = threading.Lock()


def cas_write(agent_id: str, base_version: int,
              changes: dict, trigger: str = "") -> tuple[bool, int]:
    """
    Compare-And-Swap write.

    Steps:
      1. Lock
      2. Read current version from meta
      3. If current_version != base_version → conflict, return (False, current_version)
      4. Write delta file
      5. Increment version in meta
      6. If total_deltas_since_snapshot >= SNAPSHOT_INTERVAL → compact
      7. Unlock

    Returns (success, new_version).
    """
    with _write_lock:
        meta = _load_meta()
        current_version = meta["current_version"]

        # CAS check — base must match current
        if base_version != current_version:
            logger.warning(
                f"[{agent_id}] CAS conflict: expected base={base_version}, "
                f"current={current_version}"
            )
            return False, current_version

        new_version = current_version + 1
        delta = Delta(
            version      = new_version,
            base_version = base_version,
            changes      = changes,
            agent_id     = agent_id,
            trigger      = trigger
        )
        _save_delta(delta)

        meta["current_version"]              = new_version
        meta["total_deltas_since_snapshot"] += 1
        _save_meta(meta)

        logger.info(
            f"[{agent_id}] Delta written: v{base_version} → v{new_version} "
            f"({len(changes)} changes)"
        )

        # Compact if needed
        if meta["total_deltas_since_snapshot"] >= SNAPSHOT_INTERVAL:
            _compact(meta, new_version)

        return True, new_version


def _compact(meta: dict, at_version: int):
    """
    Rebuild full state and write a new snapshot.
    Resets delta counter.
    """
    state, _ = read_current_state()
    new_snap_index = meta["latest_snapshot_index"] + 1
    snapshot = Snapshot(version=at_version, state=state)
    _save_snapshot(snapshot, new_snap_index)

    meta["latest_snapshot_version"]       = at_version
    meta["latest_snapshot_index"]         = new_snap_index
    meta["total_deltas_since_snapshot"]   = 0
    _save_meta(meta)

    logger.info(f"Compacted → S_{new_snap_index} at v{at_version}")


# ════════════════════════════════════════════════════════════════
# Initialization
# ════════════════════════════════════════════════════════════════

def init_state(initial_state: dict) -> int:
    """
    Create S_0 with the initial state.
    Must be called once before any agent writes.
    Returns initial version (0).
    """
    snapshot = Snapshot(version=0, state=initial_state)
    _save_snapshot(snapshot, index=0)

    meta = {
        "current_version":              0,
        "latest_snapshot_version":      0,
        "latest_snapshot_index":        0,
        "total_deltas_since_snapshot":  0
    }
    _save_meta(meta)
    logger.info("State initialized as S_0 at version 0")
    return 0


# ════════════════════════════════════════════════════════════════
# Local archive helpers
# ════════════════════════════════════════════════════════════════

def _local_agent_dir(agent_id: str) -> str:
    """Each agent gets its own subdirectory under LOCAL_ARCHIVE_DIR."""
    path = os.path.join(LOCAL_ARCHIVE_DIR, agent_id)
    os.makedirs(path, exist_ok=True)
    return path


def _save_local_delta(agent_id: str, delta: "Delta", local_dir: str):
    """
    Write a local copy of the delta immediately after a successful CAS write.
    Protects against transmission loss between agent and shared state store.
    File: <local_dir>/delta_<version:04d>.json
    """
    path = os.path.join(local_dir, f"delta_{delta.version:04d}.json")
    with open(path, "w") as f:
        json.dump(delta.to_dict(), f, indent=2)
    logger.info(f"[{agent_id}] Local archive saved: {path}")


# ════════════════════════════════════════════════════════════════
# Agent base class
# ════════════════════════════════════════════════════════════════

class Agent:
    """
    Base agent with WAL (Write-Ahead Log) + transmission loss recovery.

    Full lifecycle per turn:
      1. Read current shared state (V_base)
      2. Compute changes
      3. WAL: write pending delta to local archive BEFORE attempting CAS
      4. Attempt CAS write to shared store
         a. Success → mark local delta as committed, return
         b. CAS conflict (other agent advanced version) →
              - Archive the pending delta locally (stash)
              - Read new shared state (includes the other agent's delta)
              - Rebase: recompute changes on top of new state
              - Retry CAS with rebased changes
         c. Transmission loss (write appears to succeed but shared store
            did not advance) →
              - Detect via version mismatch on next read
              - Recover: restore local WAL delta to shared store
              - Continue from recovered version

    Subclass and implement compute_changes().
    """

    MAX_RETRIES = 5
    RETRY_DELAY = 0.05  # seconds

    def __init__(self, agent_id: str, local_dir: str = None):
        self.agent_id   = agent_id
        self._local_dir = local_dir or _local_agent_dir(agent_id)
        os.makedirs(self._local_dir, exist_ok=True)

    # ── Override this ─────────────────────────────────────────

    def compute_changes(self, state: dict, version: int) -> dict:
        """
        Return {key: new_value} for only the fields this agent changes.
        Called both on initial read and after rebase on conflict.
        """
        raise NotImplementedError

    # ── WAL helpers ───────────────────────────────────────────

    def _wal_path(self, version: int) -> str:
        return os.path.join(self._local_dir, f"wal_{version:04d}.json")

    def _committed_marker(self, version: int) -> str:
        return os.path.join(self._local_dir, f"wal_{version:04d}.committed")

    def _stash_path(self, version: int) -> str:
        """
        Stash: a pending delta that could not be submitted because a CAS
        conflict occurred. Kept for audit / manual replay.
        """
        return os.path.join(self._local_dir, f"stash_{version:04d}.json")

    def _write_wal(self, delta: "Delta"):
        """
        Step 3: Write the pending delta locally BEFORE attempting CAS.
        If the process dies between here and CAS success, the WAL entry
        can be inspected and replayed.
        """
        path = self._wal_path(delta.version)
        with open(path, "w") as f:
            json.dump(delta.to_dict(), f, indent=2)
        logger.debug(f"[{self.agent_id}] WAL written: {path}")

    def _commit_wal(self, version: int):
        """Mark WAL entry as successfully committed to shared store."""
        marker = self._committed_marker(version)
        with open(marker, "w") as f:
            f.write(datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
        logger.debug(f"[{self.agent_id}] WAL committed: v{version}")

    def _stash_delta(self, delta: "Delta"):
        """
        CAS conflict: the delta we computed on a stale base cannot be submitted
        as-is. Archive it as a stash so it is not lost, then rebase.
        """
        path = self._stash_path(delta.base_version)
        with open(path, "w") as f:
            json.dump({**delta.to_dict(), "stash_reason": "cas_conflict"}, f, indent=2)
        logger.info(
            f"[{self.agent_id}] Delta stashed (base=v{delta.base_version} "
            f"was stale): {path}"
        )

    def _recover_wal(self) -> list["Delta"]:
        """
        On startup or after suspected transmission loss:
        find all WAL entries that have no .committed marker.
        These represent deltas that were written locally but may not have
        reached the shared store.
        """
        pending = []
        for fname in sorted(os.listdir(self._local_dir)):
            if not fname.startswith("wal_") or not fname.endswith(".json"):
                continue
            version_str = fname[4:8]
            try:
                version = int(version_str)
            except ValueError:
                continue
            if not os.path.exists(self._committed_marker(version)):
                path = os.path.join(self._local_dir, fname)
                with open(path) as f:
                    pending.append(Delta.from_dict(json.load(f)))
        return pending

    def _try_restore_wal_to_shared(self, delta: "Delta") -> bool:
        """
        Transmission loss recovery:
        If our WAL delta is not in the shared store, copy it there.
        Only safe if no other agent has written to that version slot.
        Returns True if the delta was restored.
        """
        shared_path = _delta_path(delta.version)
        if os.path.exists(shared_path):
            # Already there (possibly our own write did go through)
            logger.info(
                f"[{self.agent_id}] WAL v{delta.version} already in shared store"
            )
            self._commit_wal(delta.version)
            return True

        # Not in shared store — copy from local WAL
        with open(self._wal_path(delta.version), "w") as f:
            json.dump(delta.to_dict(), f, indent=2)
        # Also write to shared path
        shared_write_path = _delta_path(delta.version)
        with open(shared_write_path, "w") as f:
            json.dump(delta.to_dict(), f, indent=2)
        self._commit_wal(delta.version)
        logger.warning(
            f"[{self.agent_id}] Restored missing delta v{delta.version} "
            f"from WAL to shared store"
        )
        return True

    # ── Main run loop ─────────────────────────────────────────

    def run(self, trigger: str = "") -> tuple[bool, int]:
        """
        Full WAL + CAS + rebase cycle.

        On each attempt:
          - Read shared state
          - Compute changes
          - WAL write (local, before CAS)
          - Attempt CAS
            * Success      → commit WAL marker, return
            * CAS conflict → stash pending delta, rebase on new state, retry
        """
        # ── Pre-flight: recover any uncommitted WAL entries ──
        uncommitted = self._recover_wal()
        if uncommitted:
            logger.warning(
                f"[{self.agent_id}] Found {len(uncommitted)} uncommitted WAL "
                f"entries on startup — attempting restoration"
            )
            for pending_delta in uncommitted:
                self._try_restore_wal_to_shared(pending_delta)

        # ── Main retry loop ──
        for attempt in range(self.MAX_RETRIES):

            # Step 1: read current shared state
            state, base_version = read_current_state()

            # Step 2: compute changes on current state
            changes = self.compute_changes(state, base_version)
            if not changes:
                logger.info(f"[{self.agent_id}] No changes to write")
                return True, base_version

            # Step 3: WAL — write pending delta locally BEFORE CAS
            # Version slot is optimistic: base_version + 1
            optimistic_version = base_version + 1
            pending_delta = Delta(
                version      = optimistic_version,
                base_version = base_version,
                changes      = changes,
                agent_id     = self.agent_id,
                trigger      = trigger
            )
            self._write_wal(pending_delta)

            # Step 4: attempt CAS write to shared store
            success, result_version = cas_write(
                agent_id     = self.agent_id,
                base_version = base_version,
                changes      = changes,
                trigger      = trigger
            )

            if success:
                # Commit WAL marker
                self._commit_wal(result_version)
                logger.info(
                    f"[{self.agent_id}] Write succeeded: "
                    f"v{base_version} → v{result_version} (attempt {attempt+1})"
                )
                return True, result_version

            # ── CAS conflict: another agent advanced the version ──
            #
            # The pending delta we computed is now stale (wrong base).
            # Workflow:
            #   1. Stash the pending delta locally (not lost, just deferred)
            #   2. Read the new shared state (which includes the other agent's delta)
            #   3. Rebase: recompute our changes on top of the new state
            #   4. Loop back and retry with the rebased changes
            #
            # current_i = V_base + sum(other_deltas) + our_rebased_delta
            #
            logger.info(
                f"[{self.agent_id}] CAS conflict on attempt {attempt+1} "
                f"(base=v{base_version}, current=v{result_version}) — "
                f"stashing and rebasing"
            )
            self._stash_delta(pending_delta)

            # Read the new deltas that advanced the version
            new_state, new_version = read_current_state()
            logger.info(
                f"[{self.agent_id}] Rebasing on v{new_version} "
                f"(absorbed {new_version - base_version} new delta(s))"
            )

            # Rebase: recompute changes on the new state
            # This is the key step — our compute_changes sees the updated world
            rebased_changes = self.compute_changes(new_state, new_version)
            if not rebased_changes:
                logger.info(
                    f"[{self.agent_id}] No changes needed after rebase on v{new_version}"
                )
                return True, new_version

            # Write rebased WAL entry
            rebased_delta = Delta(
                version      = new_version + 1,
                base_version = new_version,
                changes      = rebased_changes,
                agent_id     = self.agent_id,
                trigger      = f"{trigger}[rebased_on_v{new_version}]"
            )
            self._write_wal(rebased_delta)

            # Attempt CAS with rebased changes
            success2, result_version2 = cas_write(
                agent_id     = self.agent_id,
                base_version = new_version,
                changes      = rebased_changes,
                trigger      = rebased_delta.trigger
            )

            if success2:
                self._commit_wal(result_version2)
                logger.info(
                    f"[{self.agent_id}] Rebased write succeeded: "
                    f"v{new_version} → v{result_version2}"
                )
                return True, result_version2

            # Still conflicting — back to top of loop with exponential backoff
            time.sleep(self.RETRY_DELAY * (attempt + 1))

        logger.error(f"[{self.agent_id}] Max retries reached, write failed")
        return False, -1


# ════════════════════════════════════════════════════════════════
# Utility: read full history
# ════════════════════════════════════════════════════════════════

def read_history(from_version: int = 0) -> list[dict]:
    """
    Return all deltas from from_version onwards, in order.
    Useful for audit trails and debugging.
    """
    meta    = _load_meta()
    current = meta["current_version"]
    history = []

    for v in range(from_version + 1, current + 1):
        path = _delta_path(v)
        if os.path.exists(path):
            with open(path) as f:
                history.append(json.load(f))

    return history


def read_version_at(target_version: int) -> dict:
    """
    Reconstruct state at any specific past version.
    Walks from nearest snapshot up to target_version.
    """
    meta = _load_meta()

    # Find the most recent snapshot at or before target_version
    best_snap_index   = 0
    best_snap_version = 0

    for i in range(meta["latest_snapshot_index"] + 1):
        path = _snapshot_path(i)
        if not os.path.exists(path):
            continue
        snap = _load_snapshot(i)
        if snap.version <= target_version and snap.version >= best_snap_version:
            best_snap_version = snap.version
            best_snap_index   = i

    if not os.path.exists(_snapshot_path(best_snap_index)):
        return {}

    snapshot = _load_snapshot(best_snap_index)
    state    = json.loads(json.dumps(snapshot.state))

    for v in range(best_snap_version + 1, target_version + 1):
        path = _delta_path(v)
        if os.path.exists(path):
            delta = _load_delta(v)
            _apply_delta(state, delta.changes)

    return state


def recover_missing_deltas() -> list[int]:
    """
    Scan all agent local archives and restore any delta files missing
    from the shared state dir. Call this on startup or after a crash.

    Returns list of recovered version numbers.
    """
    if not os.path.exists(LOCAL_ARCHIVE_DIR):
        return []

    recovered = []
    for agent_id in os.listdir(LOCAL_ARCHIVE_DIR):
        agent_dir = os.path.join(LOCAL_ARCHIVE_DIR, agent_id)
        if not os.path.isdir(agent_dir):
            continue
        for fname in sorted(os.listdir(agent_dir)):
            if not fname.startswith("delta_") or not fname.endswith(".json"):
                continue
            shared_path = os.path.join(STATE_DIR, fname)
            local_path  = os.path.join(agent_dir, fname)
            if not os.path.exists(shared_path):
                try:
                    with open(local_path) as f:
                        data = json.load(f)
                    with open(shared_path, "w") as f:
                        json.dump(data, f, indent=2)
                    version = data.get("version", "?")
                    logger.warning(
                        f"Recovered missing delta v{version} "
                        f"from [{agent_id}] local archive"
                    )
                    recovered.append(version)
                except Exception as e:
                    logger.error(f"Failed to recover {fname} from [{agent_id}]: {e}")

    return recovered

if __name__ == "__main__":
    import shutil

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    # ── Clean slate ──
    for d in [STATE_DIR, LOCAL_ARCHIVE_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
    _ensure_dirs()

    # ── Initialize S_0 ──
    initial = {
        "global_tension": 0.0,
        "goals": {},
        "tasks": []
    }
    init_state(initial)
    print("\n── S_0 initialized ──")

    # ── Define example agents ──

    class TensionAgent(Agent):
        def __init__(self, agent_id: str, goal_id: str, new_tension: float):
            super().__init__(agent_id)
            self.goal_id     = goal_id
            self.new_tension = new_tension

        def compute_changes(self, state: dict, version: int) -> dict:
            return {f"goals.{self.goal_id}.tension": self.new_tension}


    class TaskAgent(Agent):
        def __init__(self, agent_id: str, task: str):
            super().__init__(agent_id)
            self.task = task

        def compute_changes(self, state: dict, version: int) -> dict:
            current_tasks = state.get("tasks", [])
            if self.task not in current_tasks:
                return {"tasks": current_tasks + [self.task]}
            return {}


    class GlobalTensionAgent(Agent):
        def compute_changes(self, state: dict, version: int) -> dict:
            goals = state.get("goals", {})
            tensions = [g["tension"] for g in goals.values() if "tension" in g]
            if not tensions:
                return {}
            return {"global_tension": round(sum(tensions) / len(tensions), 3)}


    # ════════════════════════════════════════════════════════════
    # Scenario 1: Normal run
    # ════════════════════════════════════════════════════════════

    print("\n════ Scenario 1: Normal run ════")
    agents = [
        TensionAgent("agent_gre",     "goal_gre",     0.72),
        TensionAgent("agent_fitness", "goal_fitness",  0.31),
        TaskAgent("agent_task_1", "Prepare GRE"),
        TaskAgent("agent_task_2", "Morning run"),
        GlobalTensionAgent("agent_global"),
    ]
    for agent in agents:
        success, version = agent.run(trigger=f"{agent.agent_id} update")
        print(f"  {agent.agent_id}: {'✓' if success else '✗'} → v{version}")

    state, version = read_current_state()
    print(f"\n  Final state v{version}: global_tension={state['global_tension']} "
          f"tasks={state['tasks']}")

    # ════════════════════════════════════════════════════════════
    # Scenario 2: CAS conflict → stash + rebase
    #
    # Two agents both read v5. agent_A writes first (v5→v6).
    # agent_B's write is rejected (base stale). It stashes its
    # pending delta, reads the new state at v6, rebases its
    # compute_changes on v6, and retries successfully.
    # ════════════════════════════════════════════════════════════

    print("\n════ Scenario 2: CAS conflict → stash + rebase ════")

    # agent_A advances the version
    agent_a = TensionAgent("agent_conflict_A", "goal_gre", 0.80)
    agent_a.run(trigger="agent_A first")

    # agent_B tries to write on the now-stale base (simulated by
    # directly calling cas_write with the old version, then running
    # through agent.run which handles the conflict automatically)
    agent_b = TensionAgent("agent_conflict_B", "goal_fitness", 0.25)
    success_b, v_b = agent_b.run(trigger="agent_B concurrent")
    print(f"  agent_B after rebase: {'✓' if success_b else '✗'} → v{v_b}")

    # Show stash files
    stash_files = [f for f in os.listdir(agent_b._local_dir)
                   if f.startswith("stash_")]
    print(f"  agent_B stash files: {stash_files or 'none (no conflict occurred in sequence)'}")

    # ════════════════════════════════════════════════════════════
    # Scenario 3: Transmission loss → WAL recovery
    #
    # Simulate: agent writes a delta, CAS succeeds (shared store
    # has the file), but then the file is deleted to simulate a
    # transmission drop or storage fault.
    # On the next agent.run(), _recover_wal() detects the uncommitted
    # WAL entry and restores the delta to the shared store.
    # ════════════════════════════════════════════════════════════

    print("\n════ Scenario 3: Transmission loss → WAL recovery ════")

    # Step 1: run an agent normally so it writes a delta + WAL entry
    recovery_agent = TaskAgent("agent_recovery", "Recovery task")
    _, written_version = recovery_agent.run(trigger="before loss")
    print(f"  Agent wrote delta v{written_version} successfully")

    # Step 2: simulate transmission loss — delete the shared delta
    lost_path = _delta_path(written_version)
    if os.path.exists(lost_path):
        os.remove(lost_path)
        print(f"  Simulated transmission loss: deleted delta v{written_version} from shared store")

    # Step 3: also remove the .committed marker so recovery_agent
    # thinks its write was not confirmed
    committed_marker = recovery_agent._committed_marker(written_version)
    if os.path.exists(committed_marker):
        os.remove(committed_marker)
        print(f"  Removed .committed marker — agent believes write is unconfirmed")

    # Step 4: re-run the agent — _recover_wal() fires on startup,
    # finds the uncommitted WAL entry, restores it to shared store
    print(f"  Re-running agent (WAL recovery should trigger)...")
    _, recovered_version = recovery_agent.run(trigger="after recovery")

    # Verify delta is back
    restored = os.path.exists(_delta_path(written_version))
    print(f"  Delta v{written_version} restored to shared store: {restored}")
    print(f"  Agent continued from v{recovered_version}")

    # ════════════════════════════════════════════════════════════
    # Final state + delta history
    # ════════════════════════════════════════════════════════════

    print("\n════ Final state ════")
    state, version = read_current_state()
    print(json.dumps(state, indent=2))

    print("\n════ Delta history ════")
    for entry in read_history():
        trigger_str = f" [{entry.get('trigger','')}]" if entry.get("trigger") else ""
        print(f"  v{entry['version']:02d} [{entry['agent_id']:<22}] "
              f"base=v{entry['base_version']:02d}  "
              f"changes={list(entry['changes'].keys())}"
              f"{trigger_str}")

    print("\n════ WAL files per agent ════")
    if os.path.exists(LOCAL_ARCHIVE_DIR):
        for agent_dir_name in sorted(os.listdir(LOCAL_ARCHIVE_DIR)):
            agent_path = os.path.join(LOCAL_ARCHIVE_DIR, agent_dir_name)
            if not os.path.isdir(agent_path):
                continue
            files = sorted(os.listdir(agent_path))
            print(f"  {agent_dir_name}:")
            for f in files:
                tag = " ✓" if f.endswith(".committed") else (
                      " [stash]" if f.startswith("stash_") else "")
                print(f"    {f}{tag}")

