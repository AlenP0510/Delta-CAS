"""
Delta-based CAS (Compare-And-Swap) State Management  v2
========================================================

V_curr = V_base + sum(Deltas)

Changes from v1:
  Bug fixes:
    1. _write_lock replaced with cross-process file lock (fcntl / msvcrt)
    2. _save_meta uses atomic write (tmp file + os.rename)
    3. _try_restore_wal_to_shared: "w" mode bug fixed — reads WAL from disk
    4. _compact: reads state BEFORE releasing lock to avoid race condition
    5. Scenario 2 in __main__ now uses real threads to trigger true CAS conflict

  New features:
    6. Intent layer  — State → Intent (via LLM) → Action → State'
       Intent validity is checked before rebase: if new state cannot
       produce the same intent, the intent is regenerated rather than
       blindly replayed.
    7. Epoch alignment — every EPOCH_INTERVAL versions, all agents pause,
       a full snapshot is written, and agents resume from the new base.
       Prevents slow agents from falling arbitrarily far behind.

Architecture:
  State → Intent → Action → Delta → CAS → State'
                     ↑
              (regenerated on conflict if old intent is no longer valid)
"""

import json
import os
import sys
import time
import logging
import tempfile
import threading
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Cross-platform file lock ──────────────────────────────────
# On POSIX (Linux/macOS): fcntl.flock — works across processes
# On Windows: msvcrt.locking — best available without third-party libs

if sys.platform == "win32":
    import msvcrt

    class _FileLock:
        def __init__(self, path: str):
            self._path = path + ".lock"
            self._fh   = None

        def __enter__(self):
            self._fh = open(self._path, "w")
            # Spin-wait — msvcrt.locking raises on conflict
            while True:
                try:
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.005)
            return self

        def __exit__(self, *_):
            msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            self._fh.close()

else:
    import fcntl

    class _FileLock:
        def __init__(self, path: str):
            self._path = path + ".lock"
            self._fh   = None

        def __enter__(self):
            self._fh = open(self._path, "w")
            fcntl.flock(self._fh, fcntl.LOCK_EX)  # blocking exclusive lock
            return self

        def __exit__(self, *_):
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            self._fh.close()


# ── Config ────────────────────────────────────────────────────

STATE_DIR         = "state"
LOCAL_ARCHIVE_DIR = "local_archive"
SNAPSHOT_INTERVAL = 10   # compact every N deltas
EPOCH_INTERVAL    = 5    # align all agents every N versions


# ════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════

class Snapshot:
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
        return Snapshot(version=d["version"], state=d["state"],
                        timestamp=d.get("timestamp"))


class Delta:
    """
    Incremental change on top of base_version.
    checksum covers (agent_id, base_version, changes) for integrity checks.
    """
    def __init__(self, version: int, base_version: int,
                 changes: dict, agent_id: str,
                 trigger: str = "", intent: str = ""):
        self.version      = version
        self.base_version = base_version
        self.changes      = changes
        self.agent_id     = agent_id
        self.trigger      = trigger
        self.intent       = intent          # human-readable intent string
        self.timestamp    = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.checksum     = self._compute_checksum()

    def _compute_checksum(self) -> str:
        payload = json.dumps(
            {"agent_id": self.agent_id,
             "base_version": self.base_version,
             "changes": self.changes},
            sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "type":         "delta",
            "version":      self.version,
            "base_version": self.base_version,
            "changes":      self.changes,
            "agent_id":     self.agent_id,
            "trigger":      self.trigger,
            "intent":       self.intent,
            "timestamp":    self.timestamp,
            "checksum":     self.checksum
        }

    @staticmethod
    def from_dict(d: dict) -> "Delta":
        delta = Delta(
            version      = d["version"],
            base_version = d["base_version"],
            changes      = d["changes"],
            agent_id     = d["agent_id"],
            trigger      = d.get("trigger", ""),
            intent       = d.get("intent", "")
        )
        delta.timestamp = d.get("timestamp", "")
        delta.checksum  = d.get("checksum", "")
        return delta

    def verify_checksum(self) -> bool:
        return self.checksum == self._compute_checksum()


# ════════════════════════════════════════════════════════════════
# Storage layer
# ════════════════════════════════════════════════════════════════

def _ensure_dirs():
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(LOCAL_ARCHIVE_DIR, exist_ok=True)

_ensure_dirs()


def _snapshot_path(index: int) -> str:
    return os.path.join(STATE_DIR, f"S_{index}.json")

def _delta_path(version: int) -> str:
    return os.path.join(STATE_DIR, f"delta_{version:04d}.json")

def _meta_path() -> str:
    return os.path.join(STATE_DIR, "meta.json")

def _lock_path() -> str:
    return os.path.join(STATE_DIR, "meta.json")  # lock file mirrors meta path

def _local_agent_dir(agent_id: str) -> str:
    path = os.path.join(LOCAL_ARCHIVE_DIR, agent_id)
    os.makedirs(path, exist_ok=True)
    return path

def _local_delta_path(agent_id: str, version: int) -> str:
    return os.path.join(_local_agent_dir(agent_id), f"delta_{version:04d}.json")


def _load_meta() -> dict:
    path = _meta_path()
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "current_version":             0,
        "latest_snapshot_version":     0,
        "latest_snapshot_index":       0,
        "total_deltas_since_snapshot": 0,
        "epoch":                       0
    }


def _save_meta(meta: dict):
    """
    Bug fix #2: atomic write via tmp file + os.rename.
    os.rename is atomic on POSIX; on Windows it replaces atomically on NTFS.
    Prevents corrupted meta.json if process dies mid-write.
    """
    path = _meta_path()
    dir_ = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp_path, path)  # atomic on both POSIX and Windows
    except Exception:
        os.unlink(tmp_path)
        raise


def _save_snapshot(snapshot: Snapshot, index: int):
    """Atomic snapshot write."""
    path = _snapshot_path(index)
    dir_ = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(snapshot.to_dict(), f, indent=2)
    os.replace(tmp, path)
    logger.info(f"Snapshot S_{index} saved at version {snapshot.version}")


def _load_snapshot(index: int) -> Snapshot:
    with open(_snapshot_path(index)) as f:
        return Snapshot.from_dict(json.load(f))


def _save_delta(delta: Delta):
    """Write to shared store + agent's local archive simultaneously."""
    data = delta.to_dict()
    with open(_delta_path(delta.version), "w") as f:
        json.dump(data, f, indent=2)
    local_path = _local_delta_path(delta.agent_id, delta.version)
    with open(local_path, "w") as f:
        json.dump(data, f, indent=2)


def _load_delta(version: int) -> Delta:
    with open(_delta_path(version)) as f:
        return Delta.from_dict(json.load(f))


# ════════════════════════════════════════════════════════════════
# Core: read state
# ════════════════════════════════════════════════════════════════

def read_current_state() -> tuple[dict, int]:
    """
    V_curr = V_base + sum(delta_i for i in base+1 ... current)
    Verifies checksum on each delta before applying.
    """
    meta            = _load_meta()
    base_version    = meta["latest_snapshot_version"]
    base_snap_index = meta["latest_snapshot_index"]
    current_version = meta["current_version"]

    if not os.path.exists(_snapshot_path(base_snap_index)):
        return {}, 0

    snapshot = _load_snapshot(base_snap_index)
    state    = json.loads(json.dumps(snapshot.state))

    for v in range(base_version + 1, current_version + 1):
        delta_file = _delta_path(v)
        if not os.path.exists(delta_file):
            logger.warning(f"Missing delta v{v} — skipping (run recovery)")
            continue
        delta = _load_delta(v)
        if not delta.verify_checksum():
            logger.error(f"Checksum mismatch on delta v{v} — skipping")
            continue
        _apply_delta(state, delta.changes)

    return state, current_version


def _apply_delta(state: dict, changes: dict):
    """
    Apply changes in-place. Supports arbitrary-depth dot notation.
    e.g. "goals.goal_001.dimensions.vocab.tension"
    """
    for key, value in changes.items():
        if "." in key:
            head, tail = key.split(".", 1)
            if head not in state or not isinstance(state[head], dict):
                state[head] = {}
            _apply_delta(state[head], {tail: value})
        else:
            state[key] = value


def read_version_at(target_version: int) -> dict:
    """Reconstruct state at any past version."""
    meta = _load_meta()
    best_snap_index, best_snap_version = 0, 0

    for i in range(meta["latest_snapshot_index"] + 1):
        if not os.path.exists(_snapshot_path(i)):
            continue
        snap = _load_snapshot(i)
        if snap.version <= target_version and snap.version >= best_snap_version:
            best_snap_version = snap.version
            best_snap_index   = i

    if not os.path.exists(_snapshot_path(best_snap_index)):
        return {}

    state = json.loads(json.dumps(_load_snapshot(best_snap_index).state))
    for v in range(best_snap_version + 1, target_version + 1):
        if os.path.exists(_delta_path(v)):
            _apply_delta(state, _load_delta(v).changes)
    return state


# ════════════════════════════════════════════════════════════════
# Core: CAS write
# ════════════════════════════════════════════════════════════════

def cas_write(agent_id: str, base_version: int,
              changes: dict, trigger: str = "",
              intent: str = "") -> tuple[bool, int]:
    """
    Cross-process safe CAS write.

    Bug fix #1: uses _FileLock (fcntl/msvcrt) instead of threading.Lock,
    so it works correctly when agents run in separate processes or machines
    sharing a filesystem.

    Bug fix #4: _compact is called while the file lock is still held,
    so no other process can sneak in a write during compaction.
    """
    with _FileLock(_lock_path()):
        meta = _load_meta()
        current_version = meta["current_version"]

        if base_version != current_version:
            logger.warning(
                f"[{agent_id}] CAS conflict: expected base=v{base_version}, "
                f"current=v{current_version}"
            )
            return False, current_version

        new_version = current_version + 1
        delta = Delta(
            version      = new_version,
            base_version = base_version,
            changes      = changes,
            agent_id     = agent_id,
            trigger      = trigger,
            intent       = intent
        )
        _save_delta(delta)

        meta["current_version"]              = new_version
        meta["total_deltas_since_snapshot"] += 1
        _save_meta(meta)

        logger.info(
            f"[{agent_id}] v{base_version} → v{new_version} "
            f"intent='{intent}' changes={list(changes.keys())}"
        )

        # Bug fix #4: compact while lock is held — no race window
        if meta["total_deltas_since_snapshot"] >= SNAPSHOT_INTERVAL:
            _compact(meta, new_version)

        return True, new_version


def _compact(meta: dict, at_version: int):
    """
    Read current state (lock already held by caller) and write new snapshot.
    Resets delta counter. Does NOT re-acquire the file lock.
    """
    state, _ = read_current_state()
    new_index = meta["latest_snapshot_index"] + 1
    _save_snapshot(Snapshot(version=at_version, state=state), new_index)
    meta["latest_snapshot_version"]     = at_version
    meta["latest_snapshot_index"]       = new_index
    meta["total_deltas_since_snapshot"] = 0
    _save_meta(meta)
    logger.info(f"Compacted → S_{new_index} at v{at_version}")


# ════════════════════════════════════════════════════════════════
# Epoch alignment
# Every EPOCH_INTERVAL versions: pause all writes, compact, resume.
# Prevents slow agents from lagging indefinitely behind fast ones.
# ════════════════════════════════════════════════════════════════

_epoch_lock = threading.Event()
_epoch_lock.set()  # starts open (not in epoch)


def _check_epoch(current_version: int):
    """
    Called after every successful CAS write.
    If we've hit an epoch boundary, compact and signal all agents to re-read.
    """
    if current_version % EPOCH_INTERVAL == 0 and current_version > 0:
        logger.info(
            f"Epoch boundary at v{current_version} — "
            f"compacting and realigning all agents"
        )
        _epoch_lock.clear()  # block new writes
        try:
            with _FileLock(_lock_path()):
                meta = _load_meta()
                if meta["current_version"] == current_version:
                    _compact(meta, current_version)
                    meta["epoch"] = meta.get("epoch", 0) + 1
                    _save_meta(meta)
                    logger.info(
                        f"Epoch {meta['epoch']} complete — "
                        f"new base S_{meta['latest_snapshot_index']} at v{current_version}"
                    )
        finally:
            _epoch_lock.set()  # unblock all agents


def _wait_for_epoch():
    """Agents call this before writing. Blocks during epoch compaction."""
    _epoch_lock.wait()


# ════════════════════════════════════════════════════════════════
# Intent layer
# State → Intent (LLM) → validate → Action → Delta
# ════════════════════════════════════════════════════════════════

def generate_intent(agent_id: str, state: dict, goal: str) -> str:
    """
    Ask an LLM to generate an intent string given the current state and goal.
    Intent describes WHAT the agent wants to do and WHY it makes sense
    given the current state.

    In production: call your LLM here.
    For testing: returns a deterministic string.
    """
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": (
                    f"Current world state: {json.dumps(state, indent=2)}\n\n"
                    f"Agent goal: {goal}\n\n"
                    f"In one sentence, describe the agent's intent — "
                    f"what it wants to do and why it makes sense given the state. "
                    f"If the state makes the intent impossible (e.g. the subject is dead), "
                    f"say 'INVALID: <reason>'."
                )
            }]
        )
        return response.content[0].text.strip()
    except Exception:
        # Fallback for testing without API key
        return f"{agent_id} intends to act on: {goal}"


def intent_is_valid(intent: str) -> bool:
    """
    Returns False if the LLM flagged the intent as impossible
    given the current state.
    """
    return not intent.upper().startswith("INVALID")


# ════════════════════════════════════════════════════════════════
# Initialization
# ════════════════════════════════════════════════════════════════

def init_state(initial_state: dict) -> int:
    _save_snapshot(Snapshot(version=0, state=initial_state), index=0)
    _save_meta({
        "current_version":             0,
        "latest_snapshot_version":     0,
        "latest_snapshot_index":       0,
        "total_deltas_since_snapshot": 0,
        "epoch":                       0
    })
    logger.info("State initialized as S_0 at version 0")
    return 0


# ════════════════════════════════════════════════════════════════
# Agent base class
# ════════════════════════════════════════════════════════════════

class Agent:
    """
    Base agent with:
      - WAL (Write-Ahead Log) for crash safety
      - Cross-process CAS via file lock
      - Intent generation and validity check on rebase
      - Epoch awareness (pauses at alignment boundaries)

    Subclass and implement:
      compute_changes(state, version) → dict
      describe_goal()                 → str   (used for intent generation)
    """

    MAX_RETRIES = 5
    RETRY_DELAY = 0.05

    def __init__(self, agent_id: str, local_dir: str = None):
        self.agent_id   = agent_id
        self._local_dir = local_dir or _local_agent_dir(agent_id)
        os.makedirs(self._local_dir, exist_ok=True)

    def compute_changes(self, state: dict, version: int) -> dict:
        raise NotImplementedError

    def describe_goal(self) -> str:
        """Return a plain-language description of what this agent is trying to do."""
        return f"{self.agent_id} goal"

    # ── WAL helpers ───────────────────────────────────────────

    def _wal_path(self, version: int) -> str:
        return os.path.join(self._local_dir, f"wal_{version:04d}.json")

    def _committed_marker(self, version: int) -> str:
        return os.path.join(self._local_dir, f"wal_{version:04d}.committed")

    def _stash_path(self, base_version: int) -> str:
        return os.path.join(self._local_dir, f"stash_{base_version:04d}.json")

    def _write_wal(self, delta: "Delta"):
        with open(self._wal_path(delta.version), "w") as f:
            json.dump(delta.to_dict(), f, indent=2)
        logger.debug(f"[{self.agent_id}] WAL written v{delta.version}")

    def _commit_wal(self, version: int):
        with open(self._committed_marker(version), "w") as f:
            f.write(datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))

    def _stash_delta(self, delta: "Delta", reason: str = "cas_conflict"):
        path = self._stash_path(delta.base_version)
        with open(path, "w") as f:
            json.dump({**delta.to_dict(), "stash_reason": reason}, f, indent=2)
        logger.info(
            f"[{self.agent_id}] Stashed delta base=v{delta.base_version} "
            f"reason={reason}"
        )

    def _recover_wal(self) -> list["Delta"]:
        """Find WAL entries without a .committed marker."""
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

    def _try_restore_wal_to_shared(self, delta: "Delta") -> bool:
        """
        Bug fix #3: read the WAL file from disk rather than re-serialising
        the in-memory delta object (which may differ if it was mutated).
        """
        shared_path = _delta_path(delta.version)
        wal_path    = self._wal_path(delta.version)

        if os.path.exists(shared_path):
            # Verify the existing shared delta matches our WAL
            with open(shared_path) as f:
                shared_data = json.load(f)
            if shared_data.get("checksum") == delta.checksum:
                logger.info(
                    f"[{self.agent_id}] WAL v{delta.version} already in shared store (checksum OK)"
                )
            else:
                logger.error(
                    f"[{self.agent_id}] WAL v{delta.version} in shared store but checksum MISMATCH — "
                    f"another agent may have written to this slot"
                )
            self._commit_wal(delta.version)
            return True

        if not os.path.exists(wal_path):
            logger.error(f"[{self.agent_id}] WAL file missing for v{delta.version}")
            return False

        # Bug fix #3: read from disk, write to shared store
        with open(wal_path, "r") as f:
            wal_data = json.load(f)
        with open(shared_path, "w") as f:
            json.dump(wal_data, f, indent=2)

        self._commit_wal(delta.version)
        logger.warning(
            f"[{self.agent_id}] Restored delta v{delta.version} from WAL to shared store"
        )
        return True

    # ── Main run loop ─────────────────────────────────────────

    def run(self, trigger: str = "") -> tuple[bool, int]:
        """
        Full cycle:
          pre-flight WAL recovery
          → epoch wait
          → read state
          → generate intent (LLM)
          → validate intent against state
          → compute changes
          → WAL write
          → CAS
          → on conflict: stash + rebase + re-validate intent
          → epoch check after success
        """
        # Pre-flight: recover uncommitted WAL entries
        uncommitted = self._recover_wal()
        if uncommitted:
            logger.warning(
                f"[{self.agent_id}] {len(uncommitted)} uncommitted WAL entries — restoring"
            )
            for pending in uncommitted:
                self._try_restore_wal_to_shared(pending)

        for attempt in range(self.MAX_RETRIES):

            # Wait if an epoch alignment is in progress
            _wait_for_epoch()

            # Step 1: read current shared state
            state, base_version = read_current_state()

            # Step 2: generate intent via LLM
            intent = generate_intent(self.agent_id, state, self.describe_goal())
            logger.info(f"[{self.agent_id}] Intent: {intent}")

            # Step 3: validate intent — if state makes it impossible, abort
            if not intent_is_valid(intent):
                logger.warning(
                    f"[{self.agent_id}] Intent invalid given current state — aborting. "
                    f"Reason: {intent}"
                )
                return False, base_version

            # Step 4: compute changes
            changes = self.compute_changes(state, base_version)
            if not changes:
                logger.info(f"[{self.agent_id}] No changes to write")
                return True, base_version

            # Step 5: WAL write before CAS
            optimistic_version = base_version + 1
            pending_delta = Delta(
                version      = optimistic_version,
                base_version = base_version,
                changes      = changes,
                agent_id     = self.agent_id,
                trigger      = trigger,
                intent       = intent
            )
            self._write_wal(pending_delta)

            # Step 6: CAS write
            success, result_version = cas_write(
                agent_id     = self.agent_id,
                base_version = base_version,
                changes      = changes,
                trigger      = trigger,
                intent       = intent
            )

            if success:
                self._commit_wal(result_version)
                logger.info(
                    f"[{self.agent_id}] Write succeeded: "
                    f"v{base_version} → v{result_version} (attempt {attempt+1})"
                )
                # Check epoch boundary
                _check_epoch(result_version)
                return True, result_version

            # ── CAS conflict ──────────────────────────────────
            # Another agent advanced the version.
            # 1. Stash our pending delta
            # 2. Read new state
            # 3. Re-generate intent — the new state may invalidate our old one
            # 4. If still valid, rebase compute_changes on new state
            # 5. Retry

            logger.info(
                f"[{self.agent_id}] CAS conflict attempt {attempt+1} "
                f"(base=v{base_version}, current=v{result_version}) — rebasing"
            )
            self._stash_delta(pending_delta, reason="cas_conflict")

            new_state, new_version = read_current_state()

            # Re-generate intent on the new state
            new_intent = generate_intent(self.agent_id, new_state, self.describe_goal())
            logger.info(f"[{self.agent_id}] Rebased intent: {new_intent}")

            if not intent_is_valid(new_intent):
                # New state makes our goal impossible — stash and abort
                logger.warning(
                    f"[{self.agent_id}] Intent invalid after rebase — aborting. "
                    f"Reason: {new_intent}"
                )
                return False, new_version

            # Intent is still valid — recompute changes on new state
            rebased_changes = self.compute_changes(new_state, new_version)
            if not rebased_changes:
                logger.info(f"[{self.agent_id}] No changes needed after rebase on v{new_version}")
                return True, new_version

            rebased_delta = Delta(
                version      = new_version + 1,
                base_version = new_version,
                changes      = rebased_changes,
                agent_id     = self.agent_id,
                trigger      = f"{trigger}[rebased_on_v{new_version}]",
                intent       = new_intent
            )
            self._write_wal(rebased_delta)

            success2, result_version2 = cas_write(
                agent_id     = self.agent_id,
                base_version = new_version,
                changes      = rebased_changes,
                trigger      = rebased_delta.trigger,
                intent       = new_intent
            )

            if success2:
                self._commit_wal(result_version2)
                logger.info(
                    f"[{self.agent_id}] Rebased write succeeded: "
                    f"v{new_version} → v{result_version2}"
                )
                _check_epoch(result_version2)
                return True, result_version2

            time.sleep(self.RETRY_DELAY * (attempt + 1))

        logger.error(f"[{self.agent_id}] Max retries reached")
        return False, -1


# ════════════════════════════════════════════════════════════════
# Global recovery utility
# ════════════════════════════════════════════════════════════════

def recover_missing_deltas() -> list[int]:
    """
    Scan all agent local archives and restore missing deltas to shared store.
    Call on startup after a crash.
    """
    if not os.path.exists(LOCAL_ARCHIVE_DIR):
        return []

    recovered = []
    for agent_id in os.listdir(LOCAL_ARCHIVE_DIR):
        agent_dir = os.path.join(LOCAL_ARCHIVE_DIR, agent_id)
        if not os.path.isdir(agent_dir):
            continue
        for fname in sorted(os.listdir(agent_dir)):
            if not (fname.startswith("delta_") and fname.endswith(".json")):
                continue
            shared_path = os.path.join(STATE_DIR, fname)
            local_path  = os.path.join(agent_dir, fname)
            if not os.path.exists(shared_path):
                try:
                    with open(local_path) as f:
                        data = json.load(f)
                    # Verify checksum before restoring
                    d = Delta.from_dict(data)
                    if not d.verify_checksum():
                        logger.error(f"Checksum fail on local {fname} from [{agent_id}] — skipping")
                        continue
                    with open(shared_path, "w") as f:
                        json.dump(data, f, indent=2)
                    logger.warning(f"Recovered v{d.version} from [{agent_id}]")
                    recovered.append(d.version)
                except Exception as e:
                    logger.error(f"Failed to recover {fname} from [{agent_id}]: {e}")

    return recovered


def read_history(from_version: int = 0) -> list[dict]:
    meta    = _load_meta()
    current = meta["current_version"]
    history = []
    for v in range(from_version + 1, current + 1):
        path = _delta_path(v)
        if os.path.exists(path):
            with open(path) as f:
                history.append(json.load(f))
    return history


# ════════════════════════════════════════════════════════════════
# Example usage
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import shutil

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    for d in [STATE_DIR, LOCAL_ARCHIVE_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
    _ensure_dirs()

    init_state({"global_tension": 0.0, "goals": {}, "tasks": [], "person": {}})
    print("\n── S_0 initialized ──\n")

    # ── Define agents ──────────────────────────────────────────

    class TensionAgent(Agent):
        def __init__(self, agent_id, goal_id, tension):
            super().__init__(agent_id)
            self.goal_id = goal_id
            self.tension = tension
        def describe_goal(self):
            return f"Set tension for {self.goal_id} to {self.tension}"
        def compute_changes(self, state, version):
            return {f"goals.{self.goal_id}.tension": self.tension}

    class TaskAgent(Agent):
        def __init__(self, agent_id, task):
            super().__init__(agent_id)
            self.task = task
        def describe_goal(self):
            return f"Add task: {self.task}"
        def compute_changes(self, state, version):
            tasks = state.get("tasks", [])
            return {"tasks": tasks + [self.task]} if self.task not in tasks else {}

    class WalkAgent(Agent):
        def describe_goal(self):
            return "Make the person start walking"
        def compute_changes(self, state, version):
            return {"person.activity": "walking"}

    class EatAgent(Agent):
        def describe_goal(self):
            return "Make the person eat — only valid if they are alive and not already eating"
        def compute_changes(self, state, version):
            activity = state.get("person", {}).get("activity", "idle")
            # Semantic merge: if walking, go eat while walking
            if activity == "walking":
                return {"person.activity": "walking to eat"}
            return {"person.activity": "eating"}

    class KillAgent(Agent):
        def describe_goal(self):
            return "The person has died"
        def compute_changes(self, state, version):
            return {"person.alive": False, "person.activity": "dead"}

    # ════════════════════════════════════════════════════════════
    # Scenario 1: Normal run + epoch alignment at v5
    # ════════════════════════════════════════════════════════════

    print("════ Scenario 1: Normal run (epoch alignment at v5) ════")
    for agent in [
        TensionAgent("agent_gre",     "goal_gre",     0.72),
        TensionAgent("agent_fitness", "goal_fitness",  0.31),
        TaskAgent("agent_task_1", "Prepare GRE"),
        TaskAgent("agent_task_2", "Morning run"),
        TensionAgent("agent_vocab",   "goal_vocab",    0.55),
    ]:
        success, v = agent.run(trigger="scenario1")
        print(f"  {agent.agent_id}: {'✓' if success else '✗'} → v{v}")

    meta = _load_meta()
    print(f"\n  Epoch after scenario 1: {meta['epoch']}")
    print(f"  Latest snapshot: S_{meta['latest_snapshot_index']} at v{meta['latest_snapshot_version']}")

    # ════════════════════════════════════════════════════════════
    # Scenario 2: True CAS conflict via threads
    # Both threads read v5, agent_A writes first, agent_B must rebase
    # ════════════════════════════════════════════════════════════

    print("\n════ Scenario 2: True CAS conflict via threads ════")

    barrier = threading.Barrier(2)
    results = {}

    def run_agent(agent, label):
        # Both threads read state simultaneously
        state, base = read_current_state()
        barrier.wait()  # synchronise — both now have the same base_version
        success, v = agent.run(trigger=f"concurrent_{label}")
        results[label] = (success, v)

    walk_agent = WalkAgent("agent_walk")
    eat_agent  = EatAgent("agent_eat")

    t1 = threading.Thread(target=run_agent, args=(walk_agent, "walk"))
    t2 = threading.Thread(target=run_agent, args=(eat_agent,  "eat"))
    t1.start(); t2.start()
    t1.join();  t2.join()

    print(f"  walk: {'✓' if results['walk'][0] else '✗'} → v{results['walk'][1]}")
    print(f"  eat:  {'✓' if results['eat'][0] else '✗'}  → v{results['eat'][1]}")

    state, v = read_current_state()
    print(f"  person.activity = '{state.get('person', {}).get('activity')}' at v{v}")

    # ════════════════════════════════════════════════════════════
    # Scenario 3: Dead person — intent validation blocks EatAgent
    # KillAgent writes first → EatAgent's intent becomes INVALID
    # ════════════════════════════════════════════════════════════

    print("\n════ Scenario 3: Dead person — intent invalidation ════")

    kill_agent = KillAgent("agent_kill")
    kill_agent.run(trigger="kill")

    state, v = read_current_state()
    print(f"  State after kill: alive={state.get('person',{}).get('alive')} v{v}")

    # EatAgent tries to act — intent should be flagged INVALID by LLM
    eat_agent2 = EatAgent("agent_eat_after_kill")
    success, v2 = eat_agent2.run(trigger="eat_after_kill")
    print(f"  EatAgent after kill: {'✓ (unexpected)' if success else '✗ correctly blocked'} v{v2}")

    # ════════════════════════════════════════════════════════════
    # Scenario 4: Transmission loss → WAL recovery
    # ════════════════════════════════════════════════════════════

    print("\n════ Scenario 4: Transmission loss → WAL recovery ════")

    recovery_agent = TaskAgent("agent_recovery", "WAL recovery task")
    _, written_v = recovery_agent.run(trigger="before_loss")
    print(f"  Wrote delta v{written_v}")

    # Simulate: shared delta lost + committed marker removed
    lost = _delta_path(written_v)
    committed = recovery_agent._committed_marker(written_v)
    if os.path.exists(lost):
        os.remove(lost)
    if os.path.exists(committed):
        os.remove(committed)
    print(f"  Simulated loss: deleted delta v{written_v} from shared store")

    _, recovered_v = recovery_agent.run(trigger="after_recovery")
    restored = os.path.exists(_delta_path(written_v))
    print(f"  Delta v{written_v} restored: {restored}")
    print(f"  Agent continued from v{recovered_v}")

    # ── Final state ──
    print("\n════ Final state ════")
    state, version = read_current_state()
    print(json.dumps(state, indent=2))

    print("\n════ Delta history ════")
    for e in read_history():
        intent_str = f" | intent: {e.get('intent','')[:60]}" if e.get("intent") else ""
        print(f"  v{e['version']:02d} [{e['agent_id']:<25}] "
              f"base=v{e['base_version']:02d}{intent_str}")

    print("\n════ Epoch summary ════")
    final_meta = _load_meta()
    print(f"  Total versions: {final_meta['current_version']}")
    print(f"  Epochs completed: {final_meta['epoch']}")
    print(f"  Latest snapshot: S_{final_meta['latest_snapshot_index']} "
          f"at v{final_meta['latest_snapshot_version']}")
