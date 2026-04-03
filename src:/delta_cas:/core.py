"""
delta_cas.core
==============
Snapshot, Delta, and Store — the CAS read/write/compact engine.

V_curr = V_base + sum(Delta_i  for i in base+1 … current)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any

from .storage import (
    FileLock, atomic_write_json, read_json,
    snapshot_path, delta_path, meta_path, lock_path,
    local_delta_path, ensure_dirs,
)

logger = logging.getLogger(__name__)

_EMPTY_META: dict = {
    "current_version":             0,
    "latest_snapshot_version":     0,
    "latest_snapshot_index":       0,
    "total_deltas_since_snapshot": 0,
    "epoch":                       0,
}


# ════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════

class Snapshot:
    """Full compacted state at a specific version (S_0, S_1, …)."""

    def __init__(self, version: int, state: dict,
                 timestamp: str | None = None) -> None:
        self.version   = version
        self.state     = state
        self.timestamp = timestamp or _now()

    def to_dict(self) -> dict:
        return {"type": "snapshot", "version": self.version,
                "state": self.state, "timestamp": self.timestamp}

    @staticmethod
    def from_dict(d: dict) -> "Snapshot":
        return Snapshot(version=d["version"], state=d["state"],
                        timestamp=d.get("timestamp"))


class Delta:
    """
    Incremental change on top of base_version.

    Fields
    ------
    version      : int   — version this delta produces
    base_version : int   — version this delta was computed from
    changes      : dict  — {key: new_value}, dot-notation for nesting
    agent_id     : str
    trigger      : str   — short human-readable description
    intent       : str   — semantic intent string (from Intent layer)
    checksum     : str   — sha256[:16] of (agent_id, base_version, changes)
    """

    def __init__(self, version: int, base_version: int,
                 changes: dict, agent_id: str,
                 trigger: str = "", intent: str = "") -> None:
        self.version      = version
        self.base_version = base_version
        self.changes      = changes
        self.agent_id     = agent_id
        self.trigger      = trigger
        self.intent       = intent
        self.timestamp    = _now()
        self.checksum     = self._compute_checksum()

    def _compute_checksum(self) -> str:
        payload = json.dumps(
            {"agent_id": self.agent_id,
             "base_version": self.base_version,
             "changes": self.changes},
            sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def verify_checksum(self) -> bool:
        return self.checksum == self._compute_checksum()

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
            "checksum":     self.checksum,
        }

    @staticmethod
    def from_dict(d: dict) -> "Delta":
        delta = Delta(
            version=d["version"], base_version=d["base_version"],
            changes=d["changes"], agent_id=d["agent_id"],
            trigger=d.get("trigger", ""), intent=d.get("intent", ""),
        )
        delta.timestamp = d.get("timestamp", "")
        delta.checksum  = d.get("checksum", "")
        return delta


# ════════════════════════════════════════════════════════════════
# Store
# ════════════════════════════════════════════════════════════════

class Store:
    """
    Delta-CAS state store.

    Parameters
    ----------
    state_dir             : shared directory for snapshots, deltas, meta
    archive_dir           : per-agent local WAL archives
    snapshot_interval     : hard-cap compact every N deltas
    circuit_breaker_ratio : compact when delta_bytes/snapshot_bytes >= ratio.
                            Set to None to disable. Default 0.8 (80%).
    """

    def __init__(self, state_dir: str,
                 archive_dir: str = "local_archive",
                 snapshot_interval: int = 20,
                 circuit_breaker_ratio: float | None = 0.8) -> None:
        self.state_dir             = state_dir
        self.archive_dir           = archive_dir
        self.snapshot_interval     = snapshot_interval
        self.circuit_breaker_ratio = circuit_breaker_ratio
        ensure_dirs(state_dir, archive_dir)

    # ── paths ─────────────────────────────────────────────────

    def _snap_path(self, i: int) -> str:
        return snapshot_path(self.state_dir, i)

    def _delta_path(self, v: int) -> str:
        return delta_path(self.state_dir, v)

    def _meta_path(self) -> str:
        return meta_path(self.state_dir)

    def _lock_path(self) -> str:
        return lock_path(self.state_dir)

    def _local_delta(self, agent_id: str, version: int) -> str:
        return local_delta_path(self.archive_dir, agent_id, version)

    # ── meta ──────────────────────────────────────────────────

    def _load_meta(self) -> dict:
        p = self._meta_path()
        return read_json(p) if os.path.exists(p) else dict(_EMPTY_META)

    def _save_meta(self, meta: dict) -> None:
        atomic_write_json(self._meta_path(), meta)

    # ── snapshots ─────────────────────────────────────────────

    def _save_snapshot(self, snap: Snapshot, index: int) -> None:
        atomic_write_json(self._snap_path(index), snap.to_dict())
        logger.info(f"Snapshot S_{index} saved at v{snap.version}")

    def _load_snapshot(self, index: int) -> Snapshot:
        return Snapshot.from_dict(read_json(self._snap_path(index)))

    # ── deltas ────────────────────────────────────────────────

    def _save_delta(self, delta: Delta) -> None:
        data = delta.to_dict()
        with open(self._delta_path(delta.version), "w") as f:
            json.dump(data, f, indent=2)
        with open(self._local_delta(delta.agent_id, delta.version), "w") as f:
            json.dump(data, f, indent=2)

    def _load_delta(self, version: int) -> Delta:
        return Delta.from_dict(read_json(self._delta_path(version)))

    # ── circuit breaker ───────────────────────────────────────

    def _delta_ratio(self, changes: dict, meta: dict) -> float | None:
        """delta_bytes / snapshot_bytes — None if unavailable or disabled."""
        if self.circuit_breaker_ratio is None:
            return None
        snap_p = self._snap_path(meta["latest_snapshot_index"])
        if not os.path.exists(snap_p):
            return None
        try:
            snap_bytes  = os.path.getsize(snap_p)
            delta_bytes = len(json.dumps(changes, separators=(",", ":")))
            return delta_bytes / snap_bytes if snap_bytes else None
        except OSError:
            return None

    # ── public API ────────────────────────────────────────────

    def init(self, initial_state: dict) -> None:
        """
        Create S_0. No-op if S_0 already exists — safe to call on every startup.
        """
        if os.path.exists(self._snap_path(0)):
            return
        self._save_snapshot(Snapshot(version=0, state=initial_state), 0)
        self._save_meta(dict(_EMPTY_META))
        logger.info("Store initialized: S_0 at v0")

    def read(self) -> tuple[dict, int]:
        """
        Reconstruct current state.
        V_curr = V_base + sum(deltas from base+1 to current).
        Verifies checksums; skips corrupted deltas with a warning.

        Returns (state, current_version).
        """
        meta            = self._load_meta()
        base_version    = meta["latest_snapshot_version"]
        base_snap_index = meta["latest_snapshot_index"]
        current_version = meta["current_version"]

        if not os.path.exists(self._snap_path(base_snap_index)):
            return {}, 0

        state = json.loads(json.dumps(
            self._load_snapshot(base_snap_index).state
        ))

        for v in range(base_version + 1, current_version + 1):
            path = self._delta_path(v)
            if not os.path.exists(path):
                logger.warning(f"Missing delta v{v} — run store.recover()")
                continue
            delta = self._load_delta(v)
            if not delta.verify_checksum():
                logger.error(f"Checksum mismatch on delta v{v} — skipping")
                continue
            _apply_delta(state, delta.changes)

        return state, current_version

    def read_at(self, target_version: int) -> dict:
        """Reconstruct state at any historical version."""
        meta = self._load_meta()
        best_index, best_version = 0, 0

        for i in range(meta["latest_snapshot_index"] + 1):
            if not os.path.exists(self._snap_path(i)):
                continue
            snap = self._load_snapshot(i)
            if snap.version <= target_version and snap.version >= best_version:
                best_version = snap.version
                best_index   = i

        if not os.path.exists(self._snap_path(best_index)):
            return {}

        state = json.loads(json.dumps(self._load_snapshot(best_index).state))
        for v in range(best_version + 1, target_version + 1):
            if os.path.exists(self._delta_path(v)):
                _apply_delta(state, self._load_delta(v).changes)
        return state

    def write(self, agent_id: str, base_version: int,
              changes: dict, trigger: str = "",
              intent: str = "") -> tuple[bool, int]:
        """
        Atomic CAS write.

        1. Acquire file lock (cross-process safe).
        2. Check base_version == current_version (CAS).
        3. Write delta to shared store + agent local archive.
        4. Update meta atomically.
        5. Check hard-cap compaction (snapshot_interval).
        6. Check circuit-breaker compaction (circuit_breaker_ratio).
        7. Release lock.

        Returns (success, new_version).
        On conflict: success=False, new_version=current_version (for rebase).
        """
        with FileLock(self._lock_path()):
            meta    = self._load_meta()
            current = meta["current_version"]

            if base_version != current:
                logger.warning(
                    f"[{agent_id}] CAS conflict: "
                    f"expected v{base_version}, current v{current}"
                )
                return False, current

            new_version = current + 1
            delta = Delta(version=new_version, base_version=base_version,
                          changes=changes, agent_id=agent_id,
                          trigger=trigger, intent=intent)
            self._save_delta(delta)

            meta["current_version"]              = new_version
            meta["total_deltas_since_snapshot"] += 1
            self._save_meta(meta)

            logger.info(
                f"[{agent_id}] v{base_version} → v{new_version} "
                f"changes={list(changes.keys())}"
            )

            # Hard-cap compaction
            if meta["total_deltas_since_snapshot"] >= self.snapshot_interval:
                logger.info(
                    f"[{agent_id}] Hard-cap compaction "
                    f"(count={meta['total_deltas_since_snapshot']})"
                )
                self._compact(meta, new_version)

            # Circuit-breaker compaction
            else:
                ratio = self._delta_ratio(changes, meta)
                if (ratio is not None
                        and self.circuit_breaker_ratio is not None
                        and ratio >= self.circuit_breaker_ratio):
                    logger.warning(
                        f"[{agent_id}] Circuit breaker: "
                        f"delta/snapshot={ratio:.1%} >= "
                        f"{self.circuit_breaker_ratio:.0%} — compacting"
                    )
                    self._compact(meta, new_version)

            return True, new_version

    def history(self, from_version: int = 0) -> list[dict]:
        """Return all delta dicts from from_version+1 onwards, in order."""
        meta    = self._load_meta()
        current = meta["current_version"]
        result  = []
        for v in range(from_version + 1, current + 1):
            path = self._delta_path(v)
            if os.path.exists(path):
                result.append(read_json(path))
        return result

    def recover(self) -> list[int]:
        """
        Scan all agent local archives for delta files missing from shared store.
        Checksums are verified before restoring.
        Returns list of recovered version numbers.
        """
        recovered = []
        if not os.path.exists(self.archive_dir):
            return recovered

        for agent_id in os.listdir(self.archive_dir):
            agent_dir = os.path.join(self.archive_dir, agent_id)
            if not os.path.isdir(agent_dir):
                continue
            for fname in sorted(os.listdir(agent_dir)):
                if not (fname.startswith("delta_") and fname.endswith(".json")):
                    continue
                shared = os.path.join(self.state_dir, fname)
                local  = os.path.join(agent_dir, fname)
                if os.path.exists(shared):
                    continue
                try:
                    data = read_json(local)
                    d    = Delta.from_dict(data)
                    if not d.verify_checksum():
                        logger.error(
                            f"Bad checksum on local {fname} [{agent_id}] — skipped"
                        )
                        continue
                    with open(shared, "w") as f:
                        json.dump(data, f, indent=2)
                    logger.warning(
                        f"Recovered v{d.version} from [{agent_id}]"
                    )
                    recovered.append(d.version)
                except Exception as e:
                    logger.error(
                        f"Recovery failed {fname} [{agent_id}]: {e}"
                    )
        return recovered

    def compact_now(self) -> None:
        """Force compaction immediately, regardless of delta count or ratio."""
        with FileLock(self._lock_path()):
            meta = self._load_meta()
            self._compact(meta, meta["current_version"])

    # ── internal ──────────────────────────────────────────────

    def _compact(self, meta: dict, at_version: int) -> None:
        """Rebuild full state from scratch and write new snapshot.
        Called while file lock is held."""
        state, _ = self.read()
        new_index = meta["latest_snapshot_index"] + 1
        self._save_snapshot(Snapshot(version=at_version, state=state), new_index)
        meta["latest_snapshot_version"]     = at_version
        meta["latest_snapshot_index"]       = new_index
        meta["total_deltas_since_snapshot"] = 0
        self._save_meta(meta)
        logger.info(f"Compacted → S_{new_index} at v{at_version}")


# ── helpers ───────────────────────────────────────────────────

def _apply_delta(state: dict, changes: dict) -> None:
    """Apply changes dict in-place. Supports arbitrary-depth dot notation."""
    for key, value in changes.items():
        if "." in key:
            head, tail = key.split(".", 1)
            if head not in state or not isinstance(state[head], dict):
                state[head] = {}
            _apply_delta(state[head], {tail: value})
        else:
            state[key] = value


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
