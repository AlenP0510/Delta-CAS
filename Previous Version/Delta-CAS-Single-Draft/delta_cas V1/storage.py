"""
delta_cas.storage
=================
Low-level file I/O: paths, atomic writes, cross-platform file locking.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time


# ── Cross-platform file lock ──────────────────────────────────

if sys.platform == "win32":
    import msvcrt

    class FileLock:
        """Exclusive file lock using msvcrt (Windows)."""

        def __init__(self, path: str):
            self._path = path + ".lock"
            self._fh   = None

        def __enter__(self) -> "FileLock":
            self._fh = open(self._path, "w")
            while True:
                try:
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.005)
            return self

        def __exit__(self, *_) -> None:
            msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            self._fh.close()

else:
    import fcntl

    class FileLock:
        """Exclusive file lock using fcntl (POSIX)."""

        def __init__(self, path: str):
            self._path = path + ".lock"
            self._fh   = None

        def __enter__(self) -> "FileLock":
            self._fh = open(self._path, "w")
            fcntl.flock(self._fh, fcntl.LOCK_EX)
            return self

        def __exit__(self, *_) -> None:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            self._fh.close()


# ── Atomic JSON helpers ───────────────────────────────────────

def atomic_write_json(path: str, data: dict) -> None:
    """Write JSON atomically via tmp file + os.replace."""
    dir_ = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Path helpers (all relative to state_dir) ──────────────────

def snapshot_path(state_dir: str, index: int) -> str:
    return os.path.join(state_dir, f"S_{index}.json")


def delta_path(state_dir: str, version: int) -> str:
    return os.path.join(state_dir, f"delta_{version:04d}.json")


def meta_path(state_dir: str) -> str:
    return os.path.join(state_dir, "meta.json")


def lock_path(state_dir: str) -> str:
    return os.path.join(state_dir, "meta.json")


def local_agent_dir(archive_dir: str, agent_id: str) -> str:
    path = os.path.join(archive_dir, agent_id)
    os.makedirs(path, exist_ok=True)
    return path


def local_delta_path(archive_dir: str, agent_id: str, version: int) -> str:
    return os.path.join(local_agent_dir(archive_dir, agent_id),
                        f"delta_{version:04d}.json")


def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)
