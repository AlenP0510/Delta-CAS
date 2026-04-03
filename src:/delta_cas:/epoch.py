"""
delta_cas.epoch
===============
Epoch-based alignment: every EPOCH_INTERVAL versions all agents pause,
a new snapshot is compacted, and agents resume from the new base.

Prevents slow agents from lagging arbitrarily behind fast ones.
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class EpochCoordinator:
    """
    Thread-safe epoch coordinator.

    Agents call wait() before writing.
    After every successful write the writer calls check(store, version).
    When version % interval == 0, all writes are paused while a new
    snapshot is compacted, then all agents are unblocked.

    Parameters
    ----------
    interval : int — versions between epoch boundaries. Default 5.
    """

    def __init__(self, interval: int = 5) -> None:
        self.interval = interval
        self._event   = threading.Event()
        self._event.set()   # starts open

    def wait(self) -> None:
        """Block if an epoch compaction is in progress."""
        self._event.wait()

    def check(self, store: "Store", current_version: int) -> None:  # noqa: F821
        """
        Called after every successful CAS write.
        Compacts and realigns all agents at epoch boundaries.
        """
        if current_version % self.interval != 0 or current_version == 0:
            return

        logger.info(
            f"Epoch boundary at v{current_version} — compacting and realigning"
        )
        self._event.clear()
        try:
            from .storage import FileLock, lock_path
            with FileLock(lock_path(store.state_dir)):
                meta = store._load_meta()
                if meta["current_version"] == current_version:
                    store._compact(meta, current_version)
                    meta["epoch"] = meta.get("epoch", 0) + 1
                    store._save_meta(meta)
                    logger.info(
                        f"Epoch {meta['epoch']} complete — "
                        f"new base S_{meta['latest_snapshot_index']} "
                        f"at v{current_version}"
                    )
        finally:
            self._event.set()
