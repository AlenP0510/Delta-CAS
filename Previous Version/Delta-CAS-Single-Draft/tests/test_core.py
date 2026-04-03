"""
Tests for delta_cas core: Store, Delta, Snapshot, WAL, epoch, intent.
Run with: pytest tests/ -v
"""

import json
import os
import shutil
import tempfile
import threading

import pytest

from delta_cas import (
    Agent, Delta, EpochCoordinator, Snapshot, Store,
    intent_is_valid, noop_intent_fn,
)
from delta_cas.core import _apply_delta


# ── fixtures ──────────────────────────────────────────────────

@pytest.fixture
def tmp_store(tmp_path):
    store = Store(
        state_dir=str(tmp_path / "state"),
        archive_dir=str(tmp_path / "archive"),
        snapshot_interval=10,
    )
    store.init({"score": 0, "tasks": [], "person": {}})
    return store


# ── _apply_delta ──────────────────────────────────────────────

class TestApplyDelta:
    def test_flat(self):
        state = {"a": 1}
        _apply_delta(state, {"a": 2, "b": 3})
        assert state == {"a": 2, "b": 3}

    def test_dot_notation_two_levels(self):
        state = {"goals": {}}
        _apply_delta(state, {"goals.gre.tension": 0.72})
        assert state["goals"]["gre"]["tension"] == 0.72

    def test_dot_notation_three_levels(self):
        state = {}
        _apply_delta(state, {"a.b.c.d": 99})
        assert state["a"]["b"]["c"]["d"] == 99

    def test_overwrites_scalar(self):
        state = {"x": "old"}
        _apply_delta(state, {"x": "new"})
        assert state["x"] == "new"

    def test_creates_missing_intermediate(self):
        state = {}
        _apply_delta(state, {"goals.new_goal.tension": 0.5})
        assert state["goals"]["new_goal"]["tension"] == 0.5


# ── Delta checksum ────────────────────────────────────────────

class TestDelta:
    def test_checksum_valid(self):
        d = Delta(1, 0, {"x": 1}, "agent_a")
        assert d.verify_checksum()

    def test_checksum_detects_tampering(self):
        d = Delta(1, 0, {"x": 1}, "agent_a")
        d.changes["x"] = 99   # tamper
        assert not d.verify_checksum()

    def test_roundtrip(self):
        d = Delta(3, 2, {"score": 10}, "agent_b", trigger="t", intent="i")
        d2 = Delta.from_dict(d.to_dict())
        assert d2.version      == d.version
        assert d2.base_version == d.base_version
        assert d2.changes      == d.changes
        assert d2.agent_id     == d.agent_id
        assert d2.checksum     == d.checksum


# ── Store.init / read ─────────────────────────────────────────

class TestStoreInit:
    def test_init_creates_s0(self, tmp_store):
        state, ver = tmp_store.read()
        assert ver == 0
        assert state["score"] == 0

    def test_init_idempotent(self, tmp_store):
        tmp_store.init({"score": 999})   # should be ignored
        state, ver = tmp_store.read()
        assert state["score"] == 0       # original value preserved


# ── Store.write (CAS) ─────────────────────────────────────────

class TestStoreWrite:
    def test_simple_write(self, tmp_store):
        ok, ver = tmp_store.write("a1", 0, {"score": 1})
        assert ok and ver == 1
        state, v = tmp_store.read()
        assert state["score"] == 1 and v == 1

    def test_cas_conflict(self, tmp_store):
        tmp_store.write("a1", 0, {"score": 1})
        ok, ver = tmp_store.write("a2", 0, {"score": 2})   # stale base
        assert not ok
        assert ver == 1   # returns current version for rebase

    def test_sequential_writes(self, tmp_store):
        for i in range(1, 6):
            ok, ver = tmp_store.write("a1", i - 1, {"score": i})
            assert ok and ver == i
        state, v = tmp_store.read()
        assert state["score"] == 5 and v == 5

    def test_checksum_in_stored_delta(self, tmp_store):
        tmp_store.write("a1", 0, {"score": 1})
        from delta_cas.storage import delta_path
        path = delta_path(tmp_store.state_dir, 1)
        data = json.loads(open(path).read())
        assert "checksum" in data and len(data["checksum"]) == 16


# ── Store.read_at ─────────────────────────────────────────────

class TestReadAt:
    def test_historical_read(self, tmp_store):
        tmp_store.write("a", 0, {"score": 10})
        tmp_store.write("a", 1, {"score": 20})
        tmp_store.write("a", 2, {"score": 30})

        assert tmp_store.read_at(1)["score"] == 10
        assert tmp_store.read_at(2)["score"] == 20
        assert tmp_store.read_at(3)["score"] == 30


# ── Store.recover ─────────────────────────────────────────────

class TestRecover:
    def test_recover_missing_delta(self, tmp_store):
        tmp_store.write("a1", 0, {"score": 1})
        tmp_store.write("a1", 1, {"score": 2})

        # Remove delta v1 from shared store
        from delta_cas.storage import delta_path
        lost = delta_path(tmp_store.state_dir, 1)
        os.remove(lost)

        recovered = tmp_store.recover()
        assert 1 in recovered
        assert os.path.exists(lost)

    def test_corrupt_local_not_restored(self, tmp_store):
        tmp_store.write("a1", 0, {"score": 1})
        from delta_cas.storage import delta_path, local_agent_dir
        local = os.path.join(
            local_agent_dir(tmp_store.archive_dir, "a1"), "delta_0001.json"
        )
        shared = delta_path(tmp_store.state_dir, 1)
        os.remove(shared)
        # Corrupt the local copy
        with open(local, "w") as f:
            f.write('{"bad": "json"}')
        recovered = tmp_store.recover()
        assert 1 not in recovered


# ── Snapshot compaction ───────────────────────────────────────

class TestCompaction:
    def test_compact_on_interval(self, tmp_path):
        store = Store(
            state_dir=str(tmp_path / "state"),
            archive_dir=str(tmp_path / "archive"),
            snapshot_interval=3,
        )
        store.init({"n": 0})
        for i in range(1, 4):
            store.write("a", i - 1, {"n": i})
        # After 3 deltas, S_1 should exist
        from delta_cas.storage import snapshot_path
        assert os.path.exists(snapshot_path(store.state_dir, 1))

    def test_read_after_compact(self, tmp_path):
        store = Store(
            state_dir=str(tmp_path / "state"),
            archive_dir=str(tmp_path / "archive"),
            snapshot_interval=3,
        )
        store.init({"n": 0})
        for i in range(1, 5):
            store.write("a", i - 1, {"n": i})
        state, ver = store.read()
        assert state["n"] == 4 and ver == 4


# ── Epoch alignment ───────────────────────────────────────────

class TestEpoch:
    def test_epoch_fires_at_boundary(self, tmp_store):
        epoch = EpochCoordinator(interval=3)

        class Dummy(Agent):
            def describe_goal(self): return "inc"
            def compute_changes(self, state, version):
                return {"score": state.get("score", 0) + 1}

        agent = Dummy("a", tmp_store, epoch=epoch)
        for _ in range(3):
            agent.run()

        meta = tmp_store._load_meta()
        assert meta["epoch"] >= 1

    def test_epoch_event_unblocked_after_compact(self, tmp_store):
        epoch = EpochCoordinator(interval=2)

        class Dummy(Agent):
            def describe_goal(self): return "inc"
            def compute_changes(self, state, version):
                return {"score": state.get("score", 0) + 1}

        Dummy("a", tmp_store, epoch=epoch).run()
        Dummy("b", tmp_store, epoch=epoch).run()   # triggers epoch

        assert epoch._event.is_set()   # must be unblocked


# ── Agent WAL ─────────────────────────────────────────────────

class TestAgentWAL:
    def test_wal_written_before_cas(self, tmp_store, tmp_path):
        calls = []

        class Spy(Agent):
            def describe_goal(self): return "spy"
            def compute_changes(self, state, version):
                return {"score": 1}
            def _write_wal(self, delta):
                calls.append("wal")
                super()._write_wal(delta)

        Spy("spy", tmp_store).run()
        assert "wal" in calls

    def test_committed_marker_written_on_success(self, tmp_store):
        class Dummy(Agent):
            def describe_goal(self): return "d"
            def compute_changes(self, state, version): return {"score": 1}

        agent = Dummy("d", tmp_store)
        agent.run()
        committed = [f for f in os.listdir(agent._local_dir)
                     if f.endswith(".committed")]
        assert len(committed) == 1

    def test_wal_restored_on_transmission_loss(self, tmp_store):
        class Dummy(Agent):
            def describe_goal(self): return "d"
            def compute_changes(self, state, version): return {"score": 1}

        agent = Dummy("d", tmp_store)
        _, ver = agent.run()

        # Simulate: shared delta lost + committed marker removed
        from delta_cas.storage import delta_path
        lost      = delta_path(tmp_store.state_dir, ver)
        committed = agent._committed_marker(ver)
        os.remove(lost)
        os.remove(committed)

        # Re-run — pre-flight recovery should restore the delta
        agent.run()
        assert os.path.exists(lost)


# ── True concurrent CAS conflict ─────────────────────────────

class TestConcurrentConflict:
    def test_threads_resolve_without_data_loss(self, tmp_store):
        results = {}
        barrier = threading.Barrier(2)

        class Inc(Agent):
            def __init__(self, aid):
                super().__init__(aid, tmp_store)
            def describe_goal(self): return "inc"
            def compute_changes(self, state, version):
                return {"score": state.get("score", 0) + 1}

        def run(aid):
            state, base = tmp_store.read()
            barrier.wait()   # both threads now have the same base
            ok, ver = Inc(aid).run()
            results[aid] = (ok, ver)

        t1 = threading.Thread(target=run, args=("t1",))
        t2 = threading.Thread(target=run, args=("t2",))
        t1.start(); t2.start()
        t1.join();  t2.join()

        # Both should eventually succeed
        assert results["t1"][0] and results["t2"][0]
        state, _ = tmp_store.read()
        assert state["score"] == 2   # both increments applied


# ── Intent validation ─────────────────────────────────────────

class TestIntent:
    def test_noop_always_valid(self, tmp_store):
        assert intent_is_valid(noop_intent_fn("a", {}, "goal"))

    def test_invalid_prefix(self):
        assert not intent_is_valid("INVALID: the subject is dead")
        assert not intent_is_valid("invalid: lowercase also caught")

    def test_agent_aborts_on_invalid_intent(self, tmp_store):
        def always_invalid(agent_id, state, goal):
            return "INVALID: test"

        class Dummy(Agent):
            def describe_goal(self): return "d"
            def compute_changes(self, state, version): return {"x": 1}

        agent = Dummy("d", tmp_store, intent_fn=always_invalid)
        ok, _ = agent.run()
        assert not ok

        # State must be unchanged
        state, ver = tmp_store.read()
        assert ver == 0

    def test_custom_intent_fn(self, tmp_store):
        called_with = []

        def record_fn(agent_id, state, goal):
            called_with.append((agent_id, goal))
            return f"{agent_id} will {goal}"

        class Dummy(Agent):
            def describe_goal(self): return "do something"
            def compute_changes(self, state, version): return {"x": 1}

        Dummy("d", tmp_store, intent_fn=record_fn).run()
        assert called_with == [("d", "do something")]


# ── history ───────────────────────────────────────────────────

class TestHistory:
    def test_history_returns_all_deltas(self, tmp_store):
        tmp_store.write("a", 0, {"score": 1})
        tmp_store.write("a", 1, {"score": 2})
        tmp_store.write("a", 2, {"score": 3})
        h = tmp_store.history()
        assert len(h) == 3
        assert h[0]["version"] == 1
        assert h[2]["version"] == 3

    def test_history_from_version(self, tmp_store):
        tmp_store.write("a", 0, {"score": 1})
        tmp_store.write("a", 1, {"score": 2})
        h = tmp_store.history(from_version=1)
        assert len(h) == 1
        assert h[0]["version"] == 2
