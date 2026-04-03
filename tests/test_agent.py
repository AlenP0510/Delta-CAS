"""Tests for delta_cas.agent: Agent WAL, CAS conflict, intent, epoch, concurrency."""
import os
import threading

import pytest

from delta_cas import Agent, EpochCoordinator, Store, intent_is_valid, noop_intent_fn
from delta_cas.storage import delta_path


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "state"), str(tmp_path / "archive"),
              snapshot_interval=20, circuit_breaker_ratio=None)
    s.init({"score": 0})
    return s


# ── Simple agents ─────────────────────────────────────────────

class Inc(Agent):
    def describe_goal(self): return "Increment score"
    def compute_changes(self, state, version):
        return {"score": state.get("score", 0) + 1}


class Noop(Agent):
    def describe_goal(self): return "Do nothing"
    def compute_changes(self, state, version): return {}


# ── WAL ───────────────────────────────────────────────────────

class TestWAL:
    def test_wal_written(self, store):
        written = []
        class Spy(Inc):
            def _write_wal(self, delta):
                written.append(delta.version)
                super()._write_wal(delta)
        Spy("a", store).run()
        assert len(written) == 1

    def test_committed_marker_on_success(self, store):
        agent = Inc("a", store)
        agent.run()
        committed = [f for f in os.listdir(agent._local_dir)
                     if f.endswith(".committed")]
        assert len(committed) == 1

    def test_transmission_loss_recovery(self, store):
        agent = Inc("a", store)
        _, ver = agent.run()
        lost      = delta_path(store.state_dir, ver)
        committed = agent._committed_marker(ver)
        os.remove(lost)
        os.remove(committed)
        # Re-run — pre-flight recovery should restore
        agent.run()
        assert os.path.exists(lost)

    def test_noop_returns_success(self, store):
        ok, ver = Noop("a", store).run()
        assert ok and ver == 0


# ── Intent ────────────────────────────────────────────────────

class TestIntent:
    def test_noop_valid(self):
        assert intent_is_valid(noop_intent_fn("a", {}, "goal"))

    def test_invalid_prefix(self):
        assert not intent_is_valid("INVALID: dead")
        assert not intent_is_valid("invalid: lowercase")

    def test_agent_aborts_on_invalid(self, store):
        class Bad(Agent):
            def describe_goal(self): return "d"
            def compute_changes(self, s, v): return {"x": 1}
        ok, _ = Bad("b", store, intent_fn=lambda *_: "INVALID: test").run()
        assert not ok
        _, ver = store.read()
        assert ver == 0

    def test_custom_intent_fn_called(self, store):
        calls = []
        def record(agent_id, state, goal):
            calls.append(goal)
            return f"{agent_id}: {goal}"
        class D(Agent):
            def describe_goal(self): return "my goal"
            def compute_changes(self, s, v): return {"x": 1}
        D("d", store, intent_fn=record).run()
        assert calls == ["my goal"]


# ── Epoch ─────────────────────────────────────────────────────

class TestEpoch:
    def test_epoch_fires(self, store):
        epoch = EpochCoordinator(interval=3)
        for _ in range(3):
            Inc("a", store, epoch=epoch).run()
        assert store._load_meta()["epoch"] >= 1

    def test_epoch_event_unblocked(self, store):
        epoch = EpochCoordinator(interval=2)
        Inc("a", store, epoch=epoch).run()
        Inc("b", store, epoch=epoch).run()
        assert epoch._event.is_set()


# ── Concurrent conflict ───────────────────────────────────────

class TestConcurrent:
    def test_both_succeed(self, store):
        results  = {}
        barrier  = threading.Barrier(2)

        def run(aid):
            store.read()         # warm read
            barrier.wait()
            ok, ver = Inc(aid, store).run()
            results[aid] = (ok, ver)

        t1 = threading.Thread(target=run, args=("t1",))
        t2 = threading.Thread(target=run, args=("t2",))
        t1.start(); t2.start()
        t1.join();  t2.join()

        assert results["t1"][0] and results["t2"][0]
        state, _ = store.read()
        assert state["score"] == 2
