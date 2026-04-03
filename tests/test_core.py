"""Tests for delta_cas.core: Store, Delta, Snapshot, compaction, circuit breaker."""
import json
import os
import shutil
import tempfile

import pytest

from delta_cas import Delta, Snapshot, Store
from delta_cas.core import _apply_delta
from delta_cas.storage import delta_path, snapshot_path


@pytest.fixture
def store(tmp_path):
    s = Store(
        state_dir=str(tmp_path / "state"),
        archive_dir=str(tmp_path / "archive"),
        snapshot_interval=20,
        circuit_breaker_ratio=0.8,
    )
    s.init({"score": 0, "tasks": []})
    return s


# ── _apply_delta ──────────────────────────────────────────────

class TestApplyDelta:
    def test_flat(self):
        state = {"a": 1}
        _apply_delta(state, {"a": 2, "b": 3})
        assert state == {"a": 2, "b": 3}

    def test_two_levels(self):
        state = {}
        _apply_delta(state, {"goals.gre.tension": 0.72})
        assert state["goals"]["gre"]["tension"] == 0.72

    def test_four_levels(self):
        state = {}
        _apply_delta(state, {"a.b.c.d": 99})
        assert state["a"]["b"]["c"]["d"] == 99

    def test_creates_missing_intermediate(self):
        state = {}
        _apply_delta(state, {"goals.new.tension": 0.5})
        assert state["goals"]["new"]["tension"] == 0.5

    def test_overwrites_existing(self):
        state = {"x": "old"}
        _apply_delta(state, {"x": "new"})
        assert state["x"] == "new"


# ── Delta ─────────────────────────────────────────────────────

class TestDelta:
    def test_checksum_valid(self):
        d = Delta(1, 0, {"x": 1}, "a")
        assert d.verify_checksum()

    def test_checksum_detects_tamper(self):
        d = Delta(1, 0, {"x": 1}, "a")
        d.changes["x"] = 99
        assert not d.verify_checksum()

    def test_roundtrip(self):
        d  = Delta(3, 2, {"score": 10}, "agent_b", "t", "i")
        d2 = Delta.from_dict(d.to_dict())
        assert d2.version      == d.version
        assert d2.base_version == d.base_version
        assert d2.changes      == d.changes
        assert d2.checksum     == d.checksum


# ── Store init ────────────────────────────────────────────────

class TestStoreInit:
    def test_creates_s0(self, store):
        state, ver = store.read()
        assert ver == 0 and state["score"] == 0

    def test_idempotent(self, store):
        store.init({"score": 999})
        state, _ = store.read()
        assert state["score"] == 0


# ── CAS write ─────────────────────────────────────────────────

class TestWrite:
    def test_simple(self, store):
        ok, ver = store.write("a", 0, {"score": 1})
        assert ok and ver == 1
        state, v = store.read()
        assert state["score"] == 1 and v == 1

    def test_conflict(self, store):
        store.write("a", 0, {"score": 1})
        ok, ver = store.write("b", 0, {"score": 2})
        assert not ok and ver == 1

    def test_sequential(self, store):
        for i in range(1, 6):
            ok, ver = store.write("a", i - 1, {"score": i})
            assert ok and ver == i
        state, v = store.read()
        assert state["score"] == 5 and v == 5

    def test_checksum_stored(self, store):
        store.write("a", 0, {"score": 1})
        data = json.loads(open(delta_path(store.state_dir, 1)).read())
        assert len(data["checksum"]) == 16


# ── read_at ───────────────────────────────────────────────────

class TestReadAt:
    def test_historical(self, store):
        store.write("a", 0, {"score": 10})
        store.write("a", 1, {"score": 20})
        store.write("a", 2, {"score": 30})
        assert store.read_at(1)["score"] == 10
        assert store.read_at(2)["score"] == 20
        assert store.read_at(3)["score"] == 30


# ── Recovery ──────────────────────────────────────────────────

class TestRecover:
    def test_recovers_missing_delta(self, store):
        store.write("a", 0, {"score": 1})
        lost = delta_path(store.state_dir, 1)
        os.remove(lost)
        recovered = store.recover()
        assert 1 in recovered and os.path.exists(lost)

    def test_skips_corrupt_local(self, store):
        store.write("a", 0, {"score": 1})
        shared = delta_path(store.state_dir, 1)
        from delta_cas.storage import local_agent_dir
        local = os.path.join(local_agent_dir(store.archive_dir, "a"),
                             "delta_0001.json")
        os.remove(shared)
        with open(local, "w") as f:
            f.write('{"bad": true}')
        recovered = store.recover()
        assert 1 not in recovered


# ── Compaction ────────────────────────────────────────────────

class TestCompaction:
    def test_hard_cap(self, tmp_path):
        s = Store(str(tmp_path / "s"), str(tmp_path / "a"),
                  snapshot_interval=3, circuit_breaker_ratio=None)
        s.init({"n": 0})
        for i in range(1, 4):
            s.write("a", i - 1, {"n": i})
        assert os.path.exists(snapshot_path(s.state_dir, 1))

    def test_state_correct_after_compact(self, tmp_path):
        s = Store(str(tmp_path / "s"), str(tmp_path / "a"),
                  snapshot_interval=3, circuit_breaker_ratio=None)
        s.init({"n": 0})
        for i in range(1, 5):
            s.write("a", i - 1, {"n": i})
        state, ver = s.read()
        assert state["n"] == 4 and ver == 4


# ── Circuit breaker ───────────────────────────────────────────

class TestCircuitBreaker:
    def test_fires_at_ratio_0(self, tmp_path):
        s = Store(str(tmp_path / "s"), str(tmp_path / "a"),
                  snapshot_interval=100, circuit_breaker_ratio=0.0)
        s.init({"data": "x" * 1000})
        s.write("a", 0, {"score": 1})
        assert os.path.exists(snapshot_path(s.state_dir, 1))

    def test_disabled_when_none(self, tmp_path):
        s = Store(str(tmp_path / "s"), str(tmp_path / "a"),
                  snapshot_interval=100, circuit_breaker_ratio=None)
        s.init({"data": "x" * 1000})
        s.write("a", 0, {"score": 1})
        assert not os.path.exists(snapshot_path(s.state_dir, 1))

    def test_no_fire_for_small_delta(self, tmp_path):
        s = Store(str(tmp_path / "s"), str(tmp_path / "a"),
                  snapshot_interval=100, circuit_breaker_ratio=1.0)
        s.init({"data": "x" * 1000})
        s.write("a", 0, {"score": 1})
        assert not os.path.exists(snapshot_path(s.state_dir, 1))

    def test_state_correct_after_breaker(self, tmp_path):
        s = Store(str(tmp_path / "s"), str(tmp_path / "a"),
                  snapshot_interval=100, circuit_breaker_ratio=0.8)
        s.init({"score": 0})
        s.write("a", 0, {"score": 10})
        s.write("a", 1, {"big": "x" * 10000})
        state, _ = s.read()
        assert state["score"] == 10
        assert state["big"] == "x" * 10000

    def test_history(self, store):
        store.write("a", 0, {"score": 1})
        store.write("a", 1, {"score": 2})
        h = store.history()
        assert len(h) == 2 and h[0]["version"] == 1
        h2 = store.history(from_version=1)
        assert len(h2) == 1 and h2[0]["version"] == 2
