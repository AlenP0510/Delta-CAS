"""
Tests for delta_cas.action:
  action_changes, find_duplicate_action, list_actions,
  Executor (claim / execute / result / timeout / stale-claim sweep),
  ActionAwareMixin (pivot / abort / retry).
"""
import time

import pytest

from delta_cas import (
    Agent, Store,
    PENDING, CLAIMED, SUCCESS, FAILED,
    action_changes, find_duplicate_action, list_actions,
    is_action_timed_out, is_claim_expired,
    Executor, ActionAwareMixin,
)


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "state"), str(tmp_path / "archive"),
              snapshot_interval=50, circuit_breaker_ratio=None)
    s.init({"score": 0, "actions": {}})
    return s


def declare(store, key="k1", action_type="email", expires=300):
    changes = action_changes(action_type, {"to": "u@e.com"},
                             "agent_a", key, expires_seconds=expires)
    store.write("agent_a", 0, changes)


# ── action_changes ────────────────────────────────────────────

class TestActionChanges:
    def test_returns_pending(self, store):
        changes = action_changes("email", {}, "a", "k1")
        statuses = [v for k, v in changes.items() if k.endswith(".status")]
        assert statuses == [PENDING]

    def test_declared_atomically(self, store):
        changes = {"score": 1}
        changes.update(action_changes("email", {"to": "u"}, "a", "k1"))
        store.write("a", 0, changes)
        state, _ = store.read()
        assert state["score"] == 1
        assert len(list_actions(state, status=PENDING)) == 1

    def test_expires_at_set(self, store):
        changes = action_changes("email", {}, "a", "k1", expires_seconds=60)
        keys = list(changes.keys())
        assert any(".expires_at" in k for k in keys)


# ── find_duplicate / list_actions ─────────────────────────────

class TestQueryHelpers:
    def test_find_duplicate(self, store):
        declare(store)
        state, _ = store.read()
        dup = find_duplicate_action(state, "email", "k1")
        assert dup is not None and dup["status"] == PENDING

    def test_no_duplicate(self, store):
        declare(store)
        state, _ = store.read()
        assert find_duplicate_action(state, "email", "other") is None

    def test_list_by_status(self, store):
        declare(store)
        state, _ = store.read()
        assert len(list_actions(state, status=PENDING)) == 1
        assert len(list_actions(state, status=SUCCESS)) == 0

    def test_list_by_type(self, store):
        declare(store, key="k1", action_type="email")
        state, ver = store.read()
        changes2 = action_changes("sms", {}, "a", "k2")
        store.write("a", ver, changes2)
        state, _ = store.read()
        assert len(list_actions(state, action_type="email")) == 1
        assert len(list_actions(state, action_type="sms")) == 1


# ── Executor — success path ───────────────────────────────────

class TestExecutorSuccess:
    def test_marks_success(self, store):
        declare(store)

        class OkExec(Executor):
            def execute(self, action): return True

        counts = OkExec("exec", store).run_pending()
        assert counts["succeeded"] == 1
        state, _ = store.read()
        actions = list_actions(state, status=SUCCESS)
        assert len(actions) == 1
        assert actions[0]["executed_at"] is not None
        assert actions[0]["claimed_by"] == "exec"

    def test_marks_failed(self, store):
        declare(store)

        class FailExec(Executor):
            def execute(self, action): return False

        counts = FailExec("exec", store).run_pending()
        assert counts["failed"] == 1
        state, _ = store.read()
        assert len(list_actions(state, status=FAILED)) == 1

    def test_captures_exception(self, store):
        declare(store)

        class RaisesExec(Executor):
            def execute(self, action): raise ValueError("smtp down")

        RaisesExec("exec", store).run_pending()
        state, _ = store.read()
        actions = list_actions(state, status=FAILED)
        assert "smtp down" in (actions[0].get("error") or "")

    def test_skips_after_max_retries(self, store):
        declare(store)

        class FailExec(Executor):
            def execute(self, action): return False

        # Run twice — second run should see retry_count >= max_retries
        exec_ = FailExec("exec", store, max_retries=1)
        exec_.run_pending()   # retry_count → 1, status becomes FAILED
        # On second run: action is FAILED (not PENDING), so run_pending ignores it
        counts = exec_.run_pending()
        assert counts["succeeded"] == 0 and counts["failed"] == 0 and counts["skipped"] == 0


# ── Executor — claim ──────────────────────────────────────────

class TestClaim:
    def test_claimed_status_set(self, store):
        declare(store)
        state, ver = store.read()
        action = list_actions(state, status=PENDING)[0]

        class SlowExec(Executor):
            def execute(self, action):
                # Check claimed status mid-execution
                s, _ = store.read()
                acts = list_actions(s, status=CLAIMED)
                assert len(acts) == 1
                return True

        SlowExec("exec", store).run_pending()

    def test_two_executors_dont_double_execute(self, store):
        declare(store)
        execute_count = [0]

        class CountExec(Executor):
            def execute(self, action):
                execute_count[0] += 1
                return True

        import threading
        barrier = threading.Barrier(2)

        def run():
            barrier.wait()
            CountExec("exec", store).run_pending()

        t1 = threading.Thread(target=run)
        t2 = threading.Thread(target=run)
        t1.start(); t2.start()
        t1.join();  t2.join()

        assert execute_count[0] == 1


# ── Timeout + stale claim sweep ───────────────────────────────

class TestTimeoutAndTakeover:
    def test_is_action_timed_out(self):
        from delta_cas.action import _future, _now
        from datetime import datetime, timedelta
        past   = (datetime.now() - timedelta(seconds=10)).strftime("%Y-%m-%dT%H:%M:%S")
        future = _future(60)
        assert is_action_timed_out({"expires_at": past})
        assert not is_action_timed_out({"expires_at": future})
        assert not is_action_timed_out({})

    def test_is_claim_expired(self):
        from datetime import datetime, timedelta
        old = (datetime.now() - timedelta(seconds=120)).strftime("%Y-%m-%dT%H:%M:%S")
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        assert is_claim_expired({"claimed_at": old}, claim_ttl=60)
        assert not is_claim_expired({"claimed_at": now}, claim_ttl=60)

    def test_sweep_timed_out(self, store):
        # Declare with 0-second TTL so it expires immediately
        changes = action_changes("email", {}, "a", "k1", expires_seconds=0)
        store.write("a", 0, changes)
        # Small sleep to ensure past deadline
        time.sleep(0.01)

        class AnyExec(Executor):
            def execute(self, action): return True

        exec_ = AnyExec("exec", store)
        n = exec_.sweep_timed_out()
        assert n == 1
        state, _ = store.read()
        assert len(list_actions(state, status=FAILED)) == 1

    def test_sweep_stale_claim_releases_to_pending(self, store):
        declare(store)
        state, ver = store.read()
        action = list_actions(state, status=PENDING)[0]
        aid    = action["id"]

        # Manually claim with a very old claimed_at
        from datetime import datetime, timedelta
        old = (datetime.now() - timedelta(seconds=200)).strftime("%Y-%m-%dT%H:%M:%S")
        store.write("fake_exec", ver, {
            f"actions.{aid}.status":     CLAIMED,
            f"actions.{aid}.claimed_by": "fake_exec",
            f"actions.{aid}.claimed_at": old,
        })

        class OkExec(Executor):
            def execute(self, action): return True

        exec_ = OkExec("exec", store, claim_ttl=60)
        n = exec_.sweep_expired_claims()
        assert n == 1
        state, _ = store.read()
        # After sweep, action is back to pending and can be re-claimed
        assert len(list_actions(state, status=PENDING)) == 1

    def test_takeover_after_sweep(self, store):
        """Full cycle: declare → stale claim → sweep → takeover by new executor."""
        declare(store)
        state, ver = store.read()
        action = list_actions(state, status=PENDING)[0]
        aid    = action["id"]

        from datetime import datetime, timedelta
        old = (datetime.now() - timedelta(seconds=200)).strftime("%Y-%m-%dT%H:%M:%S")
        store.write("dead_exec", ver, {
            f"actions.{aid}.status":     CLAIMED,
            f"actions.{aid}.claimed_by": "dead_exec",
            f"actions.{aid}.claimed_at": old,
        })

        class NewExec(Executor):
            def execute(self, action): return True

        exec_ = NewExec("new_exec", store, claim_ttl=60)
        counts = exec_.run_pending()   # sweep_expired_claims + pick up
        assert counts["succeeded"] == 1
        state, _ = store.read()
        success_actions = list_actions(state, status=SUCCESS)
        assert len(success_actions) == 1
        assert success_actions[0]["claimed_by"] == "new_exec"


# ── ActionAwareMixin ──────────────────────────────────────────

class TestActionAwareMixin:
    def test_none_when_no_duplicate(self, store):
        class A(ActionAwareMixin, Agent):
            def describe_goal(self): return "d"
            def compute_changes(self, s, v): return {}
        a = A("a", store)
        state, _ = store.read()
        assert a.check_action_conflict(state, "email", "k1") == "none"

    def test_pivot_on_pending(self, store):
        declare(store)
        state, _ = store.read()

        class A(ActionAwareMixin, Agent):
            def describe_goal(self): return "d"
            def compute_changes(self, s, v): return {}
        a = A("a", store)
        assert a.check_action_conflict(state, "email", "k1") == "pivot"

    def test_abort_on_success(self, store):
        declare(store)

        class OkExec(Executor):
            def execute(self, a): return True
        OkExec("exec", store).run_pending()

        state, _ = store.read()

        class A(ActionAwareMixin, Agent):
            def describe_goal(self): return "d"
            def compute_changes(self, s, v): return {}
        a = A("a", store)
        assert a.check_action_conflict(state, "email", "k1") == "abort"

    def test_retry_on_failed(self, store):
        declare(store)

        class FailExec(Executor):
            def execute(self, a): return False
        FailExec("exec", store, max_retries=1).run_pending()

        state, _ = store.read()

        class A(ActionAwareMixin, Agent):
            def describe_goal(self): return "d"
            def compute_changes(self, s, v): return {}
        a = A("a", store)
        assert a.check_action_conflict(state, "email", "k1") == "retry"

    def test_pivot_on_claimed(self, store):
        declare(store)
        state, ver = store.read()
        action = list_actions(state, status=PENDING)[0]
        aid    = action["id"]

        # Manually put into claimed
        store.write("exec", ver, {
            f"actions.{aid}.status":     CLAIMED,
            f"actions.{aid}.claimed_by": "exec",
            f"actions.{aid}.claimed_at": "2026-01-01T00:00:00",
        })
        state2, _ = store.read()

        class A(ActionAwareMixin, Agent):
            def describe_goal(self): return "d"
            def compute_changes(self, s, v): return {}
        a = A("a", store)
        assert a.check_action_conflict(state2, "email", "k1") == "pivot"

    def test_full_agent_aborts_on_duplicate_success(self, store):
        """End-to-end: second agent run aborts when action already succeeded."""
        class MyAgent(ActionAwareMixin, Agent):
            def describe_goal(self): return "send email"
            def compute_changes(self, state, version):
                conflict = self.check_action_conflict(state, "email", "k1")
                if conflict == "abort": return {}
                changes = {"score": state.get("score", 0) + 1}
                changes.update(action_changes("email", {}, self.agent_id, "k1"))
                return changes

        agent = MyAgent("a", store)
        agent.run()

        class OkExec(Executor):
            def execute(self, a): return True
        OkExec("exec", store).run_pending()

        agent.run()   # should abort
        state, _ = store.read()
        assert state["score"] == 1   # unchanged by second run

    def test_full_agent_pivots_on_pending(self, store):
        """End-to-end: second agent pivots to alternative when action is pending."""
        class MyAgent(ActionAwareMixin, Agent):
            def describe_goal(self): return "send email"
            def compute_changes(self, state, version):
                conflict = self.check_action_conflict(state, "email", "k1")
                if conflict == "pivot": return {"score": 99}
                changes = {"score": 1}
                changes.update(action_changes("email", {}, self.agent_id, "k1"))
                return changes

        MyAgent("a", store).run()   # score=1, action=pending
        MyAgent("b", store).run()   # should pivot → score=99
        state, _ = store.read()
        assert state["score"] == 99
