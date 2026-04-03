# delta-cas

Delta-based Compare-And-Swap state management for multi-agent systems.

```
V_curr = V_base + sum(Deltas)
```

Multiple agents read from the same base state, compute their changes as small deltas, and write back atomically. The protocol guarantees consistency, crash recovery, and — critically — correct handling of irreversible external actions even under retries and agent failures.

---

## Install

```bash
pip install delta-cas

# With Anthropic intent validation
pip install "delta-cas[anthropic]"

# With OpenAI / DeepSeek intent validation
pip install "delta-cas[openai]"

# Both
pip install "delta-cas[all]"
```

---

## Quick start

```python
from delta_cas import Store, Agent, EpochCoordinator
from delta_cas.intent import noop_intent_fn   # or anthropic_intent_fn, openai_intent_fn

store = Store("./state")
store.init({"score": 0})
epoch = EpochCoordinator(interval=5)

class ScoreAgent(Agent):
    def describe_goal(self):
        return "Increment the score by 1"
    def compute_changes(self, state, version):
        return {"score": state.get("score", 0) + 1}

agent = ScoreAgent("scorer", store, epoch=epoch)
success, version = agent.run()

state, ver = store.read()
print(state["score"])   # 1
```

---

## Core concepts

### Delta

A delta is the smallest unit of change — only the fields that actually changed, not the full state. Dot notation is supported for nested keys.

```python
{"goals.gre.tension": 0.72}             # dot notation
{"tasks": ["Prepare GRE", "Morning run"]}
```

Every delta carries a sha256 checksum. Corrupted deltas are detected and skipped on read.

### Compare-And-Swap

Every write is atomic:

1. Agent reads current state at version N
2. Agent computes changes
3. Agent calls `store.write(base_version=N, changes={...})`
4. If another agent wrote between step 1 and 3, the write is rejected

```python
ok, new_version = store.write(
    agent_id="my_agent",
    base_version=current_version,
    changes={"score": 10},
)
if not ok:
    # CAS conflict — new_version is the current version, use it to rebase
```

### WAL (Write-Ahead Log)

Every agent writes its delta locally **before** attempting CAS. If the process crashes mid-write, the WAL entry is restored automatically on the next startup.

```
local_archive/
    my_agent/
        wal_0003.json       ← written before CAS
        wal_0003.committed  ← written after CAS success
        stash_0002.json     ← stashed delta from a CAS conflict
```

### Circuit breaker

When `delta_bytes / snapshot_bytes >= circuit_breaker_ratio`, the store immediately compacts regardless of delta count. Prevents path-addressing overhead from exceeding raw snapshot cost.

```python
store = Store("./state",
              snapshot_interval=20,       # hard cap
              circuit_breaker_ratio=0.8)  # 80% threshold
```

### Intent layer

Before writing, an agent generates an intent string via an LLM function. If the current state makes the action impossible (`INVALID: ...`), the agent aborts or rebases rather than writing invalid state.

```python
from delta_cas.intent import anthropic_intent_fn

class KillAgent(Agent):
    def describe_goal(self):
        return "The person has died"
    def compute_changes(self, state, version):
        return {"person.alive": False}

class EatAgent(Agent):
    def describe_goal(self):
        return "Make the person eat — only valid if they are alive"
    def compute_changes(self, state, version):
        return {"person.activity": "eating"}

# If KillAgent runs first, EatAgent's intent will be flagged INVALID
# and it will abort rather than write a contradictory state.
```

Built-in intent functions:

| Function | Requires |
|---|---|
| `noop_intent_fn` | nothing — always valid (testing / no LLM) |
| `anthropic_intent_fn()` | `pip install anthropic` |
| `openai_intent_fn()` | `pip install openai` |
| `openai_intent_fn(base_url=...)` | any OpenAI-compatible API (DeepSeek, etc.) |

Any callable `(agent_id, state, goal) -> str` works as an intent function.

### Epoch alignment

Every N versions, all agents pause, a new snapshot is compacted, and agents resume from the new base. Prevents slow agents from lagging arbitrarily far behind fast ones.

```python
epoch = EpochCoordinator(interval=5)
agent = MyAgent("a", store, epoch=epoch)
```

---

## Action layer — irreversible external effects

The action layer solves the "did this already happen?" problem for irreversible external actions (API calls, emails, MCP tool calls, etc.).

**Core principle: Commit first. Execute only if commit succeeds.**

```
V_n = V_(n-1) + S_(n-1) + {A_i} + A_status(pending → claimed → success | failed)
```

### Action lifecycle

```
pending  →  claimed  →  success
                     →  failed
claimed  →  pending      (stale claim swept — executor died)
pending  →  failed       (timeout — expires_at exceeded)
```

### Declaring an action

```python
from delta_cas import action_changes, ActionAwareMixin, Agent

class ReportAgent(ActionAwareMixin, Agent):
    def describe_goal(self):
        return "Send weekly report email"

    def compute_changes(self, state, version):
        # Check if this action already happened
        conflict = self.check_action_conflict(
            state, "send_email", "weekly-report-2026"
        )
        if conflict == "abort":   return {}   # already succeeded
        if conflict == "pivot":   return {}   # in-flight — skip
        if conflict == "retry":   pass        # failed — re-declare

        changes = {}
        changes.update(action_changes(
            action_type="send_email",
            payload={"to": "user@example.com"},
            agent_id=self.agent_id,
            idempotency_key="weekly-report-2026",
            expires_seconds=300,
        ))
        return changes
```

### Executing actions

```python
from delta_cas import Executor

class EmailExecutor(Executor):
    def execute(self, action) -> bool:
        send_email(action["payload"])   # your actual side effect
        return True   # success; return False or raise for failure

executor = EmailExecutor("email_exec", store)
counts = executor.run_pending()
# {"succeeded": 1, "failed": 0, "skipped": 0}
```

`run_pending()` automatically:
1. Sweeps timed-out pending actions (marks failed)
2. Releases stale claims (executor died mid-execution)
3. Claims and executes remaining pending actions

### Timeout and takeover

```python
# If executor dies after claiming but before writing result,
# the claim expires after claim_ttl seconds.
# sweep_expired_claims() resets status to pending so another executor can take over.

executor = EmailExecutor("exec_B", store, claim_ttl=60)
executor.run_pending()   # will take over stale claims from exec_A
```

### Action record schema

```python
{
  "id":               "act_<12hex>",
  "type":             "send_email",
  "status":           "pending" | "claimed" | "success" | "failed",
  "payload":          {...},
  "idempotency_key":  "weekly-report-2026",
  "agent_id":         "report_agent",
  "declared_at":      "2026-04-02T10:00:00",
  "expires_at":       "2026-04-02T10:05:00",
  "claimed_by":       "email_exec",
  "claimed_at":       "2026-04-02T10:00:01",
  "executed_at":      "2026-04-02T10:00:02",
  "error":            null,
  "retry_count":      0
}
```

---

## File layout

```
state/
    meta.json               ← current version, epoch, snapshot index
    S_0.json                ← initial snapshot
    S_1.json                ← compacted snapshot
    delta_0001.json
    delta_0002.json
    ...

local_archive/
    <agent_id>/
        delta_0001.json     ← local copy of every delta written
        wal_0001.json       ← WAL entry (written before CAS)
        wal_0001.committed  ← commit marker (written after success)
        stash_0001.json     ← stashed delta from a CAS conflict
```

---

## Concurrent agents

```python
import threading

store = Store("./state")
store.init({"count": 0})
epoch = EpochCoordinator(interval=10)

class Counter(Agent):
    def describe_goal(self): return "Increment count"
    def compute_changes(self, state, version):
        return {"count": state.get("count", 0) + 1}

def run(agent_id):
    Counter(agent_id, store, epoch=epoch).run()

threads = [threading.Thread(target=run, args=(f"a{i}",)) for i in range(10)]
for t in threads: t.start()
for t in threads: t.join()

state, _ = store.read()
print(state["count"])   # 10
```

---

## Recovery

```python
# Restore deltas missing from shared store using agent local archives
recovered = store.recover()
print(f"Recovered {len(recovered)} deltas")

# Reconstruct state at any past version
state_v3 = store.read_at(3)
```

---

## Running tests

```bash
pip install "delta-cas[dev]"
pytest tests/ -v
```

---

## License

MIT — see [LICENSE](LICENSE).
