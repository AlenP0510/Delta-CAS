[README.md](https://github.com/user-attachments/files/26338315/README.md)
# delta-cas

Delta-based Compare-And-Swap state management for multi-agent systems.

```
V_curr = V_base + sum(Deltas)
```

Multiple agents read from the same base state, compute their changes as small deltas, and write back atomically. Full snapshots are compacted every N versions to keep the read chain short.

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

# Create a store
store = Store("./state")
store.init({"score": 0, "tasks": []})

# Create an epoch coordinator (realigns all agents every 5 versions)
epoch = EpochCoordinator(interval=5)

# Subclass Agent
class ScoreAgent(Agent):
    def describe_goal(self):
        return "Increment the score by 1"

    def compute_changes(self, state, version):
        return {"score": state.get("score", 0) + 1}

# Run it
agent = ScoreAgent("scorer", store, epoch=epoch)
success, version = agent.run()

state, ver = store.read()
print(state["score"])   # 1
```

---

## Concepts

### Delta

A delta is the smallest unit of change — only the fields that actually changed, not the full state.

```python
{"goals.gre.tension": 0.72}   # dot notation for nested keys
{"tasks": ["Prepare GRE", "Morning run"]}
```

Deltas are checksum-verified on read. A corrupted delta is skipped and logged.

### Compare-And-Swap

Every write is atomic:

1. Agent reads current state at version N
2. Agent computes changes
3. Agent calls `store.write(base_version=N, changes={...})`
4. If another agent wrote between step 1 and 3, the write is rejected and the agent rebases

```python
ok, new_version = store.write(
    agent_id="my_agent",
    base_version=current_version,
    changes={"score": 10},
)
if not ok:
    # CAS conflict — read new state and retry
```

### WAL (Write-Ahead Log)

Every agent writes its delta locally **before** attempting CAS. If the process crashes after a successful write but before the shared store is updated, the WAL entry is restored on the next startup.

```
local_archive/
    my_agent/
        wal_0003.json         ← written before CAS
        wal_0003.committed    ← written after CAS success
        stash_0002.json       ← stashed delta from a CAS conflict
```

### Intent layer

Before writing, an agent generates an intent string via an LLM function. If the current state makes the action impossible (`INVALID: ...`), the agent aborts rather than writing invalid state.

```python
from delta_cas.intent import anthropic_intent_fn

agent = MyAgent(
    "agent_id", store,
    intent_fn=anthropic_intent_fn(model="claude-sonnet-4-20250514")
)
```

Built-in intent functions:

| Function | Requires |
|---|---|
| `noop_intent_fn` | nothing (always valid, for testing) |
| `anthropic_intent_fn()` | `pip install anthropic` |
| `openai_intent_fn()` | `pip install openai` |
| `openai_intent_fn(base_url=...)` | any OpenAI-compatible API (DeepSeek, etc.) |

You can also pass any callable with the signature `(agent_id, state, goal) -> str`.

### Epoch alignment

Every N versions, all agents pause, a new snapshot is compacted, and agents resume from the new base. This prevents slow agents from lagging arbitrarily behind fast ones.

```python
epoch = EpochCoordinator(interval=5)
agent = MyAgent("a", store, epoch=epoch)
```

---

## Concurrent agents

```python
import threading
from delta_cas import Store, Agent, EpochCoordinator

store = Store("./state")
store.init({"count": 0})
epoch = EpochCoordinator(interval=10)

class Counter(Agent):
    def describe_goal(self):
        return "Increment count by 1"
    def compute_changes(self, state, version):
        return {"count": state.get("count", 0) + 1}

def run(agent_id):
    Counter(agent_id, store, epoch=epoch).run()

threads = [threading.Thread(target=run, args=(f"agent_{i}",)) for i in range(10)]
for t in threads: t.start()
for t in threads: t.join()

state, ver = store.read()
print(state["count"])   # 10
```

---

## File layout

```
state/
    meta.json               ← current version, epoch, snapshot index
    S_0.json                ← initial snapshot
    S_1.json                ← compacted snapshot after N deltas
    delta_0001.json         ← individual delta
    delta_0002.json
    ...

local_archive/
    <agent_id>/
        delta_0001.json     ← local copy of every delta this agent wrote
        wal_0001.json       ← WAL entry (written before CAS)
        wal_0001.committed  ← commit marker (written after CAS success)
        stash_0001.json     ← stashed delta from CAS conflict
```

---

## Recovery

```python
# Restore deltas missing from shared store using agent local archives
recovered = store.recover()
print(f"Recovered {len(recovered)} deltas: {recovered}")
```

---

## Historical reads

```python
# Reconstruct state at any past version
state_at_v3 = store.read_at(3)
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
