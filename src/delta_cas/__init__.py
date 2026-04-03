"""
delta_cas
=========
Delta-based Compare-And-Swap state management for multi-agent systems.

V_curr = V_base + sum(Deltas)

Quick start
-----------
>>> from delta_cas import Store, Agent, EpochCoordinator
>>> from delta_cas.intent import noop_intent_fn
>>>
>>> store = Store("./state")
>>> store.init({"score": 0})
>>> epoch = EpochCoordinator(interval=5)
>>>
>>> class ScoreAgent(Agent):
...     def describe_goal(self): return "Increment score by 1"
...     def compute_changes(self, state, version):
...         return {"score": state.get("score", 0) + 1}
>>>
>>> agent = ScoreAgent("scorer", store, epoch=epoch)
>>> success, version = agent.run()
>>> state, ver = store.read()
>>> state["score"]
1
"""

from .core    import Delta, Snapshot, Store
from .agent   import Agent
from .epoch   import EpochCoordinator
from .intent  import (
    IntentFn,
    intent_is_valid,
    noop_intent_fn,
    anthropic_intent_fn,
    openai_intent_fn,
)
from .action  import (
    PENDING, CLAIMED, SUCCESS, FAILED,
    action_changes,
    find_duplicate_action,
    list_actions,
    make_action_id,
    is_action_timed_out,
    is_claim_expired,
    Executor,
    ActionAwareMixin,
)
from .storage import FileLock, ensure_dirs

__version__ = "0.3.0"
__author__  = "Alen Pu"
__license__ = "MIT"

__all__ = [
    # Core
    "Store", "Snapshot", "Delta",
    # Agent
    "Agent",
    # Epoch
    "EpochCoordinator",
    # Intent
    "IntentFn", "intent_is_valid", "noop_intent_fn",
    "anthropic_intent_fn", "openai_intent_fn",
    # Action layer
    "PENDING", "CLAIMED", "SUCCESS", "FAILED",
    "action_changes", "find_duplicate_action", "list_actions",
    "make_action_id", "is_action_timed_out", "is_claim_expired",
    "Executor", "ActionAwareMixin",
    # Storage
    "FileLock", "ensure_dirs",
    # Meta
    "__version__",
]
