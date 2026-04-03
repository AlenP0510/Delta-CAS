"""
delta_cas.intent
================
Intent layer: State → Intent (via user-supplied LLM function) → validate → Action.

The intent function is fully injectable — pass any callable that takes
(agent_id, state, goal_description) and returns a string.

If the string starts with "INVALID:" the agent will not write and will
either abort or rebase depending on the caller's logic.

Built-in helpers
----------------
anthropic_intent_fn(model, api_key)   → returns a ready-to-use intent function
noop_intent_fn                        → always returns a valid intent (for testing)
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

IntentFn = Callable[[str, dict, str], str]
"""
Signature: (agent_id: str, state: dict, goal_description: str) -> str

Return "INVALID: <reason>" to signal the action is impossible given state.
Any other return value is treated as a valid intent.
"""


def intent_is_valid(intent: str) -> bool:
    """Return False if the intent string signals impossibility."""
    return not intent.upper().startswith("INVALID")


def noop_intent_fn(agent_id: str, state: dict, goal_description: str) -> str:
    """
    Default intent function — always valid.
    Use for testing or when you don't need semantic intent validation.
    """
    return f"{agent_id}: {goal_description}"


def anthropic_intent_fn(
    model: str = "claude-sonnet-4-20250514",
    api_key: str | None = None,
) -> IntentFn:
    """
    Returns an intent function backed by Anthropic Claude.

    Parameters
    ----------
    model   : Anthropic model string
    api_key : Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.

    Example
    -------
    agent = Agent(..., intent_fn=anthropic_intent_fn())
    """
    import os
    try:
        import anthropic as _anthropic
    except ImportError as e:
        raise ImportError(
            "anthropic package is required for anthropic_intent_fn. "
            "Install it with: pip install anthropic"
        ) from e

    key    = api_key or os.environ.get("ANTHROPIC_API_KEY")
    client = _anthropic.Anthropic(api_key=key)

    def _fn(agent_id: str, state: dict, goal_description: str) -> str:
        import json
        snapshot_summary = json.dumps({
            k: v for k, v in state.items()
            if k not in ("raw", "_internal")
        }, indent=2)
        try:
            response = client.messages.create(
                model=model,
                max_tokens=120,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Current world state:\n{snapshot_summary}\n\n"
                        f"Agent goal: {goal_description}\n\n"
                        "In one sentence, describe the agent's intent and confirm it "
                        "is valid given the current state. If the state makes the "
                        "action impossible or contradictory, respond with exactly: "
                        "'INVALID: <reason>'"
                    )
                }]
            )
            return response.content[0].text.strip()
        except Exception as exc:
            logger.warning(f"[{agent_id}] anthropic_intent_fn error: {exc} — falling back")
            return f"{agent_id}: {goal_description}"

    return _fn


def openai_intent_fn(
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    base_url: str | None = None,
) -> IntentFn:
    """
    Returns an intent function backed by OpenAI (or any OpenAI-compatible API,
    e.g. DeepSeek, Together, local Ollama).

    Parameters
    ----------
    model    : model string
    api_key  : API key. Falls back to OPENAI_API_KEY env var.
    base_url : override base URL for compatible APIs (e.g. https://api.deepseek.com)

    Example — DeepSeek
    ------------------
    agent = Agent(..., intent_fn=openai_intent_fn(
        model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    ))
    """
    import os
    try:
        import openai as _openai
    except ImportError as e:
        raise ImportError(
            "openai package is required for openai_intent_fn. "
            "Install it with: pip install openai"
        ) from e

    kwargs: dict = {"api_key": api_key or os.environ.get("OPENAI_API_KEY")}
    if base_url:
        kwargs["base_url"] = base_url
    client = _openai.OpenAI(**kwargs)

    def _fn(agent_id: str, state: dict, goal_description: str) -> str:
        import json
        snapshot_summary = json.dumps({
            k: v for k, v in state.items()
            if k not in ("raw", "_internal")
        }, indent=2)
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=120,
                messages=[
                    {"role": "system", "content":
                     "You are a semantic intent validator for a multi-agent system."},
                    {"role": "user", "content": (
                        f"Current world state:\n{snapshot_summary}\n\n"
                        f"Agent goal: {goal_description}\n\n"
                        "In one sentence, describe the agent's intent and confirm it "
                        "is valid given the current state. If the state makes the "
                        "action impossible, respond with: 'INVALID: <reason>'"
                    )}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning(f"[{agent_id}] openai_intent_fn error: {exc} — falling back")
            return f"{agent_id}: {goal_description}"

    return _fn
