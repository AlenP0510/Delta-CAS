"""
delta_cas.intent
================
Injectable intent layer: State → Intent (LLM) → validate → Action.

The intent function is a plain callable injected at Agent construction time.
Return "INVALID: <reason>" to prevent the agent from writing.

Built-in helpers
----------------
noop_intent_fn            → always valid, no LLM call (testing / prod without validation)
anthropic_intent_fn(...)  → Claude-backed validation
openai_intent_fn(...)     → OpenAI / DeepSeek / any OpenAI-compatible API
"""
from __future__ import annotations

import logging
import os
from typing import Callable

logger = logging.getLogger(__name__)

IntentFn = Callable[[str, dict, str], str]
"""
Signature:  (agent_id: str, state: dict, goal_description: str) -> str

Return "INVALID: <reason>" to signal that the action is impossible
given the current state. Any other return value is treated as valid.
"""


def intent_is_valid(intent: str) -> bool:
    """False if the intent string starts with INVALID (case-insensitive)."""
    return not intent.upper().startswith("INVALID")


def noop_intent_fn(agent_id: str, state: dict, goal_description: str) -> str:
    """Always-valid intent function. Use for testing or when no LLM is available."""
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
    api_key : API key. Falls back to ANTHROPIC_API_KEY env var.

    Example
    -------
    agent = MyAgent("id", store, intent_fn=anthropic_intent_fn())
    """
    try:
        import anthropic as _anthropic
    except ImportError as exc:
        raise ImportError(
            "anthropic package required. Install: pip install anthropic"
        ) from exc

    client = _anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def _fn(agent_id: str, state: dict, goal_description: str) -> str:
        import json
        summary = json.dumps(
            {k: v for k, v in state.items() if k not in ("raw", "_internal")},
            indent=2
        )
        try:
            resp = client.messages.create(
                model=model, max_tokens=120,
                messages=[{"role": "user", "content": (
                    f"Current world state:\n{summary}\n\n"
                    f"Agent goal: {goal_description}\n\n"
                    "In one sentence describe the agent's intent and confirm it is valid. "
                    "If the state makes the action impossible, respond: 'INVALID: <reason>'"
                )}]
            )
            return resp.content[0].text.strip()
        except Exception as exc:
            logger.warning(f"[{agent_id}] anthropic_intent_fn fallback: {exc}")
            return f"{agent_id}: {goal_description}"

    return _fn


def openai_intent_fn(
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    base_url: str | None = None,
) -> IntentFn:
    """
    Returns an intent function backed by OpenAI or any compatible API
    (DeepSeek, Together, Ollama, etc.).

    Parameters
    ----------
    model    : model string
    api_key  : API key. Falls back to OPENAI_API_KEY env var.
    base_url : override for compatible APIs, e.g. https://api.deepseek.com

    Example — DeepSeek
    ------------------
    agent = MyAgent("id", store, intent_fn=openai_intent_fn(
        model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    ))
    """
    try:
        import openai as _openai
    except ImportError as exc:
        raise ImportError(
            "openai package required. Install: pip install openai"
        ) from exc

    kwargs: dict = {"api_key": api_key or os.environ.get("OPENAI_API_KEY")}
    if base_url:
        kwargs["base_url"] = base_url
    client = _openai.OpenAI(**kwargs)

    def _fn(agent_id: str, state: dict, goal_description: str) -> str:
        import json
        summary = json.dumps(
            {k: v for k, v in state.items() if k not in ("raw", "_internal")},
            indent=2
        )
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=120,
                messages=[
                    {"role": "system",
                     "content": "You are a semantic intent validator for a multi-agent system."},
                    {"role": "user", "content": (
                        f"Current world state:\n{summary}\n\n"
                        f"Agent goal: {goal_description}\n\n"
                        "In one sentence describe the agent's intent and confirm it is valid. "
                        "If impossible, respond: 'INVALID: <reason>'"
                    )}
                ]
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning(f"[{agent_id}] openai_intent_fn fallback: {exc}")
            return f"{agent_id}: {goal_description}"

    return _fn
