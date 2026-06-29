"""
Stage 5: Answer generation — 6 templates × 3 modes.

Mode logic:
  kg_rag  → LLM with KG context (or without if zero hits, kg_coverage stays False)
  llm_only → LLM without context
  kg_only  → return null + "no_kg_match" if no context; else return raw context (no LLM)
"""

from __future__ import annotations

import json
import logging
import os

from ..state import QAState
from ..utils.config import get_config
from ..utils.prompts import (
    ANSWER_TEMPLATES,
    build_context_block,
    format_mcq_options,
)

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        import openai
        _client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _client


def _call_llm(system: str, user: str, cfg: dict) -> tuple[dict, int]:
    """Call GPT-4o-mini, return (parsed_dict, tokens). Falls back to error dict."""
    try:
        response = _get_client().chat.completions.create(
            model=cfg["model"],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        tokens = response.usage.total_tokens if response.usage else 0
        data = json.loads(response.choices[0].message.content)
        return data, tokens
    except Exception as exc:
        logger.warning("LLM answer call failed: %s", exc)
        return {"answer": None, "explanation": f"LLM error: {exc}"}, 0


def answer_node(state: QAState) -> QAState:
    mode = state.get("mode", "kg_rag")
    question_type = state.get("question_type", "factoid")
    raw_triples = state.get("raw_triples", [])
    cfg = get_config()["llm"]

    # kg_only + no context → null answer
    if mode == "kg_only" and not raw_triples:
        state["answer"] = None
        state["error"] = "no_kg_match"
        return state

    # kg_only + has context → return raw triples without LLM
    if mode == "kg_only":
        state["answer"] = {
            "answer": raw_triples,
            "explanation": "Raw knowledge graph context (kg_only mode — no LLM).",
        }
        return state

    # llm_only or kg_rag → call LLM
    template = ANSWER_TEMPLATES.get(question_type, ANSWER_TEMPLATES["factoid"])
    context_block = build_context_block(raw_triples=raw_triples if mode != "llm_only" else None)

    # Build options string for MEDQA
    options_str = ""
    if question_type == "mcq":
        raw_options = state.get("options", {}).get("choices", {})
        options_str = format_mcq_options(raw_options) if raw_options else ""

    # Format user prompt
    user_prompt = template["user"].format(
        context_block=context_block,
        question=state.get("query_en") or state["query_raw"],
        options=options_str,
    )

    answer_dict, tokens = _call_llm(template["system"], user_prompt, cfg)
    state["tokens_used"] = state.get("tokens_used", 0) + tokens
    state["answer"] = answer_dict

    # Ensure required keys are present for downstream consumers
    if "answer" not in answer_dict:
        state["answer"]["answer"] = None
    if "reasoning_answer" not in answer_dict:
        state["answer"]["reasoning_answer"] = None

    return state
