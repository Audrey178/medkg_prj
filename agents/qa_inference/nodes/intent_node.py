"""
Stage 2: Question type classification + relation intent extraction.

- MEDQA   → always "mcq" (rule-based, no LLM call)
- PubMedQA → always "yes_no_maybe" (rule-based)
- BioASQ  → GPT-4o-mini classifies: yes_no | factoid | list | summary
"""

from __future__ import annotations

import json
import logging
import os

from ..state import QAState
from ..utils.config import get_config
from ..utils.prompts import INTENT_SYSTEM, INTENT_USER

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        import openai
        _client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _client


def _classify_bioasq(query: str, cfg: dict) -> tuple[str, list[str], int]:
    """Return (question_type, relation_intents, tokens). Fallback to factoid on error."""
    try:
        response = _get_client().chat.completions.create(
            model=cfg["model"],
            temperature=0.0,
            max_tokens=256,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": INTENT_SYSTEM},
                {"role": "user", "content": INTENT_USER.format(query=query)},
            ],
        )
        data = json.loads(response.choices[0].message.content)
        qtype = data.get("question_type", "factoid")
        if qtype not in ("yes_no", "factoid", "list", "summary"):
            qtype = "factoid"
        intents = [str(i) for i in data.get("relation_intents", [])][:3]
        tokens = response.usage.total_tokens if response.usage else 0
        return qtype, intents, tokens
    except Exception as exc:
        logger.warning("intent classification failed: %s", exc)
        return "factoid", [], 0


def intent_node(state: QAState) -> QAState:
    benchmark = state.get("benchmark_type", "bioasq")
    query_en = state.get("query_en") or state["query_raw"]
    cfg = get_config()["llm"]

    if benchmark == "medqa":
        state["question_type"] = "mcq"
        state["relation_intents"] = []

    elif benchmark == "pubmedqa":
        state["question_type"] = "yes_no_maybe"
        state["relation_intents"] = []

    else:  # bioasq or unknown
        # If caller already provided question_type (e.g. from gold data), skip LLM
        if state.get("question_type") in ("yes_no", "factoid", "list", "summary"):
            pass
        else:
            qtype, intents, tokens = _classify_bioasq(query_en, cfg)
            state["question_type"] = qtype
            state["relation_intents"] = intents
            state["tokens_used"] = state.get("tokens_used", 0) + tokens

    return state
