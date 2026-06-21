"""
Stage 0: Language detection and query translation.

- Detects source language via GPT-4o-mini
- Translates non-English queries to English
- Passes English queries through unchanged
- On API failure: falls back to original query, marks lang as "unknown"
"""

from __future__ import annotations

import json
import logging
import os

from ..state import QAState
from ..utils.config import get_config
from ..utils.prompts import TRANSLATE_SYSTEM, TRANSLATE_USER

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        import openai
        _client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _client


def translate_node(state: QAState) -> QAState:
    query = state["query_raw"].strip()
    cfg = get_config()["llm"]

    try:
        response = _get_client().chat.completions.create(
            model=cfg["model"],
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": TRANSLATE_SYSTEM},
                {"role": "user", "content": TRANSLATE_USER.format(query=query)},
            ],
        )
        data = json.loads(response.choices[0].message.content)
        lang = str(data.get("language", "unknown")).lower()
        query_en = str(data.get("query_en", query)).strip() or query

        state["tokens_used"] = state.get("tokens_used", 0) + (
            response.usage.total_tokens if response.usage else 0
        )

    except Exception as exc:
        logger.warning("translate_node failed (%s) — using raw query", exc)
        lang = "unknown"
        query_en = query

    state["lang_detected"] = lang
    state["query_en"] = query_en
    return state
