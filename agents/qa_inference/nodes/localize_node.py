"""
Stage 6: Answer localization — translate free-text fields back to source language.

Skipped automatically (no-op) when lang_detected is "en", "unknown", or empty.
Only translates text fields appropriate to the question_type:
  factoid/list  → answer + explanation
  summary       → answer + explanation + key_points
  yes_no / yes_no_maybe / mcq → explanation only (answer is controlled vocabulary)
"""

from __future__ import annotations

import json
import logging
import os

from ..state import QAState
from ..utils.config import get_config
from ..utils.prompts import LOCALIZE_SYSTEM, LOCALIZE_USER

logger = logging.getLogger(__name__)

_client = None

_LANG_NAMES: dict[str, str] = {
    "vi": "Vietnamese",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ar": "Arabic",
    "it": "Italian",
    "nl": "Dutch",
    "ru": "Russian",
    "tr": "Turkish",
    "pl": "Polish",
    "sv": "Swedish",
}

# question_types whose "answer" field is controlled vocabulary — must not translate
_CONTROLLED_VOCAB_TYPES = {"yes_no", "yes_no_maybe", "mcq"}


def _get_client():
    global _client
    if _client is None:
        import openai
        _client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _client


def localize_node(state: QAState) -> QAState:
    lang = state.get("lang_detected", "en")
    answer_dict = state.get("answer")

    if lang in ("en", "unknown", "") or not answer_dict or state.get("error"):
        state["lang_localized"] = False
        return state

    question_type = state.get("question_type", "factoid")

    if question_type in _CONTROLLED_VOCAB_TYPES:
        candidate_fields = ["explanation"]
    elif question_type == "summary":
        candidate_fields = ["answer", "explanation", "key_points"]
    else:
        candidate_fields = ["answer", "explanation"]

    fields = [f for f in candidate_fields if answer_dict.get(f) is not None]
    if not fields:
        state["lang_localized"] = False
        return state

    cfg = get_config()["llm"]
    language_name = _LANG_NAMES.get(lang, lang.upper())

    user_prompt = LOCALIZE_USER.format(
        language_name=language_name,
        lang_code=lang,
        answer_json=json.dumps(answer_dict, ensure_ascii=False, indent=2),
        fields_to_translate=", ".join(fields),
    )

    try:
        response = _get_client().chat.completions.create(
            model=cfg["model"],
            temperature=0.0,
            max_tokens=1024,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": LOCALIZE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )
        translated = json.loads(response.choices[0].message.content)
        state["tokens_used"] = state.get("tokens_used", 0) + (
            response.usage.total_tokens if response.usage else 0
        )
        # Merge only the fields we asked to translate — never overwrite the full dict
        for field in fields:
            if field in translated:
                state["answer"][field] = translated[field]
        state["lang_localized"] = True

    except Exception as exc:
        logger.warning("localize_node failed (%s) — keeping English answer", exc)
        state["lang_localized"] = False

    return state
