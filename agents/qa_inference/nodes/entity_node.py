"""
Stage 1: Biomedical NER + 4-strategy entity matching.

Flow:
  BioASQ/PubMedQA (stem extraction):
    1. Extract entity names from English query via GPT-4o-mini (focused NER)
    2. Expand abbreviations via built-in dict
    3. cascade_match against Neo4j candidates; source="stem"

  MEDQA (MCQ-aware extraction):
    1. Extract option_entities from answer options + clinical_clues from stem
    2. cascade_match option_entities → source="option"
    3. cascade_match clinical_clues → source="clue"

  FAISS semantic search used if index is available.
  Deduplicated by CUI (highest confidence kept); capped at 10.

Mode handling:
  - llm_only: extract entities (for intent node), skip matching
  - kg_rag / kg_only: full extraction + matching
"""

from __future__ import annotations

import json
import logging
import os

from ..state import QAState, MatchedNode
from ..utils.config import get_config
from ..utils.matching import cascade_match
from ..utils.prompts import NER_SYSTEM, NER_USER, NER_SYSTEM_MCQ, NER_USER_MCQ

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Common biomedical abbreviations (supplements Neo4j aliases which are empty)
# ---------------------------------------------------------------------------

_ABBR_EXPAND: dict[str, str] = {
    "DMD": "Duchenne muscular dystrophy",
    "BMD": "Becker muscular dystrophy",
    "ALS": "amyotrophic lateral sclerosis",
    "MS": "multiple sclerosis",
    "SMA": "spinal muscular atrophy",
    "CF": "cystic fibrosis",
    "PKU": "phenylketonuria",
    "HD": "Huntington disease",
    "AD": "Alzheimer disease",
    "PD": "Parkinson disease",
    "RA": "rheumatoid arthritis",
    "SLE": "systemic lupus erythematosus",
    "IBD": "inflammatory bowel disease",
    "CKD": "chronic kidney disease",
    "T1D": "type 1 diabetes mellitus",
    "T2D": "type 2 diabetes mellitus",
    "NF1": "neurofibromatosis type 1",
    "NF2": "neurofibromatosis type 2",
    "TSC": "tuberous sclerosis complex",
    "PWS": "Prader-Willi syndrome",
    "AS": "Angelman syndrome",
    "FMR1": "fragile X mental retardation 1",
    "FXTAS": "fragile X-associated tremor/ataxia syndrome",
    "BRCA1": "breast cancer 1 gene",
    "BRCA2": "breast cancer 2 gene",
    "EGFR": "epidermal growth factor receptor",
    "TNF": "tumor necrosis factor",
    "IL6": "interleukin 6",
    "IL1B": "interleukin 1 beta",
}

# ---------------------------------------------------------------------------
# Module-level lazy caches
# ---------------------------------------------------------------------------

_openai_client = None
_candidates_cache: list[dict] | None = None
_faiss_cache = None  # FAISSIndex | None
_neo4j_driver = None


def _get_openai():
    global _openai_client
    if _openai_client is None:
        import openai
        _openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _openai_client


def _get_driver():
    global _neo4j_driver
    if _neo4j_driver is None:
        try:
            from ..utils.neo4j_queries import get_driver
            _neo4j_driver = get_driver()
        except Exception as exc:
            logger.warning("Neo4j driver init failed: %s", exc)
    return _neo4j_driver


def _get_candidates() -> list[dict]:
    global _candidates_cache
    if _candidates_cache is None:
        driver = _get_driver()
        if driver is None:
            _candidates_cache = []
            return _candidates_cache
        try:
            from ..utils.neo4j_queries import fetch_all_entities
            _candidates_cache = fetch_all_entities(driver)
            logger.info("Loaded %d KG entity candidates", len(_candidates_cache))
        except Exception as exc:
            logger.warning("fetch_all_entities failed: %s", exc)
            _candidates_cache = []
    return _candidates_cache


def _get_faiss(config: dict):
    global _faiss_cache
    if _faiss_cache is None:
        try:
            from ..utils.faiss_index import FAISSIndex
            driver = _get_driver()
            _faiss_cache = FAISSIndex.load_or_build(driver, config.get("matching", {}))
            logger.info("FAISS index loaded (%d vectors)", _faiss_cache.size)
        except FileNotFoundError:
            logger.info("FAISS cache not found — semantic search disabled")
        except Exception as exc:
            logger.warning("FAISS load failed: %s", exc)
    return _faiss_cache


# ---------------------------------------------------------------------------
# NER via GPT-4o-mini
# ---------------------------------------------------------------------------

def _extract_entities_llm(query: str, cfg: dict) -> tuple[list[str], int]:
    """BioASQ/PubMedQA: return (entity_names, tokens_used). Falls back to [] on error."""
    try:
        response = _get_openai().chat.completions.create(
            model=cfg["model"],
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": NER_SYSTEM},
                {"role": "user", "content": NER_USER.format(query=query)},
            ],
        )
        data = json.loads(response.choices[0].message.content)
        entities = [str(e).strip() for e in data.get("entities", []) if e]
        tokens = response.usage.total_tokens if response.usage else 0
        return entities, tokens
    except Exception as exc:
        logger.warning("NER extraction failed: %s", exc)
        return [], 0


def _extract_entities_mcq_llm(
    query: str, choices: dict, cfg: dict
) -> tuple[list[str], list[str], int]:
    """MEDQA: return (option_entities, clinical_clues, tokens_used)."""
    options_text = "\n".join(f"{k}. {v}" for k, v in sorted(choices.items()))
    try:
        response = _get_openai().chat.completions.create(
            model=cfg["model"],
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": NER_SYSTEM_MCQ},
                {"role": "user", "content": NER_USER_MCQ.format(
                    query=query, options_text=options_text
                )},
            ],
        )
        data = json.loads(response.choices[0].message.content)
        option_ents = [str(e).strip() for e in data.get("option_entities", []) if e]
        clues = [str(e).strip() for e in data.get("clinical_clues", []) if e]
        tokens = response.usage.total_tokens if response.usage else 0
        return option_ents, clues, tokens
    except Exception as exc:
        logger.warning("MCQ NER extraction failed: %s", exc)
        # Fallback: use option values as-is
        fallback = [str(v).strip() for v in choices.values() if v]
        return fallback, [], 0


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def _match_entities(
    entities: list[str],
    source: str,
    candidates: list[dict],
    faiss_fn,
    match_cfg: dict,
    accumulated: dict[str, MatchedNode],
) -> None:
    """Match a list of entity strings, tag with source, accumulate into dict keyed by CUI."""
    for entity in entities:
        hits = cascade_match(entity, candidates, faiss_search_fn=faiss_fn, config=match_cfg)
        for h in hits:
            h["source"] = source
            if h["cui"] not in accumulated or h["confidence"] > accumulated[h["cui"]]["confidence"]:
                accumulated[h["cui"]] = h

        expanded = _ABBR_EXPAND.get(entity.upper())
        if expanded:
            hits_exp = cascade_match(expanded, candidates, faiss_search_fn=faiss_fn, config=match_cfg)
            for h in hits_exp:
                h["source"] = source
                if h["cui"] not in accumulated or h["confidence"] > accumulated[h["cui"]]["confidence"]:
                    accumulated[h["cui"]] = h


def entity_node(state: QAState) -> QAState:
    query_en = state["query_en"] or state["query_raw"]
    mode = state.get("mode", "kg_rag")
    benchmark = state.get("benchmark_type", "bioasq")
    cfg = get_config()

    # 1. Extract entities — branch on benchmark type
    if benchmark == "medqa":
        choices = state.get("options", {}).get("choices", {})
        option_ents, clues, tokens = _extract_entities_mcq_llm(query_en, choices, cfg["llm"])
        state["extracted_entities"] = option_ents + clues
    else:
        entities, tokens = _extract_entities_llm(query_en, cfg["llm"])
        state["extracted_entities"] = entities

    state["tokens_used"] = state.get("tokens_used", 0) + tokens

    # llm_only: skip KG matching
    if mode == "llm_only":
        state["matched_nodes"] = []
        return state

    # 2. Load KG candidates + optional FAISS
    candidates = _get_candidates()
    faiss = _get_faiss(cfg)
    faiss_fn = faiss.search_fn if faiss is not None else None
    match_cfg = cfg.get("matching", {})

    # 3. Match entities with source tagging
    all_matched: dict[str, MatchedNode] = {}  # cui → best match

    if benchmark == "medqa":
        _match_entities(option_ents, "option", candidates, faiss_fn, match_cfg, all_matched)
        _match_entities(clues, "clue", candidates, faiss_fn, match_cfg, all_matched)
    else:
        _match_entities(state["extracted_entities"], "stem", candidates, faiss_fn, match_cfg, all_matched)

    # 4. Sort by confidence, cap at 10
    matched = sorted(all_matched.values(), key=lambda x: x["confidence"], reverse=True)[:10]

    state["matched_nodes"] = matched
    state["kg_coverage"] = len(matched) > 0
    return state
