"""
Stage 3: Neo4j neighborhood retrieval.

For each matched entity → fetch relationships (outgoing + incoming).
Collects raw_triples and sources (PMIDs).

Mode handling:
  - llm_only: skip retrieval, return empty
  - kg_rag / kg_only: full retrieval
"""

from __future__ import annotations

import logging

from ..state import QAState
from ..utils.config import get_config

logger = logging.getLogger(__name__)

_driver = None


def _get_driver():
    global _driver
    if _driver is None:
        try:
            from ..utils.neo4j_queries import get_driver
            _driver = get_driver()
        except Exception as exc:
            logger.warning("Neo4j driver init failed: %s", exc)
    return _driver


def retrieval_node(state: QAState) -> QAState:
    mode = state.get("mode", "kg_rag")

    if mode == "llm_only":
        state["raw_triples"] = []
        state["sources"] = []
        return state

    matched_nodes = state.get("matched_nodes", [])
    if not matched_nodes:
        state["raw_triples"] = []
        state["sources"] = []
        return state

    driver = _get_driver()
    if driver is None:
        logger.warning("Neo4j unavailable — no context retrieved")
        state["raw_triples"] = []
        state["sources"] = []
        return state

    cfg = get_config()
    retrieval_cfg = cfg["retrieval"]
    max_rels = retrieval_cfg["max_relationships_per_entity"]

    # Dedup by CUI (keep highest-confidence match per entity)
    seen_cuis: dict[str, dict] = {}
    for node in matched_nodes:
        cui = node["cui"]
        if cui not in seen_cuis or node["confidence"] > seen_cuis[cui]["confidence"]:
            seen_cuis[cui] = node

    # Apply per-source confidence floor
    floor_cfg = retrieval_cfg.get("confidence_floor", {})
    floor = {"option": floor_cfg.get("option", 0.70),
             "clue":   floor_cfg.get("clue",   0.85),
             "stem":   floor_cfg.get("stem",   0.80)}
    matched_nodes = [
        n for n in seen_cuis.values()
        if n["confidence"] >= floor.get(n.get("source", "stem"), floor["stem"])
    ]

    if not matched_nodes:
        state["raw_triples"] = []
        state["sources"] = []
        return state

    from ..utils.neo4j_queries import fetch_entity_neighborhood, extract_pmids

    all_triples: list[dict] = []
    all_pmids: set[str] = set()

    for node in matched_nodes:
        entity_id = node["cui"]
        entity_name = node["name"]
        try:
            meta, rels = fetch_entity_neighborhood(driver, entity_id, max_rels=max_rels)
        except Exception as exc:
            logger.warning("Neighborhood fetch failed for %s: %s", entity_id, exc)
            continue

        # Enrich each rel with anchor entity info and match confidence
        for rel in rels:
            rel["anchor_name"] = entity_name
            rel["anchor_id"] = entity_id
            rel["match_confidence"] = node["confidence"]

        all_triples.extend(rels)
        for pmid in extract_pmids(rels):
            all_pmids.add(pmid)

    state["raw_triples"] = all_triples
    state["sources"] = sorted(all_pmids)
    state["kg_coverage"] = len(all_triples) > 0
    return state
