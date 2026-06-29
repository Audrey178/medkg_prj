"""
Stage 3: GraphRAG retrieval over ChronoMedKG adjacency + PMID indexes.

Mode handling:
  - llm_only : skip retrieval entirely
  - kg_rag / kg_only : GraphRAGRetriever → TemporalEdge subgraph → serialize to dicts

Data flow:
  extracted_entities (from entity_node LLM) ──► _resolve_entities ──► BFS ──► ranked edges
  Falls back to heuristic entity extraction from raw question if list is empty.

Output: raw_triples (list[dict]) compatible with context_node._serialize_rel().
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from ..state import QAState
from ..utils.config import get_config
from ..utils.prompts import NER_SYSTEM, NER_USER
from core.models import TemporalEdge

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_openai_client_ner = None


def _get_openai_client_ner():
    global _openai_client_ner
    if _openai_client_ner is None:
        import openai
        _openai_client_ner = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _openai_client_ner


def _extract_entities_ner(query: str, cfg: dict) -> tuple[list[str], int]:
    """Extract biomedical entity mentions via NER prompt. Returns (entities, tokens_used)."""
    try:
        response = _get_openai_client_ner().chat.completions.create(
            model=cfg["model"],
            temperature=0.0,
            max_tokens=128,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": NER_SYSTEM},
                {"role": "user", "content": NER_USER.format(query=query)},
            ],
        )
        data = json.loads(response.choices[0].message.content)
        entities = [str(e) for e in data.get("entities", [])]
        tokens = response.usage.total_tokens if response.usage else 0
        return entities, tokens
    except Exception as exc:
        logger.warning("NER entity extraction failed: %s", exc)
        return [], 0
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
PATH_FILE =  Path("data/extracted/validated_triples.jsonl")

# Module-level singletons — built once, reused across all requests
_graph_rag_retriever = None
_primekg_index = None
_entity_normalizer = None
_llm_client = None


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

def _load_local_triples() -> list[TemporalEdge]:
    """Scan data/extracted/*/validated_triples.jsonl and load all TemporalEdges."""
    triples: list[TemporalEdge] = []

    vt = PATH_FILE
    with open(vt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                triples.append(TemporalEdge.from_dict(json.loads(line)))
            except Exception as exc:
                logger.debug("Skip malformed triple in %s: %s", vt.parent.name, exc)

    logger.info("Loaded %d TemporalEdges from %s", len(triples), EXTRACTED_DIR)
    return triples


def _get_primekg_index():
    global _primekg_index
    if _primekg_index is not None:
        return _primekg_index
    try:
        from core.schema_alignment import PrimeKGIndex
        idx = PrimeKGIndex()
        idx.load(lightweight=True)
        _primekg_index = idx
        logger.info("PrimeKGIndex loaded (lightweight)")
    except Exception as exc:
        logger.warning("PrimeKGIndex load failed: %s", exc)
    return _primekg_index


def _get_entity_normalizer(primekg_index=None):
    global _entity_normalizer
    if _entity_normalizer is not None:
        return _entity_normalizer
    try:
        from core.entity_normalizer import EntityNormalizer
        norm = EntityNormalizer(
            use_embeddings=True,  # skip SapBERT — requires separate .venv-sapbert
            use_llm=False,         # skip LLM disambiguation — dictionary-only for speed
            shared_primekg_index=primekg_index,
        )
        norm.initialize()
        _entity_normalizer = norm
        logger.info("EntityNormalizer loaded")
    except Exception as exc:
        logger.warning("EntityNormalizer load failed: %s", exc)
    return _entity_normalizer


def _get_llm_client():
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    try:
        from agents.knowledge_extractor import LLMClient
        _llm_client = LLMClient()
        logger.info("LLMClient loaded for entity extraction")
    except Exception as exc:
        logger.warning("LLMClient init failed — entity extraction will use heuristic: %s", exc)
    return _llm_client


def _get_graph_rag_retriever():
    global _graph_rag_retriever
    if _graph_rag_retriever is not None:
        return _graph_rag_retriever

    try:
        from core.retrieval import GraphRAGRetriever
        from core.retrieval_index import build_indexes

        edges = _load_local_triples()
        if not edges:
            logger.warning("No TemporalEdges loaded — GraphRAGRetriever will always return empty")

        pmid_index, adj_index = build_indexes(edges)

        primekg = _get_primekg_index()
        normalizer = _get_entity_normalizer(primekg)
        llm = _get_llm_client()

        _graph_rag_retriever = GraphRAGRetriever(
            edge_index=adj_index,
            pmid_index=pmid_index,
            entity_normalizer=normalizer,
            primekg_index=primekg,
            llm_client=llm,
        )
        logger.info(
            "GraphRAGRetriever ready: %d entities, %d edges (llm=%s)",
            len(adj_index.entity_to_edge_ids),
            len(pmid_index.edge_id_to_edge),
            "yes" if llm else "no",
        )
    except Exception as exc:
        logger.warning("GraphRAGRetriever init failed: %s", exc)

    return _graph_rag_retriever


# ---------------------------------------------------------------------------
# TemporalEdge → dict bridge (matches context_node._serialize_rel schema)
# ---------------------------------------------------------------------------

def _edge_to_dict(edge: TemporalEdge) -> dict:
    t = edge.temporal
    ev = edge.evidence
    return {
        "direction": "outgoing",
        "relation": edge.relation.value,
        "anchor_name": edge.source_name,
        "source_name": edge.source_name,
        "target_name": edge.target_name,
        "onset_age_min": t.onset_age_min if t else None,
        "onset_age_max": t.onset_age_max if t else None,
        "temporal_qualifier": (t.temporal_qualifier or "") if t else "",
        "progression_stage": (t.progression_stage or "") if t else "",
        "tier": ev.tier.value if (ev and ev.tier) else "",
        "pmid": ev.source_ids[0] if (ev and ev.source_ids) else None,
        "credibility_score": ev.credibility_score if ev else 0.5,
        "is_retracted": ev.is_retracted if ev else False,
        "disease_profile_id": edge.disease_profile_id or "",
        "evidence_text": "",
    }


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def retrieval_node(state: QAState) -> QAState:
    mode = state.get("mode", "kg_rag")

    if mode == "llm_only":
        state["raw_triples"] = []
        state["sources"] = []
        return state

    query = state.get("query_en") or state["query_raw"]
    cfg_llm = get_config()["llm"]
    cfg_ret = get_config()["retrieval"]
    top_n = cfg_ret.get("max_relationships_per_entity", 20)

    # NER: extract biomedical entities before KG retrieval
    entities, ner_tokens = _extract_entities_ner(query, cfg_llm)
    state["extracted_entities"] = entities
    state["tokens_used"] = state.get("tokens_used", 0) + ner_tokens
    logger.debug("NER extracted entities: %s", entities)

    retriever = _get_graph_rag_retriever()
    if retriever is None:
        logger.warning("GraphRAGRetriever unavailable — returning empty context")
        state["raw_triples"] = []
        state["sources"] = []
        state["kg_coverage"] = False
        return state

    try:
        result = retriever.retrieve(query, k_hops=16, top_n=top_n, pre_extracted_mentions=entities or None)
        state["raw_triples"] = [_edge_to_dict(e) for e in result.subgraph_edges]
        state["sources"] = result.cited_pmids
        state["kg_coverage"] = not result.insufficient_evidence
        state["matched_nodes"] = [
            {"name": e, "cui": "", "confidence": 1.0, "strategy": "ner+kg_match"}
            for e in result.linked_entities
        ]
        logger.info(
            "Retrieved %d triples for query %r (linked: %s)",
            len(result.subgraph_edges), query[:60], result.linked_entities,
        )
    except Exception as exc:
        logger.warning("Retrieval failed: %s", exc, exc_info=True)
        state["raw_triples"] = []
        state["sources"] = []
        state["kg_coverage"] = False

    return state
