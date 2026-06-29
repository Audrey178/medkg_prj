"""
GraphRAG retrieval for ChronoMedKG — Tier 2.1 + 2.2
======================================================
GraphRAGRetriever: question → k-hop entity subgraph → reranked TemporalEdge list.
build_patient_subgraph: case vignette → union neighborhood with age-window filtering.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from core.models import PrimeKGNodeType, TemporalEdge
from core.retrieval_index import AdjacencyIndex, PMIDEdgeIndex

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    question: str
    linked_entities: list[str]           # canonical entity names resolved from question
    subgraph_edges: list[TemporalEdge]   # reranked, capped at top_n
    cited_pmids: list[str]
    insufficient_evidence: bool = False  # True when entities or subgraph are empty


class GraphRAGRetriever:
    """
    Minimal GraphRAG retriever over the ChronoMedKG adjacency index.

    Rerank formula (from spec):
      score = credibility_score * 0.5
            + consensus_confidence * 0.3
            + (1 if quality_grade == "A" else 0.5) * 0.2
    """

    def __init__(
        self,
        edge_index: AdjacencyIndex,
        pmid_index: PMIDEdgeIndex,
        entity_normalizer=None,   # core.entity_normalizer.EntityNormalizer (optional)
        primekg_index=None,       # core.schema_alignment.PrimeKGIndex (optional fallback)
        llm_client=None,          # agents.knowledge_extractor.LLMClient (optional)
    ):
        self._adj = edge_index
        self._pmid = pmid_index
        self._normalizer = entity_normalizer
        self._primekg = primekg_index
        self._llm = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        question: str,
        k_hops: int = 2,
        top_n: int = 20,
        pre_extracted_mentions: list[str] | None = None,
    ) -> RetrievalResult:
        """Return a reranked subgraph for a free-text question."""
        if pre_extracted_mentions is not None:
            mentions = pre_extracted_mentions
        else:
            mentions = self._extract_entity_mentions(question)
        canonical = self._resolve_entities(mentions)

        if not canonical:
            return RetrievalResult(
                question=question,
                linked_entities=[],
                subgraph_edges=[],
                cited_pmids=[],
                insufficient_evidence=True,
            )

        edge_ids = self._bfs_collect(canonical, k_hops)
        edges = [self._pmid.edge_id_to_edge[eid]
                 for eid in edge_ids
                 if eid in self._pmid.edge_id_to_edge]

        if not edges:
            return RetrievalResult(
                question=question,
                linked_entities=list(canonical),
                subgraph_edges=[],
                cited_pmids=[],
                insufficient_evidence=True,
            )

        ranked = sorted(edges, key=self._score, reverse=True)
        pmids = list({
            pmid
            for e in ranked
            for pmid in e.evidence.source_ids
            if pmid
        })

        return RetrievalResult(
            question=question,
            linked_entities=list(canonical),
            subgraph_edges=ranked,
            cited_pmids=pmids,
            insufficient_evidence=False,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_entity_mentions(self, question: str) -> list[str]:
        """Extract candidate entity mentions from question text.

        Uses LLM if available; falls back to capitalised-run heuristic.
        """
        if self._llm is not None:
            try:
                return self._llm_extract_mentions(question)
            except Exception as e:
                logger.warning("LLM entity extraction failed, using heuristic: %s", e)

        # Heuristic fallback: n-gram candidates (1–4 tokens, lowercase) + capitalised runs
        import re
        stopwords = {
            "the", "a", "an", "in", "of", "for", "and", "or", "is", "are",
            "was", "were", "with", "what", "when", "which", "who", "how",
            "does", "do", "have", "has", "been", "be", "this", "that",
            "can", "could", "would", "may", "might", "will", "should",
            "by", "at", "to", "from", "than", "more", "most", "its", "it",
            "not", "no", "if", "then", "also", "but", "so", "as", "on",
        }
        words = re.findall(r'\b\w[\w\-]*\b', question.lower())
        candidates: list[str] = []
        # Slide windows of 1–4 words; skip all-stopword n-grams
        for size in range(4, 0, -1):
            for start in range(len(words) - size + 1):
                chunk = words[start:start + size]
                if any(w not in stopwords and len(w) > 2 for w in chunk):
                    candidates.append(" ".join(chunk))
        return candidates

    def _llm_extract_mentions(self, question: str) -> list[str]:
        prompt = (
            'Extract all biomedical entity mentions (diseases, drugs, genes, symptoms) '
            f'from this question as a JSON list of strings.\nQuestion: {question}\n'
            'Return only: {"entities": [...]}'
        )
        results = self._llm.extract("gpt-4.1-nano", prompt)
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict) and "entities" in first:
                return first["entities"]
        return []

    def _resolve_entities(self, mentions: list[str]) -> set[str]:
        """Map mention strings to canonical entity names (lowercase) in the adjacency index."""
        canonical: set[str] = set()
        for m in mentions:
            m_lower = m.lower()
            # Direct hit in adjacency index (exact)
            if m_lower in self._adj.entity_to_edge_ids:
                canonical.add(m_lower)
                continue
            # EntityNormalizer 3-stage lookup
            if self._normalizer is not None:
                try:
                    result = self._normalizer.normalize(m, entity_type="disease")
                    if result.normalized_name:
                        n = result.normalized_name.lower()
                        if n in self._adj.entity_to_edge_ids:
                            canonical.add(n)
                            continue
                except Exception:
                    pass
            # PrimeKGIndex fuzzy fallback — returns list[PrimeKGNode]
            if self._primekg is not None:
                try:
                    matched = False
                    for node in self._primekg.fuzzy_resolve_name(m):
                        r_lower = node.name_lower
                        if r_lower in self._adj.entity_to_edge_ids:
                            canonical.add(r_lower)
                            matched = True
                            break
                    if matched:
                        continue
                except Exception:
                    pass
            # Last-resort: partial-string match against index keys
            # Handles the case where mention is a prefix of a multi-word entity name
            # (e.g. "Duchenne" matches "duchenne muscular dystrophy")
            for entity_name in self._adj.entity_to_edge_ids:
                if m_lower in entity_name:
                    canonical.add(entity_name)
                    break
        return canonical

    def _bfs_collect(self, seeds: set[str], k_hops: int) -> set[str]:
        """BFS from seed entities to collect edge IDs up to k hops away."""
        visited_entities: set[str] = set()
        collected_edges: set[str] = set()
        queue: deque[tuple[str, int]] = deque((e, 0) for e in seeds)

        while queue:
            entity, depth = queue.popleft()
            if entity in visited_entities:
                continue
            visited_entities.add(entity)

            for eid in self._adj.edges_for_entity(entity):
                collected_edges.add(eid)
                if depth < k_hops:
                    edge = self._pmid.edge_id_to_edge.get(eid)
                    if edge:
                        for neighbour in (edge.source_name.lower(), edge.target_name.lower()):
                            if neighbour not in visited_entities:
                                queue.append((neighbour, depth + 1))

        return collected_edges

    @staticmethod
    def _score(edge: TemporalEdge) -> float:
        """Rerank score per spec: credibility*0.5 + consensus*0.3 + grade*0.2."""
        grade_bonus = 1.0 if edge.quality_grade == "A" else 0.5
        return (
            edge.evidence.credibility_score * 0.5
            + edge.evidence.consensus_confidence * 0.3
            + grade_bonus * 0.2
        )


# ---------------------------------------------------------------------------
# Tier 2.2 — Patient subgraph (one-shot, NOT persisted to KG)
# ---------------------------------------------------------------------------

def build_patient_subgraph(
    retriever: GraphRAGRetriever,
    age: Optional[float],
    symptoms: list[str],
    findings: list[str],
    risk_factors: list[str],
    k_hops: int = 2,
) -> RetrievalResult:
    """
    Build a temporary subgraph for a patient case vignette.

    Unions k-hop neighborhoods of all entities in the vignette, then applies
    age-window filtering using the temporal grounding already on each edge.
    Result is one-shot per question — NOT written back to the KG.
    """
    all_mentions = symptoms + findings + risk_factors
    if not all_mentions:
        return RetrievalResult(
            question="patient_subgraph",
            linked_entities=[],
            subgraph_edges=[],
            cited_pmids=[],
            insufficient_evidence=True,
        )

    canonical = retriever._resolve_entities(all_mentions)
    if not canonical:
        return RetrievalResult(
            question="patient_subgraph",
            linked_entities=[],
            subgraph_edges=[],
            cited_pmids=[],
            insufficient_evidence=True,
        )

    edge_ids = retriever._bfs_collect(canonical, k_hops)
    edges = [retriever._pmid.edge_id_to_edge[eid]
             for eid in edge_ids
             if eid in retriever._pmid.edge_id_to_edge]

    # Age-window filter: drop edges whose temporal onset range excludes the patient's age
    if age is not None:
        filtered = []
        for e in edges:
            t = e.temporal
            if t.onset_age_min is not None and t.onset_age_max is not None:
                if not (t.onset_age_min <= age <= t.onset_age_max):
                    continue  # patient age outside onset window — skip
            filtered.append(e)
        edges = filtered

    if not edges:
        return RetrievalResult(
            question="patient_subgraph",
            linked_entities=list(canonical),
            subgraph_edges=[],
            cited_pmids=[],
            insufficient_evidence=True,
        )

    ranked = sorted(edges, key=retriever._score, reverse=True)
    pmids = list({
        pmid
        for e in ranked
        for pmid in e.evidence.source_ids
        if pmid
    })

    return RetrievalResult(
        question="patient_subgraph",
        linked_entities=list(canonical),
        subgraph_edges=ranked,
        cited_pmids=pmids,
        insufficient_evidence=False,
    )
