"""
Hybrid retriever — queries VectorStore and MEDKG (KG) in parallel.

VectorStore gives textbook + BioASQ snippet chunks (Nhánh B).
KG side is called via kg_lookup_fn, which is a stub returning [] until
Nhánh A pipeline is complete and entity linking into MEDKG is wired up.
That integration is deferred to a separate spec.

Usage:
    retriever = HybridRetriever(vector_store=vs, kg_lookup_fn=lambda q: [])
    ctx = retriever.retrieve("What causes Hirschsprung disease?")
    print(ctx.combined_context_text)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from retrieval.vector_store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class HybridContext:
    question: str
    vector_results: list[RetrievedChunk] = field(default_factory=list)
    kg_results: list[dict] = field(default_factory=list)
    combined_context_text: str = ""


class HybridRetriever:
    """Query VectorStore + MEDKG KG in parallel, merge context for LLM prompt."""

    def __init__(
        self,
        vector_store: VectorStore,
        kg_lookup_fn: Callable[[str], list[dict]],
    ) -> None:
        self._vs = vector_store
        self._kg_fn = kg_lookup_fn

    def retrieve(self, question: str, top_k_vector: int = 5) -> HybridContext:
        """
        1. Query VectorStore (textbook + bioasq_snippet, no source filter).
        2. Call kg_lookup_fn(question) — [] if KG not wired up yet.
        3. Build combined_context_text: vector results first, KG edges after.

        TODO: rerank across vector + KG results once a unified relevance score
        is defined. Vector scores are cosine similarity (0-1); KG results have
        no comparable score — do not silently conflate them.
        """
        # VectorStore query (no source type filter — mix textbook + BioASQ)
        vector_results = self._vs.query(question, top_k=top_k_vector)

        # KG lookup (stub returns [] until Nhánh A + entity linking is ready)
        try:
            kg_results = self._kg_fn(question)
        except Exception as exc:
            logger.warning("kg_lookup_fn raised: %s — treating as []", exc)
            kg_results = []

        # Build combined context text
        parts: list[str] = []

        for rc in vector_results:
            c = rc.chunk
            source_label = c.source_name
            if c.section_heading:
                source_label = f"{source_label} - {c.section_heading}"
            parts.append(f"[Nguồn: {source_label}] {c.text}")

        for edge in kg_results:
            subj = edge.get("subject", edge.get("source", ""))
            rel = edge.get("relation", "")
            obj = edge.get("object", edge.get("target", ""))
            pmid = edge.get("pmid", "")
            entry = f"{subj} — {rel} — {obj}"
            if pmid:
                entry += f" [PMID:{pmid}]"
            parts.append(f"[KG] {entry}")

        combined = "\n\n".join(parts)

        return HybridContext(
            question=question,
            vector_results=vector_results,
            kg_results=kg_results,
            combined_context_text=combined,
        )
