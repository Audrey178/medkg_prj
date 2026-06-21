"""
FAISS-based semantic entity index for KG-RAG.

Workflow:
  1. FAISSIndex.build(driver, config) — fetch all entities from Neo4j,
     embed with all-MiniLM-L6-v2, build IndexFlatIP, cache to disk.
  2. FAISSIndex.load(config) — restore from cache.
  3. index.search_fn(query_entity) — returns list[MatchedNode] for cascade_match.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ..state import MatchedNode

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_CACHE_BIN = Path("data/faiss_kg_index.bin")



class FAISSIndex:
    """In-memory FAISS index over KG entity names."""

    def __init__(self, index, meta: list[dict], model):
        self._index = index       # faiss.IndexFlatIP
        self._meta = meta         # [{cui, name, entity_type}, ...]
        self._model = model       # SentenceTransformer

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        driver,
        config: dict | None = None,
        cache_bin: Path | None = None,
        cache_meta: Path | None = None,
    ) -> "FAISSIndex":
        """
        Fetch all entities from Neo4j, embed, build index, cache to disk.
        Returns loaded FAISSIndex.
        """
        import faiss
        from sentence_transformers import SentenceTransformer
        from .neo4j_queries import fetch_all_entities

        cfg = config or {}
        model_name = cfg.get("embedding_model", _DEFAULT_MODEL)
        bin_path = cache_bin or Path(cfg.get("faiss_cache_path", str(_DEFAULT_CACHE_BIN)))
        meta_path = cache_meta or bin_path.with_suffix(".meta.json")

        logger.info("Loading embedding model %s", model_name)
        model = SentenceTransformer(model_name)

        logger.info("Fetching entities from Neo4j")
        entities = fetch_all_entities(driver)
        if not entities:
            raise RuntimeError("No entities found in Neo4j — has the KG been imported?")

        names = [e["name"] for e in entities]
        logger.info("Embedding %d entity names", len(names))
        embeddings = model.encode(names, normalize_embeddings=True, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product on L2-normalized = cosine
        index.add(embeddings)

        # Cache
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(bin_path))
        meta_path.write_text(json.dumps(entities, ensure_ascii=False))
        logger.info("FAISS index saved → %s (%d vectors)", bin_path, index.ntotal)

        return cls(index, entities, model)

    # ------------------------------------------------------------------
    # Load from cache
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        config: dict | None = None,
        cache_bin: Path | None = None,
    ) -> "FAISSIndex":
        """Restore index from disk cache. Raises FileNotFoundError if not built yet."""
        import faiss
        from sentence_transformers import SentenceTransformer

        cfg = config or {}
        model_name = cfg.get("embedding_model", _DEFAULT_MODEL)
        bin_path = cache_bin or Path(cfg.get("faiss_cache_path", str(_DEFAULT_CACHE_BIN)))
        meta_path = bin_path.with_suffix(".meta.json")

        if not bin_path.exists():
            raise FileNotFoundError(
                f"FAISS cache not found at {bin_path}. Run FAISSIndex.build() first."
            )

        logger.info("Loading FAISS index from %s", bin_path)
        index = faiss.read_index(str(bin_path))
        meta = json.loads(meta_path.read_text())
        model = SentenceTransformer(model_name)

        return cls(index, meta, model)

    @classmethod
    def load_or_build(cls, driver, config: dict | None = None) -> "FAISSIndex":
        """Load from cache; build from Neo4j if cache is missing."""
        try:
            return cls.load(config)
        except FileNotFoundError:
            logger.info("Cache miss — building FAISS index from Neo4j")
            return cls.build(driver, config)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 10) -> list[MatchedNode]:
        """
        Embed query, search index, return top-k as MatchedNode list.
        Confidence = cosine similarity (IndexFlatIP on L2-normalized vectors).
        """
        q_emb = self._model.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype=np.float32)

        scores, indices = self._index.search(q_emb, k)
        results: list[MatchedNode] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS pads missing results with -1
                continue
            entity = self._meta[idx]
            results.append(
                MatchedNode(
                    cui=entity["cui"],
                    name=entity["name"],
                    confidence=float(round(float(score), 4)),
                    strategy="semantic",
                )
            )
        return results

    def search_fn(self, query: str) -> list[MatchedNode]:
        """Callable interface for cascade_match's faiss_search_fn parameter."""
        return self.search(query)

    @property
    def size(self) -> int:
        return self._index.ntotal
