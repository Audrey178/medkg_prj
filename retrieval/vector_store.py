"""
VectorStore — FAISS-backed corpus retrieval for Nhánh B.

Embeds VectorChunks (textbook + BioASQ snippets) with BioLORD-2023-C
(fallback: SapBERT) and stores them in a FAISS IndexFlatIP for cosine
similarity search.

IMPORTANT normalization difference from EmbeddingLinker (core/entity_normalizer.py):
  EmbeddingLinker.build_index() does NOT normalize at add time — it normalizes
  only at query time (manually). Here we normalize both at add time AND at
  query time by passing normalize_embeddings=True to model.encode(). This is
  required for IndexFlatIP to behave as true cosine similarity, since FAISS
  does not normalize vectors automatically.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from core.models import VectorChunk

logger = logging.getLogger(__name__)

_PRIMARY_MODEL = "FremyCompany/BioLORD-2023-C"
_FALLBACK_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
_ENCODE_BATCH_SIZE = 32  # BioLORD needs ~300MB/batch on GPU; 32 is safe for 4GB cards


@dataclass
class RetrievedChunk:
    chunk: VectorChunk
    score: float  # cosine similarity, 0–1


class VectorStore:
    """FAISS-backed vector store for textbook and BioASQ snippet chunks."""

    def __init__(
        self,
        model_name: str = _PRIMARY_MODEL,
        index_path: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._model = None
        self._index = None          # faiss.IndexFlatIP
        self._chunks: list[VectorChunk] = []  # parallel to FAISS index rows

        if index_path:
            self.load(index_path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        """Lazy-load BioLORD with SapBERT fallback (mirrors EmbeddingLinker).

        Always loads on CPU for corpus-scale encoding — the GPU (≤4GB) cannot
        hold BioLORD activations for large batches. CPU encoding is slower but
        stable. Query-time latency is still fast (single-sample encode).
        """
        if self._model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s (device=cpu)", self._model_name)
            self._model = SentenceTransformer(self._model_name, device="cpu")
            logger.info("Embedding model loaded: %s", self._model_name)
            return True
        except ImportError:
            logger.error("sentence-transformers not installed. "
                         "Run: pip install sentence-transformers faiss-cpu")
            return False
        except Exception as exc:
            logger.warning("Primary model failed (%s), trying SapBERT fallback", exc)
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(_FALLBACK_MODEL, device="cpu")
                self._model_name = _FALLBACK_MODEL
                logger.info("SapBERT fallback loaded (device=cpu)")
                return True
            except Exception as exc2:
                logger.error("No embedding model available: %s", exc2)
                return False

    def _init_index(self, dim: int) -> None:
        try:
            import faiss
            self._index = faiss.IndexFlatIP(dim)
        except ImportError:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[VectorChunk]) -> None:
        """Encode chunks and add to FAISS index.

        normalize_embeddings=True ensures vectors are L2-normalized before
        being added, so IndexFlatIP inner product == cosine similarity.
        This is intentionally different from EmbeddingLinker.build_index()
        which does NOT normalize at add time.
        """
        if not chunks:
            return
        if not self._ensure_loaded():
            raise RuntimeError("Embedding model could not be loaded")

        texts = [c.text for c in chunks]
        logger.info("Encoding %d chunks (batch_size=%d)...", len(texts), _ENCODE_BATCH_SIZE)
        embeddings = self._model.encode(
            texts,
            batch_size=_ENCODE_BATCH_SIZE,
            normalize_embeddings=True,  # L2-normalize for IndexFlatIP cosine sim
            show_progress_bar=len(texts) > 500,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        if self._index is None:
            self._init_index(embeddings.shape[1])

        self._index.add(embeddings)
        self._chunks.extend(chunks)
        logger.info("VectorStore: %d total chunks in index", len(self._chunks))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        text: str,
        top_k: int = 5,
        filter_source_type: str | None = None,
    ) -> list[RetrievedChunk]:
        """Encode query and return top-k chunks by cosine similarity.

        filter_source_type: "textbook" or "bioasq_snippet" to restrict results.
        Filtering is applied AFTER FAISS search (FAISS has no native filter),
        so we search top_k*3 first to avoid empty results after filtering.
        """
        if self._index is None or not self._chunks:
            return []
        if not self._ensure_loaded():
            return []

        query_emb = self._model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_emb = np.array(query_emb, dtype=np.float32)

        search_k = top_k * 3 if filter_source_type else top_k
        search_k = min(search_k, len(self._chunks))
        scores, indices = self._index.search(query_emb, search_k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            if filter_source_type and chunk.source_type != filter_source_type:
                continue
            results.append(RetrievedChunk(chunk=chunk, score=float(score)))
            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist(self, path: str) -> None:
        """Save FAISS index + chunk metadata to disk.

        Creates two files:
          {path}           — FAISS binary index
          {path}.meta.pkl  — list[VectorChunk] (without embeddings)
        """
        import faiss

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(out))

        meta_path = out.with_suffix(out.suffix + ".meta.pkl")
        chunks_no_emb = [
            VectorChunk(
                chunk_id=c.chunk_id,
                source_type=c.source_type,
                source_name=c.source_name,
                section_heading=c.section_heading,
                text=c.text,
                char_start=c.char_start,
                char_end=c.char_end,
                embedding=None,
            )
            for c in self._chunks
        ]
        with meta_path.open("wb") as fh:
            pickle.dump(chunks_no_emb, fh)

        logger.info("VectorStore persisted: %s (%d chunks)", path, len(self._chunks))

    def load(self, path: str) -> None:
        """Load FAISS index + chunk metadata from disk."""
        import faiss

        idx_path = Path(path)
        meta_path = idx_path.with_suffix(idx_path.suffix + ".meta.pkl")

        if not idx_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"VectorStore files not found at {path}")

        self._index = faiss.read_index(str(idx_path))
        with meta_path.open("rb") as fh:
            self._chunks = pickle.load(fh)

        logger.info("VectorStore loaded: %s (%d chunks)", path, len(self._chunks))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return summary of indexed chunks by source type."""
        by_type: dict[str, int] = {}
        for c in self._chunks:
            by_type[c.source_type] = by_type.get(c.source_type, 0) + 1
        return {"total_chunks": len(self._chunks), "by_source_type": by_type}
