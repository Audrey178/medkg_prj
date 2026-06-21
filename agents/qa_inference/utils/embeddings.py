"""Singleton sentence-transformer model — shared across context_node and faiss_index."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_model = None


def get_embedding_model(model_name: str = "FremyCompany/BioLORD-2023-C"):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model %s", model_name)
        _model = SentenceTransformer(model_name)
    return _model
