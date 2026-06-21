"""
Stage 4: Serialize raw_triples → natural language sentences, rank, filter.

Ranking: combined_score = α * credibility_score + (1-α) * cosine(query, sentence)
Filter: if sentences > threshold (150), keep top-k by combined_score.
Retracted relationships are always excluded.
"""

from __future__ import annotations

import logging

import numpy as np

from ..state import QAState
from ..utils.config import get_config

logger = logging.getLogger(__name__)

# Human-readable labels for Neo4j relationship types
_REL_LABELS: dict[str, str] = {
    "DISEASE_PHENOTYPE_POSITIVE": "is associated with phenotype",
    "DISEASE_PHENOTYPE_NEGATIVE": "is NOT associated with phenotype",
    "DISEASE_DISEASE": "is related to disease",
    "DISEASE_PROTEIN": "is associated with gene/protein",
    "INDICATION": "is indicated for treating",
    "CONTRAINDICATION": "is contraindicated for",
    "DRUG_EFFECT": "has drug effect",
    "DRUG_PROTEIN": "targets gene/protein",
    "BIOPROCESS_PROTEIN": "involves gene/protein in biological process",
    "PHENOTYPE_PROTEIN": "is associated with gene/protein",
    "PROTEIN_PROTEIN": "interacts with protein",
    "PATHWAY_PROTEIN": "involves gene/protein in pathway",
    "OTHER": "is related to",
}

_INCOMING_FLIP: dict[str, str] = {
    "DISEASE_PHENOTYPE_POSITIVE": "is a phenotype of",
    "DISEASE_PHENOTYPE_NEGATIVE": "is NOT a phenotype of",
    "DISEASE_DISEASE": "is related to disease",
    "DISEASE_PROTEIN": "is a gene/protein associated with",
    "INDICATION": "can be treated by",
    "CONTRAINDICATION": "is contraindicated with",
    "DRUG_EFFECT": "is an effect of drug",
    "DRUG_PROTEIN": "is targeted by",
    "BIOPROCESS_PROTEIN": "participates in biological process involving",
    "PHENOTYPE_PROTEIN": "is a gene/protein associated with phenotype",
    "PROTEIN_PROTEIN": "interacts with protein",
    "PATHWAY_PROTEIN": "is a gene/protein in pathway",
    "OTHER": "is related to",
}


def _serialize_rel(rel: dict) -> str:
    """Convert a relationship dict to a natural language sentence."""
    direction = rel.get("direction", "outgoing")
    relation = rel.get("relation", "OTHER")
    anchor = rel.get("anchor_name", "Unknown entity")

    if direction == "outgoing":
        neighbour = rel.get("target_name", "")
        verb = _REL_LABELS.get(relation, "is related to")
        sentence = f"{anchor} {verb} {neighbour}"
    else:
        neighbour = rel.get("source_name", "")
        verb = _INCOMING_FLIP.get(relation, "is related to")
        sentence = f"{neighbour} {verb} {anchor}"

    # Append temporal info if available
    parts = []
    onset_min = rel.get("onset_age_min")
    onset_max = rel.get("onset_age_max")
    if onset_min is not None or onset_max is not None:
        if onset_min is not None and onset_max is not None:
            parts.append(f"onset: {onset_min}–{onset_max} years")
        elif onset_min is not None:
            parts.append(f"onset after {onset_min} years")
        else:
            parts.append(f"onset before {onset_max} years")

    tq = rel.get("temporal_qualifier", "")
    if tq:
        parts.append(tq)

    ps = rel.get("progression_stage", "")
    if ps:
        parts.append(f"stage: {ps}")
    
    tier = rel.get("tier", "")
    if tier:
        parts.append(f"tier: {tier}")
    
    pmid = rel.get("pmid")
    if pmid:
        parts.append(f"PMID: {pmid}")
    
    disease_profile_id = rel.get("disease_profile_id")
    if disease_profile_id:
        parts.append(f"disease profile_id: {disease_profile_id}")
    
    envidence_text = rel.get("evidence_text", "")
    if envidence_text:
        parts.append(f"evidence: {envidence_text}")

    if parts:
        sentence += f" [{'\n ' .join(parts)}]"

    sentence += "."
    return sentence


def _cosine_scores(query: str, sentences: list[str], model_name: str) -> np.ndarray:
    """Return cosine similarity between query and each sentence."""
    try:
        from ..utils.embeddings import get_embedding_model
        model = get_embedding_model(model_name)
        all_texts = [query] + sentences
        embs = model.encode(all_texts, normalize_embeddings=True, show_progress_bar=False)
        q_emb = embs[0]
        s_embs = embs[1:]
        return np.dot(s_embs, q_emb)  # cosine on normalized = dot product
    except Exception as exc:
        logger.warning("Cosine scoring failed (%s) — using uniform scores", exc)
        return np.ones(len(sentences), dtype=np.float32)


def context_node(state: QAState) -> QAState:
    raw_triples = state.get("raw_triples", [])
    query_en = state.get("query_en") or state["query_raw"]
    benchmark = state.get("benchmark_type", "bioasq")
    cfg = get_config()
    retrieval_cfg = cfg["retrieval"]
    threshold = retrieval_cfg.get("context_volume_by_benchmark", {}).get(
        benchmark, retrieval_cfg["context_volume_threshold"]
    )
    alpha = cfg["ranking"]["alpha"]
    model_name = cfg["matching"]["embedding_model"]

    # Filter retracted relationships
    valid = [r for r in raw_triples if not r.get("is_retracted")]

    # Serialize to sentences
    sentences: list[str] = []
    credibility_scores: list[float] = []

    for rel in valid:
        try:
            sent = _serialize_rel(rel)
        except Exception:
            continue
        sentences.append(sent)
        score = rel.get("credibility_score")
        credibility_scores.append(float(score) if score is not None else 0.5)

    if not sentences:
        state["context_sentences"] = []
        state["context_filtered"] = False
        return state

    # Rank and filter if over threshold
    if len(sentences) > threshold:
        cosine = _cosine_scores(query_en, sentences, model_name)
        cred_arr = np.array(credibility_scores, dtype=np.float32)
        combined = alpha * cred_arr + (1 - alpha) * cosine

        top_k_idx = np.argsort(combined)[::-1][:threshold]
        sentences = [sentences[i] for i in top_k_idx]
        context_filtered = True
    else:
        context_filtered = False

    state["context_sentences"] = sentences
    state["context_filtered"] = context_filtered
    return state
