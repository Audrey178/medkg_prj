"""
Paper Credibility Scorer
========================
Six-signal credibility scoring system for source documents.

Signals:
1. Journal tier (IF-based quartile)
2. Citation velocity (citations/year, field-normalized)
3. Study type (meta-analysis=1.0 → editorial=0.1)
4. Replication status (LLM + citation analysis)
5. Retraction status (CrossRef / Retraction Watch)
6. LLM consensus (multi-model agreement — computed during extraction)

Score = weighted combination of all signals.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date
from typing import Optional

from core.models import StudyType, STUDY_TYPE_WEIGHT

logger = logging.getLogger(__name__)

# Default weights from config/default.yaml
DEFAULT_WEIGHTS = {
    "journal_tier": 0.15,
    "citation_velocity": 0.15,
    "study_type": 0.25,
    "replication_status": 0.15,
    "retraction_status": 0.15,
    "llm_consensus": 0.15,
}

# Journal Impact Factor tiers (approximate quartiles for biomedical)
# Q1 >= 10, Q2 >= 5, Q3 >= 2, Q4 < 2
JOURNAL_TIER_MAP = {
    "Q1": 1.0,  # IF >= 10 (e.g., NEJM, Lancet, Nature Medicine)
    "Q2": 0.75,  # IF 5-10 (e.g., Neuromuscular Disorders)
    "Q3": 0.5,  # IF 2-5
    "Q4": 0.25,  # IF < 2
    "preprint": 0.15,  # bioRxiv, medRxiv
    "unknown": 0.3,
}

# Known high-impact journals (hardcoded for speed, expanded over time)
HIGH_IMPACT_JOURNALS: dict[str, str] = {
    # Q1 journals (IF >= 10)
    "The New England journal of medicine": "Q1",
    "Lancet": "Q1",
    "Nature medicine": "Q1",
    "Nature genetics": "Q1",
    "Nature": "Q1",
    "Science": "Q1",
    "JAMA": "Q1",
    "BMJ": "Q1",
    "Annals of neurology": "Q1",
    "Brain": "Q1",
    "Neurology": "Q1",
    "The Lancet. Neurology": "Q1",
    "Nature reviews. Neurology": "Q1",
    "Nature machine intelligence": "Q1",
    "Genome research": "Q1",
    "American journal of human genetics": "Q1",
    "Genetics in medicine": "Q1",
    "npj digital medicine": "Q1",
    # Q2 journals (IF 5-10)
    "Neuromuscular disorders": "Q2",
    "Muscle & nerve": "Q2",
    "Journal of neurology, neurosurgery, and psychiatry": "Q2",
    "Journal of neurology": "Q2",
    "European journal of neurology": "Q2",
    "Orphanet journal of rare diseases": "Q2",
    "Human mutation": "Q2",
    "Human molecular genetics": "Q2",
    "Molecular therapy": "Q2",
    "Gene therapy": "Q2",
    "Scientific data": "Q2",
}


@dataclass
class CredibilitySignals:
    """Individual signal scores before weighting."""
    journal_tier: float = 0.3       # default unknown
    citation_velocity: float = 0.0
    study_type: float = 0.1
    replication_status: float = 0.5  # neutral default
    retraction_status: float = 1.0   # not retracted
    llm_consensus: float = 0.0      # filled during extraction

    def to_dict(self) -> dict:
        return {
            "journal_tier": self.journal_tier,
            "citation_velocity": self.citation_velocity,
            "study_type": self.study_type,
            "replication_status": self.replication_status,
            "retraction_status": self.retraction_status,
            "llm_consensus": self.llm_consensus,
        }


class CredibilityScorer:
    """Compute credibility score for a source document."""

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or DEFAULT_WEIGHTS

    def score_journal(self, journal_name: str | None) -> float:
        """Score based on journal impact factor tier."""
        if not journal_name:
            return JOURNAL_TIER_MAP["unknown"]

        # Check hardcoded map (case-insensitive)
        name_lower = journal_name.lower().strip()
        for known_journal, tier in HIGH_IMPACT_JOURNALS.items():
            if known_journal.lower() in name_lower or name_lower in known_journal.lower():
                return JOURNAL_TIER_MAP[tier]

        # Preprint detection
        if any(p in name_lower for p in ["biorxiv", "medrxiv", "arxiv", "preprint"]):
            return JOURNAL_TIER_MAP["preprint"]

        return JOURNAL_TIER_MAP["unknown"]

    def score_citation_velocity(
        self,
        citation_count: int | None,
        publication_date: date | None,
    ) -> float:
        """Score based on citations per year, normalized."""
        if not citation_count or not publication_date:
            return 0.0

        years_since = max(0.5, (date.today() - publication_date).days / 365.25)
        velocity = citation_count / years_since

        # Normalize: 0-5 citations/year = 0-0.5, 5-50 = 0.5-0.9, 50+ = 0.9-1.0
        if velocity >= 50:
            return min(1.0, 0.9 + (velocity - 50) / 500)
        elif velocity >= 5:
            return 0.5 + (velocity - 5) / 45 * 0.4
        else:
            return velocity / 5 * 0.5

    def score_study_type(self, study_type: StudyType | str | None) -> float:
        """Score based on study type hierarchy."""
        if study_type is None:
            return STUDY_TYPE_WEIGHT.get(StudyType.OTHER, 0.1)
        if isinstance(study_type, str):
            try:
                study_type = StudyType(study_type)
            except ValueError:
                return STUDY_TYPE_WEIGHT.get(StudyType.OTHER, 0.1)
        return STUDY_TYPE_WEIGHT.get(study_type, 0.1)

    def score_retraction(self, is_retracted: bool) -> float:
        """Score based on retraction status. Retracted = 0."""
        return 0.0 if is_retracted else 1.0

    def compute(
        self,
        journal_name: str | None = None,
        citation_count: int | None = None,
        publication_date: date | None = None,
        study_type: StudyType | str | None = None,
        is_retracted: bool = False,
        replication_score: float = 0.5,
        llm_consensus: float = 0.0,
    ) -> tuple[float, CredibilitySignals]:
        """
        Compute overall credibility score from all available signals.

        Returns:
            (score, signals) — overall score 0-1 and individual signal breakdown
        """
        signals = CredibilitySignals(
            journal_tier=self.score_journal(journal_name),
            citation_velocity=self.score_citation_velocity(citation_count, publication_date),
            study_type=self.score_study_type(study_type),
            replication_status=replication_score,
            retraction_status=self.score_retraction(is_retracted),
            llm_consensus=llm_consensus,
        )

        score = sum(
            self.weights[signal_name] * getattr(signals, signal_name)
            for signal_name in self.weights
        )

        return round(min(1.0, max(0.0, score)), 4), signals

    def score_tier1_source(self, source_type: str) -> tuple[float, CredibilitySignals]:
        """
        Tier 1 sources (GeneReviews, OMIM, Orphanet) get high default scores.
        """
        signals = CredibilitySignals(
            journal_tier=1.0,
            citation_velocity=0.8,
            study_type=STUDY_TYPE_WEIGHT[StudyType.DATABASE],
            replication_status=0.9,
            retraction_status=1.0,
            llm_consensus=1.0,
        )
        score = sum(
            self.weights[signal_name] * getattr(signals, signal_name)
            for signal_name in self.weights
        )
        return round(min(1.0, score), 4), signals
