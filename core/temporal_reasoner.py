"""
Temporal Reasoner
==================
Handles temporal inference for ChronoMedKG edges:

1. Supersession detection: newer evidence contradicts older → mark old as superseded
2. Validity period inference: set validity_start/validity_end from evidence dates
3. Temporal consistency: detect impossible temporal claims
4. Evidence evolution: track how understanding of a relationship changes over time

This module operates on validated TemporalEdge objects (after Quality Controller).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from core.models import (
    PrimeKGRelationType,
    TemporalEdge,
    TemporalMetadata,
    TemporalResolution,
)

logger = logging.getLogger(__name__)


@dataclass
class SupersessionEvent:
    """Records when one edge supersedes another."""
    old_edge_id: str
    new_edge_id: str
    reason: str  # "newer_evidence", "contradictory", "updated_treatment", "retracted"
    old_discovery_date: Optional[date] = None
    new_discovery_date: Optional[date] = None


@dataclass
class TemporalConsistencyIssue:
    """A detected temporal inconsistency."""
    edge_id: str
    issue_type: str  # "future_date", "age_implausible", "duration_conflict", "date_ordering"
    description: str
    severity: str = "warning"  # "warning" or "error"


@dataclass
class TemporalReasoningResult:
    """Output of temporal reasoning over a set of edges."""
    total_edges: int = 0
    supersessions_detected: int = 0
    validity_periods_inferred: int = 0
    consistency_issues: int = 0
    supersession_events: list[SupersessionEvent] = field(default_factory=list)
    issues: list[TemporalConsistencyIssue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_edges": self.total_edges,
            "supersessions_detected": self.supersessions_detected,
            "validity_periods_inferred": self.validity_periods_inferred,
            "consistency_issues": self.consistency_issues,
            "supersession_events": [
                {"old": s.old_edge_id, "new": s.new_edge_id, "reason": s.reason}
                for s in self.supersession_events
            ],
            "issues": [
                {"edge_id": i.edge_id, "type": i.issue_type, "description": i.description}
                for i in self.issues
            ],
        }


# Relations where newer evidence typically supersedes older
SUPERSEDABLE_RELATIONS = {
    PrimeKGRelationType.INDICATION,
    PrimeKGRelationType.CONTRAINDICATION,
    PrimeKGRelationType.CARRIER_TREATS,
    PrimeKGRelationType.OFF_LABEL_USE,
}

# Relations where contradictory pairs indicate supersession
CONTRADICTORY_PAIRS = [
    (PrimeKGRelationType.INDICATION, PrimeKGRelationType.CONTRAINDICATION),
    (PrimeKGRelationType.DISEASE_PHENOTYPE_POSITIVE, PrimeKGRelationType.DISEASE_PHENOTYPE_NEGATIVE),
]


class TemporalReasoner:
    """
    Infers temporal structure from extracted edges.

    Operates on a list of TemporalEdge objects for a single disease,
    detecting supersession chains and inferring validity periods.
    """

    def __init__(self):
        self._today = date.today()

    def reason(self, edges: list[TemporalEdge]) -> TemporalReasoningResult:
        """
        Run all temporal reasoning on a set of edges.

        Mutates edges in-place (sets validity_start, validity_end, superseded_by).
        Returns a summary of what was inferred.
        """
        result = TemporalReasoningResult(total_edges=len(edges))

        # Step 1: Infer validity_start from discovery dates
        for edge in edges:
            inferred = self._infer_validity_start(edge)
            if inferred:
                result.validity_periods_inferred += 1

        # Step 2: Detect supersession (newer contradicts older)
        supersessions = self._detect_supersessions(edges)
        result.supersessions_detected = len(supersessions)
        result.supersession_events = supersessions

        # Apply supersession: set validity_end on old edges
        for event in supersessions:
            old_edge = next((e for e in edges if e.edge_id == event.old_edge_id), None)
            new_edge = next((e for e in edges if e.edge_id == event.new_edge_id), None)
            if old_edge and new_edge:
                old_edge.temporal.validity_end = event.new_discovery_date
                old_edge.temporal.superseded_by = event.new_edge_id

        # Step 3: Temporal consistency checks
        issues = self._check_consistency(edges)
        result.consistency_issues = len(issues)
        result.issues = issues

        return result

    def _infer_validity_start(self, edge: TemporalEdge) -> bool:
        """
        Infer validity_start from available temporal signals.

        Priority:
        1. Explicit validity_start (already set)
        2. discovery_date (when the evidence was published)
        3. Source document publication date (fallback)

        Returns True if a validity_start was inferred.
        """
        if edge.temporal.validity_start is not None:
            return False  # Already set

        # Use discovery_date as validity_start
        if edge.temporal.discovery_date:
            edge.temporal.validity_start = edge.temporal.discovery_date
            return True

        # Fallback: use extraction_date as a lower bound
        # (we know the edge is valid at least as of when we found it)
        # Don't set this — it's not a real validity start

        return False

    def _detect_supersessions(self, edges: list[TemporalEdge]) -> list[SupersessionEvent]:
        """
        Detect when newer evidence supersedes older evidence.

        Supersession scenarios:
        1. Same (subject, object) but contradictory relations (e.g., indication → contraindication)
        2. Same (subject, relation, object) but different temporal scope
           (e.g., "Drug X treats Disease Y" with 2020 evidence replacing 2005 evidence
            when the newer explicitly negates or refines the older)
        3. Treatment evolution: newer drug replaces older for same indication
        """
        events = []

        # Group edges by (subject_id/name, object_id/name)
        entity_pairs: dict[tuple[str, str], list[TemporalEdge]] = defaultdict(list)
        for edge in edges:
            key = (
                (edge.source_id or edge.source_name).lower(),
                (edge.target_id or edge.target_name).lower(),
            )
            entity_pairs[key].append(edge)

        # Check each pair for contradictory relations
        for pair_key, group in entity_pairs.items():
            if len(group) < 2:
                continue

            # Sort by discovery date (oldest first)
            dated = [e for e in group if e.temporal.discovery_date]
            if len(dated) < 2:
                continue
            dated.sort(key=lambda e: e.temporal.discovery_date)

            # Check for contradictory relation pairs
            for contra_a, contra_b in CONTRADICTORY_PAIRS:
                edges_a = [e for e in dated if e.relation == contra_a]
                edges_b = [e for e in dated if e.relation == contra_b]

                if edges_a and edges_b:
                    # Newer contradicts older
                    oldest_a = min(edges_a, key=lambda e: e.temporal.discovery_date)
                    newest_b = max(edges_b, key=lambda e: e.temporal.discovery_date)

                    if newest_b.temporal.discovery_date > oldest_a.temporal.discovery_date:
                        events.append(SupersessionEvent(
                            old_edge_id=oldest_a.edge_id,
                            new_edge_id=newest_b.edge_id,
                            reason="contradictory",
                            old_discovery_date=oldest_a.temporal.discovery_date,
                            new_discovery_date=newest_b.temporal.discovery_date,
                        ))

        # Treatment evolution: same disease, same relation (treats/indication),
        # different drug, newer replaces older
        # Group by (disease, relation=treats/indication)
        treatment_groups: dict[str, list[TemporalEdge]] = defaultdict(list)
        for edge in edges:
            if edge.relation in SUPERSEDABLE_RELATIONS:
                # Key by the disease entity (could be source or target)
                disease_key = None
                if edge.source_type.value == "disease":
                    disease_key = (edge.source_id or edge.source_name).lower()
                elif edge.target_type.value == "disease":
                    disease_key = (edge.target_id or edge.target_name).lower()
                if disease_key:
                    treatment_groups[disease_key].append(edge)

        # Within each disease's treatments, check for explicit supersession signals
        # (e.g., "replaced by", "no longer recommended" in evidence text)
        for disease_key, treatments in treatment_groups.items():
            dated_treatments = [t for t in treatments if t.temporal.discovery_date]
            if len(dated_treatments) < 2:
                continue

            dated_treatments.sort(key=lambda e: e.temporal.discovery_date)

            # Check evidence text for supersession language
            supersession_phrases = [
                "replaced by", "no longer recommended", "superseded",
                "withdrawn", "discontinued", "obsolete", "preferred over",
            ]
            for newer in dated_treatments:
                if not newer.evidence or not newer.evidence.evidence_text:
                    continue
                text_lower = newer.evidence.evidence_text.lower()
                for phrase in supersession_phrases:
                    if phrase in text_lower:
                        # Find which older treatment is being superseded
                        for older in dated_treatments:
                            if (older.edge_id != newer.edge_id and
                                    older.temporal.discovery_date < newer.temporal.discovery_date):
                                # Check if the older treatment entity is mentioned
                                older_drug = older.source_name if older.source_type.value == "drug" else older.target_name
                                if older_drug.lower() in text_lower:
                                    events.append(SupersessionEvent(
                                        old_edge_id=older.edge_id,
                                        new_edge_id=newer.edge_id,
                                        reason="updated_treatment",
                                        old_discovery_date=older.temporal.discovery_date,
                                        new_discovery_date=newer.temporal.discovery_date,
                                    ))
                        break

        return events

    def _check_consistency(self, edges: list[TemporalEdge]) -> list[TemporalConsistencyIssue]:
        """
        Check temporal consistency of edges.

        Detects:
        1. Future discovery dates
        2. Implausible age ranges
        3. validity_end before validity_start
        4. Duration conflicts
        """
        issues = []

        for edge in edges:
            t = edge.temporal

            # Future discovery date
            if t.discovery_date and t.discovery_date > self._today:
                issues.append(TemporalConsistencyIssue(
                    edge_id=edge.edge_id,
                    issue_type="future_date",
                    description=f"Discovery date {t.discovery_date} is in the future",
                    severity="error",
                ))

            # Implausible age ranges
            if t.onset_age_min is not None and t.onset_age_max is not None:
                if t.onset_age_min > t.onset_age_max:
                    issues.append(TemporalConsistencyIssue(
                        edge_id=edge.edge_id,
                        issue_type="age_implausible",
                        description=f"onset_age_min ({t.onset_age_min}) > onset_age_max ({t.onset_age_max})",
                        severity="error",
                    ))
                if t.onset_age_max > 120:
                    issues.append(TemporalConsistencyIssue(
                        edge_id=edge.edge_id,
                        issue_type="age_implausible",
                        description=f"onset_age_max ({t.onset_age_max}) exceeds 120 years",
                        severity="warning",
                    ))

            # Date ordering: validity_end before validity_start
            if t.validity_start and t.validity_end:
                if t.validity_end < t.validity_start:
                    issues.append(TemporalConsistencyIssue(
                        edge_id=edge.edge_id,
                        issue_type="date_ordering",
                        description=f"validity_end ({t.validity_end}) before validity_start ({t.validity_start})",
                        severity="error",
                    ))

            # Very old discovery dates (before 1900) are suspicious
            if t.discovery_date and t.discovery_date.year < 1900:
                issues.append(TemporalConsistencyIssue(
                    edge_id=edge.edge_id,
                    issue_type="date_ordering",
                    description=f"Discovery date {t.discovery_date} seems too old (before 1900)",
                    severity="warning",
                ))

        return issues

    def compute_temporal_coverage(self, edges: list[TemporalEdge]) -> dict:
        """
        Compute temporal coverage statistics for a set of edges.
        Returns detailed breakdown of temporal metadata completeness.
        """
        total = len(edges)
        if total == 0:
            return {"total": 0, "coverage": 0.0}

        has_discovery_date = sum(1 for e in edges if e.temporal.discovery_date)
        has_validity_start = sum(1 for e in edges if e.temporal.validity_start)
        has_validity_end = sum(1 for e in edges if e.temporal.validity_end)
        has_onset_age = sum(
            1 for e in edges
            if e.temporal.onset_age_min is not None or e.temporal.onset_age_max is not None
        )
        has_progression = sum(1 for e in edges if e.temporal.progression_stage)
        has_duration = sum(1 for e in edges if e.temporal.duration)
        has_milestone = sum(1 for e in edges if getattr(e.temporal, 'milestone', None))
        has_temporal_qualifier = sum(1 for e in edges if getattr(e.temporal, 'temporal_qualifier', None))
        has_treatment_start_age = sum(
            1 for e in edges if getattr(e.temporal, 'treatment_start_age', None) is not None
        )
        has_any_temporal = sum(
            1 for e in edges
            if (e.temporal.discovery_date or e.temporal.onset_age_min is not None
                or e.temporal.progression_stage or e.temporal.duration
                or getattr(e.temporal, 'milestone', None)
                or getattr(e.temporal, 'temporal_qualifier', None)
                or (getattr(e.temporal, 'treatment_start_age', None) is not None))
        )

        # Resolution breakdown
        resolution_counts = defaultdict(int)
        for e in edges:
            resolution_counts[e.temporal.temporal_resolution.value] += 1

        return {
            "total": total,
            "coverage": has_any_temporal / total,
            "discovery_date": has_discovery_date / total,
            "validity_start": has_validity_start / total,
            "validity_end": has_validity_end / total,
            "onset_age": has_onset_age / total,
            "progression_stage": has_progression / total,
            "duration": has_duration / total,
            "milestone": has_milestone / total,
            "temporal_qualifier": has_temporal_qualifier / total,
            "treatment_start_age": has_treatment_start_age / total,
            "resolution": dict(resolution_counts),
        }
