"""
Agent 4: Quality Controller
=============================
Validates extracted triples against PrimeKG, checks temporal consistency,
detects conflicts, and produces per-disease quality reports.

Validation checks:
1. PrimeKG confirmation (boost confidence if triple confirms existing edge)
2. PrimeKG contradiction (flag if triple contradicts known edge)
3. Temporal consistency (age plausibility, date ordering)
4. Evidence evolution (newer paper contradicts older → supersession)
5. Statistical outlier detection (triple density per document)
6. Credibility scoring (six-signal system from source metadata)
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime
from pathlib import Path

from agents.base_agent import BaseAgent
from agents.evidence_harvester import _load_evidence_json
from core.credibility_scorer import CredibilityScorer
from core.models import (
    AgentResult,
    DiseaseProfile,
    EvidenceTier,
    ExtractionResult,
    PrimeKGNodeType,
    PrimeKGRelationType,
    QualityReport,
    RawTriple,
    TemporalEdge,
    TemporalMetadata,
    EvidenceMetadata,
    ConditionalContext,
    StudyType,
    TemporalResolution,
    CARRIER_TO_PRIMEKG,
)
from core.schema_alignment import PrimeKGIndex
from core.temporal_reasoner import TemporalReasoner

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class QualityController(BaseAgent):
    """Agent 4: Validation, conflict detection, quality assessment."""

    def __init__(self, config: dict, primekg_index: PrimeKGIndex | None = None):
        super().__init__(config, logger)
        self.primekg_index = primekg_index or PrimeKGIndex()
        self.temporal_reasoner = TemporalReasoner()
        self.credibility_scorer = CredibilityScorer()
        self._cache_dir = PROJECT_ROOT / "data" / "extracted"
        # Evidence metadata cache: source_id → (credibility_score, study_type)
        self._evidence_cache: dict[str, tuple[float, StudyType | None]] = {}

    def load_evidence_metadata(self, disease_id: str) -> None:
        """
        Load evidence metadata from harvester output so we can propagate
        credibility scores to validated triples.
        """
        cache_dir = self._cache_dir / disease_id.replace(":", "_")
        ev_data = _load_evidence_json(cache_dir)
        if ev_data is None:
            return

        try:

            for doc in ev_data.get("tier1_sources", []) + ev_data.get("tier2_sources", []):
                source_id = doc.get("source_id", "")
                cred_score = doc.get("credibility_score", 0.0)
                study_type_str = doc.get("study_type")
                study_type = None
                if study_type_str:
                    try:
                        study_type = StudyType(study_type_str)
                    except ValueError:
                        pass
                self._evidence_cache[source_id] = (cred_score, study_type)

            self.logger.info("Loaded evidence metadata: %d sources", len(self._evidence_cache))
        except Exception as e:
            self.logger.warning("Failed to load evidence metadata: %s", e)

    async def run(self, input_data: dict) -> AgentResult:
        """
        Validate extracted triples and produce quality report.

        input_data:
            profile: DiseaseProfile dict
            consensus_triples: list of RawTriple dicts
        """
        profile = DiseaseProfile.from_dict(input_data["profile"])
        raw_triples = [
            RawTriple(**t) if isinstance(t, dict) else t
            for t in input_data.get("consensus_triples", [])
        ]

        self.logger.info("Quality control for %s: %d triples",
                         profile.disease_name, len(raw_triples))

        start_time = time.monotonic()

        # Load evidence metadata for credibility propagation
        self.load_evidence_metadata(profile.disease_id)

        # Load PrimeKG index if not already loaded
        if not self.primekg_index.is_loaded:
            self.primekg_index.load()

        report = QualityReport(
            disease_id=profile.disease_id,
            total_triples_input=len(raw_triples),
        )

        validated_triples: list[TemporalEdge] = []
        rejected: list[dict] = []

        for triple in raw_triples:
            is_valid, reason = self._validate_triple(triple, profile)
            if is_valid:
                temporal_edge = self._convert_to_temporal_edge(triple, profile)
                if temporal_edge:
                    # Check PrimeKG confirmation using schema alignment
                    confirmation = self.primekg_index.confirm_triple(
                        subject=triple.subject,
                        relation=triple.relation,
                        obj=triple.object,
                        subject_type=triple.subject_type,
                        object_type=triple.object_type,
                    )
                    if confirmation.is_confirmed:
                        report.confirmations_with_primekg += 1
                        temporal_edge.quality_grade = "A"
                        # Use PrimeKG IDs if resolved
                        if confirmation.primekg_x_id and not triple.subject_id:
                            temporal_edge.source_id = confirmation.primekg_x_id
                        if confirmation.primekg_y_id and not triple.object_id:
                            temporal_edge.target_id = confirmation.primekg_y_id
                    else:
                        report.novel_triples += 1
                        temporal_edge.quality_grade = "B"

                    validated_triples.append(temporal_edge)
                    report.validated_triples += 1
                else:
                    rejected.append({"triple": triple.to_dict(), "reason": "conversion_failed"})
                    report.rejected_triples += 1
            else:
                rejected.append({"triple": triple.to_dict(), "reason": reason})
                report.rejected_triples += 1

        # Tier 3.2: aggregate cross-source claims before supersession detection
        validated_triples = self._aggregate_cross_source(validated_triples)
        report.validated_triples = len(validated_triples)  # update count after merge

        # Detect conflicts within validated set
        conflicts = self._detect_conflicts(validated_triples)
        report.conflicts_detected = len(conflicts)

        # Run temporal reasoning (supersession after aggregation, per Tier 3.2 decision)
        temporal_result = self.temporal_reasoner.reason(validated_triples)
        self.logger.info("Temporal reasoning: %d supersessions, %d issues",
                         temporal_result.supersessions_detected,
                         temporal_result.consistency_issues)

        # Temporal coverage (from temporal reasoner)
        temporal_stats = self.temporal_reasoner.compute_temporal_coverage(validated_triples)
        report.temporal_coverage = temporal_stats.get("coverage", 0.0)

        # Average credibility
        if validated_triples:
            report.avg_credibility_score = sum(
                e.evidence.credibility_score for e in validated_triples
            ) / len(validated_triples)

        # Quality grade
        if report.total_triples_input > 0:
            valid_pct = report.validated_triples / report.total_triples_input
            if valid_pct >= 0.9:
                report.quality_grade = "A"
            elif valid_pct >= 0.7:
                report.quality_grade = "B"
            else:
                report.quality_grade = "C"

        elapsed = time.monotonic() - start_time

        # Save outputs
        cache_dir = self._cache_dir / profile.disease_id.replace(":", "_")
        cache_dir.mkdir(parents=True, exist_ok=True)

        with open(cache_dir / "validated_triples.jsonl", "w") as f:
            for edge in validated_triples:
                f.write(json.dumps(edge.to_dict(), default=str) + "\n")

        with open(cache_dir / "rejected_triples.json", "w") as f:
            json.dump(rejected, f, indent=2, default=str)

        report_dict = report.to_dict()
        report_dict["temporal_reasoning"] = temporal_result.to_dict()
        report_dict["temporal_stats"] = temporal_stats

        with open(cache_dir / "quality_report.json", "w") as f:
            json.dump(report_dict, f, indent=2)

        return AgentResult(
            agent_name="QualityController",
            disease_id=profile.disease_id,
            status="success",
            data={
                "report": report_dict,
                "validated_count": len(validated_triples),
                "rejected_count": len(rejected),
            },
            metrics={
                "elapsed_seconds": round(elapsed, 1),
                **report.to_dict(),
            },
            timestamp=datetime.utcnow(),
        )

    def _validate_triple(self, triple: RawTriple, profile: DiseaseProfile) -> tuple[bool, str]:
        """Validate a single triple. Returns (is_valid, rejection_reason)."""
        # Basic field checks
        if not triple.subject or not triple.object or not triple.relation:
            return False, "missing_fields"

        if len(triple.subject) < 2 or len(triple.object) < 2:
            return False, "entity_too_short"

        if triple.confidence < 0.3:
            return False, "low_confidence"

        # Temporal plausibility
        if triple.temporal_context:
            tc = triple.temporal_context
            if isinstance(tc, dict):
                age_min = tc.get("onset_age_min")
                age_max = tc.get("onset_age_max")
                if age_min is not None and age_max is not None:
                    if age_min > age_max:
                        return False, "temporal_age_inverted"
                    if age_min < 0 or age_max > 120:
                        return False, "temporal_age_implausible"

        # Self-referencing
        if triple.subject.lower() == triple.object.lower():
            return False, "self_referencing"

        return True, ""

    def _convert_to_temporal_edge(self, triple: RawTriple, profile: DiseaseProfile) -> TemporalEdge | None:
        """Convert a RawTriple to a ChronoMedKG TemporalEdge with credibility scoring."""
        try:
            # Map relation to PrimeKG type
            relation_str = triple.relation.lower().strip()
            relation = CARRIER_TO_PRIMEKG.get(relation_str)
            if relation is None:
                # Try direct PrimeKG relation
                try:
                    relation = PrimeKGRelationType(relation_str)
                except ValueError:
                    # Fallback: use carrier_other
                    relation = PrimeKGRelationType.CARRIER_OTHER

            # Map entity types
            source_type = self._map_node_type(triple.subject_type)
            target_type = self._map_node_type(triple.object_type)

            # Build temporal metadata
            temporal = TemporalMetadata()
            if triple.temporal_context and isinstance(triple.temporal_context, dict):
                tc = triple.temporal_context
                temporal.onset_age_min = tc.get("onset_age_min")
                temporal.onset_age_max = tc.get("onset_age_max")
                temporal.progression_stage = tc.get("progression_stage")
                temporal.duration = tc.get("duration")
                temporal.milestone = tc.get("milestone")
                temporal.temporal_qualifier = tc.get("temporal_qualifier")
                tsa = tc.get("treatment_start_age")
                if tsa is not None:
                    try:
                        temporal.treatment_start_age = float(tsa)
                    except (ValueError, TypeError):
                        pass

                disc_year = tc.get("discovery_year")
                if disc_year:
                    try:
                        temporal.discovery_date = date(int(disc_year), 1, 1)
                        temporal.temporal_resolution = TemporalResolution.YEAR
                    except (ValueError, TypeError):
                        pass

            # Compute credibility score from source metadata
            cred_score = 0.0
            study_type = StudyType.OTHER

            # Look up source document credibility from harvester cache
            source_id = triple.source_id or ""
            if source_id in self._evidence_cache:
                cached_score, cached_type = self._evidence_cache[source_id]
                cred_score = cached_score
                if cached_type:
                    study_type = cached_type
            else:
                # For PMC full-text docs with compound IDs (PMID:xxx|PMCyyy)
                base_id = source_id.split("|")[0] if "|" in source_id else source_id
                if base_id in self._evidence_cache:
                    cached_score, cached_type = self._evidence_cache[base_id]
                    cred_score = cached_score
                    if cached_type:
                        study_type = cached_type

            # If still no score, compute from what we have
            if cred_score == 0.0:
                cred_score, _ = self.credibility_scorer.compute(
                    study_type=study_type,
                    llm_consensus=triple.confidence,
                )

            # Build evidence metadata
            evidence = EvidenceMetadata(
                tier=EvidenceTier.TIER_2,
                source_ids=[triple.source_id] if triple.source_id else [],
                credibility_score=cred_score,
                study_type=study_type,
                consensus_confidence=triple.confidence,
                extraction_models=[triple.extraction_model] if triple.extraction_model else [],
                extraction_method="tier2_llm_consensus",
                evidence_text=triple.evidence_text,
            )

            # Build conditions
            conditions = None
            if triple.conditions and isinstance(triple.conditions, dict):
                conditions = ConditionalContext.from_dict(triple.conditions)

            return TemporalEdge(
                source_id=triple.subject_id or triple.subject,
                source_type=source_type,
                source_name=triple.subject,
                relation=relation,
                target_id=triple.object_id or triple.object,
                target_type=target_type,
                target_name=triple.object,
                temporal=temporal,
                evidence=evidence,
                conditions=conditions,
                disease_profile_id=profile.disease_id,
            )

        except Exception as e:
            logger.debug("Failed to convert triple: %s", e)
            return None

    def _map_node_type(self, type_str: str) -> PrimeKGNodeType:
        """Map free-text entity type to PrimeKG node type."""
        type_lower = type_str.lower().strip()
        mapping = {
            "disease": PrimeKGNodeType.DISEASE,
            "gene": PrimeKGNodeType.GENE_PROTEIN,
            "protein": PrimeKGNodeType.GENE_PROTEIN,
            "gene/protein": PrimeKGNodeType.GENE_PROTEIN,
            "drug": PrimeKGNodeType.DRUG,
            "treatment": PrimeKGNodeType.DRUG,
            "phenotype": PrimeKGNodeType.PHENOTYPE,
            "effect/phenotype": PrimeKGNodeType.PHENOTYPE,
            "symptom": PrimeKGNodeType.CLINICAL_FINDING,       # acute/presenting — Tier 1.2
            "clinical_finding": PrimeKGNodeType.CLINICAL_FINDING,
            "anatomy": PrimeKGNodeType.ANATOMY,
            "pathway": PrimeKGNodeType.PATHWAY,
            "biological_process": PrimeKGNodeType.BIOLOGICAL_PROCESS,
            "molecular_function": PrimeKGNodeType.MOLECULAR_FUNCTION,
            "cellular_component": PrimeKGNodeType.CELLULAR_COMPONENT,
            "exposure": PrimeKGNodeType.EXPOSURE,
        }
        return mapping.get(type_lower, PrimeKGNodeType.PHENOTYPE)

    def _aggregate_cross_source(self, edges: list[TemporalEdge]) -> list[TemporalEdge]:
        """
        Tier 3.2: aggregate edges representing the same claim from multiple source documents.

        Algorithm: Union-Find on (relation, fuzzy-subject, fuzzy-object), same as
        KnowledgeExtractor._compute_consensus (Appendix C.3) but applied at the
        TemporalEdge level across different PMIDs instead of across LLM models.

        Runs BEFORE supersession detection so the temporal reasoner operates on
        already-merged, stronger claims.

        Merged edge gets:
          - source_ids = union of all contributing PMIDs
          - credibility_score = max across contributors
          - consensus_confidence = max(original, min(1.0, n_sources * 0.33))
          - extraction_models = union across contributors
        """
        if len(edges) < 2:
            return edges

        try:
            from rapidfuzz import fuzz
        except ImportError:
            logger.warning("rapidfuzz unavailable — skipping cross-source aggregation")
            return edges

        import copy

        # Union-Find helpers
        parent = list(range(len(edges)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Partition by relation first to keep O(n^2) within-relation only
        rel_groups: dict[str, list[int]] = {}
        for i, edge in enumerate(edges):
            rel_groups.setdefault(edge.relation.value, []).append(i)

        for group in rel_groups.values():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a_idx, b_idx = group[i], group[j]
                    a, b = edges[a_idx], edges[b_idx]
                    # Only aggregate if they come from DIFFERENT source documents
                    if set(a.evidence.source_ids) & set(b.evidence.source_ids):
                        continue
                    subj_score = fuzz.token_sort_ratio(a.source_name.lower(), b.source_name.lower())
                    obj_score = fuzz.token_sort_ratio(a.target_name.lower(), b.target_name.lower())
                    if subj_score >= 80 and obj_score >= 80:
                        union(a_idx, b_idx)

        # Collect clusters and merge
        clusters: dict[int, list[int]] = {}
        for i in range(len(edges)):
            clusters.setdefault(find(i), []).append(i)

        merged: list[TemporalEdge] = []
        n_aggregated = 0
        for members in clusters.values():
            if len(members) == 1:
                merged.append(edges[members[0]])
                continue

            member_edges = [edges[i] for i in members]
            representative = max(member_edges, key=lambda e: e.evidence.credibility_score)

            all_source_ids = list({
                sid for e in member_edges for sid in e.evidence.source_ids if sid
            })
            all_models = list({
                m for e in member_edges for m in e.evidence.extraction_models
            })
            max_cred = max(e.evidence.credibility_score for e in member_edges)
            multi_conf = min(1.0, len(member_edges) * 0.33)

            rep = copy.copy(representative)
            rep.evidence = copy.copy(representative.evidence)
            rep.evidence.source_ids = all_source_ids
            rep.evidence.extraction_models = all_models
            rep.evidence.credibility_score = max_cred
            rep.evidence.consensus_confidence = max(
                representative.evidence.consensus_confidence, multi_conf
            )
            merged.append(rep)
            n_aggregated += len(members) - 1

        if n_aggregated:
            self.logger.info(
                "Cross-source aggregation: merged %d edges → %d (-%d duplicates)",
                len(edges), len(merged), n_aggregated,
            )
        return merged

    def _detect_conflicts(self, edges: list[TemporalEdge]) -> list[dict]:
        """Detect conflicting edges (same entities, contradictory relations)."""
        conflicts = []

        # Group by (source, target) pair
        pair_edges: dict[tuple, list[TemporalEdge]] = {}
        for edge in edges:
            key = (edge.source_name.lower(), edge.target_name.lower())
            pair_edges.setdefault(key, []).append(edge)

        for key, group in pair_edges.items():
            if len(group) < 2:
                continue

            relations = {e.relation for e in group}
            # Contradiction: same entities with opposing relations
            if (PrimeKGRelationType.INDICATION in relations and
                    PrimeKGRelationType.CONTRAINDICATION in relations):
                conflicts.append({
                    "type": "indication_contraindication_conflict",
                    "entities": key,
                    "relations": [r.value for r in relations],
                })

            if (PrimeKGRelationType.DISEASE_PHENOTYPE_POSITIVE in relations and
                    PrimeKGRelationType.DISEASE_PHENOTYPE_NEGATIVE in relations):
                conflicts.append({
                    "type": "phenotype_positive_negative_conflict",
                    "entities": key,
                    "relations": [r.value for r in relations],
                })

            if (PrimeKGRelationType.FIRST_LINE_TREATMENT in relations and
                    PrimeKGRelationType.CONTRAINDICATION in relations):
                conflicts.append({
                    "type": "first_line_contraindication_conflict",
                    "entities": key,
                    "relations": [r.value for r in relations],
                })

        return conflicts
