"""
ChronoMedKG Data Models
=========================
Universal data models for the ChronoMedKG pipeline.
Extends Paper 1 (HEG-TKG) models to be disease-autonomous and PrimeKG-aligned.

Key differences from Paper 1 (tier1/models.py):
- Entity types aligned with PrimeKG's 10 node types (not custom per disease)
- Relation types mapped to PrimeKG's 30 edge types
- Temporal metadata as first-class citizen on every edge
- Evidence hierarchy with credibility scoring
- Conditional context (inspired by AutoBioKG composite triples)
- No hardcoded disease-specific entities — everything from config/LLM
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# PrimeKG-aligned enums
# ---------------------------------------------------------------------------

class PrimeKGNodeType(Enum):
    """PrimeKG node types (10 types)."""
    DISEASE = "disease"
    GENE_PROTEIN = "gene/protein"
    DRUG = "drug"
    BIOLOGICAL_PROCESS = "biological_process"
    MOLECULAR_FUNCTION = "molecular_function"
    CELLULAR_COMPONENT = "cellular_component"
    ANATOMY = "anatomy"
    PATHWAY = "pathway"
    PHENOTYPE = "phenotype"
    EXPOSURE = "exposure"


class PrimeKGRelationType(Enum):
    """PrimeKG relation types (30 types, grouped by category)."""
    # Disease-gene
    DISEASE_PROTEIN = "disease_protein"

    # Drug-gene
    DRUG_PROTEIN = "drug_protein"

    # Drug-disease
    INDICATION = "indication"
    CONTRAINDICATION = "contraindication"
    OFF_LABEL_USE = "off-label use"

    # Disease-phenotype
    PHENOTYPE_PROTEIN = "phenotype_protein"
    DISEASE_PHENOTYPE_POSITIVE = "disease_phenotype_positive"
    DISEASE_PHENOTYPE_NEGATIVE = "disease_phenotype_negative"

    # Drug effects
    DRUG_EFFECT = "drug_effect"

    # Protein-protein
    PROTEIN_PROTEIN = "protein_protein"

    # Biological scale
    BIOPROCESS_PROTEIN = "bioprocess_protein"
    MOLFUNC_PROTEIN = "molfunc_protein"
    CELLCOMP_PROTEIN = "cellcomp_protein"

    # Exposure
    EXPOSURE_DISEASE = "exposure_disease"
    EXPOSURE_PROTEIN = "exposure_protein"
    EXPOSURE_BIOPROCESS = "exposure_bioprocess"
    EXPOSURE_MOLFUNC = "exposure_molfunc"
    EXPOSURE_CELLCOMP = "exposure_cellcomp"

    # Pathway
    PATHWAY_PROTEIN = "pathway_protein"

    # Anatomy
    ANATOMY_PROTEIN_EXPRESSED = "anatomy_protein_expressed"
    ANATOMY_PROTEIN_ABSENT = "anatomy_protein_absent"

    # Disease-disease
    DISEASE_DISEASE = "disease_disease"

    # Carriers from Paper 1 that map onto the above but we need for extraction
    # These get normalized to PrimeKG types before ingestion
    CARRIER_TREATS = "treats"
    CARRIER_MANIFESTS_AS = "manifests_as"
    CARRIER_CAUSED_BY = "caused_by"
    CARRIER_BIOMARKER_FOR = "biomarker_for"
    CARRIER_PROGRESSES_TO = "progresses_to"
    CARRIER_DIFFERENTIATES = "differentiates"
    CARRIER_ONSET_AT = "onset_at"
    CARRIER_OTHER = "other"


# Mapping from carrier (extraction) relations to PrimeKG edge types
CARRIER_TO_PRIMEKG: dict[str, PrimeKGRelationType] = {
    "treats": PrimeKGRelationType.INDICATION,
    "manifests_as": PrimeKGRelationType.DISEASE_PHENOTYPE_POSITIVE,
    "caused_by": PrimeKGRelationType.DISEASE_PROTEIN,
    "biomarker_for": PrimeKGRelationType.DISEASE_PROTEIN,
    "progresses_to": PrimeKGRelationType.DISEASE_DISEASE,
    "differentiates": PrimeKGRelationType.DISEASE_DISEASE,
    "onset_at": PrimeKGRelationType.DISEASE_PHENOTYPE_POSITIVE,
}

# ---------------------------------------------------------------------------
# Credibility scoring enums and dataclasses
# ---------------------------------------------------------------------------

class EvidenceTier(Enum):
    """Evidence hierarchy tiers."""
    TIER_1 = 1  # Curated databases: GeneReviews, OMIM, Orphanet, guidelines
    TIER_2 = 2  # Literature extraction: PubMed abstracts, PMC full-text


class StudyType(Enum):
    """Study type for credibility scoring."""
    META_ANALYSIS = "meta-analysis"
    RCT = "RCT"
    COHORT = "cohort"
    CASE_CONTROL = "case-control"
    CASE_REPORT = "case-report"
    CASE_SERIES = "case-series"
    REVIEW = "review"
    GUIDELINE = "guideline"
    DATABASE = "database"
    EXPERT_OPINION = "expert-opinion"
    OTHER = "other"


STUDY_TYPE_WEIGHT: dict[StudyType, float] = {
    StudyType.META_ANALYSIS: 1.0,
    StudyType.GUIDELINE: 0.95,
    StudyType.RCT: 0.9,
    StudyType.COHORT: 0.7,
    StudyType.CASE_CONTROL: 0.6,
    StudyType.CASE_SERIES: 0.4,
    StudyType.CASE_REPORT: 0.3,
    StudyType.REVIEW: 0.5,
    StudyType.DATABASE: 0.85,
    StudyType.EXPERT_OPINION: 0.2,
    StudyType.OTHER: 0.1,
}


class TemporalResolution(Enum):
    """How precise the temporal grounding is."""
    EXACT_DATE = "exact_date"
    YEAR = "year"
    DECADE = "decade"
    UNKNOWN = "unknown"


class CoverageFlag(Enum):
    """Disease coverage quality in ChronoMedKG."""
    RICH = "rich"        # >500 PubMed articles + Tier 1 sources
    MODERATE = "moderate"  # 50-500 articles OR Tier 1 only
    SPARSE = "sparse"    # <50 articles, limited Tier 1


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------

@dataclass
class TemporalMetadata:
    """
    Temporal grounding for an edge.
    Every ChronoMedKG edge carries this — no edge is timeless.
    """
    discovery_date: Optional[date] = None
    validity_start: Optional[date] = None
    validity_end: Optional[date] = None
    superseded_by: Optional[str] = None  # edge_id of superseding edge
    temporal_resolution: TemporalResolution = TemporalResolution.UNKNOWN

    # Clinical temporal context (disease-stage-specific)
    onset_age_min: Optional[float] = None  # years
    onset_age_max: Optional[float] = None
    progression_stage: Optional[str] = None
    duration: Optional[str] = None  # acute | chronic | episodic | progressive

    # Rich temporal signals (extracted by LLMs, previously dropped)
    milestone: Optional[str] = None  # e.g., "loss of ambulation", "ventilation required"
    temporal_qualifier: Optional[str] = None  # e.g., "by 6 weeks", "before age 10", "FDA approved 2017"
    treatment_start_age: Optional[float] = None  # years — when treatment typically begins

    def to_dict(self) -> dict:
        return {
            "discovery_date": self.discovery_date.isoformat() if self.discovery_date else None,
            "validity_start": self.validity_start.isoformat() if self.validity_start else None,
            "validity_end": self.validity_end.isoformat() if self.validity_end else None,
            "superseded_by": self.superseded_by,
            "temporal_resolution": self.temporal_resolution.value,
            "onset_age_min": self.onset_age_min,
            "onset_age_max": self.onset_age_max,
            "progression_stage": self.progression_stage,
            "duration": self.duration,
            "milestone": self.milestone,
            "temporal_qualifier": self.temporal_qualifier,
            "treatment_start_age": self.treatment_start_age,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TemporalMetadata:
        if not data:
            return cls()
        return cls(
            discovery_date=date.fromisoformat(data["discovery_date"]) if data.get("discovery_date") else None,
            validity_start=date.fromisoformat(data["validity_start"]) if data.get("validity_start") else None,
            validity_end=date.fromisoformat(data["validity_end"]) if data.get("validity_end") else None,
            superseded_by=data.get("superseded_by"),
            temporal_resolution=TemporalResolution(data.get("temporal_resolution", "unknown")),
            onset_age_min=data.get("onset_age_min"),
            onset_age_max=data.get("onset_age_max"),
            progression_stage=data.get("progression_stage"),
            duration=data.get("duration"),
            milestone=data.get("milestone"),
            temporal_qualifier=data.get("temporal_qualifier"),
            treatment_start_age=data.get("treatment_start_age"),
        )


@dataclass
class EvidenceMetadata:
    """
    Evidence hierarchy and provenance for an edge.
    """
    tier: EvidenceTier
    source_ids: list[str] = field(default_factory=list)  # PMIDs, GeneReviews IDs, OMIM IDs
    credibility_score: float = 0.0  # 0-1, from six-signal scoring
    study_type: StudyType = StudyType.OTHER
    consensus_confidence: float = 0.0  # Multi-LLM agreement (1.0 = all models agree)
    extraction_models: list[str] = field(default_factory=list)
    extraction_method: str = "tier2_llm_consensus"  # "tier1_curated" | "tier2_llm_consensus"
    citation_count: Optional[int] = None
    is_retracted: bool = False
    evidence_text: Optional[str] = None  # Supporting text snippet

    def to_dict(self) -> dict:
        return {
            "tier": self.tier.value,
            "source_ids": self.source_ids,
            "credibility_score": self.credibility_score,
            "study_type": self.study_type.value,
            "consensus_confidence": self.consensus_confidence,
            "extraction_models": self.extraction_models,
            "extraction_method": self.extraction_method,
            "citation_count": self.citation_count,
            "is_retracted": self.is_retracted,
            "evidence_text": self.evidence_text,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EvidenceMetadata:
        return cls(
            tier=EvidenceTier(data["tier"]),
            source_ids=data.get("source_ids", []),
            credibility_score=data.get("credibility_score", 0.0),
            study_type=StudyType(data.get("study_type", "other")),
            consensus_confidence=data.get("consensus_confidence", 0.0),
            extraction_models=data.get("extraction_models", []),
            extraction_method=data.get("extraction_method", "tier2_llm_consensus"),
            citation_count=data.get("citation_count"),
            is_retracted=data.get("is_retracted", False),
            evidence_text=data.get("evidence_text"),
        )


@dataclass
class ConditionalContext:
    """
    Conditional context for an edge (inspired by AutoBioKG composite triples).
    Captures UNDER WHAT CONDITIONS a relationship holds.
    """
    age_group: Optional[str] = None        # pediatric | adult | elderly | all
    genetic_subtype: Optional[str] = None  # e.g., exon_deletion, missense
    disease_stage: Optional[str] = None    # e.g., ambulatory, non-ambulatory
    population: Optional[str] = None       # e.g., european, asian
    treatment_line: Optional[str] = None   # first-line | second-line | adjunct
    sex: Optional[str] = None              # male | female | all

    def to_dict(self) -> dict:
        d = {}
        for fld in ["age_group", "genetic_subtype", "disease_stage",
                     "population", "treatment_line", "sex"]:
            val = getattr(self, fld)
            if val is not None:
                d[fld] = val
        return d if d else None

    @classmethod
    def from_dict(cls, data: dict | None) -> ConditionalContext | None:
        if not data:
            return None
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TemporalEdge:
    """
    A single edge in ChronoMedKG.
    Extends PrimeKG edge with temporal + evidence + conditional metadata.
    """
    # Core triple (PrimeKG-aligned)
    source_id: str           # Ontology ID (MONDO, NCBI Gene, DrugBank, etc.)
    source_type: PrimeKGNodeType
    source_name: str
    relation: PrimeKGRelationType
    target_id: str
    target_type: PrimeKGNodeType
    target_name: str

    # ChronoMedKG extensions
    temporal: TemporalMetadata = field(default_factory=TemporalMetadata)
    evidence: EvidenceMetadata = field(default_factory=lambda: EvidenceMetadata(tier=EvidenceTier.TIER_2))
    conditions: Optional[ConditionalContext] = None

    # Provenance
    extraction_date: date = field(default_factory=date.today)
    pipeline_version: str = "1.0.0"
    disease_profile_id: Optional[str] = None
    quality_grade: Optional[str] = None  # A, B, C from Quality Controller

    @property
    def edge_id(self) -> str:
        """Deterministic edge ID for deduplication and supersession tracking."""
        content = f"{self.source_id}|{self.relation.value}|{self.target_id}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "source_type": self.source_type.value,
            "source_name": self.source_name,
            "relation": self.relation.value,
            "target_id": self.target_id,
            "target_type": self.target_type.value,
            "target_name": self.target_name,
            "temporal": self.temporal.to_dict(),
            "evidence": self.evidence.to_dict(),
            "conditions": self.conditions.to_dict() if self.conditions else None,
            "extraction_date": self.extraction_date.isoformat(),
            "pipeline_version": self.pipeline_version,
            "disease_profile_id": self.disease_profile_id,
            "quality_grade": self.quality_grade,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TemporalEdge:
        return cls(
            source_id=data["source_id"],
            source_type=PrimeKGNodeType(data["source_type"]),
            source_name=data["source_name"],
            relation=PrimeKGRelationType(data["relation"]),
            target_id=data["target_id"],
            target_type=PrimeKGNodeType(data["target_type"]),
            target_name=data["target_name"],
            temporal=TemporalMetadata.from_dict(data.get("temporal", {})),
            evidence=EvidenceMetadata.from_dict(data["evidence"]),
            conditions=ConditionalContext.from_dict(data.get("conditions")),
            extraction_date=date.fromisoformat(data["extraction_date"]) if data.get("extraction_date") else date.today(),
            pipeline_version=data.get("pipeline_version", "1.0.0"),
            disease_profile_id=data.get("disease_profile_id"),
            quality_grade=data.get("quality_grade"),
        )

    def to_primekg_csv_row(self) -> dict:
        """Export as PrimeKG-compatible CSV row with temporal extensions appended."""
        return {
            "relation": self.relation.value,
            "display_relation": self.relation.value,
            "x_id": self.source_id,
            "x_type": self.source_type.value,
            "x_name": self.source_name,
            "x_source": "ChronoMedKG",
            "y_id": self.target_id,
            "y_type": self.target_type.value,
            "y_name": self.target_name,
            "y_source": "ChronoMedKG",
            # Temporal extensions
            "discovery_date": self.temporal.discovery_date.isoformat() if self.temporal.discovery_date else "",
            "validity_start": self.temporal.validity_start.isoformat() if self.temporal.validity_start else "",
            "validity_end": self.temporal.validity_end.isoformat() if self.temporal.validity_end else "",
            "evidence_tier": self.evidence.tier.value,
            "credibility_score": self.evidence.credibility_score,
            "study_type": self.evidence.study_type.value,
            "consensus_confidence": self.evidence.consensus_confidence,
            "source_ids": json.dumps(self.evidence.source_ids),
            "conditions": json.dumps(self.conditions.to_dict()) if self.conditions else "",
        }


# ---------------------------------------------------------------------------
# Agent communication models
# ---------------------------------------------------------------------------

@dataclass
class DiseaseProfile:
    """
    Complete disease profile generated by Agent 1 (Disease Profiler).
    Parameterizes all downstream agents — no per-disease code needed.
    """
    # Identity
    disease_id: str                # Primary ID (OMIM preferred)
    disease_name: str
    synonyms: list[str] = field(default_factory=list)
    orphanet_id: Optional[str] = None
    omim_id: Optional[str] = None
    mondo_id: Optional[str] = None

    # Ontological context
    disease_category: str = ""         # neuromuscular, neurological, metabolic, etc.
    inheritance_pattern: str = ""      # AD, AR, XLR, sporadic, etc.
    disease_type: str = ""             # rare_genetic, autoimmune, infectious, etc.

    # Differential diagnosis partners
    differential_diseases: list[str] = field(default_factory=list)

    # PrimeKG context
    primekg_node_id: Optional[str] = None
    primekg_neighbor_count: int = 0
    primekg_edge_types: list[str] = field(default_factory=list)

    # Tier 1 source availability
    has_genereviews: bool = False
    genereviews_id: Optional[str] = None
    has_omim: bool = False
    has_orphanet: bool = False
    has_clinical_guidelines: list[str] = field(default_factory=list)

    # Literature profile
    pubmed_article_count: int = 0
    pmc_oa_count: int = 0
    key_genes: list[str] = field(default_factory=list)
    key_phenotypes: list[str] = field(default_factory=list)
    mesh_terms: list[str] = field(default_factory=list)

    # Extraction strategy (auto-determined)
    tier1_sources: list[str] = field(default_factory=list)
    recommended_pubmed_queries: list[str] = field(default_factory=list)
    expected_yield: str = "low"   # high (>500 papers), medium (50-500), low (<50)
    coverage_flag: CoverageFlag = CoverageFlag.SPARSE

    def has_sufficient_sources(self) -> bool:
        """Check if there are enough sources to justify extraction."""
        return self.has_genereviews or self.has_omim or self.has_orphanet or self.pubmed_article_count >= 10

    def to_dict(self) -> dict:
        d = {}
        for fld_name, fld_obj in self.__dataclass_fields__.items():
            val = getattr(self, fld_name)
            if isinstance(val, Enum):
                d[fld_name] = val.value
            else:
                d[fld_name] = val
        return d

    @classmethod
    def from_dict(cls, data: dict) -> DiseaseProfile:
        if "coverage_flag" in data and isinstance(data["coverage_flag"], str):
            data["coverage_flag"] = CoverageFlag(data["coverage_flag"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SourceDocument:
    """A single source document collected by Agent 2 (Evidence Harvester)."""
    source_id: str              # PMID, GeneReviews ID, OMIM ID, etc.
    source_type: str            # "genereviews" | "omim" | "orphanet" | "pubmed_abstract" | "pmc_fulltext" | "guideline"
    tier: EvidenceTier
    title: str
    text: str                   # Full text or abstract
    sections: Optional[dict[str, str]] = None  # For full-text: {"introduction": ..., "results": ...}
    publication_date: Optional[date] = None
    journal: Optional[str] = None
    credibility_score: float = 0.0
    study_type: Optional[StudyType] = None
    citation_count: Optional[int] = None
    is_retracted: bool = False

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "tier": self.tier.value,
            "title": self.title,
            "text": self.text[:500] + "..." if len(self.text) > 500 else self.text,
            "text_length": len(self.text),
            "sections": list(self.sections.keys()) if self.sections else None,
            "publication_date": self.publication_date.isoformat() if self.publication_date else None,
            "journal": self.journal,
            "credibility_score": self.credibility_score,
            "study_type": self.study_type.value if self.study_type else None,
            "citation_count": self.citation_count,
            "is_retracted": self.is_retracted,
        }


@dataclass
class EvidenceCollection:
    """All evidence collected for a disease by Agent 2."""
    disease_id: str
    tier1_documents: list[SourceDocument] = field(default_factory=list)
    tier2_documents: list[SourceDocument] = field(default_factory=list)
    harvest_metrics: dict = field(default_factory=dict)

    @property
    def total_sources(self) -> int:
        return len(self.tier1_documents) + len(self.tier2_documents)

    @property
    def coverage_quality(self) -> CoverageFlag:
        t1_count = len(self.tier1_documents)
        t2_count = len(self.tier2_documents)
        if t1_count >= 2 and t2_count >= 200:
            return CoverageFlag.RICH
        if t1_count >= 1 or t2_count >= 50:
            return CoverageFlag.MODERATE
        return CoverageFlag.SPARSE


@dataclass
class RawTriple:
    """A single triple extracted by Agent 3 (Knowledge Extractor)."""
    subject: str
    subject_type: str           # PrimeKG node type string
    relation: str               # Free-text or PrimeKG relation
    object: str
    object_type: str
    subject_id: Optional[str] = None
    object_id: Optional[str] = None
    temporal_context: Optional[dict] = None  # onset_age, progression, duration
    conditions: Optional[dict] = None        # contextual conditions
    evidence_text: str = ""
    source_id: str = ""         # PMID or Tier 1 source ID
    extraction_model: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "subject_type": self.subject_type,
            "subject_id": self.subject_id,
            "relation": self.relation,
            "object": self.object,
            "object_type": self.object_type,
            "object_id": self.object_id,
            "temporal_context": self.temporal_context,
            "conditions": self.conditions,
            "evidence_text": self.evidence_text,
            "source_id": self.source_id,
            "extraction_model": self.extraction_model,
            "confidence": self.confidence,
        }


@dataclass
class ExtractionResult:
    """Output of Agent 3: all extracted triples for a disease."""
    disease_id: str
    raw_triples: list[RawTriple] = field(default_factory=list)
    consensus_triples: list[RawTriple] = field(default_factory=list)
    model_agreement_stats: dict = field(default_factory=dict)
    extraction_metrics: dict = field(default_factory=dict)


@dataclass
class QualityReport:
    """Output of Agent 4: quality assessment for a disease's extraction."""
    disease_id: str
    total_triples_input: int = 0
    validated_triples: int = 0
    rejected_triples: int = 0
    conflicts_detected: int = 0
    confirmations_with_primekg: int = 0
    novel_triples: int = 0
    temporal_coverage: float = 0.0
    avg_credibility_score: float = 0.0
    quality_grade: str = "C"  # A (>90% valid), B (70-90%), C (<70%)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "disease_id": self.disease_id,
            "total_triples_input": self.total_triples_input,
            "validated_triples": self.validated_triples,
            "rejected_triples": self.rejected_triples,
            "conflicts_detected": self.conflicts_detected,
            "confirmations_with_primekg": self.confirmations_with_primekg,
            "novel_triples": self.novel_triples,
            "temporal_coverage": self.temporal_coverage,
            "avg_credibility_score": self.avg_credibility_score,
            "quality_grade": self.quality_grade,
            "issues": self.issues,
        }


@dataclass
class AgentResult:
    """Standard result envelope for any agent."""
    agent_name: str
    disease_id: str
    status: str  # "success" | "partial" | "failed" | "skipped"
    data: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
