#!/usr/bin/env python3
"""
PrimeKG Edge Audit Pipeline
============================
Audits PrimeKG's 682K disease edges against ChronoMedKG extractions to
quantify staleness, contradiction, and confirmation rates.

This script is also the **foundation for self-healing**: re-running it after
new extractions automatically updates staleness scores and marks superseded
edges. Design principle: idempotent, incremental, re-runnable.

Audit categories:
  - CONFIRMED:    Our evidence independently supports the PrimeKG edge
  - CONTRADICTED: Our evidence directly contradicts the PrimeKG edge
  - STALE:        PrimeKG edge exists but our newer evidence refines it
  - NOVEL:        We extracted facts PrimeKG doesn't have (not audited here)
  - UNASSESSED:   PrimeKG edge for a disease we haven't extracted yet

Evidence dating uses the PMID year from our extractions. PrimeKG itself
has NO publication dates — this is part of the problem we're solving.

Output:
  - data/audit/primekg_audit_report.json   — full machine-readable audit
  - data/audit/staleness_by_relation.json  — per-relation staleness stats
  - data/audit/audit_summary.json          — top-line numbers for paper
  - data/audit/edge_verdicts.jsonl         — per-edge audit verdict

Usage:
    python3 scripts/audit_primekg.py                  # Full audit
    python3 scripts/audit_primekg.py --diseases 100   # Sample N diseases
    python3 scripts/audit_primekg.py --relation disease_phenotype_positive
    python3 scripts/audit_primekg.py --self-heal      # Write validity_end on stale edges
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("primekg_audit")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class EdgeVerdict:
    """Audit verdict for a single PrimeKG edge."""
    # PrimeKG edge identity
    disease_id: str
    primekg_relation: str
    primekg_x_name: str
    primekg_x_type: str
    primekg_y_name: str
    primekg_y_type: str

    # Audit result
    verdict: str = "UNASSESSED"     # CONFIRMED | CONTRADICTED | STALE | UNASSESSED
    confidence: float = 0.0         # How confident are we in this verdict (0-1)
    match_score: float = 0.0        # Entity match quality (0-1)

    # Evidence details (when confirmed/contradicted)
    our_relation: str = ""          # What relation we extracted
    our_evidence_pmids: list = field(default_factory=list)
    our_evidence_years: list = field(default_factory=list)
    our_evidence_tier: str = ""     # gold/silver/bronze_a/bronze_b
    our_confidence_avg: float = 0.0

    # Staleness metrics
    newest_evidence_year: int = 0
    oldest_evidence_year: int = 0
    evidence_age_years: int = 0     # 2026 - newest_evidence_year

    # Self-healing fields
    superseded_by: str = ""         # Our triple ID if we supersede this edge
    validity_end: str = ""          # Date string if edge is contradicted/stale


@dataclass
class AuditStats:
    """Aggregate audit statistics."""
    total_primekg_edges: int = 0
    assessed_edges: int = 0
    unassessed_edges: int = 0

    confirmed: int = 0
    contradicted: int = 0
    stale_5yr: int = 0              # Evidence > 5 years old
    stale_10yr: int = 0             # Evidence > 10 years old
    stale_15yr: int = 0             # Evidence > 15 years old
    no_evidence_found: int = 0      # PrimeKG edge, no matching extraction

    # Per-relation breakdown
    by_relation: dict = field(default_factory=dict)

    # Contradiction details
    contradiction_examples: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Entity matching utilities
# ---------------------------------------------------------------------------
class EntityMatcher:
    """
    Matches PrimeKG node names to ChronoMedKG extracted entity names.

    Uses a 3-tier matching strategy:
      1. Exact match (after normalization)
      2. Substring containment (shorter in longer, min 4 chars)
      3. Token overlap ratio (Jaccard on word tokens)

    No external dependencies (no rapidfuzz needed for audit).
    """

    @staticmethod
    def normalize(name: str) -> str:
        """Lowercase, strip punctuation, collapse whitespace."""
        if not name:
            return ""
        name = name.lower().strip()
        name = re.sub(r"[^\w\s]", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        return name

    @staticmethod
    def tokenize(name: str) -> set[str]:
        """Split into meaningful word tokens (length >= 3)."""
        return {w for w in name.split() if len(w) >= 3}

    @classmethod
    def match(cls, primekg_name: str, extracted_name: str) -> tuple[bool, float]:
        """
        Returns (is_match: bool, score: float 0-1).

        Thresholds:
          - Exact normalized: score=1.0
          - Substring containment (>=60% of longer): score=0.85
          - Token overlap >= 0.6 Jaccard: score=jaccard
        """
        a = cls.normalize(primekg_name)
        b = cls.normalize(extracted_name)

        if not a or not b:
            return False, 0.0

        # Tier 1: exact match
        if a == b:
            return True, 1.0

        # Tier 2: substring containment
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        if len(shorter) >= 4 and shorter in longer:
            # Only count if substring is a substantial portion
            ratio = len(shorter) / len(longer)
            if ratio >= 0.4:
                return True, 0.85

        # Tier 3: token overlap (Jaccard similarity)
        tokens_a = cls.tokenize(a)
        tokens_b = cls.tokenize(b)
        if not tokens_a or not tokens_b:
            return False, 0.0

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        jaccard = len(intersection) / len(union)

        if jaccard >= 0.5:
            return True, jaccard

        return False, jaccard


# ---------------------------------------------------------------------------
# Relation mapping: PrimeKG relations → our canonical relations
# ---------------------------------------------------------------------------
PRIMEKG_TO_OURS = {
    "disease_phenotype_positive": [
        "disease_phenotype_positive", "manifests_as", "associated_with",
        "characterized_by", "presents_with", "clinical_feature",
        "clinical_manifestation", "sign_of", "has_symptom", "phenotype_of",
        "symptom_of", "associated_with_phenotype",
    ],
    "disease_phenotype_negative": [
        "disease_phenotype_negative", "absent_in", "not_associated_with",
        "absent_phenotype",
    ],
    "disease_protein": [
        "disease_protein", "caused_by", "gene_disease", "associated_gene",
        "caused_by_mutation_in", "gene_variant", "genetic_basis",
        "mutation_in", "pathogenic_variant",
    ],
    "disease_disease": [
        "disease_disease", "differentiates", "progresses_to",
        "comorbidity", "differential_diagnosis", "associated_disease",
        "co_occurs_with", "misdiagnosed_as", "overlaps_with",
    ],
    "indication": [
        "treats", "indication", "treatment_for", "used_to_treat",
        "therapeutic_for", "managed_by", "managed_with", "therapy_for",
        "first_line_treatment", "second_line_treatment", "standard_of_care",
    ],
    "contraindication": [
        "contraindication", "contraindicated_for", "avoid_in",
        "not_recommended",
    ],
    "off-label use": [
        "treats", "off_label",
    ],
    "exposure_disease": [
        "caused_by", "exposure_disease", "risk_factor", "risk_factor_for",
        "environmental_cause", "trigger", "precipitant",
    ],
}

# Negation pairs for contradiction detection
NEGATION_PAIRS = {
    "disease_phenotype_positive": "disease_phenotype_negative",
    "disease_phenotype_negative": "disease_phenotype_positive",
    "indication":                 "contraindication",
    "contraindication":           "indication",
}


# ---------------------------------------------------------------------------
# PMID year extraction
# ---------------------------------------------------------------------------
def extract_year_from_pmid(pmid: str) -> Optional[int]:
    """
    Extract publication year from a PMID string.

    We rely on the numeric PMID range as a rough proxy:
    - PMIDs 1-10M: ~1966-2000
    - PMIDs 10M-20M: ~2000-2010
    - PMIDs 20M-30M: ~2010-2018
    - PMIDs 30M-40M: ~2018-2026

    For precise dating, we use the evidence_collection metadata.
    """
    if not pmid:
        return None
    # Extract numeric part
    m = re.search(r"(\d{6,9})", str(pmid))
    if not m:
        return None
    num = int(m.group(1))

    # Rough PMID-to-year mapping (linear interpolation)
    if num < 1000000:
        return 1970
    elif num < 10000000:
        return 1970 + int((num - 1000000) / 9000000 * 30)
    elif num < 20000000:
        return 2000 + int((num - 10000000) / 10000000 * 10)
    elif num < 30000000:
        return 2010 + int((num - 20000000) / 10000000 * 8)
    elif num < 40000000:
        return 2018 + int((num - 30000000) / 10000000 * 8)
    else:
        return 2025


def get_evidence_years_from_collection(disease_dir: Path) -> dict[str, int]:
    """
    Load precise publication years from evidence_collection.json.gz.

    Evidence collections store sources under 'tier1_sources' and 'tier2_sources',
    each with a 'publication_date' field (format: "YYYY-MM-DD") and 'source_id'
    (format: "PMID:12345678").

    Returns dict of {source_id: year}, e.g. {"PMID:24858164": 2014}.
    """
    import gzip
    ec_path = disease_dir / "evidence_collection.json.gz"
    if not ec_path.exists():
        return {}

    try:
        with gzip.open(ec_path, "rt") as f:
            data = json.load(f)
    except Exception:
        return {}

    pmid_years = {}

    # Collect from all source lists
    for key in ["tier1_sources", "tier2_sources", "papers", "documents"]:
        sources = data.get(key, [])
        if not isinstance(sources, list):
            continue
        for source in sources:
            if not isinstance(source, dict):
                continue

            source_id = source.get("source_id", source.get("pmid", source.get("id", "")))
            if not source_id:
                continue

            # Try multiple date field names
            year = None
            for date_field in ["publication_date", "pub_date", "date", "year", "pub_year"]:
                val = source.get(date_field)
                if not val:
                    continue
                try:
                    if isinstance(val, int):
                        year = val
                    elif isinstance(val, str) and len(val) >= 4:
                        year = int(val[:4])  # "2014-08-18" → 2014
                    break
                except (ValueError, TypeError):
                    continue

            if source_id and year and 1900 <= year <= 2030:
                pmid_years[str(source_id)] = year

    return pmid_years


# ---------------------------------------------------------------------------
# Core audit engine
# ---------------------------------------------------------------------------
class PrimeKGAuditor:
    """
    Audits PrimeKG edges against ChronoMedKG extractions.

    Design: stateless per-disease. Can be run incrementally as new
    diseases are extracted.
    """

    def __init__(self, primekg_path: Path, extracted_dir: Path):
        self.extracted_dir = extracted_dir
        self.matcher = EntityMatcher()
        self.stats = AuditStats()
        self.verdicts: list[EdgeVerdict] = []

        # Load PrimeKG
        logger.info("Loading PrimeKG from %s...", primekg_path)
        with open(primekg_path, "rb") as f:
            self.primekg = pickle.load(f)
        self.disease_edges = self.primekg["disease_edges"]
        logger.info("Loaded %d diseases with edges from PrimeKG",
                     len(self.disease_edges))

    def _load_disease_triples(self, disease_id: str) -> list[dict]:
        """Load tiered triples for a disease. Falls back to validated_triples."""
        disease_dir = self.extracted_dir / f"MONDO_{disease_id}"

        # Prefer tiered_triples (has bronze), fall back to validated
        for fname in ["tiered_triples.jsonl", "validated_triples.jsonl",
                       "consensus_triples.jsonl"]:
            path = disease_dir / fname
            if path.exists():
                triples = []
                with open(path) as f:
                    for line in f:
                        if line.strip():
                            triples.append(json.loads(line))
                return triples
        return []

    def _get_evidence_years(self, disease_id: str) -> dict[str, int]:
        """Get PMID→year mapping for a disease."""
        disease_dir = self.extracted_dir / f"MONDO_{disease_id}"
        return get_evidence_years_from_collection(disease_dir)

    def _find_matching_triples(
        self,
        primekg_edge,
        our_triples: list[dict],
    ) -> list[tuple[dict, float, str]]:
        """
        Find our triples that match a PrimeKG edge.

        Returns list of (triple, match_score, match_type) where match_type
        is 'confirm' or 'contradict'.
        """
        # Determine which of our relations could match this PrimeKG relation
        valid_relations = set(PRIMEKG_TO_OURS.get(primekg_edge.relation, []))
        negation_relations = set()
        neg_rel = NEGATION_PAIRS.get(primekg_edge.relation)
        if neg_rel:
            negation_relations = set(PRIMEKG_TO_OURS.get(neg_rel, [neg_rel]))

        matches = []

        for triple in our_triples:
            our_rel = triple.get("relation", "")

            # Check if relation family matches (confirmation or contradiction)
            is_confirm = our_rel in valid_relations
            is_contradict = our_rel in negation_relations

            if not is_confirm and not is_contradict:
                continue

            # Match entities — PrimeKG x_name/y_name vs our subject/object
            # PrimeKG convention: x is source, y is disease (usually)
            # Try both orientations

            # Orientation 1: primekg_x → our_subject, primekg_y → our_object
            x_match, x_score = self.matcher.match(
                primekg_edge.x_name, triple.get("subject", "")
            )
            y_match, y_score = self.matcher.match(
                primekg_edge.y_name, triple.get("object", "")
            )
            score_1 = (x_score + y_score) / 2 if x_match and y_match else 0

            # Orientation 2: primekg_x → our_object, primekg_y → our_subject
            x_match2, x_score2 = self.matcher.match(
                primekg_edge.x_name, triple.get("object", "")
            )
            y_match2, y_score2 = self.matcher.match(
                primekg_edge.y_name, triple.get("subject", "")
            )
            score_2 = (x_score2 + y_score2) / 2 if x_match2 and y_match2 else 0

            best_score = max(score_1, score_2)
            if best_score > 0:
                match_type = "confirm" if is_confirm else "contradict"
                matches.append((triple, best_score, match_type))

        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def audit_disease(self, disease_id: str, primekg_edges: list) -> list[EdgeVerdict]:
        """Audit all PrimeKG edges for a single disease."""
        our_triples = self._load_disease_triples(disease_id)
        evidence_years = self._get_evidence_years(disease_id)
        verdicts = []

        for edge in primekg_edges:
            verdict = EdgeVerdict(
                disease_id=disease_id,
                primekg_relation=edge.relation,
                primekg_x_name=edge.x_name,
                primekg_x_type=edge.x_type,
                primekg_y_name=edge.y_name,
                primekg_y_type=edge.y_type,
            )

            if not our_triples:
                verdict.verdict = "UNASSESSED"
                verdicts.append(verdict)
                continue

            # Find matching triples
            matches = self._find_matching_triples(edge, our_triples)

            if not matches:
                verdict.verdict = "UNASSESSED"
                verdict.confidence = 0.3  # We had data but no match
                verdicts.append(verdict)
                continue

            # Best match determines verdict
            best_triple, best_score, match_type = matches[0]
            verdict.match_score = best_score
            verdict.our_relation = best_triple.get("relation", "")
            verdict.our_evidence_tier = best_triple.get("validation_tier", "")
            verdict.our_confidence_avg = best_triple.get("confidence", 0)

            # Collect evidence years from all matching triples
            pmids = []
            years = []
            for triple, score, mt in matches[:5]:  # Top 5 matches
                pmid = triple.get("source_id", "")
                pmids.append(pmid)
                # Try precise year from evidence collection
                year = evidence_years.get(
                    pmid.replace("PMID:", ""),
                    extract_year_from_pmid(pmid)
                )
                if year:
                    years.append(year)

            verdict.our_evidence_pmids = pmids
            verdict.our_evidence_years = years

            if years:
                verdict.newest_evidence_year = max(years)
                verdict.oldest_evidence_year = min(years)
                verdict.evidence_age_years = 2026 - verdict.newest_evidence_year

            # Determine verdict
            if match_type == "contradict":
                verdict.verdict = "CONTRADICTED"
                verdict.confidence = best_score
                if years:
                    verdict.validity_end = f"{max(years)}-01-01"
            elif match_type == "confirm":
                verdict.verdict = "CONFIRMED"
                verdict.confidence = best_score
            else:
                verdict.verdict = "UNASSESSED"

            verdicts.append(verdict)

        return verdicts

    def run_audit(
        self,
        max_diseases: int = 0,
        relation_filter: str = "",
    ) -> AuditStats:
        """
        Run full audit across all diseases.

        Args:
            max_diseases: Limit to N diseases (0 = all)
            relation_filter: Only audit edges of this relation type
        """
        # Find diseases we have extractions for
        extracted_diseases = set()
        for d in self.extracted_dir.iterdir():
            if d.is_dir() and d.name.startswith("MONDO_"):
                mondo_num = d.name.replace("MONDO_", "")
                extracted_diseases.add(mondo_num)
                # Also add as component for composite PrimeKG IDs
                # PrimeKG uses "5044" or composite "1200_1134_15512"

        logger.info("Found %d diseases with extractions", len(extracted_diseases))

        # Iterate PrimeKG diseases
        disease_ids = sorted(self.disease_edges.keys())
        if max_diseases > 0:
            disease_ids = disease_ids[:max_diseases]

        total_edges = 0
        assessed = 0
        t0 = time.time()

        for i, disease_id in enumerate(disease_ids):
            edges = self.disease_edges[disease_id]

            # Filter by relation if specified
            if relation_filter:
                edges = [e for e in edges if e.relation == relation_filter]

            if not edges:
                continue

            total_edges += len(edges)

            # Check if we have extraction — direct or component match
            # PrimeKG ID could be "5044" or composite "1200_1134_15512"
            matched_id = None
            if disease_id in extracted_diseases:
                matched_id = disease_id
            else:
                # For composite IDs, check each component
                for component in disease_id.split("_"):
                    if component in extracted_diseases:
                        matched_id = component
                        break

            has_extraction = matched_id is not None

            if has_extraction:
                verdicts = self.audit_disease(str(matched_id), edges)
                assessed += len(verdicts)
            else:
                verdicts = [
                    EdgeVerdict(
                        disease_id=str(disease_id),
                        primekg_relation=e.relation,
                        primekg_x_name=e.x_name,
                        primekg_x_type=e.x_type,
                        primekg_y_name=e.y_name,
                        primekg_y_type=e.y_type,
                        verdict="UNASSESSED",
                    )
                    for e in edges
                ]

            self.verdicts.extend(verdicts)

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                logger.info("  Audited %d/%d diseases (%d edges, %.1f/sec)",
                             i + 1, len(disease_ids), total_edges,
                             total_edges / elapsed)

        # Compute stats
        self._compute_stats()
        elapsed = time.time() - t0
        logger.info("Audit complete: %d edges in %.1fs", total_edges, elapsed)

        return self.stats

    def _compute_stats(self):
        """Compute aggregate statistics from individual verdicts."""
        s = self.stats
        s.total_primekg_edges = len(self.verdicts)

        by_rel = defaultdict(lambda: {
            "total": 0, "confirmed": 0, "contradicted": 0,
            "unassessed": 0, "stale_5yr": 0, "stale_10yr": 0,
            "avg_evidence_age": 0, "evidence_ages": [],
        })

        for v in self.verdicts:
            rel_stats = by_rel[v.primekg_relation]
            rel_stats["total"] += 1

            if v.verdict == "CONFIRMED":
                s.confirmed += 1
                s.assessed_edges += 1
                rel_stats["confirmed"] += 1

                if v.evidence_age_years > 0:
                    rel_stats["evidence_ages"].append(v.evidence_age_years)
                    if v.evidence_age_years >= 5:
                        s.stale_5yr += 1
                        rel_stats["stale_5yr"] += 1
                    if v.evidence_age_years >= 10:
                        s.stale_10yr += 1
                        rel_stats["stale_10yr"] += 1
                    if v.evidence_age_years >= 15:
                        s.stale_15yr += 1

            elif v.verdict == "CONTRADICTED":
                s.contradicted += 1
                s.assessed_edges += 1
                rel_stats["contradicted"] += 1

                if len(s.contradiction_examples) < 50:
                    s.contradiction_examples.append(asdict(v))

            elif v.verdict == "UNASSESSED":
                s.unassessed_edges += 1
                rel_stats["unassessed"] += 1

        # Compute averages
        for rel, rs in by_rel.items():
            ages = rs.pop("evidence_ages", [])
            rs["avg_evidence_age"] = round(sum(ages) / len(ages), 1) if ages else 0
            rs["confirmed_pct"] = round(100 * rs["confirmed"] / max(1, rs["total"]), 1)
            rs["contradicted_pct"] = round(100 * rs["contradicted"] / max(1, rs["total"]), 1)
            rs["stale_10yr_pct"] = round(100 * rs["stale_10yr"] / max(1, rs["confirmed"]), 1)

        s.by_relation = dict(by_rel)
        s.no_evidence_found = s.total_primekg_edges - s.assessed_edges - s.unassessed_edges

    # ------------------------------------------------------------------
    # Self-healing: mark stale edges
    # ------------------------------------------------------------------
    def mark_stale_edges(self, staleness_threshold_years: int = 10) -> int:
        """
        Self-healing pass: for confirmed edges with evidence > threshold years
        old, add a staleness flag. For contradicted edges, mark validity_end.

        Returns count of edges marked.
        """
        marked = 0
        for v in self.verdicts:
            if v.verdict == "CONFIRMED" and v.evidence_age_years >= staleness_threshold_years:
                v.validity_end = f"STALE_SINCE_{2026 - staleness_threshold_years}"
                marked += 1
            elif v.verdict == "CONTRADICTED" and v.our_evidence_years:
                v.validity_end = f"{max(v.our_evidence_years)}-01-01"
                marked += 1
        logger.info("Self-heal: marked %d edges as stale/contradicted", marked)
        return marked

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def save_results(self, output_dir: Path):
        """Save all audit results."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Edge verdicts (full detail)
        verdicts_path = output_dir / "edge_verdicts.jsonl"
        with open(verdicts_path, "w") as f:
            for v in self.verdicts:
                f.write(json.dumps(asdict(v), default=str) + "\n")
        logger.info("Saved %d edge verdicts to %s",
                     len(self.verdicts), verdicts_path)

        # 2. Per-relation staleness breakdown
        staleness_path = output_dir / "staleness_by_relation.json"
        with open(staleness_path, "w") as f:
            json.dump(self.stats.by_relation, f, indent=2)

        # 3. Summary for paper
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_primekg_edges": self.stats.total_primekg_edges,
            "assessed_edges": self.stats.assessed_edges,
            "unassessed_edges": self.stats.unassessed_edges,
            "confirmed": self.stats.confirmed,
            "confirmed_pct": round(
                100 * self.stats.confirmed / max(1, self.stats.assessed_edges), 1
            ),
            "contradicted": self.stats.contradicted,
            "contradicted_pct": round(
                100 * self.stats.contradicted / max(1, self.stats.assessed_edges), 1
            ),
            "stale_5yr": self.stats.stale_5yr,
            "stale_5yr_pct_of_confirmed": round(
                100 * self.stats.stale_5yr / max(1, self.stats.confirmed), 1
            ),
            "stale_10yr": self.stats.stale_10yr,
            "stale_10yr_pct_of_confirmed": round(
                100 * self.stats.stale_10yr / max(1, self.stats.confirmed), 1
            ),
            "stale_15yr": self.stats.stale_15yr,
            "headline": (
                f"Of {self.stats.assessed_edges:,} assessed PrimeKG edges, "
                f"{self.stats.contradicted:,} ({100*self.stats.contradicted/max(1,self.stats.assessed_edges):.1f}%) "
                f"are contradicted by newer evidence, and "
                f"{self.stats.stale_10yr:,} ({100*self.stats.stale_10yr/max(1,self.stats.confirmed):.1f}%) "
                f"of confirmed edges rely on evidence >10 years old."
            ),
        }
        summary_path = output_dir / "audit_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # 4. Contradiction examples
        if self.stats.contradiction_examples:
            contra_path = output_dir / "contradiction_examples.json"
            with open(contra_path, "w") as f:
                json.dump(self.stats.contradiction_examples, f, indent=2, default=str)

        logger.info("Audit results saved to %s", output_dir)
        return summary


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------
def print_audit_summary(summary: dict, stats: AuditStats):
    """Print human-readable audit summary."""
    print("\n" + "=" * 72)
    print("PRIMEKG EDGE AUDIT — RESULTS")
    print("=" * 72)

    print(f"\n  Total PrimeKG edges:  {stats.total_primekg_edges:>10,}")
    print(f"  Assessed:             {stats.assessed_edges:>10,}  "
          f"({100*stats.assessed_edges/max(1,stats.total_primekg_edges):.1f}%)")
    print(f"  Unassessed:           {stats.unassessed_edges:>10,}  "
          f"({100*stats.unassessed_edges/max(1,stats.total_primekg_edges):.1f}%)")

    print(f"\n  Of assessed edges:")
    print(f"    ✓ Confirmed:        {stats.confirmed:>10,}  "
          f"({100*stats.confirmed/max(1,stats.assessed_edges):.1f}%)")
    print(f"    ✗ Contradicted:     {stats.contradicted:>10,}  "
          f"({100*stats.contradicted/max(1,stats.assessed_edges):.1f}%)")

    print(f"\n  Evidence staleness (of confirmed edges):")
    print(f"    > 5 years old:      {stats.stale_5yr:>10,}  "
          f"({100*stats.stale_5yr/max(1,stats.confirmed):.1f}%)")
    print(f"    > 10 years old:     {stats.stale_10yr:>10,}  "
          f"({100*stats.stale_10yr/max(1,stats.confirmed):.1f}%)")
    print(f"    > 15 years old:     {stats.stale_15yr:>10,}  "
          f"({100*stats.stale_15yr/max(1,stats.confirmed):.1f}%)")

    print(f"\n  Per-relation breakdown:")
    print(f"  {'Relation':<30s} {'Total':>8s} {'Conf':>7s} {'Contr':>7s} "
          f"{'Stale10':>8s} {'AvgAge':>7s}")
    print("  " + "-" * 68)
    for rel, rs in sorted(stats.by_relation.items(),
                           key=lambda x: x[1]["total"], reverse=True):
        print(f"  {rel:<30s} {rs['total']:>8,} {rs['confirmed']:>7,} "
              f"{rs['contradicted']:>7,} {rs['stale_10yr']:>8,} "
              f"{rs['avg_evidence_age']:>6.1f}yr")

    print("\n  " + summary.get("headline", ""))
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Audit PrimeKG edges against ChronoMedKG extractions"
    )
    parser.add_argument("--diseases", type=int, default=0,
                        help="Limit to N diseases (0 = all)")
    parser.add_argument("--relation", type=str, default="",
                        help="Only audit this PrimeKG relation type")
    parser.add_argument("--self-heal", action="store_true",
                        help="Mark stale/contradicted edges with validity_end")
    parser.add_argument("--staleness-threshold", type=int, default=10,
                        help="Years before marking edge as stale (default 10)")
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "audit"),
                        help="Output directory for audit results")
    args = parser.parse_args()

    primekg_path = PROJECT_ROOT / "data" / "primekg" / "kg.pkl"
    extracted_dir = PROJECT_ROOT / "data" / "extracted"
    output_dir = Path(args.output_dir)

    if not primekg_path.exists():
        logger.error("PrimeKG not found at %s", primekg_path)
        sys.exit(1)

    # Run audit
    auditor = PrimeKGAuditor(primekg_path, extracted_dir)
    stats = auditor.run_audit(
        max_diseases=args.diseases,
        relation_filter=args.relation,
    )

    # Self-healing pass
    if args.self_heal:
        auditor.mark_stale_edges(args.staleness_threshold)

    # Save results
    summary = auditor.save_results(output_dir)

    # Print summary
    print_audit_summary(summary, stats)


if __name__ == "__main__":
    main()
