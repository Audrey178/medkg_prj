#!/usr/bin/env python3
"""
Experiment 1: PrimeKG Evidence Decay Audit
============================================
Quantifies the temporal blindness of PrimeKG by analyzing evidence ages
in ChronoMedKG vs PrimeKG's complete lack of evidence dating.

Key questions:
  1. What is the publication year distribution of ChronoMedKG evidence?
  2. For PrimeKG-confirmed edges, how old is the supporting evidence?
  3. How many TA edges are from recent evidence (2020+) not captured by PrimeKG?
  4. What fraction of knowledge would be "stale" (>10 years old) without TA?

Outputs:
  data/benchmark/evidence_decay_audit.json — full results

Usage:
  .venv-sapbert/bin/python scripts/experiment_evidence_decay.py
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("evidence_decay")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
PRIMEKG_DIR = PROJECT_ROOT / "data" / "primekg"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"


def load_primekg_disease_edges():
    """Load PrimeKG disease-involved edges for overlap analysis."""
    kg_file = PRIMEKG_DIR / "kg.csv"
    if not kg_file.exists():
        logger.error("PrimeKG kg.csv not found")
        return {}, {}

    # Index: disease_name_lower -> set of (relation, entity_name_lower)
    disease_edges = defaultdict(set)
    source_dist = Counter()
    total = 0

    with open(kg_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            rel = row.get("display_relation", "")
            x_name = row.get("x_name", "").lower().strip()
            y_name = row.get("y_name", "").lower().strip()
            x_type = row.get("x_type", "")
            y_type = row.get("y_type", "")
            x_source = row.get("x_source", "")

            source_dist[x_source] += 1

            if x_type == "disease":
                disease_edges[x_name].add((rel, y_name))
            if y_type == "disease":
                disease_edges[y_name].add((rel, x_name))

    logger.info("PrimeKG: %d total edges, %d disease entries", total, len(disease_edges))
    return dict(disease_edges), dict(source_dist)


def build_pmid_year_index():
    """Build PMID -> publication year index from evidence collection files."""
    pmid_year = {}
    diseases_processed = 0

    for d in sorted(EXTRACTED_DIR.iterdir()):
        ec = d / "evidence_collection.json.gz"
        if not ec.exists():
            continue
        diseases_processed += 1

        try:
            with gzip.open(ec, "rt") as f:
                data = json.load(f)
        except Exception:
            continue

        for src in data.get("tier2_sources", []) + data.get("tier1_sources", []):
            source_id = src.get("source_id", "")
            pub_date = src.get("publication_date")
            if not source_id or not pub_date or pub_date == "None":
                continue
            try:
                year = int(str(pub_date)[:4])
                if 1900 <= year <= 2026:
                    pmid_year[source_id] = year
            except (ValueError, TypeError):
                pass

    logger.info("Built PMID-year index: %d PMIDs from %d diseases", len(pmid_year), diseases_processed)
    return pmid_year


def analyze_ta_evidence_ages(pmid_year):
    """Analyze evidence ages across all ChronoMedKG validated triples."""
    year_dist = Counter()
    quality_year = {"A": [], "B": []}  # A = PrimeKG-confirmed, B = novel
    total_triples = 0
    triples_with_year = 0
    recent_triples = 0  # 2020+
    old_triples = 0     # before 2010

    for d in sorted(EXTRACTED_DIR.iterdir()):
        vf = d / "validated_triples.jsonl"
        if not vf.exists():
            continue

        with open(vf) as f:
            for line in f:
                if not line.strip():
                    continue
                t = json.loads(line)
                total_triples += 1

                ev = t.get("evidence", {})
                grade = t.get("quality_grade", "B")
                source_ids = ev.get("source_ids", [])

                # Get year for this triple's primary evidence
                year = None
                for sid in source_ids:
                    if sid in pmid_year:
                        year = pmid_year[sid]
                        break

                if year is None:
                    continue

                triples_with_year += 1
                year_dist[year] += 1

                if grade in quality_year:
                    quality_year[grade].append(year)

                if year >= 2020:
                    recent_triples += 1
                if year < 2010:
                    old_triples += 1

    logger.info("TA triples: %d total, %d with year (%0.1f%%)",
                total_triples, triples_with_year, 100 * triples_with_year / max(1, total_triples))

    return {
        "total_triples": total_triples,
        "triples_with_year": triples_with_year,
        "year_distribution": dict(year_dist),
        "quality_grade_years": {k: v for k, v in quality_year.items()},
        "recent_2020_plus": recent_triples,
        "old_pre_2010": old_triples,
    }


def compute_staleness_metrics(ta_years):
    """Compute staleness metrics from TA evidence age analysis."""
    all_years = []
    for year, count in ta_years["year_distribution"].items():
        all_years.extend([int(year)] * count)

    if not all_years:
        return {}

    total = len(all_years)

    # Age brackets
    brackets = {
        "0-5 years (2021-2026)": sum(1 for y in all_years if y >= 2021),
        "5-10 years (2016-2020)": sum(1 for y in all_years if 2016 <= y <= 2020),
        "10-15 years (2011-2015)": sum(1 for y in all_years if 2011 <= y <= 2015),
        "15-20 years (2006-2010)": sum(1 for y in all_years if 2006 <= y <= 2010),
        "20+ years (before 2006)": sum(1 for y in all_years if y < 2006),
    }

    # PrimeKG-confirmed vs novel
    grade_a_years = ta_years["quality_grade_years"].get("A", [])
    grade_b_years = ta_years["quality_grade_years"].get("B", [])

    grade_a_median = statistics.median(grade_a_years) if grade_a_years else None
    grade_b_median = statistics.median(grade_b_years) if grade_b_years else None

    grade_a_recent = sum(1 for y in grade_a_years if y >= 2020) / max(1, len(grade_a_years))
    grade_b_recent = sum(1 for y in grade_b_years if y >= 2020) / max(1, len(grade_b_years))

    return {
        "total_evidence_dated": total,
        "median_year": statistics.median(all_years),
        "mean_year": round(statistics.mean(all_years), 1),
        "age_brackets": {k: {"count": v, "pct": round(100 * v / total, 1)} for k, v in brackets.items()},
        "primekg_confirmed_edges": {
            "count": len(grade_a_years),
            "median_year": grade_a_median,
            "pct_recent_2020": round(100 * grade_a_recent, 1),
        },
        "novel_edges": {
            "count": len(grade_b_years),
            "median_year": grade_b_median,
            "pct_recent_2020": round(100 * grade_b_recent, 1),
        },
    }


def analyze_primekg_limitations(primekg_edges, primekg_sources):
    """Quantify what PrimeKG is missing."""
    total_edges = sum(primekg_sources.values()) // 2  # bidirectional
    return {
        "total_edges": total_edges,
        "has_publication_dates": False,
        "has_evidence_pmids": False,
        "has_temporal_metadata": False,
        "has_evidence_grading": False,
        "source_databases": dict(sorted(primekg_sources.items(), key=lambda x: -x[1])[:10]),
        "disease_count": len(primekg_edges),
        "note": "PrimeKG edges have NO evidence dating, NO PMIDs, NO temporal metadata. "
                "Edges are database snapshots with no way to determine when facts were "
                "established or whether they have been superseded by newer evidence."
    }


def find_supersession_examples(pmid_year):
    """Find concrete examples of newer evidence potentially superseding older facts."""
    # For each disease, find triples on the same (subject, relation, object)
    # with different temporal contexts from different years
    examples = []

    for d in sorted(EXTRACTED_DIR.iterdir()):
        vf = d / "validated_triples.jsonl"
        if not vf.exists():
            continue

        # Group triples by (source_name, relation, target_name)
        triple_groups = defaultdict(list)
        with open(vf) as f:
            for line in f:
                if not line.strip():
                    continue
                t = json.loads(line)
                key = (
                    t.get("source_name", "").lower(),
                    t.get("relation", ""),
                    t.get("target_name", "").lower(),
                )
                ev = t.get("evidence", {})
                source_ids = ev.get("source_ids", [])
                year = None
                for sid in source_ids:
                    if sid in pmid_year:
                        year = pmid_year[sid]
                        break
                if year:
                    triple_groups[key].append({
                        "year": year,
                        "temporal": t.get("temporal", {}),
                        "pmid": source_ids[0] if source_ids else "",
                        "disease_dir": d.name,
                    })

        # Find groups with evidence from different decades
        for key, triples in triple_groups.items():
            if len(triples) < 2:
                continue
            years = [t["year"] for t in triples]
            if max(years) - min(years) >= 10:
                examples.append({
                    "subject": key[0],
                    "relation": key[1],
                    "target": key[2],
                    "oldest_evidence": min(years),
                    "newest_evidence": max(years),
                    "year_span": max(years) - min(years),
                    "n_evidence_sources": len(triples),
                    "disease_dir": triples[0]["disease_dir"],
                })

        if len(examples) >= 200:
            break

    examples.sort(key=lambda x: -x["year_span"])
    return examples[:50]


def main():
    logger.info("=" * 75)
    logger.info("Experiment 1: PrimeKG Evidence Decay Audit")
    logger.info("=" * 75)

    # Step 1: Load PrimeKG
    logger.info("\n[1/5] Loading PrimeKG edges...")
    primekg_edges, primekg_sources = load_primekg_disease_edges()

    # Step 2: Build PMID-year index from ChronoMedKG evidence collections
    logger.info("\n[2/5] Building PMID-year index from evidence collections...")
    pmid_year = build_pmid_year_index()

    # Step 3: Analyze TA evidence ages
    logger.info("\n[3/5] Analyzing ChronoMedKG evidence ages...")
    ta_years = analyze_ta_evidence_ages(pmid_year)

    # Step 4: Compute staleness metrics
    logger.info("\n[4/5] Computing staleness metrics...")
    staleness = compute_staleness_metrics(ta_years)

    # Step 5: PrimeKG limitations analysis
    logger.info("\n[5/5] Analyzing PrimeKG limitations...")
    primekg_limits = analyze_primekg_limitations(primekg_edges, primekg_sources)

    # Step 6: Find supersession examples
    logger.info("\n[6/5] Finding evidence evolution examples...")
    supersession_examples = find_supersession_examples(pmid_year)

    # === PRINT RESULTS ===
    print(f"\n{'=' * 75}")
    print("EVIDENCE DECAY AUDIT RESULTS")
    print(f"{'=' * 75}")

    print(f"\n1. PrimeKG's Temporal Blindness:")
    print(f"   Total edges: {primekg_limits['total_edges']:,}")
    print(f"   Has publication dates: {primekg_limits['has_publication_dates']}")
    print(f"   Has evidence PMIDs: {primekg_limits['has_evidence_pmids']}")
    print(f"   Has temporal metadata: {primekg_limits['has_temporal_metadata']}")
    print(f"   → PrimeKG is a frozen snapshot with NO evidence aging information")

    print(f"\n2. ChronoMedKG Evidence Age Distribution:")
    print(f"   Total triples with dated evidence: {staleness.get('total_evidence_dated', 0):,}")
    print(f"   Median evidence year: {staleness.get('median_year', '?')}")
    print(f"   Mean evidence year: {staleness.get('mean_year', '?')}")
    for bracket, info in staleness.get("age_brackets", {}).items():
        bar = "#" * (info["count"] // 2000)
        print(f"   {bracket:>30}: {info['count']:>8,} ({info['pct']:>5.1f}%) {bar}")

    print(f"\n3. PrimeKG-Confirmed vs Novel Edges:")
    conf = staleness.get("primekg_confirmed_edges", {})
    novel = staleness.get("novel_edges", {})
    print(f"   PrimeKG-confirmed (Grade A): {conf.get('count', 0):,} triples")
    print(f"     Median evidence year: {conf.get('median_year', '?')}")
    print(f"     % from 2020+: {conf.get('pct_recent_2020', 0)}%")
    print(f"   Novel (Grade B): {novel.get('count', 0):,} triples")
    print(f"     Median evidence year: {novel.get('median_year', '?')}")
    print(f"     % from 2020+: {novel.get('pct_recent_2020', 0)}%")

    if conf.get("median_year") and novel.get("median_year"):
        diff = novel["median_year"] - conf["median_year"]
        print(f"   → Novel edges are based on evidence {abs(diff):.0f} years "
              f"{'newer' if diff > 0 else 'older'} than PrimeKG-confirmed edges")

    print(f"\n4. Evidence Evolution Examples (top 10 by year span):")
    for i, ex in enumerate(supersession_examples[:10]):
        print(f"   [{i+1}] {ex['subject'][:25]} --{ex['relation'][:20]}--> {ex['target'][:25]}")
        print(f"       Evidence: {ex['oldest_evidence']} to {ex['newest_evidence']} "
              f"({ex['year_span']}y span, {ex['n_evidence_sources']} sources)")

    # Key headline numbers for the paper
    recent_pct = staleness.get("age_brackets", {}).get("0-5 years (2021-2026)", {}).get("pct", 0)
    old_pct = staleness.get("age_brackets", {}).get("20+ years (before 2006)", {}).get("pct", 0)

    print(f"\n{'=' * 75}")
    print("HEADLINE NUMBERS FOR PAPER")
    print(f"{'=' * 75}")
    print(f"  PrimeKG: {primekg_limits['total_edges']:,} edges, ZERO evidence dates")
    print(f"  ChronoMedKG: {staleness.get('total_evidence_dated', 0):,} triples with dated evidence")
    print(f"  {recent_pct}% of TA evidence from last 5 years (2021-2026)")
    print(f"  {old_pct}% of TA evidence >20 years old")
    print(f"  Median evidence year: {staleness.get('median_year', '?')}")
    if conf.get("pct_recent_2020") and novel.get("pct_recent_2020"):
        print(f"  Novel edges: {novel['pct_recent_2020']}% from 2020+ vs "
              f"PrimeKG-confirmed: {conf['pct_recent_2020']}% from 2020+")

    # Save
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "PrimeKG Evidence Decay Audit",
        "primekg_limitations": primekg_limits,
        "ta_evidence_age_distribution": {
            "total_triples": ta_years["total_triples"],
            "triples_with_year": ta_years["triples_with_year"],
            "year_distribution": ta_years["year_distribution"],
            "recent_2020_plus": ta_years["recent_2020_plus"],
            "old_pre_2010": ta_years["old_pre_2010"],
        },
        "staleness_metrics": staleness,
        "supersession_examples": supersession_examples[:20],
    }
    out_file = BENCHMARK_DIR / "evidence_decay_audit.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
