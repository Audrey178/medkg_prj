#!/usr/bin/env python3
"""Quality audit on all validated triples across extracted diseases.

Reads ALL validated_triples.jsonl files from data/extracted/ and computes
comprehensive quality metrics. Saves results to data/benchmark/quality_audit_v3.json.
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
OUTPUT_PATH = PROJECT_ROOT / "data" / "benchmark" / "quality_audit_v3.json"

# Temporal field names to check
TEMPORAL_FIELDS = [
    "onset_age_min", "onset_age_max", "temporal_qualifier", "milestone",
    "progression_stage", "duration", "discovery_date", "treatment_start_age",
    "validity_start", "validity_end",
]


def has_value(val):
    """Check if a value is non-null and non-empty."""
    if val is None:
        return False
    if isinstance(val, str) and val.strip() in ("", "unknown", "null", "None"):
        return False
    return True


def main():
    if not EXTRACTED_DIR.exists():
        print(f"ERROR: Extracted directory not found: {EXTRACTED_DIR}")
        sys.exit(1)

    # Find all disease directories with validated_triples.jsonl
    disease_dirs = sorted([
        d for d in EXTRACTED_DIR.iterdir()
        if d.is_dir() and (d / "validated_triples.jsonl").exists()
    ])
    print(f"Found {len(disease_dirs)} disease directories with validated_triples.jsonl")

    # Counters and accumulators
    total_triples = 0
    total_diseases = 0
    triples_per_disease = []
    disease_names = []

    # Temporal coverage
    temporal_any = 0
    temporal_field_counts = Counter()

    # Relation types
    relation_counts = Counter()

    # Entity types
    source_type_counts = Counter()
    target_type_counts = Counter()

    # Evidence quality
    has_evidence_text_count = 0
    has_pmid_count = 0
    confidence_values = []
    confidence_buckets = Counter()  # 0.0-0.2, 0.2-0.4, ...

    # Quality grades
    grade_counts = Counter()

    # Model attribution
    model_counts = Counter()
    model_combo_counts = Counter()

    # Type mismatch
    type_mismatch_count = 0

    # Study types
    study_type_counts = Counter()

    # Credibility scores
    credibility_values = []

    # Evidence tier
    tier_counts = Counter()

    # Process each disease
    for disease_dir in disease_dirs:
        triples_file = disease_dir / "validated_triples.jsonl"
        disease_triple_count = 0

        try:
            with open(triples_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        triple = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    total_triples += 1
                    disease_triple_count += 1

                    # --- Temporal coverage ---
                    temporal = triple.get("temporal", {}) or {}
                    has_any_temporal = False
                    for field in TEMPORAL_FIELDS:
                        val = temporal.get(field)
                        if has_value(val):
                            temporal_field_counts[field] += 1
                            has_any_temporal = True
                    if has_any_temporal:
                        temporal_any += 1

                    # --- Relation type ---
                    relation = triple.get("relation", "unknown")
                    relation_counts[relation] += 1

                    # --- Entity types ---
                    src_type = triple.get("source_type", "unknown")
                    tgt_type = triple.get("target_type", "unknown")
                    source_type_counts[src_type] += 1
                    target_type_counts[tgt_type] += 1

                    # --- Type mismatch check ---
                    # Check if source/target types are inconsistent with relation
                    relation_lower = relation.lower()
                    if "phenotype" in relation_lower:
                        if src_type not in ("disease", "phenotype") and tgt_type not in ("disease", "phenotype"):
                            type_mismatch_count += 1
                    elif relation_lower == "indication":
                        if src_type != "drug" and tgt_type != "drug":
                            type_mismatch_count += 1

                    # --- Evidence quality ---
                    evidence = triple.get("evidence", {}) or {}
                    if has_value(evidence.get("evidence_text")):
                        has_evidence_text_count += 1
                    source_ids = evidence.get("source_ids", []) or []
                    if any(str(s).startswith("PMID") for s in source_ids):
                        has_pmid_count += 1

                    conf = evidence.get("consensus_confidence")
                    if conf is not None:
                        try:
                            conf = float(conf)
                            confidence_values.append(conf)
                            bucket = min(int(conf * 5), 4)  # 0-4 for 0.0-1.0
                            bucket_label = f"{bucket*0.2:.1f}-{(bucket+1)*0.2:.1f}"
                            confidence_buckets[bucket_label] += 1
                        except (ValueError, TypeError):
                            pass

                    cred = evidence.get("credibility_score")
                    if cred is not None:
                        try:
                            credibility_values.append(float(cred))
                        except (ValueError, TypeError):
                            pass

                    study_type = evidence.get("study_type", "unknown")
                    study_type_counts[study_type] += 1

                    tier = evidence.get("tier", "unknown")
                    tier_counts[str(tier)] += 1

                    # Model attribution
                    models = evidence.get("extraction_models", []) or []
                    for m in models:
                        model_counts[m] += 1
                    combo = tuple(sorted(models))
                    if combo:
                        model_combo_counts[str(combo)] += 1

                    # --- Quality grade ---
                    grade = triple.get("quality_grade", "unknown")
                    grade_counts[grade] += 1

        except OSError as e:
            print(f"  WARNING: Failed to read {triples_file}: {e}", file=sys.stderr)
            continue

        if disease_triple_count > 0:
            total_diseases += 1
            triples_per_disease.append(disease_triple_count)
            disease_names.append(disease_dir.name)

    # Compute distribution stats for triples per disease
    if triples_per_disease:
        arr = np.array(triples_per_disease)
        disease_dist = {
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
            "mean": round(float(np.mean(arr)), 2),
            "median": round(float(np.median(arr)), 2),
            "std": round(float(np.std(arr)), 2),
            "p5": round(float(np.percentile(arr, 5)), 2),
            "p25": round(float(np.percentile(arr, 25)), 2),
            "p75": round(float(np.percentile(arr, 75)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
        }
    else:
        disease_dist = {}

    # Confidence stats
    if confidence_values:
        conf_arr = np.array(confidence_values)
        confidence_stats = {
            "mean": round(float(np.mean(conf_arr)), 4),
            "median": round(float(np.median(conf_arr)), 4),
            "std": round(float(np.std(conf_arr)), 4),
            "min": round(float(np.min(conf_arr)), 4),
            "max": round(float(np.max(conf_arr)), 4),
        }
    else:
        confidence_stats = {}

    # Credibility stats
    if credibility_values:
        cred_arr = np.array(credibility_values)
        credibility_stats = {
            "mean": round(float(np.mean(cred_arr)), 4),
            "median": round(float(np.median(cred_arr)), 4),
            "std": round(float(np.std(cred_arr)), 4),
            "min": round(float(np.min(cred_arr)), 4),
            "max": round(float(np.max(cred_arr)), 4),
        }
    else:
        credibility_stats = {}

    # Build results
    results = {
        "audit_version": "v3",
        "audit_date": "2026-04-03",
        "summary": {
            "total_triples": total_triples,
            "total_diseases": total_diseases,
            "total_disease_dirs": len(disease_dirs),
        },
        "temporal_coverage": {
            "any_temporal": temporal_any,
            "any_temporal_pct": round(temporal_any / max(total_triples, 1) * 100, 2),
            "field_counts": dict(temporal_field_counts.most_common()),
            "field_percentages": {
                k: round(v / max(total_triples, 1) * 100, 2)
                for k, v in temporal_field_counts.most_common()
            },
        },
        "relation_types": {
            "distribution": dict(relation_counts.most_common()),
            "unique_count": len(relation_counts),
        },
        "entity_types": {
            "source_types": dict(source_type_counts.most_common()),
            "target_types": dict(target_type_counts.most_common()),
        },
        "evidence_quality": {
            "has_evidence_text": has_evidence_text_count,
            "has_evidence_text_pct": round(has_evidence_text_count / max(total_triples, 1) * 100, 2),
            "has_pmid": has_pmid_count,
            "has_pmid_pct": round(has_pmid_count / max(total_triples, 1) * 100, 2),
            "confidence_stats": confidence_stats,
            "confidence_distribution": dict(sorted(confidence_buckets.items())),
            "credibility_stats": credibility_stats,
            "study_types": dict(study_type_counts.most_common()),
            "evidence_tiers": dict(tier_counts.most_common()),
        },
        "quality_grades": {
            "distribution": dict(grade_counts.most_common()),
            "percentages": {
                k: round(v / max(total_triples, 1) * 100, 2)
                for k, v in grade_counts.most_common()
            },
        },
        "model_attribution": {
            "per_model": dict(model_counts.most_common()),
            "top_combos": dict(Counter(model_combo_counts).most_common(15)),
        },
        "type_mismatches": {
            "count": type_mismatch_count,
            "percentage": round(type_mismatch_count / max(total_triples, 1) * 100, 2),
        },
        "disease_triple_distribution": disease_dist,
        "comparison_with_v1": {
            "v1_triples": 284000,
            "v3_triples": total_triples,
            "growth_factor": round(total_triples / max(284000, 1), 2),
            "v1_diseases": "~5",
            "v3_diseases": total_diseases,
        },
    }

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print full report
    print(f"\n{'='*70}")
    print(f"CHRONOMEDKG QUALITY AUDIT v3")
    print(f"{'='*70}")

    print(f"\n--- 1. SCALE ---")
    print(f"Total triples:                 {total_triples:,}")
    print(f"Total diseases with triples:   {total_diseases:,}")
    print(f"Disease dirs scanned:          {len(disease_dirs):,}")

    print(f"\n--- 2. TEMPORAL COVERAGE ---")
    print(f"Triples with ANY temporal:     {temporal_any:,} ({temporal_any/max(total_triples,1)*100:.1f}%)")
    for field, count in temporal_field_counts.most_common():
        pct = count / max(total_triples, 1) * 100
        print(f"  {field:25s}   {count:>8,} ({pct:5.1f}%)")

    print(f"\n--- 3. RELATION TYPE DISTRIBUTION ---")
    for rel, count in relation_counts.most_common():
        pct = count / max(total_triples, 1) * 100
        print(f"  {rel:35s}   {count:>8,} ({pct:5.1f}%)")

    print(f"\n--- 4. ENTITY TYPE DISTRIBUTION ---")
    print(f"  Source types:")
    for t, c in source_type_counts.most_common():
        print(f"    {t:25s}   {c:>8,} ({c/max(total_triples,1)*100:5.1f}%)")
    print(f"  Target types:")
    for t, c in target_type_counts.most_common():
        print(f"    {t:25s}   {c:>8,} ({c/max(total_triples,1)*100:5.1f}%)")

    print(f"\n--- 5. EVIDENCE QUALITY ---")
    print(f"Has evidence text:             {has_evidence_text_count:,} ({has_evidence_text_count/max(total_triples,1)*100:.1f}%)")
    print(f"Has PMID:                      {has_pmid_count:,} ({has_pmid_count/max(total_triples,1)*100:.1f}%)")
    if confidence_stats:
        print(f"Avg consensus confidence:      {confidence_stats['mean']:.4f}")
        print(f"Median consensus confidence:   {confidence_stats['median']:.4f}")
    if credibility_stats:
        print(f"Avg credibility score:         {credibility_stats['mean']:.4f}")
        print(f"Median credibility score:      {credibility_stats['median']:.4f}")
    print(f"  Confidence distribution:")
    for bucket, count in sorted(confidence_buckets.items()):
        print(f"    {bucket}: {count:>8,}")
    print(f"  Study types:")
    for st, count in study_type_counts.most_common():
        print(f"    {st:25s}   {count:>8,}")
    print(f"  Evidence tiers:")
    for t, c in tier_counts.most_common():
        print(f"    Tier {t}: {c:>8,}")

    print(f"\n--- 6. QUALITY GRADES ---")
    for g, c in grade_counts.most_common():
        print(f"  Grade {g}: {c:>8,} ({c/max(total_triples,1)*100:5.1f}%)")

    print(f"\n--- 7. MODEL ATTRIBUTION ---")
    for m, c in model_counts.most_common():
        print(f"  {m:25s}   {c:>8,}")
    print(f"  Top model combos:")
    for combo, c in Counter(model_combo_counts).most_common(10):
        print(f"    {combo}: {c:>8,}")

    print(f"\n--- 8. TYPE MISMATCHES ---")
    print(f"Type mismatch count:           {type_mismatch_count:,} ({type_mismatch_count/max(total_triples,1)*100:.2f}%)")

    print(f"\n--- 9. TRIPLE DISTRIBUTION ACROSS DISEASES ---")
    if disease_dist:
        print(f"  Min:    {disease_dist['min']}")
        print(f"  P5:     {disease_dist['p5']}")
        print(f"  P25:    {disease_dist['p25']}")
        print(f"  Median: {disease_dist['median']}")
        print(f"  Mean:   {disease_dist['mean']}")
        print(f"  P75:    {disease_dist['p75']}")
        print(f"  P95:    {disease_dist['p95']}")
        print(f"  P99:    {disease_dist['p99']}")
        print(f"  Max:    {disease_dist['max']}")
        print(f"  Std:    {disease_dist['std']}")

    print(f"\n--- 10. COMPARISON WITH v1 (284K triples) ---")
    print(f"  v1 triples: ~284,000")
    print(f"  v3 triples: {total_triples:,}")
    print(f"  Growth:     {total_triples/284000:.1f}x")
    print(f"  v1 diseases: ~5")
    print(f"  v3 diseases: {total_diseases:,}")

    print(f"\n{'='*70}")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
