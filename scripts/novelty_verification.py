#!/usr/bin/env python3
"""
Novelty Verification Script for ChronoMedKG

Samples 100 diseases that have onset information in ChronoMedKG but NOT
in any external gold standard (Orphadata, HPOA, GeneReviews). Outputs a
structured JSON + human-readable report for manual PMID spot-checking.

Usage:
    python3 scripts/novelty_verification.py

Output:
    data/benchmark/novelty_verification_sample.json
"""

import json
import os
import pickle
import random
import sys
import yaml
from collections import defaultdict
from pathlib import Path

# Reproducibility
random.seed(42)

BASE_DIR = Path("/Users/shamim/Desktop/S4C/primekg-t")
CONFIG_DIR = BASE_DIR / "config" / "diseases"
EXTRACTED_DIR = BASE_DIR / "data" / "extracted"
VALIDATION_DIR = BASE_DIR / "data" / "validation_sources"
OUTPUT_DIR = BASE_DIR / "data" / "benchmark"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pct(n, d):
    """Safe percentage."""
    if d == 0:
        return 0.0
    return round(100.0 * n / d, 1)


def safe_float(v):
    """Convert to float, return None on failure."""
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def pad_mondo_id(raw_id: str) -> str:
    """
    Zero-pad MONDO IDs to 7-digit format.
    MONDO:270 -> MONDO:0000270
    MONDO:0000270 -> MONDO:0000270  (already padded)
    """
    if not raw_id or ":" not in raw_id:
        return raw_id
    prefix, num_str = raw_id.split(":", 1)
    return f"{prefix}:{int(num_str):07d}"


def dir_name_to_mondo(dirname: str) -> str:
    """MONDO_270 -> MONDO:0000270"""
    num = dirname.replace("MONDO_", "")
    return f"MONDO:{int(num):07d}"


def mondo_to_dirname(mondo_id: str) -> str:
    """MONDO:0000270 -> MONDO_270"""
    num = mondo_id.split(":")[1]
    return f"MONDO_{int(num)}"


def get_literature_tier(pubmed_count: int) -> str:
    """Classify disease into literature tier based on PubMed article count."""
    if pubmed_count >= 100:
        return "Standard"
    elif pubmed_count >= 20:
        return "Light"
    else:
        return "Minimal"


def get_onset_bucket(n_triples: int) -> str:
    """Classify disease by number of onset triples."""
    if n_triples <= 2:
        return "1-2"
    elif n_triples <= 10:
        return "3-10"
    else:
        return ">10"


# ---------------------------------------------------------------------------
# Step 1: Load all gold standard disease IDs (with onset info)
# ---------------------------------------------------------------------------

def load_orphadata_diseases() -> set:
    """Load MONDO IDs of diseases that have onset data in Orphadata."""
    fp = VALIDATION_DIR / "orphadata_with_ids.json"
    with open(fp) as f:
        data = json.load(f)

    # Load crosswalk: orpha -> mondo
    cw_fp = VALIDATION_DIR / "mondo_crosswalk.json"
    with open(cw_fp) as f:
        crosswalk = json.load(f)

    orpha_to_mondo = crosswalk.get("orpha_to_mondo", {})
    mondo_ids = set()

    for orpha_id, entry in data.get("by_orpha_id", {}).items():
        # Only count if it has actual onset data
        if entry.get("min_age") is not None or entry.get("max_age") is not None:
            mondo = orpha_to_mondo.get(orpha_id)
            if mondo:
                mondo_ids.add(pad_mondo_id(mondo))

    return mondo_ids


def load_hpoa_diseases() -> set:
    """Load MONDO IDs of diseases that have onset data in HPOA."""
    fp = VALIDATION_DIR / "hpoa_with_ids.json"
    with open(fp) as f:
        data = json.load(f)

    # HPOA is keyed by OMIM IDs
    cw_fp = VALIDATION_DIR / "mondo_crosswalk.json"
    with open(cw_fp) as f:
        crosswalk = json.load(f)

    omim_to_mondo = crosswalk.get("omim_to_mondo", {})
    mondo_ids = set()

    for omim_id, entry in data.items():
        if entry.get("min_age") is not None or entry.get("max_age") is not None:
            mondo = omim_to_mondo.get(omim_id)
            if mondo:
                mondo_ids.add(pad_mondo_id(mondo))

    return mondo_ids


def load_genereviews_disease_names() -> set:
    """Load disease names from GeneReviews (name-matched only)."""
    fp = VALIDATION_DIR / "genereviews_parsed.pkl"
    with open(fp, "rb") as f:
        data = pickle.load(f)

    names = set()
    for entry in data.values():
        # Only include entries that have onset data
        onset = entry.get("temporal", {}).get("onset_ages", [])
        if onset:
            for name in entry.get("disease_names", []):
                names.add(name.lower().strip())

    return names


# ---------------------------------------------------------------------------
# Step 2: Load all TA diseases with onset triples
# ---------------------------------------------------------------------------

def load_ta_onset_diseases() -> dict:
    """
    Scan all extracted diseases and find those with valid onset triples.

    Returns dict: mondo_id -> {
        'disease_name': str,
        'onset_triples': list of triple dicts,
        'n_onset_triples': int,
        'onset_age_min_median': float or None,
        'onset_age_max_median': float or None,
        'pubmed_article_count': int,
        'coverage_flag': str,
        'literature_tier': str,
    }
    """
    results = {}

    if not EXTRACTED_DIR.exists():
        print(f"ERROR: {EXTRACTED_DIR} does not exist")
        sys.exit(1)

    disease_dirs = sorted([d for d in os.listdir(EXTRACTED_DIR) if d.startswith("MONDO_")])
    print(f"Scanning {len(disease_dirs)} disease directories...")

    for dirname in disease_dirs:
        triples_fp = EXTRACTED_DIR / dirname / "validated_triples.jsonl"
        if not triples_fp.exists():
            continue

        mondo_id = dir_name_to_mondo(dirname)

        # Load config for metadata
        config_fp = CONFIG_DIR / f"{dirname}.yaml"
        disease_name = mondo_id
        pubmed_count = 0
        coverage_flag = "unknown"
        if config_fp.exists():
            with open(config_fp) as f:
                cfg = yaml.safe_load(f)
            disease_name = cfg.get("disease_name", mondo_id)
            pubmed_count = cfg.get("pubmed_article_count", 0) or 0
            coverage_flag = cfg.get("coverage_flag", "unknown")

        # Scan triples for onset data
        onset_triples = []
        with open(triples_fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    triple = json.loads(line)
                except json.JSONDecodeError:
                    continue

                temporal = triple.get("temporal", {})
                onset_min = safe_float(temporal.get("onset_age_min"))
                onset_max = safe_float(temporal.get("onset_age_max"))

                # Must have at least one valid onset age
                if onset_min is None and onset_max is None:
                    continue

                # Filter onset_age > 120 (year-as-age extraction bug)
                if onset_min is not None and onset_min > 120:
                    continue
                if onset_max is not None and onset_max > 120:
                    continue

                onset_triples.append(triple)

        if not onset_triples:
            continue

        # Compute median onset ages
        mins = [safe_float(t["temporal"]["onset_age_min"]) for t in onset_triples
                if safe_float(t["temporal"]["onset_age_min"]) is not None]
        maxs = [safe_float(t["temporal"]["onset_age_max"]) for t in onset_triples
                if safe_float(t["temporal"]["onset_age_max"]) is not None]

        def median(vals):
            if not vals:
                return None
            s = sorted(vals)
            n = len(s)
            if n % 2 == 0:
                return round((s[n // 2 - 1] + s[n // 2]) / 2, 1)
            return round(s[n // 2], 1)

        results[mondo_id] = {
            "disease_name": disease_name,
            "onset_triples": onset_triples,
            "n_onset_triples": len(onset_triples),
            "onset_age_min_median": median(mins),
            "onset_age_max_median": median(maxs),
            "pubmed_article_count": pubmed_count,
            "coverage_flag": coverage_flag,
            "literature_tier": get_literature_tier(pubmed_count),
        }

    return results


# ---------------------------------------------------------------------------
# Step 3: Identify novel diseases (not in any gold standard)
# ---------------------------------------------------------------------------

def identify_novel_diseases(ta_diseases: dict, orphadata_mondos: set,
                            hpoa_mondos: set, genereviews_names: set) -> dict:
    """
    Filter to diseases NOT covered by any gold standard.
    GeneReviews is name-matched (case-insensitive).
    """
    novel = {}
    gs_covered = 0
    gs_orphadata = 0
    gs_hpoa = 0
    gs_genereviews = 0

    for mondo_id, info in ta_diseases.items():
        in_orphadata = mondo_id in orphadata_mondos
        in_hpoa = mondo_id in hpoa_mondos
        in_genereviews = info["disease_name"].lower().strip() in genereviews_names

        if in_orphadata:
            gs_orphadata += 1
        if in_hpoa:
            gs_hpoa += 1
        if in_genereviews:
            gs_genereviews += 1

        if in_orphadata or in_hpoa or in_genereviews:
            gs_covered += 1
        else:
            novel[mondo_id] = info

    print(f"\n--- Gold Standard Coverage ---")
    print(f"TA diseases with onset: {len(ta_diseases)}")
    print(f"  Covered by Orphadata:    {gs_orphadata} ({pct(gs_orphadata, len(ta_diseases))}%)")
    print(f"  Covered by HPOA:         {gs_hpoa} ({pct(gs_hpoa, len(ta_diseases))}%)")
    print(f"  Covered by GeneReviews:  {gs_genereviews} ({pct(gs_genereviews, len(ta_diseases))}%)")
    print(f"  Any gold standard:       {gs_covered} ({pct(gs_covered, len(ta_diseases))}%)")
    print(f"  NOVEL (no gold std):     {len(novel)} ({pct(len(novel), len(ta_diseases))}%)")

    return novel


# ---------------------------------------------------------------------------
# Step 4: Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(novel: dict, n_target: int = 100) -> list:
    """
    Stratify by (literature_tier, onset_bucket), then proportional sample.
    Returns list of mondo_ids.
    """
    strata = defaultdict(list)
    for mondo_id, info in novel.items():
        tier = info["literature_tier"]
        bucket = get_onset_bucket(info["n_onset_triples"])
        strata[(tier, bucket)].append(mondo_id)

    print(f"\n--- Stratification (novel diseases) ---")
    total = sum(len(v) for v in strata.values())
    for key in sorted(strata.keys()):
        print(f"  {key[0]:10s} | {key[1]:5s} | {len(strata[key]):5d} ({pct(len(strata[key]), total)}%)")

    # Proportional allocation with minimum 1 per non-empty stratum
    sampled = []
    allocations = {}
    for key, ids in strata.items():
        # Proportional share
        share = max(1, round(n_target * len(ids) / total))
        allocations[key] = min(share, len(ids))

    # Adjust if we overshoot
    while sum(allocations.values()) > n_target:
        # Reduce the largest stratum
        largest = max(allocations, key=lambda k: allocations[k])
        allocations[largest] -= 1

    # Adjust if we undershoot
    while sum(allocations.values()) < n_target:
        # Add to the largest stratum that has room
        for key in sorted(strata.keys(), key=lambda k: len(strata[k]), reverse=True):
            if allocations[key] < len(strata[key]):
                allocations[key] += 1
                break
        else:
            break  # No room to add more

    print(f"\n--- Sampling Allocation ---")
    for key in sorted(allocations.keys()):
        print(f"  {key[0]:10s} | {key[1]:5s} | {allocations[key]:3d} of {len(strata[key])}")

    for key, n_sample in allocations.items():
        chosen = random.sample(strata[key], n_sample)
        sampled.extend(chosen)

    random.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Step 5: Build verification records
# ---------------------------------------------------------------------------

def build_verification_records(sampled_ids: list, novel: dict) -> list:
    """Build structured records for each sampled disease."""
    records = []

    for mondo_id in sampled_ids:
        info = novel[mondo_id]
        triples = info["onset_triples"]

        # Collect unique PMIDs and evidence texts
        all_pmids = set()
        onset_details = []

        for t in triples:
            evidence = t.get("evidence", {})
            temporal = t.get("temporal", {})

            pmids = [s for s in evidence.get("source_ids", []) if s.startswith("PMID:")]
            all_pmids.update(pmids)

            onset_min = safe_float(temporal.get("onset_age_min"))
            onset_max = safe_float(temporal.get("onset_age_max"))
            qualifier = temporal.get("temporal_qualifier", "")
            ev_text = evidence.get("evidence_text", "")
            credibility = safe_float(evidence.get("credibility_score"))
            relation = t.get("relation", "")
            source_name = t.get("source_name", "")
            target_name = t.get("target_name", "")

            onset_details.append({
                "relation": relation,
                "source_name": source_name,
                "target_name": target_name,
                "onset_age_min": onset_min,
                "onset_age_max": onset_max,
                "temporal_qualifier": qualifier,
                "evidence_text": ev_text[:500] if ev_text else "",
                "pmids": pmids,
                "credibility_score": credibility,
            })

        record = {
            "mondo_id": mondo_id,
            "disease_name": info["disease_name"],
            "literature_tier": info["literature_tier"],
            "pubmed_article_count": info["pubmed_article_count"],
            "coverage_flag": info["coverage_flag"],
            "n_onset_triples": info["n_onset_triples"],
            "onset_age_min_median": info["onset_age_min_median"],
            "onset_age_max_median": info["onset_age_max_median"],
            "all_pmids": sorted(all_pmids),
            "n_unique_pmids": len(all_pmids),
            "onset_details": onset_details,
        }
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Step 6: Audit — double-check no sampled disease is in gold standards
# ---------------------------------------------------------------------------

def audit_sample(records: list, orphadata_mondos: set, hpoa_mondos: set,
                 genereviews_names: set) -> list:
    """
    Double-check that no sampled disease appears in any gold standard.
    Returns list of any violations found.
    """
    violations = []
    for rec in records:
        mondo_id = rec["mondo_id"]
        name = rec["disease_name"].lower().strip()

        in_orpha = mondo_id in orphadata_mondos
        in_hpoa = mondo_id in hpoa_mondos
        in_gr = name in genereviews_names

        if in_orpha or in_hpoa or in_gr:
            violations.append({
                "mondo_id": mondo_id,
                "disease_name": rec["disease_name"],
                "in_orphadata": in_orpha,
                "in_hpoa": in_hpoa,
                "in_genereviews": in_gr,
            })

    return violations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ChronoMedKG Novelty Verification")
    print("=" * 70)

    # Step 1: Load gold standards
    print("\n[1/6] Loading gold standard datasets...")
    orphadata_mondos = load_orphadata_diseases()
    print(f"  Orphadata: {len(orphadata_mondos)} MONDO IDs with onset data")

    hpoa_mondos = load_hpoa_diseases()
    print(f"  HPOA: {len(hpoa_mondos)} MONDO IDs with onset data")

    genereviews_names = load_genereviews_disease_names()
    print(f"  GeneReviews: {len(genereviews_names)} disease names with onset data")

    # Step 2: Load TA onset diseases
    print("\n[2/6] Loading ChronoMedKG onset data...")
    ta_diseases = load_ta_onset_diseases()
    print(f"  TA diseases with onset triples: {len(ta_diseases)}")

    # Step 3: Identify novel diseases
    print("\n[3/6] Identifying novel diseases...")
    novel = identify_novel_diseases(ta_diseases, orphadata_mondos, hpoa_mondos, genereviews_names)

    if len(novel) == 0:
        print("ERROR: No novel diseases found. Check data loading.")
        sys.exit(1)

    # Step 4: Stratified sample
    print("\n[4/6] Stratified sampling 100 diseases...")
    n_target = min(100, len(novel))
    sampled_ids = stratified_sample(novel, n_target)
    print(f"  Sampled: {len(sampled_ids)} diseases")

    # Step 5: Build records
    print("\n[5/6] Building verification records...")
    records = build_verification_records(sampled_ids, novel)

    # Compute summary stats
    total_pmids = sum(r["n_unique_pmids"] for r in records)
    diseases_with_pmids = sum(1 for r in records if r["n_unique_pmids"] > 0)
    tier_counts = defaultdict(int)
    bucket_counts = defaultdict(int)
    for r in records:
        tier_counts[r["literature_tier"]] += 1
        bucket_counts[get_onset_bucket(r["n_onset_triples"])] += 1

    # Step 6: Audit
    print("\n[6/6] Auditing sample against gold standards...")
    violations = audit_sample(records, orphadata_mondos, hpoa_mondos, genereviews_names)

    if violations:
        print(f"\n  WARNING: {len(violations)} VIOLATIONS found!")
        for v in violations:
            print(f"    {v['mondo_id']} ({v['disease_name']}): "
                  f"Orphadata={v['in_orphadata']}, HPOA={v['in_hpoa']}, "
                  f"GeneReviews={v['in_genereviews']}")
    else:
        print("  PASS: All 100 sampled diseases are confirmed NOT in any gold standard.")

    # Build output
    output = {
        "metadata": {
            "description": "Novelty verification sample: 100 TA diseases with onset "
                           "info NOT in any external gold standard",
            "generation_date": "2026-04-14",
            "random_seed": 42,
            "n_sampled": len(records),
            "n_novel_diseases_total": len(novel),
            "n_ta_onset_diseases": len(ta_diseases),
            "gold_standard_sizes": {
                "orphadata_mondo_ids": len(orphadata_mondos),
                "hpoa_mondo_ids": len(hpoa_mondos),
                "genereviews_names": len(genereviews_names),
            },
            "stratification": {
                "by_tier": dict(tier_counts),
                "by_onset_bucket": dict(bucket_counts),
            },
            "summary_stats": {
                "total_unique_pmids": total_pmids,
                "diseases_with_pmids": diseases_with_pmids,
                "diseases_without_pmids": len(records) - diseases_with_pmids,
                "audit_violations": len(violations),
            },
        },
        "audit_violations": violations,
        "samples": records,
    }

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_fp = OUTPUT_DIR / "novelty_verification_sample.json"
    with open(out_fp, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Output written to: {out_fp}")

    # Print summary report
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    print(f"TA diseases with onset data:     {len(ta_diseases)}")
    print(f"Covered by any gold standard:    {len(ta_diseases) - len(novel)}")
    print(f"Novel (no gold standard):        {len(novel)} ({pct(len(novel), len(ta_diseases))}%)")
    print(f"Sampled for verification:        {len(records)}")
    print(f"  By tier:  {dict(tier_counts)}")
    print(f"  By onset: {dict(bucket_counts)}")
    print(f"Total unique PMIDs in sample:    {total_pmids}")
    print(f"Diseases with PMID evidence:     {diseases_with_pmids}/{len(records)}")
    print(f"Audit violations:                {len(violations)}")
    print("=" * 70)

    # Print first 5 records as preview
    print("\n--- Sample Preview (first 5) ---")
    for r in records[:5]:
        print(f"\n  {r['mondo_id']} — {r['disease_name']}")
        print(f"    Tier: {r['literature_tier']} | PubMed: {r['pubmed_article_count']} | "
              f"Onset triples: {r['n_onset_triples']}")
        print(f"    Onset range (median): {r['onset_age_min_median']} - {r['onset_age_max_median']} years")
        print(f"    PMIDs: {', '.join(r['all_pmids'][:5])}"
              + (" ..." if len(r['all_pmids']) > 5 else ""))
        if r["onset_details"]:
            d = r["onset_details"][0]
            print(f"    First triple: {d['source_name']} -> {d['target_name']} "
                  f"({d['temporal_qualifier']})")
            if d["evidence_text"]:
                print(f"    Evidence: {d['evidence_text'][:120]}...")


if __name__ == "__main__":
    main()
