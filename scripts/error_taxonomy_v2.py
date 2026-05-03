#!/usr/bin/env python3
"""
Error Taxonomy v2 (Strict) — ChronoMedKG vs Orphadata + GeneReviews
======================================================================
Strict classification rules (audit-corrected):
  - ADJACENT_STAGE only if gap ≤ 10 years
  - GOLD_POSSIBLY_OUTDATED removed as a category (too generous)
  - Adds GeneReviews as a third gold standard (462 diseases)
  - Reports per-triple taxonomy (not just median aggregation)

Categories (strict):
  1. GRANULARITY_MISMATCH — ranges overlap but TA extends beyond gold
  2. ADJACENT_STAGE — no overlap, gap ≤ 10 years (one developmental stage)
  3. TA_WIDER_BUT_CORRECT — TA range encompasses gold (gold ⊆ TA)
  4. SINGLE_TRIPLE_NOISE — only 1 onset triple
  5. PHENOTYPE_VS_DISEASE — non-phenotype relations pull the range
  6. GENUINELY_WRONG — no overlap, gap > 10 years, or no excuse

Output:
  data/benchmark/error_taxonomy_v2.json
"""

from __future__ import annotations

import json
import logging
import pickle
import statistics
import yaml
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation_sources"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
CONFIG_DIR = PROJECT_ROOT / "config" / "diseases"

MAX_AGE_CUTOFF = 120  # Filter onset values above this (year-as-age bug)


def pct(n, d):
    return round(100 * n / d, 2) if d > 0 else 0.0


def classify_strict(ta_med_min, ta_med_max, ref_min, ref_max,
                    n_triples, non_pheno_ratio):
    """Strict error classification. Returns (category, explanation)."""
    ta_overlaps = ta_med_min <= ref_max and ta_med_max >= ref_min
    ta_contains_gold = ref_min >= ta_med_min and ref_max <= ta_med_max
    ta_mid = (ta_med_min + ta_med_max) / 2
    gold_mid = (ref_min + ref_max) / 2
    gap = abs(ta_mid - gold_mid)

    if n_triples == 1:
        return "SINGLE_TRIPLE_NOISE", (
            f"Only 1 onset triple — TA=[{ta_med_min:.1f}-{ta_med_max:.1f}] "
            f"vs gold=[{ref_min}-{ref_max}]"
        )

    if ta_contains_gold and ta_overlaps:
        return "TA_WIDER_BUT_CORRECT", (
            f"TA=[{ta_med_min:.1f}-{ta_med_max:.1f}] encompasses "
            f"gold=[{ref_min}-{ref_max}]"
        )

    if ta_overlaps:
        if non_pheno_ratio > 0.4:
            return "PHENOTYPE_VS_DISEASE", (
                f"TA=[{ta_med_min:.1f}-{ta_med_max:.1f}] has {non_pheno_ratio:.0%} "
                f"non-phenotype relations pulling range beyond gold=[{ref_min}-{ref_max}]"
            )
        return "GRANULARITY_MISMATCH", (
            f"TA=[{ta_med_min:.1f}-{ta_med_max:.1f}] overlaps "
            f"gold=[{ref_min}-{ref_max}] but extends beyond"
        )

    # No overlap
    if gap <= 10:
        return "ADJACENT_STAGE", (
            f"TA midpoint {ta_mid:.1f}yr vs gold midpoint {gold_mid:.1f}yr — "
            f"gap={gap:.1f}yr (≤10yr threshold)"
        )

    return "GENUINELY_WRONG", (
        f"TA=[{ta_med_min:.1f}-{ta_med_max:.1f}] vs gold=[{ref_min}-{ref_max}] — "
        f"gap={gap:.1f}yr, no overlap"
    )


def load_configs_padded():
    """Load disease configs with zero-padded MONDO IDs."""
    configs = {}
    for yf in CONFIG_DIR.glob("*.yaml"):
        try:
            with open(yf) as f:
                cfg = yaml.safe_load(f)
            if cfg and cfg.get("mondo_id"):
                padded = f"MONDO:{cfg['mondo_id'].split(':')[1].zfill(7)}"
                configs[yf.stem] = {
                    "mondo_id": padded,
                    "disease_name": (cfg.get("disease_name") or "").lower().strip(),
                }
        except Exception:
            pass
    return configs


def load_onset_triples(disease_dir, use_raw=False):
    """Load onset triples for a single disease. Returns list of dicts."""
    fname = "raw_triples.jsonl" if use_raw else "validated_triples.jsonl"
    tf = disease_dir / fname
    if not tf.exists() or tf.stat().st_size == 0:
        return []

    PHENOTYPE_RELS = {
        "disease_phenotype_positive", "disease_phenotype_negative",
        "manifests_as", "onset_at",
    }

    triples = []
    with open(tf) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except Exception:
                continue

            temporal = t.get("temporal") or t.get("temporal_context") or {}
            omin = temporal.get("onset_age_min")
            omax = temporal.get("onset_age_max")
            try:
                omin = float(omin) if omin is not None else None
            except (ValueError, TypeError):
                omin = None
            try:
                omax = float(omax) if omax is not None else None
            except (ValueError, TypeError):
                omax = None

            if omin is None:
                continue
            if omax is None:
                omax = omin
            # Filter year-as-age outliers
            if omin > MAX_AGE_CUTOFF or omax > MAX_AGE_CUTOFF:
                continue

            rel = (t.get("relation") or "").lower()
            is_phenotype = rel in PHENOTYPE_RELS

            triples.append({
                "min": omin,
                "max": omax,
                "is_phenotype_rel": is_phenotype,
            })

    return triples


def run_taxonomy(ta_triples_by_dir, gold_lookup, gold_name, configs, mondo_to_gold):
    """Run error taxonomy against a gold standard.

    Args:
        ta_triples_by_dir: dict[dir_name] → list of triple dicts
        gold_lookup: dict[gold_id] → {min_age, max_age, ...}
        gold_name: str (for logging)
        configs: dict[dir_name] → {mondo_id, disease_name}
        mondo_to_gold: dict[mondo_id] → [gold_id, ...]
    """
    taxonomy = Counter()
    contained_count = 0
    total_compared = 0
    skipped_allages = 0
    no_match = 0
    no_triples = 0
    all_errors = []

    for dir_name, triples in ta_triples_by_dir.items():
        cfg = configs.get(dir_name)
        if not cfg:
            continue

        mondo_id = cfg["mondo_id"]
        disease_name = cfg["disease_name"]

        # Match to gold standard via ID
        gold_ids = mondo_to_gold.get(mondo_id, [])
        gold_entry = None
        for gid in gold_ids:
            if gid in gold_lookup:
                gold_entry = gold_lookup[gid]
                break
        if gold_entry is None:
            no_match += 1
            continue

        ref_min = gold_entry.get("min_age", 0)
        ref_max = gold_entry.get("max_age", 120)

        if ref_min == 0 and ref_max >= 100:
            skipped_allages += 1
            continue

        if not triples:
            no_triples += 1
            continue

        total_compared += 1
        all_mins = [t["min"] for t in triples]
        all_maxs = [t["max"] for t in triples]

        med_min = statistics.median(all_mins)
        med_max = statistics.median(all_maxs)
        if med_min > med_max:
            med_min, med_max = med_max, med_min

        # Check containment
        if med_min >= ref_min and med_max <= ref_max:
            contained_count += 1
            continue

        # Classify error
        non_pheno = sum(1 for t in triples if not t["is_phenotype_rel"])
        non_pheno_ratio = non_pheno / len(triples) if triples else 0

        category, explanation = classify_strict(
            med_min, med_max, ref_min, ref_max,
            len(triples), non_pheno_ratio,
        )
        taxonomy[category] += 1
        all_errors.append({
            "disease": disease_name,
            "mondo_id": mondo_id,
            "ta_median": [round(med_min, 2), round(med_max, 2)],
            "gold_range": [ref_min, ref_max],
            "n_triples": len(triples),
            "category": category,
            "explanation": explanation,
        })

    errors = sum(taxonomy.values())
    genuine = taxonomy.get("GENUINELY_WRONG", 0)
    benign = errors - genuine

    return {
        "gold_standard": gold_name,
        "total_compared": total_compared,
        "contained": contained_count,
        "contained_pct": pct(contained_count, total_compared),
        "errors": errors,
        "taxonomy": dict(taxonomy),
        "genuine_errors": genuine,
        "genuine_error_pct": pct(genuine, total_compared),
        "benign_mismatches": benign,
        "effective_accuracy": pct(contained_count + benign, total_compared),
        "skipped_allages": skipped_allages,
        "no_match": no_match,
        "no_triples": no_triples,
        "all_errors": all_errors,
    }


def main():
    logger.info("=" * 75)
    logger.info("Error Taxonomy v2 (Strict) — ChronoMedKG")
    logger.info("=" * 75)

    # Load crosswalk
    with open(VALIDATION_DIR / "mondo_crosswalk.json") as f:
        xwalk = json.load(f)

    # Load gold standards
    # 1. Orphadata (by Orphanet ID)
    with open(VALIDATION_DIR / "orphadata_with_ids.json") as f:
        orpha_data = json.load(f)
    orpha_by_id = orpha_data["by_orpha_id"]
    logger.info(f"  Orphadata: {len(orpha_by_id)} diseases")

    # 2. HPOA (by OMIM ID)
    with open(VALIDATION_DIR / "hpoa_with_ids.json") as f:
        hpoa_by_id = json.load(f)
    logger.info(f"  HPOA: {len(hpoa_by_id)} diseases")

    # 3. GeneReviews (by MONDO ID via OMIM crosswalk + name fallback)
    gr_by_mondo_file = VALIDATION_DIR / "genereviews_by_mondo.json"
    gr_by_mondo = {}
    if gr_by_mondo_file.exists():
        with open(gr_by_mondo_file) as f:
            gr_by_mondo = json.load(f)
        logger.info(f"  GeneReviews: {len(gr_by_mondo)} diseases (by MONDO ID)")
    else:
        logger.warning("  GeneReviews: genereviews_by_mondo.json NOT FOUND — run build_ontology_crosswalk.py first")

    # Load disease configs
    configs = load_configs_padded()
    logger.info(f"  Disease configs: {len(configs)}")

    # Load triples for both validated and raw
    for triple_type, use_raw in [("validated", False), ("raw", True)]:
        logger.info(f"\n{'#' * 75}")
        logger.info(f"# {triple_type.upper()} TRIPLES")
        logger.info(f"{'#' * 75}")

        # Load all onset triples
        ta_triples = {}
        for disease_dir in sorted(EXTRACTED_DIR.iterdir()):
            if not disease_dir.is_dir():
                continue
            triples = load_onset_triples(disease_dir, use_raw=use_raw)
            if triples:
                ta_triples[disease_dir.name] = triples

        total_triples = sum(len(t) for t in ta_triples.values())
        logger.info(f"  Loaded {len(ta_triples)} diseases, {total_triples:,} onset triples")

        # Run taxonomy against each gold standard
        # 1. Orphadata
        orpha_result = run_taxonomy(
            ta_triples, orpha_by_id, "Orphadata",
            configs, xwalk["mondo_to_orpha"],
        )
        print_results(orpha_result)

        # 2. HPOA
        hpoa_result = run_taxonomy(
            ta_triples, hpoa_by_id, "HPOA",
            configs, xwalk["mondo_to_omim"],
        )
        print_results(hpoa_result)

        # 3. GeneReviews (MONDO ID matching via OMIM crosswalk)
        # gr_by_mondo is already keyed by MONDO ID → {min_age, max_age, ...}
        # Build a trivial mondo_to_gr mapping: MONDO → [MONDO] (self-reference)
        mondo_to_gr = {mondo_id: [mondo_id] for mondo_id in gr_by_mondo}

        gr_result = run_taxonomy(
            ta_triples, gr_by_mondo, "GeneReviews",
            configs, mondo_to_gr,
        )
        print_results(gr_result)

        # Save results for this triple type
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        out = {
            "triple_type": triple_type,
            "orphadata": {k: v for k, v in orpha_result.items() if k != "all_errors"},
            "hpoa": {k: v for k, v in hpoa_result.items() if k != "all_errors"},
            "genereviews": {k: v for k, v in gr_result.items() if k != "all_errors"},
            "orphadata_errors": orpha_result["all_errors"],
        }
        out_file = BENCHMARK_DIR / f"error_taxonomy_v2_{triple_type}.json"
        with open(out_file, "w") as f:
            json.dump(out, f, indent=2, default=str)
        logger.info(f"\n  Saved: {out_file}")

    # Final summary table
    logger.info(f"\n{'=' * 75}")
    logger.info("CROSS-GOLD-STANDARD SUMMARY")
    logger.info(f"{'=' * 75}")


def print_results(result):
    """Print formatted taxonomy results."""
    gold = result["gold_standard"]
    total = result["total_compared"]
    contained = result["contained"]
    errors = result["errors"]
    genuine = result["genuine_errors"]
    benign = result["benign_mismatches"]

    logger.info(f"\n  --- {gold} (n={total}) ---")
    if total == 0:
        logger.info(f"    No diseases compared")
        return

    logger.info(f"    Contained:          {contained:>5} ({pct(contained, total):.1f}%)")

    taxonomy = result["taxonomy"]
    for cat in ["GRANULARITY_MISMATCH", "ADJACENT_STAGE", "TA_WIDER_BUT_CORRECT",
                "SINGLE_TRIPLE_NOISE", "PHENOTYPE_VS_DISEASE", "GENUINELY_WRONG"]:
        count = taxonomy.get(cat, 0)
        if count > 0:
            logger.info(f"    {cat:<25} {count:>5} ({pct(count, total):.1f}%)")

    logger.info(f"    ────────────────────────────────────")
    logger.info(f"    Effective accuracy:  {result['effective_accuracy']:.1f}%")
    logger.info(f"    True error rate:     {result['genuine_error_pct']:.1f}%")
    logger.info(f"    (no match: {result['no_match']}, no triples: {result['no_triples']}, skipped all-ages: {result['skipped_allages']})")


if __name__ == "__main__":
    main()
