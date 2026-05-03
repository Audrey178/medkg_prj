#!/usr/bin/env python3
"""
Corrected Validation Metrics — ChronoMedKG
=============================================
The original deep_validation_analysis.py used min/max aggregation across
ALL triples per disease, producing artificially wide ranges (avg 39.4 years)
that almost never fit inside narrow Orphadata bins (4-15 years).

This script computes metrics using THREE aggregation strategies:
  1. Original min/max (baseline, for comparison)
  2. Median aggregation (robust to outliers)
  3. Per-triple comparison (each triple independently vs gold standard)

Also adds fuzzy entity matching (Hypothesis A recovery) and computes
precision against BOTH validated (460K) and raw (13M) triple sets.

Output:
  data/benchmark/corrected_validation_metrics.json
  stdout — comparison tables
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


def age_ranges_overlap(a_min, a_max, b_min, b_max):
    return a_min <= b_max and a_max >= b_min


def strict_containment(inner_min, inner_max, outer_min, outer_max):
    """True if [inner_min, inner_max] ⊆ [outer_min, outer_max]."""
    return inner_min >= outer_min and inner_max <= outer_max


def pct(n, d):
    return round(100 * n / d, 2) if d > 0 else 0.0


# ──────────────────────────────────────────────
# Data Loaders
# ──────────────────────────────────────────────

def load_disease_configs():
    """Load disease name + synonyms from YAML configs."""
    configs = {}
    if not CONFIG_DIR.exists():
        return configs
    for yf in CONFIG_DIR.glob("*.yaml"):
        try:
            with open(yf) as f:
                cfg = yaml.safe_load(f)
            if cfg and "disease_name" in cfg:
                configs[yf.stem] = {
                    "disease_name": cfg["disease_name"].lower().strip(),
                    "synonyms": [s.lower().strip() for s in cfg.get("synonyms", []) if s],
                }
        except Exception:
            pass
    return configs


def load_onset_triples_per_disease(use_raw=False):
    """Load onset triples grouped by disease.

    Returns: dict[disease_name_lower] = {
        'triples': list of (onset_min, onset_max, relation, credibility_score),
        'disease_dir': str,
        'synonyms': list[str],
    }
    """
    configs = load_disease_configs()
    diseases = {}

    for disease_dir in sorted(EXTRACTED_DIR.iterdir()):
        if not disease_dir.is_dir():
            continue

        fname = "raw_triples.jsonl" if use_raw else "validated_triples.jsonl"
        triple_file = disease_dir / fname
        if not triple_file.exists() or triple_file.stat().st_size == 0:
            continue

        dir_name = disease_dir.name
        cfg = configs.get(dir_name, {})
        disease_name = cfg.get("disease_name")

        triples = []
        with open(triple_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                except Exception:
                    continue

                temporal = t.get("temporal") or t.get("temporal_context") or {}
                onset_min = temporal.get("onset_age_min")
                onset_max = temporal.get("onset_age_max")

                try:
                    onset_min = float(onset_min) if onset_min is not None else None
                except (ValueError, TypeError):
                    onset_min = None
                try:
                    onset_max = float(onset_max) if onset_max is not None else None
                except (ValueError, TypeError):
                    onset_max = None

                if onset_min is None:
                    continue

                if onset_max is None:
                    onset_max = onset_min

                relation = (t.get("relation") or "unknown").lower()
                evidence = t.get("evidence") or t.get("provenance") or {}
                cred = evidence.get("credibility_score")
                try:
                    cred = float(cred) if cred is not None else None
                except (ValueError, TypeError):
                    cred = None

                triples.append((onset_min, onset_max, relation, cred))

                # Fallback disease name
                if not disease_name:
                    if t.get("source_type") == "disease":
                        disease_name = (t.get("source_name") or "").lower().strip()
                    elif t.get("target_type") == "disease":
                        disease_name = (t.get("target_name") or "").lower().strip()
                    # Raw format
                    if not disease_name:
                        if t.get("subject_type") == "disease":
                            disease_name = (t.get("subject") or "").lower().strip()

        if disease_name and triples:
            diseases[disease_name] = {
                "triples": triples,
                "disease_dir": dir_name,
                "synonyms": cfg.get("synonyms", []),
            }

    return diseases


def load_orphadata():
    pkl = VALIDATION_DIR / "orpha_parsed.pkl"
    if not pkl.exists():
        logger.error(f"Not found: {pkl}")
        return {}
    with open(pkl, "rb") as f:
        return pickle.load(f)


def load_hpo():
    pkl = VALIDATION_DIR / "hpo_parsed.pkl"
    if not pkl.exists():
        logger.error(f"Not found: {pkl}")
        return {}
    with open(pkl, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────
# Fuzzy Matching (Hypothesis A recovery)
# ──────────────────────────────────────────────

def build_disease_matcher(ta_diseases):
    """Build exact + synonym + fuzzy matching index.

    Returns a function: match(gold_name) → ta_disease_name or None
    """
    exact = set(ta_diseases.keys())

    # Synonym → canonical name
    syn_map = {}
    for name, data in ta_diseases.items():
        for syn in data.get("synonyms", []):
            if syn and syn not in exact:
                syn_map[syn] = name

    try:
        from rapidfuzz import fuzz, process
        has_rf = True
        ta_name_list = list(exact)
    except ImportError:
        has_rf = False
        ta_name_list = []

    # Cache for fuzzy matches to avoid redundant computation
    fuzzy_cache = {}

    def match(gold_name):
        gold_lower = gold_name.lower().strip() if isinstance(gold_name, str) else gold_name

        # 1. Exact
        if gold_lower in exact:
            return gold_lower

        # 2. Synonym
        if gold_lower in syn_map:
            return syn_map[gold_lower]

        # 3. Fuzzy
        if has_rf:
            if gold_lower in fuzzy_cache:
                return fuzzy_cache[gold_lower]
            result = process.extractOne(
                gold_lower, ta_name_list,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=85,
            )
            match_name = result[0] if result else None
            fuzzy_cache[gold_lower] = match_name
            return match_name

        return None

    return match


# ──────────────────────────────────────────────
# Validation with Multiple Aggregation Strategies
# ──────────────────────────────────────────────

def validate_multi_strategy(ta_diseases, gold_standard, gold_name, matcher):
    """Validate TA against a gold standard using 3 aggregation strategies.

    gold_standard: dict[disease_name_lower] = {min_age, max_age, categories?}
    """
    # Strategy results
    strategies = {
        "minmax": {"overlap": 0, "contained": 0, "compared": 0},
        "median": {"overlap": 0, "contained": 0, "compared": 0},
        "per_triple": {"overlap": 0, "contained": 0, "total": 0, "diseases": 0},
        "iqr": {"overlap": 0, "contained": 0, "compared": 0},
    }
    skipped_allages = 0
    match_stats = {"exact": 0, "synonym": 0, "fuzzy": 0, "unmatched": 0}

    # Per-disease details for analysis
    disease_details = []

    for gold_disease, gold_entry in gold_standard.items():
        ref_min = gold_entry.get("min_age", 0)
        ref_max = gold_entry.get("max_age", 120)

        # Skip uninformative "all ages"
        if ref_min == 0 and ref_max >= 100:
            skipped_allages += 1
            continue

        # Match to TA disease
        ta_name = matcher(gold_disease)
        if ta_name is None:
            match_stats["unmatched"] += 1
            continue

        ta_data = ta_diseases.get(ta_name)
        if ta_data is None:
            match_stats["unmatched"] += 1
            continue

        # Track match type
        if gold_disease == ta_name:
            match_stats["exact"] += 1
        elif gold_disease in ta_data.get("synonyms", []):
            match_stats["synonym"] += 1
        else:
            match_stats["fuzzy"] += 1

        triples = ta_data["triples"]  # list of (onset_min, onset_max, relation, cred)
        if not triples:
            continue

        all_mins = [t[0] for t in triples]
        all_maxs = [t[1] for t in triples]

        # ── Strategy 1: Min/Max (original) ──
        agg_min = min(all_mins)
        agg_max = max(all_maxs)
        strategies["minmax"]["compared"] += 1
        if age_ranges_overlap(agg_min, agg_max, ref_min, ref_max):
            strategies["minmax"]["overlap"] += 1
        if strict_containment(agg_min, agg_max, ref_min, ref_max):
            strategies["minmax"]["contained"] += 1

        # ── Strategy 2: Median ──
        med_min = statistics.median(all_mins)
        med_max = statistics.median(all_maxs)
        # Ensure med_min <= med_max
        if med_min > med_max:
            med_min, med_max = med_max, med_min
        strategies["median"]["compared"] += 1
        if age_ranges_overlap(med_min, med_max, ref_min, ref_max):
            strategies["median"]["overlap"] += 1
        if strict_containment(med_min, med_max, ref_min, ref_max):
            strategies["median"]["contained"] += 1

        # ── Strategy 3: Per-triple ──
        strategies["per_triple"]["diseases"] += 1
        for t_min, t_max, _, _ in triples:
            strategies["per_triple"]["total"] += 1
            if age_ranges_overlap(t_min, t_max, ref_min, ref_max):
                strategies["per_triple"]["overlap"] += 1
            if strict_containment(t_min, t_max, ref_min, ref_max):
                strategies["per_triple"]["contained"] += 1

        # ── Strategy 4: IQR (25th-75th percentile) ──
        if len(all_mins) >= 4:
            q1_min = sorted(all_mins)[len(all_mins) // 4]
            q3_max = sorted(all_maxs)[3 * len(all_maxs) // 4]
            if q1_min > q3_max:
                q1_min, q3_max = q3_max, q1_min
            strategies["iqr"]["compared"] += 1
            if age_ranges_overlap(q1_min, q3_max, ref_min, ref_max):
                strategies["iqr"]["overlap"] += 1
            if strict_containment(q1_min, q3_max, ref_min, ref_max):
                strategies["iqr"]["contained"] += 1
        else:
            # Fall back to median for small triple counts
            strategies["iqr"]["compared"] += 1
            if age_ranges_overlap(med_min, med_max, ref_min, ref_max):
                strategies["iqr"]["overlap"] += 1
            if strict_containment(med_min, med_max, ref_min, ref_max):
                strategies["iqr"]["contained"] += 1

        disease_details.append({
            "gold_disease": gold_disease,
            "ta_disease": ta_name,
            "gold_range": [ref_min, ref_max],
            "minmax_range": [agg_min, agg_max],
            "median_range": [round(med_min, 2), round(med_max, 2)],
            "triple_count": len(triples),
            "minmax_width": round(agg_max - agg_min, 2),
            "median_width": round(med_max - med_min, 2),
            "gold_width": round(ref_max - ref_min, 2),
        })

    return strategies, skipped_allages, match_stats, disease_details


def print_results(gold_name, strategies, skipped, match_stats, details):
    """Print formatted comparison table."""
    logger.info(f"\n{'=' * 75}")
    logger.info(f"VALIDATION vs {gold_name.upper()}")
    logger.info(f"{'=' * 75}")

    logger.info(f"\n  Match stats: exact={match_stats['exact']}, "
                f"synonym={match_stats['synonym']}, fuzzy={match_stats['fuzzy']}, "
                f"unmatched={match_stats['unmatched']}")
    logger.info(f"  Skipped (all-ages): {skipped}")

    logger.info(f"\n  {'Strategy':<25} {'N':>6} {'Consistency':>12} {'Strict Prec':>12}")
    logger.info(f"  {'-' * 58}")

    for name, data in strategies.items():
        if name == "per_triple":
            n = data["total"]
            overlap = pct(data["overlap"], n)
            contained = pct(data["contained"], n)
            label = f"per_triple ({data['diseases']}d)"
        else:
            n = data["compared"]
            overlap = pct(data["overlap"], n)
            contained = pct(data["contained"], n)
            label = name
        logger.info(f"  {label:<25} {n:>6,} {overlap:>11.1f}% {contained:>11.1f}%")

    # Range width statistics
    if details:
        minmax_widths = [d["minmax_width"] for d in details]
        median_widths = [d["median_width"] for d in details]
        gold_widths = [d["gold_width"] for d in details]

        avg_mm = sum(minmax_widths) / len(minmax_widths)
        avg_med = sum(median_widths) / len(median_widths)
        avg_gold = sum(gold_widths) / len(gold_widths)

        med_mm = statistics.median(minmax_widths)
        med_med = statistics.median(median_widths)
        med_gold = statistics.median(gold_widths)

        logger.info(f"\n  Range widths (years):")
        logger.info(f"  {'Source':<20} {'Mean':>8} {'Median':>8}")
        logger.info(f"  {'-' * 38}")
        logger.info(f"  {'Gold standard':<20} {avg_gold:>8.1f} {med_gold:>8.1f}")
        logger.info(f"  {'TA min/max':<20} {avg_mm:>8.1f} {med_mm:>8.1f}")
        logger.info(f"  {'TA median':<20} {avg_med:>8.1f} {med_med:>8.1f}")


# ──────────────────────────────────────────────
# Credibility-Stratified Analysis
# ──────────────────────────────────────────────

def credibility_stratified_analysis(ta_diseases, gold_standard, matcher):
    """Compare per-triple precision at different credibility score quartiles."""
    # Collect all (triple, gold_range) pairs
    pairs = []
    for gold_disease, gold_entry in gold_standard.items():
        ref_min = gold_entry.get("min_age", 0)
        ref_max = gold_entry.get("max_age", 120)
        if ref_min == 0 and ref_max >= 100:
            continue

        ta_name = matcher(gold_disease)
        if ta_name is None:
            continue
        ta_data = ta_diseases.get(ta_name)
        if ta_data is None:
            continue

        for t_min, t_max, relation, cred in ta_data["triples"]:
            pairs.append({
                "onset_min": t_min,
                "onset_max": t_max,
                "cred": cred,
                "relation": relation,
                "ref_min": ref_min,
                "ref_max": ref_max,
            })

    if not pairs:
        return {}

    # Sort by credibility
    scored = [p for p in pairs if p["cred"] is not None]
    unscored = [p for p in pairs if p["cred"] is None]

    if not scored:
        return {"note": "no credibility scores available"}

    scored.sort(key=lambda x: x["cred"], reverse=True)
    n = len(scored)
    quartiles = {
        "Q1 (top 25%)": scored[:n // 4],
        "Q2 (25-50%)": scored[n // 4:n // 2],
        "Q3 (50-75%)": scored[n // 2:3 * n // 4],
        "Q4 (bottom 25%)": scored[3 * n // 4:],
    }
    if unscored:
        quartiles["Unscored"] = unscored

    logger.info(f"\n  Credibility-stratified per-triple precision:")
    logger.info(f"  {'Quartile':<20} {'N':>7} {'Overlap':>10} {'Contained':>10} {'Avg Cred':>10}")
    logger.info(f"  {'-' * 60}")

    results = {}
    for label, triples in quartiles.items():
        if not triples:
            continue
        overlap = sum(1 for t in triples if age_ranges_overlap(t["onset_min"], t["onset_max"], t["ref_min"], t["ref_max"]))
        contained = sum(1 for t in triples if strict_containment(t["onset_min"], t["onset_max"], t["ref_min"], t["ref_max"]))
        avg_cred = statistics.mean([t["cred"] for t in triples if t["cred"] is not None]) if any(t["cred"] is not None for t in triples) else 0
        nt = len(triples)
        logger.info(f"  {label:<20} {nt:>7,} {pct(overlap, nt):>9.1f}% {pct(contained, nt):>9.1f}% {avg_cred:>10.3f}")
        results[label] = {
            "n": nt,
            "overlap_pct": pct(overlap, nt),
            "contained_pct": pct(contained, nt),
            "avg_credibility": round(avg_cred, 4),
        }

    return results


# ──────────────────────────────────────────────
# Per-Relation-Type Precision
# ──────────────────────────────────────────────

def per_relation_analysis(ta_diseases, gold_standard, matcher):
    """Per-triple precision broken down by relation type."""
    relation_stats = defaultdict(lambda: {"overlap": 0, "contained": 0, "total": 0})

    for gold_disease, gold_entry in gold_standard.items():
        ref_min = gold_entry.get("min_age", 0)
        ref_max = gold_entry.get("max_age", 120)
        if ref_min == 0 and ref_max >= 100:
            continue

        ta_name = matcher(gold_disease)
        if ta_name is None:
            continue
        ta_data = ta_diseases.get(ta_name)
        if ta_data is None:
            continue

        for t_min, t_max, relation, _ in ta_data["triples"]:
            relation_stats[relation]["total"] += 1
            if age_ranges_overlap(t_min, t_max, ref_min, ref_max):
                relation_stats[relation]["overlap"] += 1
            if strict_containment(t_min, t_max, ref_min, ref_max):
                relation_stats[relation]["contained"] += 1

    logger.info(f"\n  Per-relation per-triple precision:")
    logger.info(f"  {'Relation':<35} {'N':>7} {'Overlap':>10} {'Contained':>10}")
    logger.info(f"  {'-' * 65}")

    results = {}
    for rel in sorted(relation_stats, key=lambda r: relation_stats[r]["total"], reverse=True):
        s = relation_stats[rel]
        if s["total"] < 10:
            continue
        logger.info(f"  {rel:<35} {s['total']:>7,} {pct(s['overlap'], s['total']):>9.1f}% {pct(s['contained'], s['total']):>9.1f}%")
        results[rel] = {
            "total": s["total"],
            "overlap_pct": pct(s["overlap"], s["total"]),
            "contained_pct": pct(s["contained"], s["total"]),
        }

    return results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    logger.info("=" * 75)
    logger.info("ChronoMedKG — Corrected Validation Metrics")
    logger.info("=" * 75)

    # Load data
    logger.info("\nLoading validated triples...")
    ta_validated = load_onset_triples_per_disease(use_raw=False)
    total_diseases = len(ta_validated)
    total_triples = sum(len(d["triples"]) for d in ta_validated.values())
    logger.info(f"  {total_diseases} diseases, {total_triples:,} onset triples")

    logger.info("\nLoading gold standards...")
    orphadata = load_orphadata()
    hpo = load_hpo()
    logger.info(f"  Orphadata: {len(orphadata)}, HPO: {len(hpo)}")

    # Build matcher
    logger.info("\nBuilding entity matcher (exact + synonym + fuzzy)...")
    matcher = build_disease_matcher(ta_validated)

    # ── Orphadata validation ──
    orpha_strats, orpha_skip, orpha_match, orpha_details = validate_multi_strategy(
        ta_validated, orphadata, "Orphadata", matcher
    )
    print_results("Orphadata", orpha_strats, orpha_skip, orpha_match, orpha_details)

    orpha_cred = credibility_stratified_analysis(ta_validated, orphadata, matcher)
    orpha_rel = per_relation_analysis(ta_validated, orphadata, matcher)

    # ── HPO validation ──
    hpo_strats, hpo_skip, hpo_match, hpo_details = validate_multi_strategy(
        ta_validated, hpo, "HPO", matcher
    )
    print_results("HPO", hpo_strats, hpo_skip, hpo_match, hpo_details)

    hpo_cred = credibility_stratified_analysis(ta_validated, hpo, matcher)

    # ── Summary comparison ──
    logger.info(f"\n{'=' * 75}")
    logger.info("SUMMARY: Original vs Corrected Metrics")
    logger.info(f"{'=' * 75}")

    logger.info(f"\n  Orphadata (original → corrected):")
    orig_o = orpha_strats["minmax"]["overlap"]
    orig_c = orpha_strats["minmax"]["contained"]
    orig_n = orpha_strats["minmax"]["compared"]
    med_o = orpha_strats["median"]["overlap"]
    med_c = orpha_strats["median"]["contained"]
    med_n = orpha_strats["median"]["compared"]
    pt_o = orpha_strats["per_triple"]["overlap"]
    pt_c = orpha_strats["per_triple"]["contained"]
    pt_n = orpha_strats["per_triple"]["total"]

    logger.info(f"    {'Metric':<25} {'MinMax':>10} {'Median':>10} {'PerTriple':>10}")
    logger.info(f"    {'-' * 58}")
    logger.info(f"    {'Consistency':<25} {pct(orig_o, orig_n):>9.1f}% {pct(med_o, med_n):>9.1f}% {pct(pt_o, pt_n):>9.1f}%")
    logger.info(f"    {'Strict precision':<25} {pct(orig_c, orig_n):>9.1f}% {pct(med_c, med_n):>9.1f}% {pct(pt_c, pt_n):>9.1f}%")

    logger.info(f"\n  HPO (original → corrected):")
    orig_o_h = hpo_strats["minmax"]["overlap"]
    orig_c_h = hpo_strats["minmax"]["contained"]
    orig_n_h = hpo_strats["minmax"]["compared"]
    med_o_h = hpo_strats["median"]["overlap"]
    med_c_h = hpo_strats["median"]["contained"]
    med_n_h = hpo_strats["median"]["compared"]
    pt_o_h = hpo_strats["per_triple"]["overlap"]
    pt_c_h = hpo_strats["per_triple"]["contained"]
    pt_n_h = hpo_strats["per_triple"]["total"]

    logger.info(f"    {'Metric':<25} {'MinMax':>10} {'Median':>10} {'PerTriple':>10}")
    logger.info(f"    {'-' * 58}")
    logger.info(f"    {'Consistency':<25} {pct(orig_o_h, orig_n_h):>9.1f}% {pct(med_o_h, med_n_h):>9.1f}% {pct(pt_o_h, pt_n_h):>9.1f}%")
    logger.info(f"    {'Strict precision':<25} {pct(orig_c_h, orig_n_h):>9.1f}% {pct(med_c_h, med_n_h):>9.1f}% {pct(pt_c_h, pt_n_h):>9.1f}%")

    # ── Save results ──
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "orphadata": {
            "strategies": {k: dict(v) for k, v in orpha_strats.items()},
            "skipped_allages": orpha_skip,
            "match_stats": orpha_match,
            "credibility_stratified": orpha_cred,
            "per_relation": orpha_rel,
            "range_stats": {
                "avg_minmax_width": round(statistics.mean([d["minmax_width"] for d in orpha_details]), 2) if orpha_details else 0,
                "avg_median_width": round(statistics.mean([d["median_width"] for d in orpha_details]), 2) if orpha_details else 0,
                "avg_gold_width": round(statistics.mean([d["gold_width"] for d in orpha_details]), 2) if orpha_details else 0,
                "median_minmax_width": round(statistics.median([d["minmax_width"] for d in orpha_details]), 2) if orpha_details else 0,
                "median_median_width": round(statistics.median([d["median_width"] for d in orpha_details]), 2) if orpha_details else 0,
            },
        },
        "hpo": {
            "strategies": {k: dict(v) for k, v in hpo_strats.items()},
            "skipped_allages": hpo_skip,
            "match_stats": hpo_match,
            "credibility_stratified": hpo_cred,
        },
    }

    out_file = BENCHMARK_DIR / "corrected_validation_metrics.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()
