#!/usr/bin/env python3
"""
Validation with ID-Based Matching — ChronoMedKG
==================================================
Uses MONDO → Orphanet/OMIM crosswalk for disease matching instead of
string names. Runs on BOTH validated (460K) and raw (13M) triples.

Comparison strategies per gold standard:
  1. MinMax aggregation (original, for reference)
  2. Median aggregation (corrected)
  3. Per-triple comparison
  4. IQR aggregation (25th-75th percentile)

Output:
  data/benchmark/validation_id_based.json
"""

from __future__ import annotations

import json
import logging
import statistics
import yaml
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation_sources"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
CONFIG_DIR = PROJECT_ROOT / "config" / "diseases"


def pct(n, d):
    return round(100 * n / d, 2) if d > 0 else 0.0


def age_ranges_overlap(a_min, a_max, b_min, b_max):
    return a_min <= b_max and a_max >= b_min


def strict_containment(inner_min, inner_max, outer_min, outer_max):
    return inner_min >= outer_min and inner_max <= outer_max


# ──────────────────────────────────────────────
# Data Loaders
# ──────────────────────────────────────────────

def load_disease_configs():
    """Load configs with zero-padded MONDO IDs."""
    configs = {}  # dir_name → {mondo_id_padded, disease_name}
    if not CONFIG_DIR.exists():
        return configs
    for yf in CONFIG_DIR.glob("*.yaml"):
        try:
            with open(yf) as f:
                cfg = yaml.safe_load(f)
            if not cfg or not cfg.get("mondo_id"):
                continue
            raw_id = cfg["mondo_id"]
            prefix, num = raw_id.split(":", 1)
            padded = f"{prefix}:{num.zfill(7)}"
            configs[yf.stem] = {
                "mondo_id": padded,
                "disease_name": (cfg.get("disease_name") or "").lower().strip(),
            }
        except Exception:
            pass
    return configs


def load_onset_triples_by_mondo_id(use_raw=False):
    """Load onset triples keyed by zero-padded MONDO ID.

    Returns: dict[mondo_id] = {
        'triples': list of (onset_min, onset_max, relation, cred),
        'disease_name': str,
        'disease_dir': str,
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
        cfg = configs.get(dir_name)
        if not cfg:
            continue

        mondo_id = cfg["mondo_id"]
        disease_name = cfg["disease_name"]

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

        if triples:
            diseases[mondo_id] = {
                "triples": triples,
                "disease_name": disease_name,
                "disease_dir": dir_name,
            }

    return diseases


def load_crosswalk():
    """Load MONDO crosswalk."""
    xwalk_file = VALIDATION_DIR / "mondo_crosswalk.json"
    if not xwalk_file.exists():
        logger.error(f"Crosswalk not found: {xwalk_file}")
        return None
    with open(xwalk_file) as f:
        return json.load(f)


def load_orphadata_by_id():
    """Load Orphadata keyed by Orphanet ID."""
    f = VALIDATION_DIR / "orphadata_with_ids.json"
    if not f.exists():
        logger.error(f"Not found: {f}")
        return None
    with open(f) as fh:
        data = json.load(fh)
    return data.get("by_orpha_id", {})


def load_hpoa_by_id():
    """Load HPOA keyed by OMIM ID."""
    f = VALIDATION_DIR / "hpoa_with_ids.json"
    if not f.exists():
        logger.error(f"Not found: {f}")
        return None
    with open(f) as fh:
        return json.load(fh)


# ──────────────────────────────────────────────
# Matching via ID crosswalk
# ──────────────────────────────────────────────

def match_diseases_to_orphadata(ta_diseases, crosswalk, orpha_by_id):
    """Match TA diseases to Orphadata via MONDO → Orphanet ID chain.

    Returns list of (mondo_id, ta_data, gold_entry) tuples.
    """
    mondo_to_orpha = crosswalk.get("mondo_to_orpha", {})
    matched = []
    unmatched = 0

    for mondo_id, ta_data in ta_diseases.items():
        orpha_ids = mondo_to_orpha.get(mondo_id, [])
        gold_entry = None
        for oid in orpha_ids:
            if oid in orpha_by_id:
                gold_entry = orpha_by_id[oid]
                break
        if gold_entry:
            matched.append((mondo_id, ta_data, gold_entry))
        else:
            unmatched += 1

    return matched, unmatched


def match_diseases_to_hpoa(ta_diseases, crosswalk, hpoa_by_id):
    """Match TA diseases to HPOA via MONDO → OMIM ID chain."""
    mondo_to_omim = crosswalk.get("mondo_to_omim", {})
    matched = []
    unmatched = 0

    for mondo_id, ta_data in ta_diseases.items():
        omim_ids = mondo_to_omim.get(mondo_id, [])
        gold_entry = None
        for oid in omim_ids:
            if oid in hpoa_by_id:
                gold_entry = hpoa_by_id[oid]
                break
        if gold_entry:
            matched.append((mondo_id, ta_data, gold_entry))
        else:
            unmatched += 1

    return matched, unmatched


# ──────────────────────────────────────────────
# Multi-Strategy Validation
# ──────────────────────────────────────────────

def validate(matched_pairs, gold_name):
    """Run 4-strategy validation on matched pairs.

    matched_pairs: list of (mondo_id, ta_data, gold_entry)
    """
    strategies = {
        "minmax": {"overlap": 0, "contained": 0, "compared": 0},
        "median": {"overlap": 0, "contained": 0, "compared": 0},
        "per_triple": {"overlap": 0, "contained": 0, "total": 0, "diseases": 0},
        "iqr": {"overlap": 0, "contained": 0, "compared": 0},
    }
    skipped_allages = 0
    widths = {"minmax": [], "median": [], "gold": []}

    for mondo_id, ta_data, gold_entry in matched_pairs:
        ref_min = gold_entry.get("min_age", 0)
        ref_max = gold_entry.get("max_age", 120)

        # Skip "all ages"
        if ref_min == 0 and ref_max >= 100:
            skipped_allages += 1
            continue

        triples = ta_data["triples"]
        if not triples:
            continue

        all_mins = [t[0] for t in triples]
        all_maxs = [t[1] for t in triples]

        # Strategy 1: MinMax
        agg_min, agg_max = min(all_mins), max(all_maxs)
        strategies["minmax"]["compared"] += 1
        if age_ranges_overlap(agg_min, agg_max, ref_min, ref_max):
            strategies["minmax"]["overlap"] += 1
        if strict_containment(agg_min, agg_max, ref_min, ref_max):
            strategies["minmax"]["contained"] += 1

        # Strategy 2: Median
        med_min = statistics.median(all_mins)
        med_max = statistics.median(all_maxs)
        if med_min > med_max:
            med_min, med_max = med_max, med_min
        strategies["median"]["compared"] += 1
        if age_ranges_overlap(med_min, med_max, ref_min, ref_max):
            strategies["median"]["overlap"] += 1
        if strict_containment(med_min, med_max, ref_min, ref_max):
            strategies["median"]["contained"] += 1

        # Strategy 3: Per-triple
        strategies["per_triple"]["diseases"] += 1
        for t_min, t_max, _, _ in triples:
            strategies["per_triple"]["total"] += 1
            if age_ranges_overlap(t_min, t_max, ref_min, ref_max):
                strategies["per_triple"]["overlap"] += 1
            if strict_containment(t_min, t_max, ref_min, ref_max):
                strategies["per_triple"]["contained"] += 1

        # Strategy 4: IQR
        if len(all_mins) >= 4:
            q1_min = sorted(all_mins)[len(all_mins) // 4]
            q3_max = sorted(all_maxs)[3 * len(all_maxs) // 4]
            if q1_min > q3_max:
                q1_min, q3_max = q3_max, q1_min
        else:
            q1_min, q3_max = med_min, med_max
        strategies["iqr"]["compared"] += 1
        if age_ranges_overlap(q1_min, q3_max, ref_min, ref_max):
            strategies["iqr"]["overlap"] += 1
        if strict_containment(q1_min, q3_max, ref_min, ref_max):
            strategies["iqr"]["contained"] += 1

        # Track widths
        widths["minmax"].append(agg_max - agg_min)
        widths["median"].append(med_max - med_min)
        widths["gold"].append(ref_max - ref_min)

    return strategies, skipped_allages, widths


def print_validation(label, strategies, skipped, widths, n_matched, n_unmatched):
    """Print formatted results."""
    logger.info(f"\n{'=' * 75}")
    logger.info(f"{label}")
    logger.info(f"{'=' * 75}")
    logger.info(f"  Matched: {n_matched}, Unmatched: {n_unmatched}, Skipped all-ages: {skipped}")

    logger.info(f"\n  {'Strategy':<25} {'N':>8} {'Consistency':>13} {'Strict Prec':>13}")
    logger.info(f"  {'-' * 62}")

    for name, data in strategies.items():
        if name == "per_triple":
            n = data["total"]
            label_s = f"per_triple ({data['diseases']}d)"
        else:
            n = data["compared"]
            label_s = name
        overlap = pct(data["overlap"], n)
        contained = pct(data["contained"], n)
        logger.info(f"  {label_s:<25} {n:>8,} {overlap:>12.1f}% {contained:>12.1f}%")

    if widths["gold"]:
        logger.info(f"\n  Range widths (years):")
        logger.info(f"  {'Source':<20} {'Mean':>8} {'Median':>8}")
        logger.info(f"  {'-' * 38}")
        for src in ["gold", "minmax", "median"]:
            w = widths[src]
            if w:
                logger.info(f"  {src:<20} {statistics.mean(w):>8.1f} {statistics.median(w):>8.1f}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    logger.info("=" * 75)
    logger.info("ChronoMedKG — Validation with ID-Based Matching")
    logger.info("=" * 75)

    # Load crosswalk
    crosswalk = load_crosswalk()
    if crosswalk is None:
        return

    # Load gold standards (by ID)
    orpha_by_id = load_orphadata_by_id()
    hpoa_by_id = load_hpoa_by_id()
    if orpha_by_id is None or hpoa_by_id is None:
        return

    logger.info(f"  Orphadata diseases (by Orphanet ID): {len(orpha_by_id)}")
    logger.info(f"  HPOA diseases (by OMIM ID): {len(hpoa_by_id)}")

    results = {}

    for triple_type, use_raw in [("validated", False), ("raw", True)]:
        logger.info(f"\n{'#' * 75}")
        logger.info(f"# Loading {triple_type.upper()} triples")
        logger.info(f"{'#' * 75}")

        ta_diseases = load_onset_triples_by_mondo_id(use_raw=use_raw)
        total_diseases = len(ta_diseases)
        total_triples = sum(len(d["triples"]) for d in ta_diseases.values())
        logger.info(f"  {total_diseases} diseases, {total_triples:,} onset triples")

        # ── Orphadata ──
        orpha_matched, orpha_unmatched = match_diseases_to_orphadata(
            ta_diseases, crosswalk, orpha_by_id
        )
        logger.info(f"  Orphadata ID matches: {len(orpha_matched)}, unmatched: {orpha_unmatched}")

        strats, skip, widths = validate(orpha_matched, "Orphadata")
        print_validation(
            f"ORPHADATA — {triple_type.upper()} triples (ID-based matching)",
            strats, skip, widths, len(orpha_matched), orpha_unmatched,
        )

        results[f"orphadata_{triple_type}"] = {
            "matched": len(orpha_matched),
            "unmatched": orpha_unmatched,
            "skipped_allages": skip,
            "strategies": {k: dict(v) for k, v in strats.items()},
            "range_widths": {
                "gold_mean": round(statistics.mean(widths["gold"]), 2) if widths["gold"] else 0,
                "gold_median": round(statistics.median(widths["gold"]), 2) if widths["gold"] else 0,
                "minmax_mean": round(statistics.mean(widths["minmax"]), 2) if widths["minmax"] else 0,
                "median_mean": round(statistics.mean(widths["median"]), 2) if widths["median"] else 0,
            },
        }

        # ── HPOA ──
        hpoa_matched, hpoa_unmatched = match_diseases_to_hpoa(
            ta_diseases, crosswalk, hpoa_by_id
        )
        logger.info(f"  HPOA ID matches: {len(hpoa_matched)}, unmatched: {hpoa_unmatched}")

        strats_h, skip_h, widths_h = validate(hpoa_matched, "HPOA")
        print_validation(
            f"HPOA — {triple_type.upper()} triples (ID-based matching)",
            strats_h, skip_h, widths_h, len(hpoa_matched), hpoa_unmatched,
        )

        results[f"hpoa_{triple_type}"] = {
            "matched": len(hpoa_matched),
            "unmatched": hpoa_unmatched,
            "skipped_allages": skip_h,
            "strategies": {k: dict(v) for k, v in strats_h.items()},
        }

    # ── Summary comparison ──
    logger.info(f"\n{'=' * 75}")
    logger.info("FINAL SUMMARY: Name-based vs ID-based × Validated vs Raw")
    logger.info(f"{'=' * 75}")

    # Compare to previous name-based results
    logger.info(f"\n  Orphadata strict precision (median aggregation):")
    logger.info(f"  {'Method':<35} {'Diseases':>10} {'Precision':>10}")
    logger.info(f"  {'-' * 58}")
    logger.info(f"  {'Name-based, validated':<35} {'2,445':>10} {'51.3%':>10}")

    ov = results.get("orphadata_validated", {})
    ov_strats = ov.get("strategies", {})
    ov_med = ov_strats.get("median", {})
    ov_n = ov_med.get("compared", 0)
    ov_p = pct(ov_med.get("contained", 0), ov_n)
    logger.info(f"  {'ID-based, validated':<35} {ov_n:>10,} {ov_p:>9.1f}%")

    orr = results.get("orphadata_raw", {})
    orr_strats = orr.get("strategies", {})
    orr_med = orr_strats.get("median", {})
    orr_n = orr_med.get("compared", 0)
    orr_p = pct(orr_med.get("contained", 0), orr_n)
    logger.info(f"  {'ID-based, raw':<35} {orr_n:>10,} {orr_p:>9.1f}%")

    # Save
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    out_file = BENCHMARK_DIR / "validation_id_based.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()
