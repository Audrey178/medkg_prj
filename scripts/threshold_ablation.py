#!/usr/bin/env python3
"""
Threshold Ablation — ChronoMedKG Consensus
=============================================
Answers: "What happens if we require 3/3 consensus instead of 2/N?"

Reads extraction_models field from each validated triple to determine
how many models agreed. Then recomputes validation metrics at different
consensus thresholds: 1+ (any model), 2+ (current), 3 (strict).

Also computes: triple count, disease count, and onset coverage at each
threshold to show the precision-recall tradeoff.

Output:
  data/benchmark/threshold_ablation.json
"""

from __future__ import annotations

import json
import logging
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

MAX_AGE = 120


def pct(n, d):
    return round(100 * n / d, 2) if d > 0 else 0.0


def load_configs_padded():
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


def main():
    logger.info("=" * 75)
    logger.info("Threshold Ablation — Consensus Requirements")
    logger.info("=" * 75)

    # Load crosswalk + Orphadata for validation
    with open(VALIDATION_DIR / "mondo_crosswalk.json") as f:
        xwalk = json.load(f)
    with open(VALIDATION_DIR / "orphadata_with_ids.json") as f:
        orpha_by_id = json.load(f)["by_orpha_id"]

    mondo_to_orpha = xwalk["mondo_to_orpha"]
    configs = load_configs_padded()

    # Collect all validated triples with model count
    logger.info("\nLoading validated triples with model counts...")

    # Per-threshold data: threshold → {disease_dir → list of onset triples}
    thresholds = {1: "any_model", 2: "2_plus (current)", 3: "3_models"}
    # Collect all triples with their model count
    all_data = defaultdict(list)  # dir_name → [(onset_min, onset_max, n_models)]

    model_count_dist = Counter()
    total_triples = 0
    total_onset = 0

    for disease_dir in sorted(EXTRACTED_DIR.iterdir()):
        if not disease_dir.is_dir():
            continue
        vf = disease_dir / "validated_triples.jsonl"
        if not vf.exists() or vf.stat().st_size == 0:
            continue

        dir_name = disease_dir.name
        with open(vf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                except Exception:
                    continue

                total_triples += 1
                evidence = t.get("evidence") or {}
                models = evidence.get("extraction_models") or []
                n_models = len(models)
                model_count_dist[n_models] += 1

                temporal = t.get("temporal") or {}
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
                if omin > MAX_AGE or omax > MAX_AGE:
                    continue

                total_onset += 1
                all_data[dir_name].append((omin, omax, n_models))

    logger.info(f"  Total validated triples: {total_triples:,}")
    logger.info(f"  Total onset triples: {total_onset:,}")
    logger.info(f"\n  Model count distribution (all triples):")
    for n_models in sorted(model_count_dist):
        count = model_count_dist[n_models]
        logger.info(f"    {n_models} model(s): {count:>8,} ({pct(count, total_triples):.1f}%)")

    # Run validation at each threshold
    results = {}
    for threshold in [1, 2, 3]:
        label = thresholds[threshold]
        logger.info(f"\n{'=' * 75}")
        logger.info(f"THRESHOLD: {threshold}+ models ({label})")
        logger.info(f"{'=' * 75}")

        # Filter triples to those meeting threshold
        onset_count = 0
        disease_count = 0
        diseases_with_onset = {}

        for dir_name, triples in all_data.items():
            filtered = [(omin, omax) for omin, omax, nm in triples if nm >= threshold]
            if filtered:
                disease_count += 1
                onset_count += len(filtered)
                diseases_with_onset[dir_name] = filtered

        logger.info(f"  Onset triples passing threshold: {onset_count:,}")
        logger.info(f"  Diseases with onset: {disease_count}")

        # Validate against Orphadata (median aggregation)
        compared = 0
        contained = 0
        genuinely_wrong = 0
        total_per_triple = 0
        per_triple_contained = 0

        for dir_name, triples in diseases_with_onset.items():
            cfg = configs.get(dir_name)
            if not cfg:
                continue

            mondo_id = cfg["mondo_id"]
            olist = mondo_to_orpha.get(mondo_id, [])
            gold = None
            for oid in olist:
                if oid in orpha_by_id:
                    gold = orpha_by_id[oid]
                    break
            if not gold:
                continue

            ref_min = gold.get("min_age", 0)
            ref_max = gold.get("max_age", 120)
            if ref_min == 0 and ref_max >= 100:
                continue

            compared += 1
            all_mins = [t[0] for t in triples]
            all_maxs = [t[1] for t in triples]

            med_min = statistics.median(all_mins)
            med_max = statistics.median(all_maxs)
            if med_min > med_max:
                med_min, med_max = med_max, med_min

            if med_min >= ref_min and med_max <= ref_max:
                contained += 1

            # Check genuinely wrong (no overlap, gap > 10)
            overlaps = med_min <= ref_max and med_max >= ref_min
            ta_mid = (med_min + med_max) / 2
            gold_mid = (ref_min + ref_max) / 2
            if not overlaps and abs(ta_mid - gold_mid) > 10:
                genuinely_wrong += 1

            # Per-triple
            for tmin, tmax in triples:
                total_per_triple += 1
                if tmin >= ref_min and tmax <= ref_max:
                    per_triple_contained += 1

        strict_prec = pct(contained, compared)
        error_rate = pct(genuinely_wrong, compared)
        per_triple_prec = pct(per_triple_contained, total_per_triple)

        logger.info(f"\n  Orphadata validation (median aggregation):")
        logger.info(f"    Diseases compared:     {compared}")
        logger.info(f"    Strict precision:      {strict_prec}%")
        logger.info(f"    Genuine error rate:    {error_rate}%")
        logger.info(f"    Per-triple precision:  {per_triple_prec}% (n={total_per_triple:,})")

        results[threshold] = {
            "label": label,
            "onset_triples": onset_count,
            "diseases_with_onset": disease_count,
            "orphadata_compared": compared,
            "strict_precision": strict_prec,
            "genuine_error_rate": error_rate,
            "per_triple_precision": per_triple_prec,
            "per_triple_n": total_per_triple,
        }

    # Summary comparison
    logger.info(f"\n{'=' * 75}")
    logger.info("ABLATION SUMMARY")
    logger.info(f"{'=' * 75}")
    logger.info(f"\n  {'Threshold':<25} {'Triples':>10} {'Diseases':>10} {'Strict P':>10} {'Error %':>10} {'Per-triple':>12}")
    logger.info(f"  {'-' * 80}")
    for threshold in [1, 2, 3]:
        r = results[threshold]
        logger.info(
            f"  {r['label']:<25} {r['onset_triples']:>10,} {r['diseases_with_onset']:>10,} "
            f"{r['strict_precision']:>9.1f}% {r['genuine_error_rate']:>9.1f}% "
            f"{r['per_triple_precision']:>11.1f}%"
        )

    # Save
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    out_file = BENCHMARK_DIR / "threshold_ablation.json"
    with open(out_file, "w") as f:
        json.dump({
            "model_count_distribution": dict(model_count_dist),
            "thresholds": results,
        }, f, indent=2)
    logger.info(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
