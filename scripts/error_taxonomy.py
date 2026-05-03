#!/usr/bin/env python3
"""
Error Taxonomy — ChronoMedKG vs Orphadata
=============================================
Classifies ALL non-matching diseases (where median TA range is NOT fully
contained within the gold standard range) into error categories.

Categories:
  1. GRANULARITY_MISMATCH: TA and gold overlap, but TA range spans a
     wider developmental window (e.g., TA=2-12yr vs gold=1-5yr "Childhood")
  2. ADJACENT_STAGE: TA range is in the next developmental stage
     (e.g., gold=Childhood 1-5, TA=Juvenile 5-15)
  3. PHENOTYPE_VS_DISEASE: TA extracted phenotype-specific timing,
     gold is disease-level onset (TA=specific feature at age 30,
     gold=onset at birth — both correct, different scope)
  4. TA_WIDER_BUT_CORRECT: TA range encompasses gold range (gold ⊆ TA)
     — not wrong, just imprecise
  5. GENUINELY_WRONG: No overlap, not adjacent — extraction error
  6. GOLD_POSSIBLY_OUTDATED: TA cites recent evidence (>2020) that
     contradicts older gold standard categorization
  7. SINGLE_TRIPLE_NOISE: Only 1 onset triple — high variance expected

Output:
  data/benchmark/error_taxonomy.json
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


def pct(n, d):
    return round(100 * n / d, 2) if d > 0 else 0.0


# Orphadata onset bins as ordered stages for adjacency detection
ONSET_STAGES_ORDERED = [
    ("Antenatal", 0, 0),
    ("Neonatal", 0, 0.08),
    ("Infancy", 0.08, 1),
    ("Childhood", 1, 5),
    ("Adolescent", 5, 15),     # Orphadata uses "Juvenile" sometimes
    ("Adult", 15, 120),
    ("Elderly", 60, 120),
]

# Map age to stage index for adjacency check
def age_to_stage_index(age):
    """Return the index of the most specific stage containing this age."""
    for i, (name, smin, smax) in enumerate(ONSET_STAGES_ORDERED):
        if smin <= age <= smax:
            return i
    return len(ONSET_STAGES_ORDERED) - 1  # default to last


def classify_mismatch(ta_med_min, ta_med_max, ref_min, ref_max,
                      gold_categories, n_triples, triples_data):
    """Classify a non-contained disease into an error category.

    Returns: (category, explanation)
    """
    ta_overlaps = ta_med_min <= ref_max and ta_med_max >= ref_min
    gold_contains_ta = ta_med_min >= ref_min and ta_med_max <= ref_max
    ta_contains_gold = ref_min >= ta_med_min and ref_max <= ta_med_max
    gold_width = ref_max - ref_min
    ta_width = ta_med_max - ta_med_min

    # Single triple = inherently noisy
    if n_triples == 1:
        return "SINGLE_TRIPLE_NOISE", "Only 1 onset triple — insufficient data for reliable aggregation"

    # TA range encompasses gold range (gold ⊆ TA) — imprecise but not wrong
    if ta_contains_gold and ta_overlaps:
        return "TA_WIDER_BUT_CORRECT", (
            f"TA range [{ta_med_min:.1f}-{ta_med_max:.1f}] encompasses "
            f"gold [{ref_min}-{ref_max}] — imprecise, not wrong"
        )

    # Overlapping but not contained — check granularity
    if ta_overlaps:
        # How much does TA extend beyond gold?
        overshoot_low = max(0, ref_min - ta_med_min)
        overshoot_high = max(0, ta_med_max - ref_max)
        total_overshoot = overshoot_low + overshoot_high

        # Check if the non-onset relations are pulling the range
        phenotype_relations = {"disease_phenotype_positive", "disease_phenotype_negative",
                               "manifests_as", "onset_at"}
        non_phenotype_count = sum(
            1 for t in triples_data
            if t.get("relation", "").lower() not in phenotype_relations
        )
        non_pheno_pct = non_phenotype_count / len(triples_data) if triples_data else 0

        if non_pheno_pct > 0.4:
            return "PHENOTYPE_VS_DISEASE", (
                f"TA range [{ta_med_min:.1f}-{ta_med_max:.1f}] includes "
                f"{non_pheno_pct:.0%} non-phenotype relations (drug timing, gene associations) "
                f"pulling the range beyond gold [{ref_min}-{ref_max}]"
            )

        if total_overshoot <= gold_width * 0.5:
            return "GRANULARITY_MISMATCH", (
                f"TA [{ta_med_min:.1f}-{ta_med_max:.1f}] overlaps gold [{ref_min}-{ref_max}] "
                f"but extends {total_overshoot:.1f}yr beyond — within half the gold bin width"
            )
        else:
            return "GRANULARITY_MISMATCH", (
                f"TA [{ta_med_min:.1f}-{ta_med_max:.1f}] overlaps gold [{ref_min}-{ref_max}] "
                f"but extends {total_overshoot:.1f}yr beyond — exceeds gold bin width"
            )

    # No overlap — check adjacency
    ta_stage = age_to_stage_index((ta_med_min + ta_med_max) / 2)
    gold_stage_min = age_to_stage_index(ref_min)
    gold_stage_max = age_to_stage_index(ref_max)

    stage_distance = min(abs(ta_stage - gold_stage_min), abs(ta_stage - gold_stage_max))

    if stage_distance <= 1:
        ta_stage_name = ONSET_STAGES_ORDERED[ta_stage][0] if ta_stage < len(ONSET_STAGES_ORDERED) else "Unknown"
        gold_stage_names = [c for c in gold_categories if c not in ("All ages", "No data available")]
        return "ADJACENT_STAGE", (
            f"TA median age {(ta_med_min+ta_med_max)/2:.1f}yr ({ta_stage_name}) "
            f"is one stage away from gold ({', '.join(gold_stage_names)})"
        )

    # Check for recent evidence (proxy: look at PMID years if available)
    recent_evidence = False
    for t in triples_data:
        pmids = t.get("source_ids", [])
        for pmid in pmids:
            # PMIDs > 33000000 are roughly 2020+
            try:
                if int(str(pmid).replace("PMID:", "")) > 33000000:
                    recent_evidence = True
                    break
            except (ValueError, TypeError):
                pass
        if recent_evidence:
            break

    if recent_evidence:
        return "GOLD_POSSIBLY_OUTDATED", (
            f"TA [{ta_med_min:.1f}-{ta_med_max:.1f}] cites post-2020 evidence "
            f"contradicting gold [{ref_min}-{ref_max}] ({', '.join(gold_categories)})"
        )

    return "GENUINELY_WRONG", (
        f"TA [{ta_med_min:.1f}-{ta_med_max:.1f}] has no overlap with "
        f"gold [{ref_min}-{ref_max}] ({', '.join(gold_categories)}) — likely extraction error"
    )


def main():
    logger.info("=" * 75)
    logger.info("Error Taxonomy — ChronoMedKG vs Orphadata")
    logger.info("=" * 75)

    # Load crosswalk + gold standard
    with open(VALIDATION_DIR / "mondo_crosswalk.json") as f:
        xwalk = json.load(f)
    with open(VALIDATION_DIR / "orphadata_with_ids.json") as f:
        orpha_data = json.load(f)

    orpha_by_id = orpha_data["by_orpha_id"]
    mondo_to_orpha = xwalk["mondo_to_orpha"]

    # Load disease configs
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

    # Process all diseases
    taxonomy = Counter()
    examples_per_cat = defaultdict(list)
    all_classified = []

    contained_count = 0
    skipped_allages = 0
    no_gold = 0
    no_triples = 0
    total_compared = 0

    for disease_dir in sorted(EXTRACTED_DIR.iterdir()):
        if not disease_dir.is_dir():
            continue

        dir_name = disease_dir.name
        cfg = configs.get(dir_name)
        if not cfg:
            continue

        mondo_id = cfg["mondo_id"]
        disease_name = cfg["disease_name"]

        # Match to Orphadata via ID
        olist = mondo_to_orpha.get(mondo_id, [])
        gold = None
        for oid in olist:
            if oid in orpha_by_id:
                gold = orpha_by_id[oid]
                break
        if not gold:
            no_gold += 1
            continue

        ref_min = gold.get("min_age", 0)
        ref_max = gold.get("max_age", 120)
        categories = gold.get("categories", [])

        if ref_min == 0 and ref_max >= 100:
            skipped_allages += 1
            continue

        # Load validated triples
        vf = disease_dir / "validated_triples.jsonl"
        if not vf.exists() or vf.stat().st_size == 0:
            no_triples += 1
            continue

        onset_triples = []
        with open(vf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                except Exception:
                    continue

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

                evidence = t.get("evidence") or {}
                onset_triples.append({
                    "min": omin,
                    "max": omax,
                    "relation": (t.get("relation") or "").lower(),
                    "target": (t.get("target_name") or "")[:60],
                    "source_ids": evidence.get("source_ids", []),
                    "temporal_qualifier": temporal.get("temporal_qualifier", ""),
                    "progression_stage": temporal.get("progression_stage", ""),
                })

        if not onset_triples:
            no_triples += 1
            continue

        total_compared += 1
        all_mins = [t["min"] for t in onset_triples]
        all_maxs = [t["max"] for t in onset_triples]

        med_min = statistics.median(all_mins)
        med_max = statistics.median(all_maxs)
        if med_min > med_max:
            med_min, med_max = med_max, med_min

        # Check containment
        if med_min >= ref_min and med_max <= ref_max:
            contained_count += 1
            continue

        # Classify the error
        category, explanation = classify_mismatch(
            med_min, med_max, ref_min, ref_max,
            categories, len(onset_triples), onset_triples,
        )

        taxonomy[category] += 1
        entry = {
            "disease": disease_name,
            "mondo_id": mondo_id,
            "ta_median_range": [round(med_min, 2), round(med_max, 2)],
            "gold_range": [ref_min, ref_max],
            "gold_categories": categories,
            "n_onset_triples": len(onset_triples),
            "category": category,
            "explanation": explanation,
        }
        all_classified.append(entry)

        if len(examples_per_cat[category]) < 5:
            examples_per_cat[category].append(entry)

    # ── Results ──
    total_errors = sum(taxonomy.values())
    logger.info(f"\n  Total compared: {total_compared}")
    logger.info(f"  Contained (correct): {contained_count} ({pct(contained_count, total_compared)}%)")
    logger.info(f"  Not contained (errors): {total_errors} ({pct(total_errors, total_compared)}%)")
    logger.info(f"  No gold match: {no_gold}")
    logger.info(f"  No onset triples: {no_triples}")
    logger.info(f"  Skipped all-ages: {skipped_allages}")

    logger.info(f"\n{'=' * 75}")
    logger.info("ERROR TAXONOMY BREAKDOWN")
    logger.info(f"{'=' * 75}")

    # Sort by count
    sorted_cats = sorted(taxonomy.items(), key=lambda x: x[1], reverse=True)
    logger.info(f"\n  {'Category':<30} {'Count':>7} {'% of errors':>12} {'% of total':>12}")
    logger.info(f"  {'-' * 65}")

    for cat, count in sorted_cats:
        logger.info(f"  {cat:<30} {count:>7} {pct(count, total_errors):>11.1f}% {pct(count, total_compared):>11.1f}%")

    # Reinterpretation: what % are "not really wrong"?
    benign = sum(taxonomy[c] for c in [
        "GRANULARITY_MISMATCH", "TA_WIDER_BUT_CORRECT",
        "PHENOTYPE_VS_DISEASE", "SINGLE_TRIPLE_NOISE",
        "GOLD_POSSIBLY_OUTDATED", "ADJACENT_STAGE",
    ])
    genuine = taxonomy.get("GENUINELY_WRONG", 0)

    logger.info(f"\n  ── REINTERPRETATION ──")
    logger.info(f"  Contained (truly correct):     {contained_count:>5} ({pct(contained_count, total_compared):.1f}%)")
    logger.info(f"  Benign mismatches:             {benign:>5} ({pct(benign, total_compared):.1f}%)")
    logger.info(f"  Genuinely wrong:               {genuine:>5} ({pct(genuine, total_compared):.1f}%)")
    logger.info(f"  ────────────────────────────────────────")
    logger.info(f"  Effective accuracy:            {pct(contained_count + benign, total_compared):.1f}%")
    logger.info(f"  True error rate:               {pct(genuine, total_compared):.1f}%")

    # Print examples
    logger.info(f"\n{'=' * 75}")
    logger.info("EXAMPLES PER CATEGORY")
    logger.info(f"{'=' * 75}")

    for cat, count in sorted_cats:
        logger.info(f"\n  --- {cat} ({count} diseases) ---")
        for ex in examples_per_cat[cat][:3]:
            logger.info(f"    {ex['disease'][:50]}")
            logger.info(f"      TA: {ex['ta_median_range']}  Gold: {ex['gold_range']} ({ex['gold_categories']})")
            logger.info(f"      {ex['explanation'][:90]}")

    # Save
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "summary": {
            "total_compared": total_compared,
            "contained": contained_count,
            "contained_pct": pct(contained_count, total_compared),
            "not_contained": total_errors,
            "taxonomy": dict(taxonomy),
            "benign_mismatches": benign,
            "genuinely_wrong": genuine,
            "effective_accuracy": pct(contained_count + benign, total_compared),
            "true_error_rate": pct(genuine, total_compared),
        },
        "all_errors": all_classified,
        "examples": {k: v for k, v in examples_per_cat.items()},
    }

    out_file = BENCHMARK_DIR / "error_taxonomy.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()
