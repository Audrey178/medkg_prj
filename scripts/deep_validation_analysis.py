#!/usr/bin/env python3
"""
Deep Validation Analysis for ChronoMedKG
==========================================
Answers three key methodological questions:

1. Are overlaps with Orphadata/HPO/HPOA "precision" or "consistency"?
   → Computes both precision AND recall where possible.

2. Do results differ between 460K consensus triples vs ALL raw triples?
   → Runs validation on both sets, side by side.

3. If a disease misses in one source (e.g., 10% miss in HPO),
   is it covered by another source?
   → Cross-source gap analysis.

Outputs:
  data/benchmark/deep_validation_analysis.json
  Prints detailed tables to stdout
"""

from __future__ import annotations
import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation_sources"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"

# HPO onset term -> age range mapping
HPO_ONSET_TO_AGE = {
    "HP:0030674": (0, 0),       # Antenatal
    "HP:0003577": (0, 0),       # Congenital
    "HP:0003623": (0, 0.08),    # Neonatal (0-28d)
    "HP:0003593": (0.08, 1),    # Infantile (28d-1yr)
    "HP:0011463": (1, 5),       # Childhood (1-5yr)
    "HP:0003621": (5, 15),      # Juvenile (5-15yr)
    "HP:0011462": (15, 40),     # Young adult
    "HP:0003584": (15, 120),    # Late onset
    "HP:0003581": (40, 120),    # Adult onset
    "HP:0025708": (60, 120),    # Middle age
}


def age_ranges_overlap(our_min, our_max, ref_min, ref_max):
    return our_min <= ref_max and our_max >= ref_min


def load_disease_onsets_from_triples(use_raw=False):
    """
    Load disease onset data from either validated (consensus) or raw triples.

    Returns: dict[disease_name_lower] = {
        'aggregate_min': float,
        'aggregate_max': float,
        'phenotype_onsets': {name: (min, max)},
        'triple_count': int,
        'disease_dir': str,
    }
    """
    import yaml

    # Load disease name mapping
    config_dir = PROJECT_ROOT / "config" / "diseases"
    disease_name_map = {}
    if config_dir.exists():
        for yf in config_dir.glob("*.yaml"):
            try:
                with open(yf) as f:
                    cfg = yaml.safe_load(f)
                if cfg and "disease_name" in cfg:
                    disease_name_map[yf.stem] = cfg["disease_name"].lower().strip()
            except Exception:
                pass

    onsets = {}

    for disease_dir in sorted(EXTRACTED_DIR.iterdir()):
        if not disease_dir.is_dir():
            continue

        if use_raw:
            triple_file = disease_dir / "raw_triples.jsonl"
        else:
            triple_file = disease_dir / "validated_triples.jsonl"

        if not triple_file.exists() or triple_file.stat().st_size == 0:
            continue

        dir_name = disease_dir.name
        disease_name = disease_name_map.get(dir_name, None)

        all_onset_mins = []
        all_onset_maxs = []
        phenotype_onsets = {}

        with open(triple_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    triple = json.loads(line)
                except Exception:
                    continue

                # Handle both validated (temporal) and raw (temporal_context) formats
                temporal = triple.get("temporal") or triple.get("temporal_context") or {}
                onset_min = temporal.get("onset_age_min")
                onset_max = temporal.get("onset_age_max")

                # Clean numeric values (raw triples may have 'unknown' etc.)
                try:
                    onset_min = float(onset_min) if onset_min is not None else None
                except (ValueError, TypeError):
                    onset_min = None
                try:
                    onset_max = float(onset_max) if onset_max is not None else None
                except (ValueError, TypeError):
                    onset_max = None

                if onset_min is not None:
                    all_onset_mins.append(onset_min)
                    if onset_max is not None:
                        all_onset_maxs.append(onset_max)
                    else:
                        all_onset_maxs.append(onset_min)

                    relation = (triple.get("relation") or "").lower()
                    if "phenotype" in relation or relation in ("manifests_as", "onset_at"):
                        # Handle both naming conventions
                        pheno_name = (triple.get("target_name") or triple.get("object") or "").lower().strip()
                        if pheno_name and pheno_name != (disease_name or ""):
                            phenotype_onsets[pheno_name] = (
                                onset_min,
                                onset_max if onset_max is not None else onset_min
                            )

                    if not disease_name:
                        # Validated format
                        for field in ("source_name", "target_name"):
                            if triple.get(f"{field.replace('_name', '_type')}") == "disease":
                                disease_name = (triple.get(field) or "").lower().strip()
                                break
                        # Raw format
                        if not disease_name:
                            if triple.get("subject_type") == "disease":
                                disease_name = (triple.get("subject") or "").lower().strip()
                            elif triple.get("object_type") == "disease":
                                disease_name = (triple.get("object") or "").lower().strip()

        if disease_name and all_onset_mins:
            onsets[disease_name] = {
                "aggregate_min": min(all_onset_mins),
                "aggregate_max": max(all_onset_maxs),
                "phenotype_onsets": phenotype_onsets,
                "triple_count": len(all_onset_mins),
                "disease_dir": dir_name,
            }

    return onsets


def validate_against_orphadata(disease_onsets, orpha):
    """Validate against Orphadata. Returns detailed results."""
    results = {"consistent": [], "inconsistent": [], "skipped_allages": []}

    for disease_name, onset in disease_onsets.items():
        orpha_entry = orpha.get(disease_name)
        if not orpha_entry:
            continue

        our_min = onset["aggregate_min"]
        our_max = onset["aggregate_max"]
        ref_min = orpha_entry.get("min_age", 0)
        ref_max = orpha_entry.get("max_age", 120)

        # Skip "all ages" diseases (0-120) — uninformative
        categories = orpha_entry.get("categories", [])
        if ref_min == 0 and ref_max >= 100:
            results["skipped_allages"].append(disease_name)
            continue

        entry = {
            "disease": disease_name,
            "ours": [our_min, our_max],
            "orphadata": [ref_min, ref_max],
            "categories": categories,
        }

        if age_ranges_overlap(our_min, our_max, ref_min, ref_max):
            results["consistent"].append(entry)
        else:
            results["inconsistent"].append(entry)

    return results


def validate_against_hpo(disease_onsets, hpo):
    """Validate against HPO disease-level onsets."""
    results = {"consistent": [], "inconsistent": []}

    for disease_name, onset in disease_onsets.items():
        hpo_entry = hpo.get(disease_name)
        if not hpo_entry:
            continue

        our_min = onset["aggregate_min"]
        our_max = onset["aggregate_max"]
        ref_min = max(hpo_entry.get("min_age", 0), 0)
        ref_max = hpo_entry.get("max_age", 120)

        entry = {"disease": disease_name, "ours": [our_min, our_max], "hpo": [ref_min, ref_max]}

        if age_ranges_overlap(our_min, our_max, ref_min, ref_max):
            results["consistent"].append(entry)
        else:
            results["inconsistent"].append(entry)

    return results


def validate_against_hpoa(disease_onsets, hpoa_per_pheno):
    """Per-phenotype validation against HPOA."""
    results = {"consistent": 0, "inconsistent": 0, "ambiguous": 0,
               "diseases_checked": set(), "total_comparisons": 0}

    for disease_name, onset in disease_onsets.items():
        hpoa_entry = hpoa_per_pheno.get(disease_name)
        if not hpoa_entry:
            continue

        our_phenotypes = onset.get("phenotype_onsets", {})
        if not our_phenotypes:
            continue

        results["diseases_checked"].add(disease_name)

        for hpo_id, (ref_min, ref_max) in hpoa_entry.items():
            found_overlap = False
            for pheno_name, (our_min, our_max) in our_phenotypes.items():
                results["total_comparisons"] += 1
                if age_ranges_overlap(our_min, our_max, ref_min, ref_max):
                    results["consistent"] += 1
                    found_overlap = True
                    break
                elif our_min > ref_max:
                    results["ambiguous"] += 1
                    found_overlap = True
                    break
            if not found_overlap and our_phenotypes:
                results["total_comparisons"] += 1
                results["inconsistent"] += 1

    results["diseases_checked"] = len(results["diseases_checked"])
    return results


def load_hpoa_per_phenotype():
    hpoa_file = VALIDATION_DIR / "phenotype.hpoa"
    if not hpoa_file.exists():
        return {}
    result = defaultdict(dict)
    with open(hpoa_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 7 or parts[0] == "database_id":
                continue
            disease_name = parts[1].lower().strip()
            onset_term = parts[6].strip() if len(parts) > 6 else ""
            hpo_id = parts[3]
            if onset_term and onset_term in HPO_ONSET_TO_AGE:
                onset_min, onset_max = HPO_ONSET_TO_AGE[onset_term]
                result[disease_name][hpo_id] = (onset_min, onset_max)
    return dict(result)


def main():
    # Load gold standards
    logger.info("Loading gold standards...")
    with open(VALIDATION_DIR / "orpha_parsed.pkl", "rb") as f:
        orpha = pickle.load(f)
    with open(VALIDATION_DIR / "hpo_parsed.pkl", "rb") as f:
        hpo = pickle.load(f)
    hpoa = load_hpoa_per_phenotype()
    logger.info("  Orphadata: %d diseases", len(orpha))
    logger.info("  HPO:       %d diseases", len(hpo))
    logger.info("  HPOA:      %d diseases (per-phenotype)", len(hpoa))

    # ================================================================
    # PART 1: Run validation on BOTH raw and consensus triples
    # ================================================================
    all_results = {}

    for label, use_raw in [("consensus_460k", False), ("raw_13M", True)]:
        logger.info("\n" + "=" * 70)
        logger.info("  VALIDATING: %s triples", label.upper())
        logger.info("=" * 70)

        onsets = load_disease_onsets_from_triples(use_raw=use_raw)
        logger.info("Diseases with onset data: %d", len(onsets))

        orpha_res = validate_against_orphadata(onsets, orpha)
        hpo_res = validate_against_hpo(onsets, hpo)
        hpoa_res = validate_against_hpoa(onsets, hpoa)

        n_orpha = len(orpha_res["consistent"]) + len(orpha_res["inconsistent"])
        n_hpo = len(hpo_res["consistent"]) + len(hpo_res["inconsistent"])

        orpha_rate = len(orpha_res["consistent"]) / n_orpha * 100 if n_orpha else 0
        hpo_rate = len(hpo_res["consistent"]) / n_hpo * 100 if n_hpo else 0
        hpoa_rate = hpoa_res["consistent"] / hpoa_res["total_comparisons"] * 100 if hpoa_res["total_comparisons"] else 0

        logger.info("\n  %s Results:", label)
        logger.info("  %-30s  %6s  %6s  %6s  %6s", "Source", "Match", "Total", "Rate", "Skipped")
        logger.info("  " + "-" * 75)
        logger.info("  %-30s  %6d  %6d  %5.1f%%  %6d",
                     "Orphadata (disease-level)",
                     len(orpha_res["consistent"]), n_orpha, orpha_rate,
                     len(orpha_res.get("skipped_allages", [])))
        logger.info("  %-30s  %6d  %6d  %5.1f%%  %6s",
                     "HPO (disease-level)",
                     len(hpo_res["consistent"]), n_hpo, hpo_rate, "-")
        logger.info("  %-30s  %6d  %6d  %5.1f%%  %6s",
                     "HPOA (per-phenotype)",
                     hpoa_res["consistent"], hpoa_res["total_comparisons"], hpoa_rate, "-")

        all_results[label] = {
            "diseases_with_onset": len(onsets),
            "orphadata": {
                "consistent": len(orpha_res["consistent"]),
                "inconsistent": len(orpha_res["inconsistent"]),
                "skipped_allages": len(orpha_res.get("skipped_allages", [])),
                "total_benchmarked": n_orpha,
                "consistency_rate": round(orpha_rate, 2),
                "inconsistent_diseases": [e["disease"] for e in orpha_res["inconsistent"][:20]],
            },
            "hpo": {
                "consistent": len(hpo_res["consistent"]),
                "inconsistent": len(hpo_res["inconsistent"]),
                "total_benchmarked": n_hpo,
                "consistency_rate": round(hpo_rate, 2),
                "inconsistent_diseases": [e["disease"] for e in hpo_res["inconsistent"][:20]],
            },
            "hpoa": {
                "consistent": hpoa_res["consistent"],
                "inconsistent": hpoa_res["inconsistent"],
                "ambiguous": hpoa_res["ambiguous"],
                "diseases_checked": hpoa_res["diseases_checked"],
                "total_comparisons": hpoa_res["total_comparisons"],
                "consistency_rate": round(hpoa_rate, 2),
            },
        }

    # ================================================================
    # PART 2: What EXACTLY is the 95.1% metric?
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  METRIC DEFINITIONS")
    logger.info("=" * 70)

    logger.info("""
  The 95.1%% Orphadata number is CONSISTENCY (overlap), not precision or accuracy.

  What it means:
    - We take each disease where BOTH TA and Orphadata have onset age data
    - We check: does TA's aggregate onset range [min, max] overlap with
      Orphadata's categorical onset range?
    - 95.1%% of the time, YES — they overlap.

  What it is NOT:
    - NOT precision: we don't check if TA claims X that Orphadata says is wrong.
      Overlap ≠ exact match. E.g., TA says 2-18yr, Orphadata says "Childhood" (1-5yr)
      → overlap=True, but TA's range is broader.
    - NOT recall: we don't measure what fraction of Orphadata diseases TA covers.
    - NOT accuracy: accuracy implies a binary correct/incorrect judgment.

  Better terminology for the paper:
    - "Consistency rate" or "concordance rate" (age-range overlap)
    - We can ALSO compute:
      * Precision: % of TA onset claims that fall WITHIN Orphadata's range
      * Recall: % of Orphadata diseases that TA covers with onset data
      * Exact match: % where ranges match closely (within ±2 years)
    """)

    # ================================================================
    # PART 3: Compute PRECISION and RECALL explicitly
    # ================================================================
    logger.info("=" * 70)
    logger.info("  PRECISION & RECALL ANALYSIS")
    logger.info("=" * 70)

    consensus_onsets = load_disease_onsets_from_triples(use_raw=False)

    # Against Orphadata
    orpha_total = len(orpha)  # all Orphadata diseases
    orpha_with_onset = sum(1 for v in orpha.values()
                          if v.get("min_age", 0) != 0 or v.get("max_age", 120) != 120)

    ta_in_orpha = 0       # TA diseases that have a match in Orphadata
    ta_consistent = 0     # TA diseases consistent with Orphadata
    ta_contained = 0      # TA range fully contained within Orphadata range (strict precision)
    orpha_covered_by_ta = 0  # Orphadata diseases that TA has onset data for (recall)

    orpha_inconsistent_in_hpo = 0  # Orphadata misses that ARE in HPO
    orpha_inconsistent_diseases = []

    for disease_name, onset in consensus_onsets.items():
        orpha_entry = orpha.get(disease_name)
        if not orpha_entry:
            continue

        ref_min = orpha_entry.get("min_age", 0)
        ref_max = orpha_entry.get("max_age", 120)
        if ref_min == 0 and ref_max >= 100:
            continue  # skip "all ages"

        ta_in_orpha += 1
        our_min = onset["aggregate_min"]
        our_max = onset["aggregate_max"]

        if age_ranges_overlap(our_min, our_max, ref_min, ref_max):
            ta_consistent += 1
        else:
            orpha_inconsistent_diseases.append(disease_name)

        # Strict: TA range fully within Orphadata
        if our_min >= ref_min and our_max <= ref_max:
            ta_contained += 1

    # Recall: fraction of Orphadata diseases covered by TA
    for disease_name in orpha:
        if disease_name in consensus_onsets:
            orpha_covered_by_ta += 1

    # Cross-check: are Orphadata misses covered by HPO?
    for disease_name in orpha_inconsistent_diseases:
        if disease_name in hpo:
            hpo_entry = hpo[disease_name]
            onset = consensus_onsets.get(disease_name, {})
            if onset:
                our_min = onset["aggregate_min"]
                our_max = onset["aggregate_max"]
                ref_min = max(hpo_entry.get("min_age", 0), 0)
                ref_max = hpo_entry.get("max_age", 120)
                if age_ranges_overlap(our_min, our_max, ref_min, ref_max):
                    orpha_inconsistent_in_hpo += 1

    logger.info("\n  Orphadata Analysis:")
    logger.info("  %-40s  %d", "Total Orphadata diseases:", orpha_total)
    logger.info("  %-40s  %d", "Orphadata with informative onset:", orpha_with_onset)
    logger.info("  %-40s  %d / %d (%.1f%%)", "TA diseases overlapping with Orphadata:",
                ta_in_orpha, len(consensus_onsets),
                ta_in_orpha / len(consensus_onsets) * 100 if consensus_onsets else 0)
    logger.info("")
    logger.info("  %-40s  %d / %d = %.1f%%", "CONSISTENCY (range overlap):",
                ta_consistent, ta_in_orpha,
                ta_consistent / ta_in_orpha * 100 if ta_in_orpha else 0)
    logger.info("  %-40s  %d / %d = %.1f%%", "STRICT PRECISION (TA ⊆ Orphadata):",
                ta_contained, ta_in_orpha,
                ta_contained / ta_in_orpha * 100 if ta_in_orpha else 0)
    logger.info("  %-40s  %d / %d = %.1f%%", "RECALL (Orphadata covered by TA):",
                orpha_covered_by_ta, orpha_total,
                orpha_covered_by_ta / orpha_total * 100 if orpha_total else 0)

    # HPO same analysis
    hpo_in_ta = 0
    hpo_consistent_count = 0
    hpo_contained = 0
    hpo_covered = 0
    hpo_inconsistent_diseases = []

    for disease_name, onset in consensus_onsets.items():
        hpo_entry = hpo.get(disease_name)
        if not hpo_entry:
            continue
        ref_min = max(hpo_entry.get("min_age", 0), 0)
        ref_max = hpo_entry.get("max_age", 120)
        hpo_in_ta += 1
        our_min = onset["aggregate_min"]
        our_max = onset["aggregate_max"]

        if age_ranges_overlap(our_min, our_max, ref_min, ref_max):
            hpo_consistent_count += 1
        else:
            hpo_inconsistent_diseases.append(disease_name)

        if our_min >= ref_min and our_max <= ref_max:
            hpo_contained += 1

    for disease_name in hpo:
        if disease_name in consensus_onsets:
            hpo_covered += 1

    logger.info("\n  HPO Analysis:")
    logger.info("  %-40s  %d / %d = %.1f%%", "CONSISTENCY:",
                hpo_consistent_count, hpo_in_ta,
                hpo_consistent_count / hpo_in_ta * 100 if hpo_in_ta else 0)
    logger.info("  %-40s  %d / %d = %.1f%%", "STRICT PRECISION (TA ⊆ HPO):",
                hpo_contained, hpo_in_ta,
                hpo_contained / hpo_in_ta * 100 if hpo_in_ta else 0)
    logger.info("  %-40s  %d / %d = %.1f%%", "RECALL (HPO covered by TA):",
                hpo_covered, len(hpo),
                hpo_covered / len(hpo) * 100 if hpo else 0)

    # ================================================================
    # PART 4: Cross-source gap analysis
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  CROSS-SOURCE GAP ANALYSIS")
    logger.info("=" * 70)
    logger.info("  Q: If a disease is inconsistent in source A, is it consistent in source B?")

    # Orphadata misses → covered by HPO?
    orpha_miss_in_hpo = 0
    orpha_miss_in_hpoa = 0
    orpha_miss_nowhere = 0

    for disease_name in orpha_inconsistent_diseases:
        in_hpo = disease_name in hpo
        in_hpoa = disease_name in hpoa

        if in_hpo:
            onset = consensus_onsets.get(disease_name, {})
            if onset:
                hpo_entry = hpo[disease_name]
                our_min = onset["aggregate_min"]
                our_max = onset["aggregate_max"]
                ref_min = max(hpo_entry.get("min_age", 0), 0)
                ref_max = hpo_entry.get("max_age", 120)
                if age_ranges_overlap(our_min, our_max, ref_min, ref_max):
                    orpha_miss_in_hpo += 1
                    continue

        if in_hpoa:
            orpha_miss_in_hpoa += 1
            continue

        orpha_miss_nowhere += 1

    logger.info("\n  Orphadata inconsistencies (%d diseases):", len(orpha_inconsistent_diseases))
    logger.info("    %-45s  %d (%.1f%%)", "Consistent with HPO instead:",
                orpha_miss_in_hpo,
                orpha_miss_in_hpo / len(orpha_inconsistent_diseases) * 100 if orpha_inconsistent_diseases else 0)
    logger.info("    %-45s  %d (%.1f%%)", "Has HPOA data (per-phenotype):",
                orpha_miss_in_hpoa,
                orpha_miss_in_hpoa / len(orpha_inconsistent_diseases) * 100 if orpha_inconsistent_diseases else 0)
    logger.info("    %-45s  %d (%.1f%%)", "Not covered by any other source:",
                orpha_miss_nowhere,
                orpha_miss_nowhere / len(orpha_inconsistent_diseases) * 100 if orpha_inconsistent_diseases else 0)

    # HPO misses → covered by Orphadata?
    hpo_miss_in_orpha = 0
    hpo_miss_nowhere = 0

    for disease_name in hpo_inconsistent_diseases:
        in_orpha = disease_name in orpha
        if in_orpha:
            onset = consensus_onsets.get(disease_name, {})
            if onset:
                orpha_entry = orpha[disease_name]
                our_min = onset["aggregate_min"]
                our_max = onset["aggregate_max"]
                ref_min = orpha_entry.get("min_age", 0)
                ref_max = orpha_entry.get("max_age", 120)
                if age_ranges_overlap(our_min, our_max, ref_min, ref_max):
                    hpo_miss_in_orpha += 1
                    continue
        hpo_miss_nowhere += 1

    logger.info("\n  HPO inconsistencies (%d diseases):", len(hpo_inconsistent_diseases))
    logger.info("    %-45s  %d (%.1f%%)", "Consistent with Orphadata instead:",
                hpo_miss_in_orpha,
                hpo_miss_in_orpha / len(hpo_inconsistent_diseases) * 100 if hpo_inconsistent_diseases else 0)
    logger.info("    %-45s  %d (%.1f%%)", "Not covered by any other source:",
                hpo_miss_nowhere,
                hpo_miss_nowhere / len(hpo_inconsistent_diseases) * 100 if hpo_inconsistent_diseases else 0)

    # ================================================================
    # Save all results
    # ================================================================
    all_results["precision_recall"] = {
        "orphadata": {
            "consistency": round(ta_consistent / ta_in_orpha * 100, 2) if ta_in_orpha else 0,
            "strict_precision": round(ta_contained / ta_in_orpha * 100, 2) if ta_in_orpha else 0,
            "recall": round(orpha_covered_by_ta / orpha_total * 100, 2) if orpha_total else 0,
            "n_matched": ta_in_orpha,
            "n_consistent": ta_consistent,
            "n_contained": ta_contained,
            "n_orpha_total": orpha_total,
            "n_covered": orpha_covered_by_ta,
        },
        "hpo": {
            "consistency": round(hpo_consistent_count / hpo_in_ta * 100, 2) if hpo_in_ta else 0,
            "strict_precision": round(hpo_contained / hpo_in_ta * 100, 2) if hpo_in_ta else 0,
            "recall": round(hpo_covered / len(hpo) * 100, 2) if hpo else 0,
            "n_matched": hpo_in_ta,
            "n_consistent": hpo_consistent_count,
            "n_contained": hpo_contained,
            "n_hpo_total": len(hpo),
            "n_covered": hpo_covered,
        },
    }

    all_results["cross_source_gaps"] = {
        "orphadata_misses": {
            "total": len(orpha_inconsistent_diseases),
            "covered_by_hpo": orpha_miss_in_hpo,
            "has_hpoa": orpha_miss_in_hpoa,
            "not_covered_elsewhere": orpha_miss_nowhere,
            "diseases": orpha_inconsistent_diseases[:20],
        },
        "hpo_misses": {
            "total": len(hpo_inconsistent_diseases),
            "covered_by_orphadata": hpo_miss_in_orpha,
            "not_covered_elsewhere": hpo_miss_nowhere,
            "diseases": hpo_inconsistent_diseases[:20],
        },
    }

    all_results["definitions"] = {
        "consistency": "Age-range overlap between TA aggregate onset and gold standard onset bin. NOT precision or accuracy.",
        "strict_precision": "TA onset range fully contained within gold standard range (TA ⊆ GS).",
        "recall": "Fraction of gold standard diseases for which TA has onset data.",
        "note": "Consistency is the most appropriate metric: TA provides per-phenotype granularity while gold standards provide disease-level bins, so exact containment is often inappropriate.",
    }

    output = BENCHMARK_DIR / "deep_validation_analysis.json"
    with open(output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("\nSaved: %s", output)


if __name__ == "__main__":
    main()
