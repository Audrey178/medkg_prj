#!/usr/bin/env python3
"""
Temporal Validation: ChronoMedKG vs Orphadata + HPO
=====================================================
Recomputes disease-level and per-phenotype onset validation
against independent gold standard sources.

Two validation levels:
1. Disease-level: aggregate onset age per disease -> check overlap with Orphadata bins
2. Per-phenotype: individual phenotype onset ages -> check overlap with HPOA annotations

Usage:
    python3 scripts/compute_temporal_validation.py
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

# HPO onset term -> age range mapping (standard)
HPO_ONSET_TO_AGE: dict[str, tuple[float, float]] = {
    "HP:0030674": (0, 0),       # Antenatal onset
    "HP:0003577": (0, 0),       # Congenital onset
    "HP:0003623": (0, 0.08),    # Neonatal onset (0-28 days)
    "HP:0003593": (0.08, 2),    # Infantile onset (28d - 1yr)
    "HP:0011463": (1, 5),       # Childhood onset (1-5 yr)
    "HP:0003621": (5, 15),      # Juvenile onset (5-15 yr)
    "HP:0011462": (15, 40),     # Young adult onset
    "HP:0003584": (15, 120),    # Late onset
    "HP:0003581": (40, 120),    # Adult onset
    "HP:0025708": (60, 120),    # Middle age onset
}


def age_ranges_overlap(our_min: float, our_max: float, ref_min: float, ref_max: float) -> bool:
    """Check if two age ranges overlap."""
    return our_min <= ref_max and our_max >= ref_min


def is_more_granular(our_min: float, our_max: float, ref_min: float, ref_max: float) -> bool:
    """Check if our range is narrower (more granular) than reference."""
    our_span = our_max - our_min
    ref_span = ref_max - ref_min
    return our_span < ref_span


# ──────────────────────────────────────────────────────────────────────
# Load all disease onset data from extracted triples
# ──────────────────────────────────────────────────────────────────────

def load_disease_onsets() -> dict:
    """
    Returns:
        disease_onsets[disease_name_lower] = {
            'aggregate_min': float,  # min of all onset_age_min
            'aggregate_max': float,  # max of all onset_age_max
            'phenotype_onsets': {phenotype_name_lower: (min, max)},
            'triple_count': int,
        }
    """
    # Load disease name mapping from config
    config_dir = PROJECT_ROOT / "config" / "diseases"
    disease_name_map = {}  # MONDO_XXX -> disease_name
    if config_dir.exists():
        for yf in config_dir.glob("*.yaml"):
            try:
                import yaml
                with open(yf) as f:
                    cfg = yaml.safe_load(f)
                if cfg and "disease_name" in cfg:
                    disease_name_map[yf.stem] = cfg["disease_name"].lower().strip()
            except Exception:
                pass

    onsets = {}

    for disease_dir in EXTRACTED_DIR.iterdir():
        if not disease_dir.is_dir():
            continue
        vt = disease_dir / "validated_triples.jsonl"
        if not vt.exists():
            continue

        # Get disease name
        dir_name = disease_dir.name
        disease_name = disease_name_map.get(dir_name, None)

        all_onset_mins = []
        all_onset_maxs = []
        phenotype_onsets = {}

        with open(vt) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    triple = json.loads(line)
                except json.JSONDecodeError:
                    continue

                temporal = triple.get("temporal", {})
                onset_min = temporal.get("onset_age_min")
                onset_max = temporal.get("onset_age_max")

                if onset_min is not None:
                    all_onset_mins.append(float(onset_min))
                    if onset_max is not None:
                        all_onset_maxs.append(float(onset_max))
                    else:
                        all_onset_maxs.append(float(onset_min))

                    # Per-phenotype: use target_name for phenotypes
                    relation = (triple.get("relation") or "").lower()
                    if "phenotype" in relation or relation in ("manifests_as", "onset_at"):
                        phenotype_name = (triple.get("target_name") or "").lower().strip()
                        if phenotype_name and phenotype_name != (disease_name or ""):
                            phenotype_onsets[phenotype_name] = (
                                float(onset_min),
                                float(onset_max) if onset_max is not None else float(onset_min)
                            )

                    # Also get disease name from triple if not in config
                    if not disease_name:
                        for field in ("source_name", "target_name"):
                            if triple.get(f"{field.replace('_name', '_type')}") == "disease":
                                disease_name = (triple.get(field) or "").lower().strip()
                                break

        if disease_name and all_onset_mins:
            onsets[disease_name] = {
                "aggregate_min": min(all_onset_mins),
                "aggregate_max": max(all_onset_maxs),
                "phenotype_onsets": phenotype_onsets,
                "triple_count": len(all_onset_mins),
            }

    return onsets


# ──────────────────────────────────────────────────────────────────────
# Parse raw HPOA for per-phenotype onset annotations
# ──────────────────────────────────────────────────────────────────────

def load_hpoa_per_phenotype() -> dict:
    """
    Returns:
        hpoa[disease_name_lower][hpo_phenotype_id] = (onset_min, onset_max)
    """
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
            hpo_id = parts[3]  # phenotype HPO ID
            onset_term = parts[6].strip() if len(parts) > 6 else ""

            if onset_term and onset_term in HPO_ONSET_TO_AGE:
                onset_min, onset_max = HPO_ONSET_TO_AGE[onset_term]
                result[disease_name][hpo_id] = (onset_min, onset_max)

    return dict(result)


def main():
    logger.info("Loading disease onsets from extracted triples...")
    disease_onsets = load_disease_onsets()
    logger.info("Found onset data for %d diseases", len(disease_onsets))

    logger.info("Loading Orphadata...")
    with open(VALIDATION_DIR / "orpha_parsed.pkl", "rb") as f:
        orpha = pickle.load(f)
    logger.info("Loaded %d Orphadata diseases", len(orpha))

    logger.info("Loading HPO disease-level...")
    with open(VALIDATION_DIR / "hpo_parsed.pkl", "rb") as f:
        hpo = pickle.load(f)
    logger.info("Loaded %d HPO diseases", len(hpo))

    logger.info("Parsing HPOA per-phenotype onset annotations...")
    hpoa_per_pheno = load_hpoa_per_phenotype()
    pheno_count = sum(len(v) for v in hpoa_per_pheno.values())
    logger.info("Loaded %d per-phenotype annotations across %d diseases",
                pheno_count, len(hpoa_per_pheno))

    # ──────────────────────────────────────────────────────────────
    # Disease-level validation vs Orphadata
    # ──────────────────────────────────────────────────────────────
    logger.info("\n--- Disease-Level Validation vs Orphadata ---")

    dl_benchmarked = 0
    dl_consistent = 0
    dl_inconsistent = 0
    dl_more_granular = 0
    dl_inconsistent_examples = []

    for disease_name, onset in disease_onsets.items():
        orpha_entry = orpha.get(disease_name)
        if not orpha_entry:
            continue

        dl_benchmarked += 1
        our_min = onset["aggregate_min"]
        our_max = onset["aggregate_max"]
        ref_min = orpha_entry.get("min_age", 0)
        ref_max = orpha_entry.get("max_age", 120)

        if age_ranges_overlap(our_min, our_max, ref_min, ref_max):
            dl_consistent += 1
            if is_more_granular(our_min, our_max, ref_min, ref_max):
                dl_more_granular += 1
        else:
            dl_inconsistent += 1
            if len(dl_inconsistent_examples) < 10:
                dl_inconsistent_examples.append({
                    "disease": disease_name,
                    "ours": [our_min, our_max],
                    "orphadata": [ref_min, ref_max],
                    "categories": orpha_entry.get("categories", []),
                })

    # ──────────────────────────────────────────────────────────────
    # Disease-level validation vs HPO
    # ──────────────────────────────────────────────────────────────
    logger.info("--- Disease-Level Validation vs HPO ---")

    hpo_benchmarked = 0
    hpo_consistent = 0
    hpo_inconsistent = 0

    for disease_name, onset in disease_onsets.items():
        hpo_entry = hpo.get(disease_name)
        if not hpo_entry:
            continue

        hpo_benchmarked += 1
        our_min = onset["aggregate_min"]
        our_max = onset["aggregate_max"]
        ref_min = max(hpo_entry.get("min_age", 0), 0)  # HPO has -1 for antenatal
        ref_max = hpo_entry.get("max_age", 120)

        if age_ranges_overlap(our_min, our_max, ref_min, ref_max):
            hpo_consistent += 1
        else:
            hpo_inconsistent += 1

    # ──────────────────────────────────────────────────────────────
    # Per-phenotype validation vs HPOA
    # ──────────────────────────────────────────────────────────────
    logger.info("--- Per-Phenotype Validation vs HPOA ---")

    pp_benchmarked = 0
    pp_consistent = 0
    pp_inconsistent = 0
    pp_ambiguous = 0  # late phenotype in early-onset disease

    for disease_name, onset in disease_onsets.items():
        hpoa_entry = hpoa_per_pheno.get(disease_name)
        if not hpoa_entry:
            continue

        our_phenotypes = onset.get("phenotype_onsets", {})
        if not our_phenotypes:
            continue

        # We can't directly match HPO IDs to phenotype names easily,
        # so we compare disease-level: for each HPOA phenotype onset,
        # check if ANY of our phenotype onsets for this disease overlap
        for hpo_id, (ref_min, ref_max) in hpoa_entry.items():
            # Find our closest phenotype onset
            found_overlap = False
            for pheno_name, (our_min, our_max) in our_phenotypes.items():
                pp_benchmarked += 1
                if age_ranges_overlap(our_min, our_max, ref_min, ref_max):
                    pp_consistent += 1
                    found_overlap = True
                    break
                elif our_min > ref_max:
                    # Our onset is later than HPOA — could be a late-appearing phenotype
                    pp_ambiguous += 1
                    found_overlap = True
                    break
            if not found_overlap and our_phenotypes:
                # Compare against first available phenotype
                pp_benchmarked += 1
                pp_inconsistent += 1

    # ──────────────────────────────────────────────────────────────
    # Build results
    # ──────────────────────────────────────────────────────────────

    results = {
        "disease_level_vs_orphadata": {
            "total_benchmarked": dl_benchmarked,
            "consistent": dl_consistent,
            "inconsistent": dl_inconsistent,
            "more_granular": dl_more_granular,
            "consistency_rate": dl_consistent / dl_benchmarked if dl_benchmarked else 0,
            "granularity_advantage_rate": dl_more_granular / dl_benchmarked if dl_benchmarked else 0,
            "inconsistent_examples": dl_inconsistent_examples,
        },
        "disease_level_vs_hpo": {
            "total_benchmarked": hpo_benchmarked,
            "consistent": hpo_consistent,
            "inconsistent": hpo_inconsistent,
            "consistency_rate": hpo_consistent / hpo_benchmarked if hpo_benchmarked else 0,
        },
        "per_phenotype_vs_hpoa": {
            "total_comparisons": pp_benchmarked,
            "consistent": pp_consistent,
            "inconsistent": pp_inconsistent,
            "ambiguous": pp_ambiguous,
            "consistency_rate": pp_consistent / pp_benchmarked if pp_benchmarked else 0,
            "note": "Conservative: phenotype name-to-HPO-ID matching is approximate",
        },
        "diseases_with_onset_data": len(disease_onsets),
        "comparison_with_v1": {
            "v1_disease_consistency_vs_orphadata": 0.928,
            "v1_per_phenotype_consistency": 0.696,
            "note": "v1 numbers from session_context_20260331.md",
        },
    }

    # Save
    output_file = BENCHMARK_DIR / "temporal_validation_results_v2.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TEMPORAL VALIDATION RESULTS")
    logger.info("=" * 60)

    logger.info("\nDisease-Level vs Orphadata:")
    logger.info("  Benchmarked:        %d diseases", dl_benchmarked)
    logger.info("  Consistent:         %d (%.1f%%)", dl_consistent,
                dl_consistent / dl_benchmarked * 100 if dl_benchmarked else 0)
    logger.info("  More granular:      %d (%.1f%%)", dl_more_granular,
                dl_more_granular / dl_benchmarked * 100 if dl_benchmarked else 0)
    logger.info("  Inconsistent:       %d (%.1f%%)", dl_inconsistent,
                dl_inconsistent / dl_benchmarked * 100 if dl_benchmarked else 0)

    logger.info("\nDisease-Level vs HPO:")
    logger.info("  Benchmarked:        %d diseases", hpo_benchmarked)
    logger.info("  Consistent:         %d (%.1f%%)", hpo_consistent,
                hpo_consistent / hpo_benchmarked * 100 if hpo_benchmarked else 0)

    logger.info("\nPer-Phenotype vs HPOA:")
    logger.info("  Comparisons:        %d", pp_benchmarked)
    logger.info("  Consistent:         %d (%.1f%%)", pp_consistent,
                pp_consistent / pp_benchmarked * 100 if pp_benchmarked else 0)
    logger.info("  Ambiguous:          %d (%.1f%%)", pp_ambiguous,
                pp_ambiguous / pp_benchmarked * 100 if pp_benchmarked else 0)
    logger.info("  Inconsistent:       %d (%.1f%%)", pp_inconsistent,
                pp_inconsistent / pp_benchmarked * 100 if pp_benchmarked else 0)

    logger.info("\nPrevious (v1): disease=92.8%%, per-phenotype=69.6%%")
    logger.info("\nSaved to %s", output_file)


if __name__ == "__main__":
    main()
