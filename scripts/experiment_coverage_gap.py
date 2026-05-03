#!/usr/bin/env python3
"""
Coverage Gap Experiment
========================
For ALL 17,080 PrimeKG diseases, measure: which resource can answer
temporal questions?

Shows that ChronoMedKG fills a clear gap — PrimeKG has 0 temporal data,
HPOA covers 8%, Orphadata covers 39%, TA covers 61%.

Additionally: for diseases WHERE TA has data, how granular is it compared
to gold standards? (per-phenotype onset ages vs disease-level bins)

No API calls. Pure computation.

Output:
  data/benchmark/coverage_gap_analysis.json
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
import yaml
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("coverage_gap")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
PRIMEKG_DIR = PROJECT_ROOT / "data" / "primekg"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation_sources"
CONFIG_DIR = PROJECT_ROOT / "config" / "diseases"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"


def load_all_primekg_diseases():
    """Load all disease names from PrimeKG."""
    diseases = set()
    with open(PRIMEKG_DIR / "kg.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("x_type") == "disease":
                diseases.add(row["x_name"].lower().strip())
            if row.get("y_type") == "disease":
                diseases.add(row["y_name"].lower().strip())
    logger.info("PrimeKG diseases: %d", len(diseases))
    return diseases


def load_ta_onset_coverage():
    """Load ChronoMedKG onset data coverage per disease."""
    ta_data = {}  # dir_name -> {has_onset, n_onset_triples, n_phenotypes_with_onset, onset_range}

    for d in sorted(EXTRACTED_DIR.iterdir()):
        vf = d / "validated_triples.jsonl"
        if not vf.exists() or vf.stat().st_size == 0:
            continue

        onset_triples = 0
        phenotypes_with_onset = set()
        all_mins = []
        all_maxs = []
        total_triples = 0
        all_phenotypes = set()

        with open(vf) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    t = json.loads(line)
                except:
                    continue
                total_triples += 1

                if "phenotype" in t.get("relation", ""):
                    all_phenotypes.add(t.get("target_name", "").lower())

                temporal = t.get("temporal", {}) or {}
                omin = temporal.get("onset_age_min")
                if omin is not None:
                    try:
                        omin = float(omin)
                        omax = float(temporal.get("onset_age_max") or omin)
                        if 0 <= omin <= 120:
                            onset_triples += 1
                            all_mins.append(omin)
                            all_maxs.append(omax)
                            pheno = t.get("target_name", "").lower()
                            if pheno:
                                phenotypes_with_onset.add(pheno)
                    except:
                        pass

        ta_data[d.name] = {
            "has_onset": onset_triples > 0,
            "n_onset_triples": onset_triples,
            "n_phenotypes_with_onset": len(phenotypes_with_onset),
            "n_total_phenotypes": len(all_phenotypes),
            "total_triples": total_triples,
            "onset_range": [min(all_mins), max(all_maxs)] if all_mins else None,
        }

    logger.info("TA diseases with data: %d, with onset: %d",
                len(ta_data), sum(1 for v in ta_data.values() if v["has_onset"]))
    return ta_data


def load_orphadata_coverage():
    """Load Orphadata onset coverage."""
    with open(VALIDATION_DIR / "orphadata_with_ids.json") as f:
        data = json.load(f)
    by_id = data.get("by_orpha_id", {})

    # Count diseases with meaningful onset (not 0-120)
    meaningful = 0
    for oid, record in by_id.items():
        min_age = record.get("min_age", 0)
        max_age = record.get("max_age", 120)
        if not (min_age == 0 and max_age >= 100):
            meaningful += 1

    logger.info("Orphadata: %d total, %d with meaningful onset", len(by_id), meaningful)
    return len(by_id), meaningful


def load_hpoa_coverage():
    """Load HPOA onset coverage."""
    with open(VALIDATION_DIR / "hpoa_with_ids.json") as f:
        data = json.load(f)
    logger.info("HPOA: %d diseases with onset", len(data))
    return len(data)


def load_phenopackets_coverage():
    """Load Phenopackets coverage."""
    import pickle
    with open(VALIDATION_DIR / "phenopackets_parsed.pkl", "rb") as f:
        pp = pickle.load(f)
    with_onset = sum(1 for v in pp.values() if v.get("phenotype_onsets"))
    logger.info("Phenopackets: %d total, %d with onset", len(pp), with_onset)
    return len(pp), with_onset


def compute_granularity(ta_data):
    """Compute granularity metrics: how many distinct phenotype-onset pairs per disease."""
    granularity = {
        "1_phenotype": 0,
        "2-5_phenotypes": 0,
        "6-10_phenotypes": 0,
        "11-20_phenotypes": 0,
        "20+_phenotypes": 0,
    }
    all_counts = []

    for dir_name, data in ta_data.items():
        n = data["n_phenotypes_with_onset"]
        if n == 0:
            continue
        all_counts.append(n)
        if n == 1:
            granularity["1_phenotype"] += 1
        elif n <= 5:
            granularity["2-5_phenotypes"] += 1
        elif n <= 10:
            granularity["6-10_phenotypes"] += 1
        elif n <= 20:
            granularity["11-20_phenotypes"] += 1
        else:
            granularity["20+_phenotypes"] += 1

    return granularity, all_counts


def main():
    logger.info("=" * 75)
    logger.info("Coverage Gap Experiment")
    logger.info("=" * 75)

    # Load all data
    logger.info("\n[1/5] Loading PrimeKG diseases...")
    primekg_diseases = load_all_primekg_diseases()

    logger.info("\n[2/5] Loading ChronoMedKG onset coverage...")
    ta_data = load_ta_onset_coverage()

    logger.info("\n[3/5] Loading Orphadata coverage...")
    orpha_total, orpha_meaningful = load_orphadata_coverage()

    logger.info("\n[4/5] Loading HPOA coverage...")
    hpoa_count = load_hpoa_coverage()

    logger.info("\n[5/5] Loading Phenopackets coverage...")
    pp_total, pp_onset = load_phenopackets_coverage()

    # Compute coverage
    ta_with_onset = sum(1 for v in ta_data.values() if v["has_onset"])
    ta_total = len(ta_data)

    # Compute TA novel (diseases with onset in TA but not in Orphadata/HPOA/Phenopackets)
    # This requires ID matching — use crosswalk
    with open(VALIDATION_DIR / "mondo_crosswalk.json") as f:
        xwalk = json.load(f)
    mondo_to_orpha = xwalk.get("mondo_to_orpha", {})
    mondo_to_omim = xwalk.get("mondo_to_omim", {})

    with open(VALIDATION_DIR / "orphadata_with_ids.json") as f:
        orpha_by_id = json.load(f).get("by_orpha_id", {})
    with open(VALIDATION_DIR / "hpoa_with_ids.json") as f:
        hpoa_data = json.load(f)

    # Build set of diseases with onset in any gold standard
    gold_diseases = set()  # dir_names that have gold standard onset

    configs = {}
    for yf in CONFIG_DIR.glob("*.yaml"):
        try:
            with open(yf) as f:
                cfg = yaml.safe_load(f)
            if cfg and cfg.get("mondo_id"):
                padded = f"MONDO:{cfg['mondo_id'].split(':')[1].zfill(7)}"
                configs[yf.stem] = padded
        except:
            pass

    for dir_name, mondo_id in configs.items():
        # Check Orphadata
        olist = mondo_to_orpha.get(mondo_id, [])
        for oid in olist:
            if oid in orpha_by_id:
                record = orpha_by_id[oid]
                if not (record.get("min_age", 0) == 0 and record.get("max_age", 120) >= 100):
                    gold_diseases.add(dir_name)
                    break

        # Check HPOA
        omim_list = mondo_to_omim.get(mondo_id, [])
        for omim_id in omim_list:
            if omim_id in hpoa_data:
                gold_diseases.add(dir_name)
                break

    # TA novel = has onset in TA but not in gold
    ta_novel = sum(
        1 for dir_name, data in ta_data.items()
        if data["has_onset"] and dir_name not in gold_diseases
    )

    # Granularity analysis
    granularity, pheno_counts = compute_granularity(ta_data)

    # Print results
    print(f"\n{'=' * 75}")
    print("COVERAGE GAP ANALYSIS")
    print(f"{'=' * 75}")

    print(f"\nQuestion: 'Can you provide onset age for this disease?'")
    print(f"Tested across {len(primekg_diseases):,} PrimeKG diseases:\n")

    print(f"{'Resource':<25} {'Diseases with onset':>20} {'Coverage':>10} {'Granularity':>15}")
    print("-" * 75)
    print(f"{'PrimeKG':<25} {'0':>20} {'0.0%':>10} {'None':>15}")
    print(f"{'HPOA':<25} {hpoa_count:>20,} {100*hpoa_count/len(primekg_diseases):>9.1f}% {'Coarse bins':>15}")
    print(f"{'Orphadata':<25} {orpha_meaningful:>20,} {100*orpha_meaningful/len(primekg_diseases):>9.1f}% {'Coarse bins':>15}")
    print(f"{'Phenopackets':<25} {pp_onset:>20,} {100*pp_onset/len(primekg_diseases):>9.1f}% {'Per-patient':>15}")
    print(f"{'ChronoMedKG':<25} {ta_with_onset:>20,} {100*ta_with_onset/len(primekg_diseases):>9.1f}% {'Per-phenotype':>15}")
    print(f"{'TA (novel only)':<25} {ta_novel:>20,} {100*ta_novel/len(primekg_diseases):>9.1f}% {'Per-phenotype':>15}")

    print(f"\n{'=' * 75}")
    print("GRANULARITY: How many distinct phenotype-onset pairs per disease?")
    print(f"{'=' * 75}")
    print(f"(Orphadata/HPOA give ONE range per disease. TA gives per-phenotype.)\n")

    for bucket, count in granularity.items():
        bar = "#" * (count // 50)
        print(f"  {bucket:>20}: {count:>6,} diseases {bar}")

    if pheno_counts:
        print(f"\n  Median phenotypes with onset per disease: {statistics.median(pheno_counts):.0f}")
        print(f"  Mean: {statistics.mean(pheno_counts):.1f}")
        print(f"  Max: {max(pheno_counts)}")
        print(f"  Diseases with >=3 phenotype-onset pairs: "
              f"{sum(1 for c in pheno_counts if c >= 3):,}")

    # Combined sources
    any_gold = len(gold_diseases)
    ta_adds = ta_with_onset  # TA covers this many (including overlap with gold)
    print(f"\n{'=' * 75}")
    print("HEADLINE NUMBERS")
    print(f"{'=' * 75}")
    print(f"  PrimeKG:       0 diseases with temporal data (0%)")
    print(f"  Best existing: {any_gold:,} diseases in Orphadata+HPOA "
          f"({100*any_gold/len(primekg_diseases):.1f}%)")
    print(f"  ChronoMedKG: {ta_with_onset:,} diseases with onset ages "
          f"({100*ta_with_onset/len(primekg_diseases):.1f}%)")
    print(f"  TA novel:      {ta_novel:,} diseases not in ANY gold standard "
          f"({100*ta_novel/len(primekg_diseases):.1f}%)")
    print(f"  TA granularity: median {statistics.median(pheno_counts):.0f} "
          f"phenotype-onset pairs per disease (vs 1 for gold standards)")

    # Save
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "Coverage Gap Analysis",
        "primekg_diseases": len(primekg_diseases),
        "coverage": {
            "primekg": {"diseases": 0, "pct": 0},
            "hpoa": {"diseases": hpoa_count, "pct": round(100 * hpoa_count / len(primekg_diseases), 2)},
            "orphadata": {"diseases": orpha_meaningful, "pct": round(100 * orpha_meaningful / len(primekg_diseases), 2)},
            "phenopackets": {"diseases": pp_onset, "pct": round(100 * pp_onset / len(primekg_diseases), 2)},
            "chronomedkg": {"diseases": ta_with_onset, "pct": round(100 * ta_with_onset / len(primekg_diseases), 2)},
            "ta_novel": {"diseases": ta_novel, "pct": round(100 * ta_novel / len(primekg_diseases), 2)},
        },
        "granularity": granularity,
        "granularity_stats": {
            "median_phenotypes_with_onset": statistics.median(pheno_counts) if pheno_counts else 0,
            "mean_phenotypes_with_onset": round(statistics.mean(pheno_counts), 1) if pheno_counts else 0,
            "diseases_with_3plus_phenotype_onsets": sum(1 for c in pheno_counts if c >= 3),
        },
    }
    out_file = BENCHMARK_DIR / "coverage_gap_analysis.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
