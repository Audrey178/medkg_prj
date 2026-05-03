#!/usr/bin/env python3
"""
Experiment 2: Temporal Diagnostic Simulation
=============================================
Demonstrates that ChronoMedKG enables temporal differential diagnosis
that static KGs cannot support.

Setup:
  For each Phenopackets disease with multiple phenotype onsets:
  1. Get known phenotype-onset pairs from Phenopackets (independent ground truth)
  2. Find distractor diseases from PrimeKG that share phenotypes
  3. At each age point (0, 2, 5, 10, 15, 20, 30, 50), score candidates:
     a. PrimeKG score: binary phenotype match (does disease have this phenotype?)
     b. ChronoMedKG score: temporal match (does disease have this phenotype AT THIS AGE?)
  4. Measure: how often does the correct diagnosis rank #1?

Ground truth: Phenopackets patient cases (independent of TA extraction).
Key finding: TA-based scoring should rank correct diagnosis higher, especially
at early time points where temporal specificity matters most.

No LLM API calls — fully computational, reproducible.

Output:
  data/benchmark/temporal_diagnosis_simulation.json

Usage:
  .venv-sapbert/bin/python scripts/experiment_temporal_diagnosis.py
"""

from __future__ import annotations

import csv
import json
import logging
import pickle
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("temporal_dx")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
PRIMEKG_DIR = PROJECT_ROOT / "data" / "primekg"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation_sources"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"

AGE_POINTS = [0, 1, 2, 5, 10, 15, 20, 30, 50]


def load_phenopackets():
    """Load Phenopackets data with phenotype-onset information."""
    pp_file = VALIDATION_DIR / "phenopackets_parsed.pkl"
    with open(pp_file, "rb") as f:
        pp = pickle.load(f)

    # Filter to diseases with >=3 phenotype onsets (enough for meaningful simulation)
    rich = {}
    for disease_name, data in pp.items():
        po = data.get("phenotype_onsets", {})
        if len(po) >= 3:
            rich[disease_name.lower().strip()] = {
                "disease_name": disease_name,
                "disease_ids": data.get("disease_ids", []),
                "cases": data.get("cases", 0),
                "phenotype_onsets": {
                    pheno: ages for pheno, ages in po.items()
                },
            }

    logger.info("Phenopackets: %d diseases with >=3 phenotype onsets", len(rich))
    return rich


def load_primekg_phenotype_index():
    """Load PrimeKG disease-phenotype edges indexed by disease name."""
    kg_file = PRIMEKG_DIR / "kg.csv"
    # disease_name_lower -> set of phenotype_name_lower
    index = defaultdict(set)

    with open(kg_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = row.get("display_relation", "")
            if "phenotype" not in rel:
                continue

            x_name = row.get("x_name", "").lower().strip()
            y_name = row.get("y_name", "").lower().strip()
            x_type = row.get("x_type", "")
            y_type = row.get("y_type", "")

            if x_type == "disease" and y_type == "effect/phenotype":
                index[x_name].add(y_name)
            elif y_type == "disease" and x_type == "effect/phenotype":
                index[y_name].add(x_name)

    logger.info("PrimeKG phenotype index: %d diseases", len(index))
    return dict(index)


def load_ta_temporal_index():
    """Load ChronoMedKG disease -> (phenotype, onset_min, onset_max) index."""
    index = defaultdict(list)  # disease_name_lower -> [(phenotype, onset_min, onset_max)]

    for d in sorted(EXTRACTED_DIR.iterdir()):
        vf = d / "validated_triples.jsonl"
        if not vf.exists():
            continue

        with open(vf) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    t = json.loads(line)
                except Exception:
                    continue

                # Only disease-phenotype edges with onset data
                if t.get("source_type") != "disease":
                    continue
                if "phenotype" not in t.get("relation", ""):
                    continue

                temporal = t.get("temporal", {}) or {}
                omin = temporal.get("onset_age_min")
                if omin is None:
                    continue

                try:
                    omin = float(omin)
                    omax = float(temporal.get("onset_age_max") or omin)
                except (ValueError, TypeError):
                    continue

                if omin > 120 or omax > 120:
                    continue

                disease_name = t.get("source_name", "").lower().strip()
                phenotype = t.get("target_name", "").lower().strip()
                if disease_name and phenotype:
                    index[disease_name].append((phenotype, omin, omax))

    logger.info("TA temporal index: %d diseases with phenotype onset data", len(index))
    return dict(index)


def find_distractors(target_disease, target_phenotypes, primekg_index, n_distractors=4):
    """Find diseases that share phenotypes with the target (plausible misdiagnoses)."""
    target_phenos = set(p.lower() for p in target_phenotypes)
    candidates = []

    for disease, phenos in primekg_index.items():
        if disease == target_disease.lower():
            continue
        overlap = target_phenos & phenos
        if len(overlap) >= 2:  # At least 2 shared phenotypes
            candidates.append({
                "disease": disease,
                "shared_phenotypes": len(overlap),
                "total_phenotypes": len(phenos),
            })

    # Sort by overlap (most confusing distractors first)
    candidates.sort(key=lambda x: (-x["shared_phenotypes"], x["total_phenotypes"]))
    return candidates[:n_distractors]


def _fuzzy_lookup_primekg(disease_name, primekg_index):
    """Fuzzy disease name lookup in PrimeKG index."""
    lower = disease_name.lower().strip()
    if lower in primekg_index:
        return primekg_index[lower]

    # Try substring match (e.g., "retinitis pigmentosa 19" in "retinitis pigmentosa")
    for pk_name, phenos in primekg_index.items():
        if lower in pk_name or pk_name in lower:
            return phenos

    # Try word overlap (at least 60% of words match)
    target_words = set(lower.split())
    if len(target_words) < 2:
        return set()
    for pk_name, phenos in primekg_index.items():
        pk_words = set(pk_name.split())
        overlap = target_words & pk_words
        if len(overlap) >= max(2, 0.6 * len(target_words)):
            return phenos

    return set()


def score_primekg(candidate_disease, presented_phenotypes, primekg_index):
    """Score a candidate disease using PrimeKG (static, no temporal info).
    Score = fraction of presented phenotypes that this disease has in PrimeKG."""
    disease_phenos = _fuzzy_lookup_primekg(candidate_disease, primekg_index)
    if not presented_phenotypes:
        return 0.0

    matches = 0
    for pheno in presented_phenotypes:
        pheno_lower = pheno.lower()
        # Fuzzy match: check if any PrimeKG phenotype contains this term or vice versa
        if pheno_lower in disease_phenos:
            matches += 1
        elif any(pheno_lower in dp or dp in pheno_lower for dp in disease_phenos):
            matches += 0.5

    return matches / len(presented_phenotypes)


def score_chronomedkg(candidate_disease, presented_phenotypes, patient_age, ta_index):
    """Score a candidate using ChronoMedKG (temporal match at specific age).
    Score = fraction of presented phenotypes that this disease has AT THIS AGE."""
    ta_entries = ta_index.get(candidate_disease.lower(), [])
    if not presented_phenotypes or not ta_entries:
        return 0.0

    matches = 0
    for pheno in presented_phenotypes:
        pheno_lower = pheno.lower()
        best_match = 0

        for ta_pheno, omin, omax in ta_entries:
            # Check phenotype match (fuzzy)
            pheno_match = False
            if pheno_lower == ta_pheno:
                pheno_match = True
            elif pheno_lower in ta_pheno or ta_pheno in pheno_lower:
                pheno_match = True
            elif any(w in ta_pheno for w in pheno_lower.split() if len(w) > 4):
                pheno_match = True

            if pheno_match:
                # Temporal match: is patient_age within this phenotype's onset window?
                # Give partial credit for being near the window
                if omin <= patient_age <= omax:
                    best_match = max(best_match, 1.0)
                else:
                    # Distance penalty: how far is patient_age from the window?
                    dist = min(abs(patient_age - omin), abs(patient_age - omax))
                    if dist <= 5:
                        best_match = max(best_match, 0.5)
                    elif dist <= 10:
                        best_match = max(best_match, 0.25)

        matches += best_match

    return matches / len(presented_phenotypes)


def simulate_diagnostic_journey(pp_disease, distractors, primekg_index, ta_index):
    """Simulate a diagnostic journey for one Phenopackets disease.

    At each age point, determine which phenotypes have appeared,
    then rank candidates using PrimeKG vs ChronoMedKG scoring.
    """
    disease_name = pp_disease["disease_name"]
    phenotype_onsets = pp_disease["phenotype_onsets"]

    # All candidate diseases: correct + distractors (RANDOMIZED to avoid tie-break bias)
    import random
    candidates = [disease_name.lower()] + [d["disease"] for d in distractors]
    random.seed(hash(disease_name))  # Deterministic per disease but shuffled
    random.shuffle(candidates)

    results_by_age = {}

    for age in AGE_POINTS:
        # Determine which phenotypes would be present at this age
        # (phenotype appears if any case reported onset <= this age)
        presented = []
        for pheno, age_pairs in phenotype_onsets.items():
            for pair in age_pairs:
                onset_age = pair[0] if isinstance(pair, (list, tuple)) else pair
                try:
                    if float(onset_age) <= age:
                        presented.append(pheno)
                        break
                except (ValueError, TypeError):
                    pass

        if not presented:
            continue  # No phenotypes at this age yet

        # Score each candidate
        pk_scores = {}
        ta_scores = {}
        for cand in candidates:
            pk_scores[cand] = score_primekg(cand, presented, primekg_index)
            ta_scores[cand] = score_chronomedkg(cand, presented, age, ta_index)

        # Rank
        pk_ranked = sorted(pk_scores.items(), key=lambda x: -x[1])
        ta_ranked = sorted(ta_scores.items(), key=lambda x: -x[1])

        correct = disease_name.lower()
        pk_rank = next((i for i, (d, _) in enumerate(pk_ranked) if d == correct), len(pk_ranked))
        ta_rank = next((i for i, (d, _) in enumerate(ta_ranked) if d == correct), len(ta_ranked))

        results_by_age[age] = {
            "n_phenotypes_presented": len(presented),
            "phenotypes": presented[:5],  # Truncate for output
            "primekg_rank": pk_rank + 1,  # 1-indexed
            "ta_rank": ta_rank + 1,
            "primekg_correct_at_1": pk_rank == 0,
            "ta_correct_at_1": ta_rank == 0,
            "primekg_top_score": pk_ranked[0][1] if pk_ranked else 0,
            "ta_top_score": ta_ranked[0][1] if ta_ranked else 0,
            "n_candidates": len(candidates),
        }

    return results_by_age


def main():
    logger.info("=" * 75)
    logger.info("Experiment 2: Temporal Diagnostic Simulation")
    logger.info("=" * 75)

    # Load data
    logger.info("\n[1/4] Loading Phenopackets (ground truth)...")
    pp = load_phenopackets()

    logger.info("\n[2/4] Loading PrimeKG phenotype index...")
    primekg_index = load_primekg_phenotype_index()

    logger.info("\n[3/4] Loading ChronoMedKG temporal index...")
    ta_index = load_ta_temporal_index()

    # Run simulations
    logger.info("\n[4/4] Running diagnostic simulations...")

    all_results = []
    diseases_simulated = 0
    diseases_skipped = 0

    for disease_lower, pp_data in pp.items():
        # Find distractors
        phenotype_names = list(pp_data["phenotype_onsets"].keys())
        distractors = find_distractors(disease_lower, phenotype_names, primekg_index)

        if len(distractors) < 2:
            diseases_skipped += 1
            continue

        # Check if disease exists in TA index
        ta_has = disease_lower in ta_index

        # Run simulation
        journey = simulate_diagnostic_journey(pp_data, distractors, primekg_index, ta_index)

        if not journey:
            diseases_skipped += 1
            continue

        all_results.append({
            "disease": pp_data["disease_name"],
            "n_distractors": len(distractors),
            "ta_coverage": ta_has,
            "journey": journey,
        })
        diseases_simulated += 1

    logger.info("Simulated: %d diseases, Skipped: %d", diseases_simulated, diseases_skipped)

    # Aggregate results
    pk_correct = defaultdict(list)  # age -> [0/1]
    ta_correct = defaultdict(list)
    pk_correct_ta_covered = defaultdict(list)
    ta_correct_ta_covered = defaultdict(list)

    for result in all_results:
        for age, data in result["journey"].items():
            pk_correct[age].append(1 if data["primekg_correct_at_1"] else 0)
            ta_correct[age].append(1 if data["ta_correct_at_1"] else 0)
            if result["ta_coverage"]:
                pk_correct_ta_covered[age].append(1 if data["primekg_correct_at_1"] else 0)
                ta_correct_ta_covered[age].append(1 if data["ta_correct_at_1"] else 0)

    # Print results
    print(f"\n{'=' * 75}")
    print("TEMPORAL DIAGNOSTIC SIMULATION RESULTS")
    print(f"{'=' * 75}")
    print(f"Diseases simulated: {diseases_simulated}")
    print(f"Diseases skipped (insufficient distractors): {diseases_skipped}")
    print(f"TA coverage: {sum(1 for r in all_results if r['ta_coverage'])}/{diseases_simulated}")

    print(f"\nRank-1 Accuracy by Patient Age (all diseases):")
    print(f"{'Age':>5} | {'PrimeKG':>10} | {'ChronoMedKG':>14} | {'Gain':>8} | {'N':>5}")
    print("-" * 55)

    for age in AGE_POINTS:
        if age not in pk_correct:
            continue
        pk_acc = 100 * sum(pk_correct[age]) / max(1, len(pk_correct[age]))
        ta_acc = 100 * sum(ta_correct[age]) / max(1, len(ta_correct[age]))
        gain = ta_acc - pk_acc
        n = len(pk_correct[age])
        marker = " ***" if gain > 5 else ""
        print(f"{age:>5} | {pk_acc:>9.1f}% | {ta_acc:>13.1f}% | {gain:>+7.1f}pp | {n:>5}{marker}")

    # TA-covered subset
    ta_covered = sum(1 for r in all_results if r["ta_coverage"])
    if ta_covered > 10:
        print(f"\nRank-1 Accuracy — TA-covered diseases only (n={ta_covered}):")
        print(f"{'Age':>5} | {'PrimeKG':>10} | {'ChronoMedKG':>14} | {'Gain':>8} | {'N':>5}")
        print("-" * 55)

        for age in AGE_POINTS:
            if age not in pk_correct_ta_covered:
                continue
            pk_acc = 100 * sum(pk_correct_ta_covered[age]) / max(1, len(pk_correct_ta_covered[age]))
            ta_acc = 100 * sum(ta_correct_ta_covered[age]) / max(1, len(ta_correct_ta_covered[age]))
            gain = ta_acc - pk_acc
            n = len(pk_correct_ta_covered[age])
            print(f"{age:>5} | {pk_acc:>9.1f}% | {ta_acc:>13.1f}% | {gain:>+7.1f}pp | {n:>5}")

    # Overall
    all_pk = [v for vals in pk_correct.values() for v in vals]
    all_ta = [v for vals in ta_correct.values() for v in vals]
    pk_overall = 100 * sum(all_pk) / max(1, len(all_pk))
    ta_overall = 100 * sum(all_ta) / max(1, len(all_ta))

    print(f"\n{'=' * 75}")
    print("HEADLINE NUMBERS")
    print(f"{'=' * 75}")
    print(f"  Overall rank-1 accuracy: PrimeKG {pk_overall:.1f}% vs ChronoMedKG {ta_overall:.1f}% "
          f"({ta_overall - pk_overall:+.1f}pp)")
    print(f"  Diseases simulated: {diseases_simulated}")
    print(f"  Ground truth: GA4GH Phenopackets (independent patient cases)")

    # Save
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "Temporal Diagnostic Simulation",
        "ground_truth": "GA4GH Phenopackets (independent patient cases)",
        "diseases_simulated": diseases_simulated,
        "diseases_skipped": diseases_skipped,
        "ta_coverage": sum(1 for r in all_results if r["ta_coverage"]),
        "age_points": AGE_POINTS,
        "accuracy_by_age": {
            age: {
                "primekg": round(100 * sum(pk_correct[age]) / max(1, len(pk_correct[age])), 2),
                "chronomedkg": round(100 * sum(ta_correct[age]) / max(1, len(ta_correct[age])), 2),
                "n": len(pk_correct[age]),
            }
            for age in AGE_POINTS if age in pk_correct
        },
        "overall_accuracy": {
            "primekg": round(pk_overall, 2),
            "chronomedkg": round(ta_overall, 2),
        },
        "per_disease_results": all_results[:50],  # Save first 50 for inspection
    }

    out_file = BENCHMARK_DIR / "temporal_diagnosis_simulation.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
