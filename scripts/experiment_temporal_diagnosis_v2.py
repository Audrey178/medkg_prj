#!/usr/bin/env python3
"""
Experiment 2 v2: Temporal Diagnostic Simulation (Fair Methodology)
===================================================================
Demonstrates that TEMPORAL information specifically (not just coverage)
improves differential diagnosis.

Key methodology fixes from v1:
  1. OVERLAP SET ONLY: diseases where BOTH PrimeKG AND TA have phenotype data
  2. TEMPORAL ABLATION: 3 conditions that isolate temporal contribution:
     - PrimeKG: static phenotype match (binary)
     - TA-static: phenotype match from TA data (binary, no temporal)
     - TA-temporal: phenotype match WITH onset window scoring
  3. Condition 2 vs 3 uses SAME data, ONLY difference is temporal weighting
  4. Random baseline included for calibration
  5. Candidate order randomized (no tie-break bias)
  6. Distractors selected from the OVERLAP set (fair for all conditions)

Ground truth: GA4GH Phenopackets patient cases (independent).

Output:
  data/benchmark/temporal_diagnosis_v2.json

Usage:
  .venv-sapbert/bin/python scripts/experiment_temporal_diagnosis_v2.py
"""

from __future__ import annotations

import csv
import json
import logging
import pickle
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("temporal_dx_v2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
PRIMEKG_DIR = PROJECT_ROOT / "data" / "primekg"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation_sources"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"

AGE_POINTS = [0, 1, 2, 5, 10, 15, 20, 30, 50]


def load_phenopackets():
    """Load Phenopackets diseases with >=3 phenotype onsets."""
    with open(VALIDATION_DIR / "phenopackets_parsed.pkl", "rb") as f:
        pp = pickle.load(f)

    rich = {}
    for name, data in pp.items():
        po = data.get("phenotype_onsets", {})
        if len(po) >= 3:
            rich[name.lower().strip()] = {
                "disease_name": name,
                "disease_ids": data.get("disease_ids", []),
                "cases": data.get("cases", 0),
                "phenotype_onsets": po,
            }
    logger.info("Phenopackets: %d diseases with >=3 phenotype onsets", len(rich))
    return rich


def load_primekg_phenotype_index():
    """Load PrimeKG disease -> set of phenotype names."""
    index = defaultdict(set)
    with open(PRIMEKG_DIR / "kg.csv") as f:
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


def load_ta_phenotype_index():
    """Load TA disease -> [(phenotype, onset_min, onset_max)] for temporal scoring,
    AND disease -> set of phenotype names for binary scoring."""
    temporal_index = defaultdict(list)
    binary_index = defaultdict(set)

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
                if t.get("source_type") != "disease":
                    continue
                if "phenotype" not in t.get("relation", ""):
                    continue

                disease = t.get("source_name", "").lower().strip()
                phenotype = t.get("target_name", "").lower().strip()
                if not disease or not phenotype:
                    continue

                binary_index[disease].add(phenotype)

                temporal = t.get("temporal", {}) or {}
                omin = temporal.get("onset_age_min")
                if omin is not None:
                    try:
                        omin = float(omin)
                        omax = float(temporal.get("onset_age_max") or omin)
                        if 0 <= omin <= 120 and 0 <= omax <= 120:
                            temporal_index[disease].append((phenotype, omin, omax))
                    except (ValueError, TypeError):
                        pass

    logger.info("TA binary index: %d diseases, temporal index: %d diseases",
                len(binary_index), len(temporal_index))
    return dict(temporal_index), dict(binary_index)


def find_overlap_diseases(phenopackets, primekg_index, ta_binary_index):
    """Find diseases in ALL THREE sources (fair evaluation set)."""
    overlap = []
    for pp_name, pp_data in phenopackets.items():
        # Check PrimeKG (fuzzy name match)
        pk_match = None
        if pp_name in primekg_index:
            pk_match = pp_name
        else:
            for pk_name in primekg_index:
                if pp_name in pk_name or pk_name in pp_name:
                    pk_match = pk_name
                    break

        # Check TA (fuzzy name match)
        ta_match = None
        if pp_name in ta_binary_index:
            ta_match = pp_name
        else:
            for ta_name in ta_binary_index:
                if pp_name in ta_name or ta_name in pp_name:
                    ta_match = ta_name
                    break

        if pk_match and ta_match:
            overlap.append({
                "pp_name": pp_name,
                "pk_name": pk_match,
                "ta_name": ta_match,
                "pp_data": pp_data,
            })

    logger.info("Overlap diseases (in PP + PK + TA): %d", len(overlap))
    return overlap


def phenotype_match(query_pheno, candidate_phenos):
    """Check if a phenotype matches any in a candidate set. Returns match score."""
    q = query_pheno.lower()
    if q in candidate_phenos:
        return 1.0
    # Substring match
    for cp in candidate_phenos:
        if q in cp or cp in q:
            return 0.7
    # Word overlap (at least one significant word)
    q_words = set(w for w in q.split() if len(w) > 4)
    for cp in candidate_phenos:
        cp_words = set(w for w in cp.split() if len(w) > 4)
        if q_words & cp_words:
            return 0.5
    return 0.0


def score_static(candidate_phenos, presented_phenotypes):
    """Binary phenotype match score (used for PrimeKG and TA-static)."""
    if not presented_phenotypes:
        return 0.0
    total = sum(phenotype_match(p, candidate_phenos) for p in presented_phenotypes)
    return total / len(presented_phenotypes)


def score_temporal(candidate_temporal, presented_phenotypes, patient_age):
    """Temporal phenotype match: does the candidate have these phenotypes AT THIS AGE?"""
    if not presented_phenotypes or not candidate_temporal:
        return 0.0

    # Build phenotype -> list of (onset_min, onset_max) for this candidate
    pheno_windows = defaultdict(list)
    for pheno, omin, omax in candidate_temporal:
        pheno_windows[pheno].append((omin, omax))

    total_score = 0.0
    for query_pheno in presented_phenotypes:
        q = query_pheno.lower()
        best = 0.0

        for cand_pheno, windows in pheno_windows.items():
            # Check phenotype name match
            name_score = 0.0
            if q == cand_pheno:
                name_score = 1.0
            elif q in cand_pheno or cand_pheno in q:
                name_score = 0.7
            else:
                q_words = set(w for w in q.split() if len(w) > 4)
                cp_words = set(w for w in cand_pheno.split() if len(w) > 4)
                if q_words & cp_words:
                    name_score = 0.5

            if name_score == 0:
                continue

            # Temporal match: is patient_age within any onset window?
            for omin, omax in windows:
                if omin <= patient_age <= omax:
                    temporal_score = 1.0
                else:
                    dist = min(abs(patient_age - omin), abs(patient_age - omax))
                    if dist <= 3:
                        temporal_score = 0.5
                    elif dist <= 8:
                        temporal_score = 0.25
                    else:
                        temporal_score = 0.1

                combined = name_score * temporal_score
                best = max(best, combined)

        total_score += best

    return total_score / len(presented_phenotypes)


def find_distractors_from_overlap(target_pp_name, target_phenotypes, overlap_diseases, n=4):
    """Find distractor diseases from the overlap set (fair for all conditions)."""
    target_phenos = set(p.lower() for p in target_phenotypes)
    candidates = []

    for disease in overlap_diseases:
        if disease["pp_name"] == target_pp_name:
            continue
        other_phenos = set(p.lower() for p in disease["pp_data"]["phenotype_onsets"].keys())
        # Count shared phenotype names (fuzzy)
        shared = 0
        for tp in target_phenos:
            if any(tp in op or op in tp for op in other_phenos):
                shared += 1
        if shared >= 1:
            candidates.append({"disease": disease, "shared": shared})

    candidates.sort(key=lambda x: -x["shared"])
    return [c["disease"] for c in candidates[:n]]


def simulate_one(target, distractors, primekg_index, ta_temporal_index, ta_binary_index):
    """Run simulation for one disease across all age points."""
    pp_data = target["pp_data"]
    disease_name = pp_data["disease_name"]
    phenotype_onsets = pp_data["phenotype_onsets"]

    # All candidates (randomized to avoid tie-break bias)
    all_candidates = [target] + distractors
    random.seed(hash(disease_name))
    random.shuffle(all_candidates)

    results = {}

    for age in AGE_POINTS:
        # Phenotypes present at this age
        presented = []
        for pheno, age_pairs in phenotype_onsets.items():
            for pair in age_pairs:
                onset = pair[0] if isinstance(pair, (list, tuple)) else pair
                try:
                    if float(onset) <= age:
                        presented.append(pheno)
                        break
                except (ValueError, TypeError):
                    pass

        if not presented:
            continue

        # Score each candidate under 3 conditions + random
        scores = {"primekg": {}, "ta_static": {}, "ta_temporal": {}}

        for cand in all_candidates:
            cand_name = cand["pp_name"]
            pk_name = cand["pk_name"]
            ta_name = cand["ta_name"]

            # PrimeKG: static phenotype match
            pk_phenos = primekg_index.get(pk_name, set())
            scores["primekg"][cand_name] = score_static(pk_phenos, presented)

            # TA-static: binary phenotype match from TA (same matching, no temporal)
            ta_phenos = ta_binary_index.get(ta_name, set())
            scores["ta_static"][cand_name] = score_static(ta_phenos, presented)

            # TA-temporal: phenotype match WITH temporal weighting
            ta_temp = ta_temporal_index.get(ta_name, [])
            scores["ta_temporal"][cand_name] = score_temporal(ta_temp, presented, age)

        correct = target["pp_name"]
        age_result = {"n_phenotypes": len(presented), "n_candidates": len(all_candidates)}

        for condition in ["primekg", "ta_static", "ta_temporal"]:
            ranked = sorted(scores[condition].items(), key=lambda x: -x[1])
            rank = next((i for i, (d, _) in enumerate(ranked) if d == correct), len(ranked))
            age_result[f"{condition}_rank"] = rank + 1
            age_result[f"{condition}_correct_at_1"] = rank == 0
            age_result[f"{condition}_score"] = scores[condition].get(correct, 0)

        results[age] = age_result

    return results


def main():
    logger.info("=" * 75)
    logger.info("Experiment 2 v2: Temporal Diagnostic Simulation (Fair Methodology)")
    logger.info("=" * 75)

    logger.info("\n[1/5] Loading Phenopackets...")
    pp = load_phenopackets()

    logger.info("\n[2/5] Loading PrimeKG phenotype index...")
    pk_index = load_primekg_phenotype_index()

    logger.info("\n[3/5] Loading TA phenotype indices (binary + temporal)...")
    ta_temporal, ta_binary = load_ta_phenotype_index()

    logger.info("\n[4/5] Finding overlap diseases (in PP + PK + TA)...")
    overlap = find_overlap_diseases(pp, pk_index, ta_binary)

    if len(overlap) < 10:
        logger.error("Too few overlap diseases (%d). Cannot run experiment.", len(overlap))
        return

    logger.info("\n[5/5] Running simulations on %d overlap diseases...", len(overlap))

    all_results = []
    skipped = 0

    for target in overlap:
        distractors = find_distractors_from_overlap(
            target["pp_name"],
            list(target["pp_data"]["phenotype_onsets"].keys()),
            overlap,
            n=4,
        )
        if len(distractors) < 2:
            skipped += 1
            continue

        journey = simulate_one(target, distractors, pk_index, ta_temporal, ta_binary)
        if not journey:
            skipped += 1
            continue

        all_results.append({
            "disease": target["pp_data"]["disease_name"],
            "n_distractors": len(distractors),
            "n_candidates": len(distractors) + 1,
            "journey": journey,
        })

    logger.info("Simulated: %d, Skipped: %d", len(all_results), skipped)

    # Aggregate
    conditions = ["primekg", "ta_static", "ta_temporal"]
    acc_by_age = {c: defaultdict(list) for c in conditions}

    for result in all_results:
        for age, data in result["journey"].items():
            for cond in conditions:
                acc_by_age[cond][age].append(1 if data[f"{cond}_correct_at_1"] else 0)

    # Print results
    n_cands = all_results[0]["n_candidates"] if all_results else 5
    random_baseline = 100.0 / n_cands

    print(f"\n{'=' * 85}")
    print("TEMPORAL DIAGNOSTIC SIMULATION v2 — FAIR METHODOLOGY")
    print(f"{'=' * 85}")
    print(f"Diseases simulated: {len(all_results)} (overlap: in Phenopackets + PrimeKG + TA)")
    print(f"Candidates per sim: {n_cands} (1 correct + {n_cands-1} distractors)")
    print(f"Random baseline: {random_baseline:.1f}%")
    print(f"Ground truth: GA4GH Phenopackets (independent patient cases)")

    print(f"\n{'Age':>5} | {'PrimeKG':>10} | {'TA-static':>10} | {'TA-temporal':>12} | "
          f"{'Static gain':>12} | {'Temporal gain':>13} | {'N':>5}")
    print("-" * 85)

    for age in AGE_POINTS:
        if age not in acc_by_age["primekg"]:
            continue
        n = len(acc_by_age["primekg"][age])
        pk = 100 * sum(acc_by_age["primekg"][age]) / max(1, n)
        ts = 100 * sum(acc_by_age["ta_static"][age]) / max(1, n)
        tt = 100 * sum(acc_by_age["ta_temporal"][age]) / max(1, n)
        static_gain = ts - pk
        temporal_gain = tt - ts  # THIS is the key: temporal vs static from SAME source
        print(f"{age:>5} | {pk:>9.1f}% | {ts:>9.1f}% | {tt:>11.1f}% | "
              f"{static_gain:>+11.1f}pp | {temporal_gain:>+12.1f}pp | {n:>5}")

    # Overall
    all_pk = [v for vals in acc_by_age["primekg"].values() for v in vals]
    all_ts = [v for vals in acc_by_age["ta_static"].values() for v in vals]
    all_tt = [v for vals in acc_by_age["ta_temporal"].values() for v in vals]

    pk_overall = 100 * sum(all_pk) / max(1, len(all_pk))
    ts_overall = 100 * sum(all_ts) / max(1, len(all_ts))
    tt_overall = 100 * sum(all_tt) / max(1, len(all_tt))

    print(f"\n{'=' * 85}")
    print("HEADLINE (Overall)")
    print(f"{'=' * 85}")
    print(f"  Random baseline:   {random_baseline:.1f}%")
    print(f"  PrimeKG (static):  {pk_overall:.1f}%")
    print(f"  TA-static:         {ts_overall:.1f}%  (coverage gain: {ts_overall - pk_overall:+.1f}pp)")
    print(f"  TA-temporal:       {tt_overall:.1f}%  (temporal gain: {tt_overall - ts_overall:+.1f}pp)")
    print(f"")
    print(f"  Total TA advantage: {tt_overall - pk_overall:+.1f}pp")
    print(f"    = {ts_overall - pk_overall:+.1f}pp from better coverage")
    print(f"    + {tt_overall - ts_overall:+.1f}pp from temporal matching")

    # Save
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "Temporal Diagnostic Simulation v2 (Fair)",
        "methodology": {
            "overlap_only": True,
            "temporal_ablation": True,
            "randomized_candidates": True,
            "conditions": ["primekg", "ta_static", "ta_temporal"],
        },
        "diseases_simulated": len(all_results),
        "candidates_per_sim": n_cands,
        "random_baseline": round(random_baseline, 1),
        "accuracy_by_age": {
            age: {
                "primekg": round(100 * sum(acc_by_age["primekg"][age]) / max(1, len(acc_by_age["primekg"][age])), 2),
                "ta_static": round(100 * sum(acc_by_age["ta_static"][age]) / max(1, len(acc_by_age["ta_static"][age])), 2),
                "ta_temporal": round(100 * sum(acc_by_age["ta_temporal"][age]) / max(1, len(acc_by_age["ta_temporal"][age])), 2),
                "n": len(acc_by_age["primekg"][age]),
            }
            for age in AGE_POINTS if age in acc_by_age["primekg"]
        },
        "overall": {
            "random": round(random_baseline, 2),
            "primekg": round(pk_overall, 2),
            "ta_static": round(ts_overall, 2),
            "ta_temporal": round(tt_overall, 2),
            "coverage_gain": round(ts_overall - pk_overall, 2),
            "temporal_gain": round(tt_overall - ts_overall, 2),
        },
        "per_disease": all_results[:30],
    }

    out_file = BENCHMARK_DIR / "temporal_diagnosis_v2.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
