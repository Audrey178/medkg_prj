#!/usr/bin/env python3
"""
Experiment: Quantitative PMC Clinical Case Analysis
====================================================
For each of 10 PMC clinical cases with diagnostic odysseys:
  - At each time point in the patient journey, compute diagnostic ranking
  - Compare: PrimeKG (static phenotype matching) vs ChronoMedKG (temporal matching)
  - Measure: rank of CORRECT diagnosis at initial presentation

Ground truth: PMC case reports (completely independent of TA)
Question: could temporal reasoning have prevented misdiagnosis?

Output:
  data/benchmark/pmc_quantitative_analysis.json
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("pmc_quant")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
PRIMEKG_DIR = PROJECT_ROOT / "data" / "primekg"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation_sources"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"


def load_pmc_cases():
    """Load 10 curated PMC clinical cases."""
    with open(VALIDATION_DIR / "pmc_clinical_cases.json") as f:
        return json.load(f)


def load_primekg_disease_phenotypes():
    """Build disease -> {phenotype set} from PrimeKG."""
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
    logger.info(f"PrimeKG: {len(index)} diseases with phenotypes")
    return dict(index)


def load_ta_disease_phenotypes():
    """Build disease -> [(phenotype, onset_min, onset_max)] from ChronoMedKG."""
    index = defaultdict(list)
    binary = defaultdict(set)
    for d in EXTRACTED_DIR.iterdir():
        vf = d / "validated_triples.jsonl"
        if not vf.exists():
            continue
        with open(vf) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    t = json.loads(line)
                except:
                    continue
                if "phenotype" not in t.get("relation", ""):
                    continue
                disease = t.get("source_name", "").lower().strip()
                phenotype = t.get("target_name", "").lower().strip()
                if not disease or not phenotype:
                    continue
                binary[disease].add(phenotype)

                temporal = t.get("temporal", {}) or {}
                omin = temporal.get("onset_age_min")
                if omin is not None:
                    try:
                        omin = float(omin)
                        omax = float(temporal.get("onset_age_max") or omin)
                        if 0 <= omin <= 120:
                            index[disease].append((phenotype, omin, omax))
                    except:
                        pass
    logger.info(f"TA: {len(binary)} diseases binary, {len(index)} with temporal")
    return dict(index), dict(binary)


def find_candidate_diseases(correct_diagnosis, misdiagnosis, primekg_idx, ta_binary):
    """Build candidate list: correct diagnosis + similar diseases + misdiagnosis."""
    candidates = [correct_diagnosis.lower()]
    if misdiagnosis and misdiagnosis.lower() != "none":
        candidates.append(misdiagnosis.lower())

    # Find more plausible distractors (diseases with similar phenotype profiles)
    correct_phenos = primekg_idx.get(correct_diagnosis.lower(), set())
    if not correct_phenos:
        # Try fuzzy match
        for d in primekg_idx:
            if correct_diagnosis.lower() in d or d in correct_diagnosis.lower():
                correct_phenos = primekg_idx[d]
                candidates[0] = d  # Update to actual PrimeKG name
                break

    # Find distractors (>=3 shared phenotypes)
    distractors = []
    for d, phenos in primekg_idx.items():
        if d in [c.lower() for c in candidates]:
            continue
        overlap = len(correct_phenos & phenos)
        if overlap >= 3:
            distractors.append((overlap, d))

    distractors.sort(key=lambda x: -x[0])
    for _, d in distractors[:5]:  # Top 5 distractors
        candidates.append(d)

    return candidates


def phenotype_match_score(query_phenotypes, candidate_phenotypes):
    """Simple match score: fraction of query phenotypes found in candidate."""
    if not query_phenotypes or not candidate_phenotypes:
        return 0.0
    matches = 0
    for q in query_phenotypes:
        q_low = q.lower()
        # Exact
        if q_low in candidate_phenotypes:
            matches += 1
            continue
        # Substring
        if any(q_low in cp or cp in q_low for cp in candidate_phenotypes):
            matches += 0.7
            continue
        # Word overlap
        q_words = set(w for w in q_low.split() if len(w) > 4)
        for cp in candidate_phenotypes:
            cp_words = set(w for w in cp.lower().split() if len(w) > 4)
            if q_words & cp_words:
                matches += 0.4
                break
    return matches / len(query_phenotypes)


def temporal_match_score(query_phenotypes, candidate_temporal, patient_age):
    """Temporal match: phenotype match weighted by onset window fit."""
    if not query_phenotypes or not candidate_temporal:
        return 0.0

    total = 0.0
    for q in query_phenotypes:
        q_low = q.lower()
        best = 0.0

        for pheno, omin, omax in candidate_temporal:
            # Phenotype match score
            pheno_score = 0.0
            if q_low == pheno:
                pheno_score = 1.0
            elif q_low in pheno or pheno in q_low:
                pheno_score = 0.7
            else:
                q_words = set(w for w in q_low.split() if len(w) > 4)
                p_words = set(w for w in pheno.split() if len(w) > 4)
                if q_words & p_words:
                    pheno_score = 0.4
            if pheno_score == 0:
                continue

            # Temporal match
            if omin <= patient_age <= omax:
                temporal_score = 1.0
            else:
                dist = min(abs(patient_age - omin), abs(patient_age - omax))
                if dist <= 2:
                    temporal_score = 0.7
                elif dist <= 5:
                    temporal_score = 0.4
                else:
                    temporal_score = 0.1

            best = max(best, pheno_score * temporal_score)

        total += best

    return total / len(query_phenotypes)


def rank_diseases(candidates, query_phenotypes, patient_age, primekg_idx, ta_temporal, ta_binary):
    """Rank candidate diseases using PrimeKG and TA scoring."""
    pk_scores = {}
    ta_scores = {}
    ta_static_scores = {}

    for cand in candidates:
        cand_low = cand.lower()

        # PrimeKG: static phenotype match
        pk_phenos = primekg_idx.get(cand_low, set())
        # Try fuzzy
        if not pk_phenos:
            for d in primekg_idx:
                if cand_low in d or d in cand_low:
                    pk_phenos = primekg_idx[d]
                    break
        pk_scores[cand] = phenotype_match_score(query_phenotypes, pk_phenos)

        # TA: temporal match
        ta_temp = ta_temporal.get(cand_low, [])
        if not ta_temp:
            for d in ta_temporal:
                if cand_low in d or d in cand_low:
                    ta_temp = ta_temporal[d]
                    break
        ta_scores[cand] = temporal_match_score(query_phenotypes, ta_temp, patient_age)

        # TA static (binary phenotype match from TA)
        ta_bin = ta_binary.get(cand_low, set())
        if not ta_bin:
            for d in ta_binary:
                if cand_low in d or d in cand_low:
                    ta_bin = ta_binary[d]
                    break
        ta_static_scores[cand] = phenotype_match_score(query_phenotypes, ta_bin)

    # Rank
    pk_ranked = sorted(pk_scores.items(), key=lambda x: -x[1])
    ta_ranked = sorted(ta_scores.items(), key=lambda x: -x[1])
    ta_static_ranked = sorted(ta_static_scores.items(), key=lambda x: -x[1])

    return pk_ranked, ta_ranked, ta_static_ranked


def find_rank(ranked, target):
    """Find 1-indexed rank of target in ranked list."""
    target_low = target.lower()
    for i, (disease, _) in enumerate(ranked):
        if disease.lower() == target_low or target_low in disease.lower() or disease.lower() in target_low:
            return i + 1
    return len(ranked) + 1  # Not found


def main():
    logger.info("=" * 75)
    logger.info("PMC Quantitative Clinical Case Analysis")
    logger.info("=" * 75)

    # Load data
    logger.info("\n[1/4] Loading PMC cases...")
    cases = load_pmc_cases()
    logger.info(f"  Loaded {len(cases)} cases")

    logger.info("\n[2/4] Loading PrimeKG phenotype index...")
    primekg_idx = load_primekg_disease_phenotypes()

    logger.info("\n[3/4] Loading ChronoMedKG indices...")
    ta_temporal, ta_binary = load_ta_disease_phenotypes()

    logger.info("\n[4/4] Running diagnostic simulation...")

    all_results = []
    pk_ranks_initial = []
    ta_ranks_initial = []
    ta_static_ranks_initial = []

    for case_idx, case in enumerate(cases):
        correct = case["correct_diagnosis"]
        misdiag = case.get("misdiagnosis", "None")
        timeline = case.get("phenotype_timeline", [])

        logger.info(f"\nCase {case_idx+1}: {correct[:50]}")
        logger.info(f"  Misdiagnosis: {misdiag[:50] if misdiag != 'None' else 'none'}")
        logger.info(f"  Timepoints: {len(timeline)}")

        # Find candidate diseases
        candidates = find_candidate_diseases(correct, misdiag, primekg_idx, ta_binary)
        logger.info(f"  Candidates: {len(candidates)}")

        case_result = {
            "pmc_id": case.get("pmc_id"),
            "correct_diagnosis": correct,
            "misdiagnosis": misdiag,
            "n_candidates": len(candidates),
            "timepoints": [],
        }

        for tp in timeline:
            age = tp.get("age")
            phenotypes = tp.get("phenotypes", [])
            if not phenotypes or age is None:
                continue

            pk_ranked, ta_ranked, ta_static_ranked = rank_diseases(
                candidates, phenotypes, age, primekg_idx, ta_temporal, ta_binary
            )

            pk_rank = find_rank(pk_ranked, correct)
            ta_rank = find_rank(ta_ranked, correct)
            ta_static_rank = find_rank(ta_static_ranked, correct)

            case_result["timepoints"].append({
                "age": age,
                "n_phenotypes": len(phenotypes),
                "pk_rank": pk_rank,
                "ta_temporal_rank": ta_rank,
                "ta_static_rank": ta_static_rank,
                "pk_top": pk_ranked[0][0] if pk_ranked else None,
                "ta_top": ta_ranked[0][0] if ta_ranked else None,
            })

        # Get rank at INITIAL presentation (earliest timepoint)
        if case_result["timepoints"]:
            initial = case_result["timepoints"][0]
            case_result["initial_age"] = initial["age"]
            case_result["initial_pk_rank"] = initial["pk_rank"]
            case_result["initial_ta_rank"] = initial["ta_temporal_rank"]
            case_result["initial_ta_static_rank"] = initial["ta_static_rank"]

            pk_ranks_initial.append(initial["pk_rank"])
            ta_ranks_initial.append(initial["ta_temporal_rank"])
            ta_static_ranks_initial.append(initial["ta_static_rank"])

            logger.info(f"  Initial (age {initial['age']}): PK rank={initial['pk_rank']}, "
                       f"TA-temporal={initial['ta_temporal_rank']}, TA-static={initial['ta_static_rank']}")

        all_results.append(case_result)

    # Summary
    print(f"\n{'=' * 75}")
    print("PMC QUANTITATIVE ANALYSIS — RANK AT INITIAL PRESENTATION")
    print(f"{'=' * 75}")
    print(f"Lower rank is better (rank 1 = correct diagnosis at top of ranking)")
    print(f"Cases: {len(all_results)}")

    print(f"\n{'Case':<40} {'Age':>6} {'PrimeKG':>8} {'TA-temp':>8} {'TA-static':>10}")
    print("-" * 80)
    for r in all_results:
        if "initial_age" not in r:
            continue
        print(f"{r['correct_diagnosis'][:38]:<40} {r['initial_age']:>6.1f} "
              f"{r['initial_pk_rank']:>8} {r['initial_ta_rank']:>8} {r['initial_ta_static_rank']:>10}")

    # Metrics
    if pk_ranks_initial:
        print(f"\n{'='*75}")
        print("METRICS AT INITIAL PRESENTATION")
        print(f"{'='*75}")

        def metrics(ranks):
            mean_rank = statistics.mean(ranks)
            rank1 = sum(1 for r in ranks if r == 1) / len(ranks) * 100
            rank3 = sum(1 for r in ranks if r <= 3) / len(ranks) * 100
            return mean_rank, rank1, rank3

        for name, ranks in [("PrimeKG", pk_ranks_initial),
                             ("TA-temporal", ta_ranks_initial),
                             ("TA-static", ta_static_ranks_initial)]:
            mean_r, r1, r3 = metrics(ranks)
            print(f"  {name:<15}: Mean rank = {mean_r:.2f}, Rank@1 = {r1:.0f}%, Rank@3 = {r3:.0f}%")

    # Save
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "PMC Quantitative Clinical Case Analysis",
        "ground_truth": "10 PMC open-access clinical case reports with diagnostic odysseys",
        "n_cases": len(all_results),
        "per_case_results": all_results,
        "summary": {
            "primekg": {
                "mean_rank": statistics.mean(pk_ranks_initial) if pk_ranks_initial else None,
                "rank_1_pct": round(100 * sum(1 for r in pk_ranks_initial if r == 1) / len(pk_ranks_initial), 1) if pk_ranks_initial else 0,
                "rank_3_pct": round(100 * sum(1 for r in pk_ranks_initial if r <= 3) / len(pk_ranks_initial), 1) if pk_ranks_initial else 0,
            },
            "ta_temporal": {
                "mean_rank": statistics.mean(ta_ranks_initial) if ta_ranks_initial else None,
                "rank_1_pct": round(100 * sum(1 for r in ta_ranks_initial if r == 1) / len(ta_ranks_initial), 1) if ta_ranks_initial else 0,
                "rank_3_pct": round(100 * sum(1 for r in ta_ranks_initial if r <= 3) / len(ta_ranks_initial), 1) if ta_ranks_initial else 0,
            },
            "ta_static": {
                "mean_rank": statistics.mean(ta_static_ranks_initial) if ta_static_ranks_initial else None,
                "rank_1_pct": round(100 * sum(1 for r in ta_static_ranks_initial if r == 1) / len(ta_static_ranks_initial), 1) if ta_static_ranks_initial else 0,
                "rank_3_pct": round(100 * sum(1 for r in ta_static_ranks_initial if r <= 3) / len(ta_static_ranks_initial), 1) if ta_static_ranks_initial else 0,
            },
        },
    }
    out_file = BENCHMARK_DIR / "pmc_quantitative_analysis.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
