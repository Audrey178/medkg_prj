#!/usr/bin/env python3
"""
Experiment 2 v3: LLM-Based Temporal Diagnostic Simulation
===========================================================
Tests whether ChronoMedKG temporal context helps LLMs reason about
differential diagnosis at specific patient ages.

Non-circular design:
  - Ground truth: GA4GH Phenopackets (real patient cases, independent of TA)
  - Task: MCQ "Patient at age X with phenotypes Y. Which disease?"
  - TA was built from population-level literature, not patient cases
  - LLM handles phenotype name matching (not ad-hoc string scoring)

3 conditions (same diseases, same distractors, different context):
  1. No KG: LLM uses only parametric knowledge
  2. PrimeKG: "Disease X has phenotypes: A, B, C" (static, no timing)
  3. ChronoMedKG: "Disease X has: A (onset 2-5y), B (onset 10-18y)" (temporal)

The key comparison is PrimeKG vs TA — same diseases, same phenotypes,
only difference is whether onset ages are included.

Output:
  data/benchmark/temporal_diagnosis_llm.json
"""

from __future__ import annotations

import csv
import json
import logging
import os
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("temporal_dx_llm")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
PRIMEKG_DIR = PROJECT_ROOT / "data" / "primekg"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation_sources"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"

# Load env
for line in open(PROJECT_ROOT / ".env"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        key, val = k.strip(), v.strip()
        if val and not os.environ.get(key):
            os.environ[key] = val

AGE_POINTS = [0, 2, 5, 10, 20]
N_DISEASES = 100
N_DISTRACTORS = 4


# ============================================================================
# DATA LOADING (reuse from v2)
# ============================================================================

def load_phenopackets():
    with open(VALIDATION_DIR / "phenopackets_parsed.pkl", "rb") as f:
        pp = pickle.load(f)
    rich = {}
    for name, data in pp.items():
        po = data.get("phenotype_onsets", {})
        if len(po) >= 3:
            rich[name.lower().strip()] = {
                "disease_name": name,
                "phenotype_onsets": po,
            }
    return rich


def load_primekg_phenotype_index():
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
    return dict(index)


def load_ta_indices():
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
    return dict(temporal_index), dict(binary_index)


def find_overlap(pp, pk_index, ta_binary):
    overlap = []
    for pp_name, pp_data in pp.items():
        pk_match = pp_name if pp_name in pk_index else None
        ta_match = pp_name if pp_name in ta_binary else None
        if not pk_match:
            for pk_name in pk_index:
                if pp_name in pk_name or pk_name in pp_name:
                    pk_match = pk_name
                    break
        if not ta_match:
            for ta_name in ta_binary:
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
    return overlap


# ============================================================================
# CONTEXT BUILDERS
# ============================================================================

def build_primekg_context(candidates, pk_index):
    """Build static phenotype context from PrimeKG for each candidate."""
    lines = []
    for i, cand in enumerate(candidates):
        letter = chr(65 + i)
        pk_name = cand["pk_name"]
        phenos = pk_index.get(pk_name, set())
        pheno_list = sorted(phenos)[:15]  # Top 15 phenotypes
        if pheno_list:
            lines.append(f"{letter}. {cand['pp_data']['disease_name']}: "
                        f"Associated phenotypes: {', '.join(pheno_list)}")
        else:
            lines.append(f"{letter}. {cand['pp_data']['disease_name']}: No phenotype data available")
    return "Known disease-phenotype associations (from PrimeKG, no temporal information):\n" + "\n".join(lines)


def build_ta_context(candidates, ta_temporal, ta_binary):
    """Build temporal phenotype context from ChronoMedKG for each candidate."""
    lines = []
    for i, cand in enumerate(candidates):
        letter = chr(65 + i)
        ta_name = cand["ta_name"]
        temporal_entries = ta_temporal.get(ta_name, [])
        binary_phenos = ta_binary.get(ta_name, set())

        if temporal_entries:
            # Group by phenotype, show onset ages
            pheno_ages = defaultdict(list)
            for pheno, omin, omax in temporal_entries:
                if omin == omax:
                    pheno_ages[pheno].append(f"{omin:.0f}y")
                else:
                    pheno_ages[pheno].append(f"{omin:.0f}-{omax:.0f}y")

            pheno_strs = []
            for pheno, ages in sorted(pheno_ages.items())[:12]:
                age_str = ages[0] if len(ages) == 1 else ages[0]  # Use first
                pheno_strs.append(f"{pheno} (onset: {age_str})")

            lines.append(f"{letter}. {cand['pp_data']['disease_name']}: "
                        f"Phenotypes with onset ages: {', '.join(pheno_strs)}")
        elif binary_phenos:
            pheno_list = sorted(binary_phenos)[:12]
            lines.append(f"{letter}. {cand['pp_data']['disease_name']}: "
                        f"Associated phenotypes (no onset data): {', '.join(pheno_list)}")
        else:
            lines.append(f"{letter}. {cand['pp_data']['disease_name']}: No data available")

    return ("Known disease-phenotype associations with onset ages "
            "(from ChronoMedKG temporal knowledge graph):\n" + "\n".join(lines))


def build_prompt(patient_age, presented_phenotypes, candidates, condition, context=""):
    """Build the MCQ prompt."""
    options = "\n".join(
        f"  {chr(65 + i)}. {c['pp_data']['disease_name']}"
        for i, c in enumerate(candidates)
    )

    pheno_str = ", ".join(presented_phenotypes[:10])

    base = (f"A patient presents at age {patient_age} years with the following clinical features: "
            f"{pheno_str}.\n\n"
            f"Which of the following diseases is the most likely diagnosis?\n{options}\n\n")

    if condition == "no_retrieval":
        base += ("Use your medical knowledge to determine the most likely diagnosis. "
                "Consider the patient's age and which disease typically presents with these "
                "features at this age.\n\n"
                "Answer with ONLY the letter (A, B, C, D, or E).")
    else:
        base += (f"{context}\n\n"
                "Use the disease information above along with your medical knowledge. "
                "Consider the patient's age and which disease typically presents with these "
                "features at this age.\n\n"
                "Answer with ONLY the letter (A, B, C, D, or E).")

    return base


# ============================================================================
# LLM CLIENTS
# ============================================================================

def call_deepseek(prompt):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=50,
    )
    return response.choices[0].message.content.strip()


def call_openai(prompt):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=50,
    )
    return response.choices[0].message.content.strip()


def call_model(model, prompt):
    if model == "deepseek-v3":
        return call_deepseek(prompt)
    elif model == "gpt-4o-mini":
        return call_openai(prompt)
    raise ValueError(f"Unknown model: {model}")


def parse_answer(llm_answer, n_options=5):
    """Extract letter answer from LLM response."""
    text = llm_answer.strip().upper()
    valid = [chr(65 + i) for i in range(n_options)]
    # Direct letter
    if text and text[0] in valid:
        return text[0]
    # Look for "The answer is X" pattern
    for letter in valid:
        if f"ANSWER IS {letter}" in text or f"ANSWER: {letter}" in text:
            return letter
    # Look for any standalone letter
    for letter in valid:
        if letter in text:
            return letter
    return None


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    logger.info("=" * 75)
    logger.info("Experiment 2 v3: LLM-Based Temporal Diagnostic Simulation")
    logger.info("=" * 75)

    # Load data
    logger.info("Loading data...")
    pp = load_phenopackets()
    logger.info("  Phenopackets: %d diseases", len(pp))
    pk_index = load_primekg_phenotype_index()
    logger.info("  PrimeKG: %d diseases", len(pk_index))
    ta_temporal, ta_binary = load_ta_indices()
    logger.info("  TA: %d temporal, %d binary", len(ta_temporal), len(ta_binary))

    overlap = find_overlap(pp, pk_index, ta_binary)
    logger.info("  Overlap: %d diseases", len(overlap))

    # Sample N diseases
    random.seed(42)
    sample = random.sample(overlap, min(N_DISEASES, len(overlap)))
    logger.info("Sampled %d diseases for simulation", len(sample))

    # Setup
    models = ["deepseek-v3", "gpt-4o-mini"]
    conditions = ["no_retrieval", "primekg_rag", "chronomedkg_rag"]
    total_calls = len(sample) * len(AGE_POINTS) * len(conditions) * len(models)
    logger.info("Total API calls: ~%d (est cost: ~$%.2f)", total_calls,
                total_calls * 0.001)  # Rough estimate

    # Checkpoint
    results_dir = BENCHMARK_DIR / "temporal_dx_llm_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = results_dir / "checkpoint.jsonl"

    completed = set()
    all_results = []
    if ckpt_file.exists():
        with open(ckpt_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    all_results.append(r)
                    completed.add((r["disease"], r["age"], r["model"], r["condition"]))
        logger.info("Resumed: %d completed results", len(completed))

    ckpt_f = open(ckpt_file, "a")
    calls_made = 0
    errors = 0
    start = time.monotonic()

    for si, target in enumerate(sample):
        disease_name = target["pp_data"]["disease_name"]
        phenotype_onsets = target["pp_data"]["phenotype_onsets"]

        # Find distractors from overlap set
        distractors = []
        target_phenos = set(p.lower() for p in phenotype_onsets.keys())
        for other in overlap:
            if other["pp_name"] == target["pp_name"]:
                continue
            other_phenos = set(p.lower() for p in other["pp_data"]["phenotype_onsets"].keys())
            shared = sum(1 for tp in target_phenos
                        if any(tp in op or op in tp for op in other_phenos))
            if shared >= 1:
                distractors.append((shared, other))

        distractors.sort(key=lambda x: -x[0])
        distractors = [d[1] for d in distractors[:N_DISTRACTORS]]

        if len(distractors) < 2:
            continue

        # Candidates: correct + distractors, shuffled
        candidates = [target] + distractors
        random.seed(hash(disease_name))
        random.shuffle(candidates)
        correct_idx = next(i for i, c in enumerate(candidates) if c["pp_name"] == target["pp_name"])
        correct_letter = chr(65 + correct_idx)

        # Build contexts once per disease
        pk_context = build_primekg_context(candidates, pk_index)
        ta_context = build_ta_context(candidates, ta_temporal, ta_binary)

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

            for model in models:
                for condition in conditions:
                    key = (disease_name, age, model, condition)
                    if key in completed:
                        continue

                    # Build context
                    if condition == "no_retrieval":
                        context = ""
                    elif condition == "primekg_rag":
                        context = pk_context
                    else:
                        context = ta_context

                    prompt = build_prompt(age, presented, candidates, condition, context)

                    try:
                        llm_answer = call_model(model, prompt)
                        parsed = parse_answer(llm_answer, len(candidates))
                        correct = parsed == correct_letter
                    except Exception as e:
                        llm_answer = f"ERROR: {e}"
                        parsed = None
                        correct = False
                        errors += 1

                    result = {
                        "disease": disease_name,
                        "age": age,
                        "model": model,
                        "condition": condition,
                        "n_phenotypes": len(presented),
                        "n_candidates": len(candidates),
                        "correct_letter": correct_letter,
                        "llm_answer": llm_answer[:100],
                        "parsed_answer": parsed,
                        "correct": correct,
                    }
                    all_results.append(result)
                    ckpt_f.write(json.dumps(result) + "\n")
                    ckpt_f.flush()
                    calls_made += 1

                    time.sleep(0.15)  # Rate limit

        if (si + 1) % 10 == 0:
            elapsed = time.monotonic() - start
            rate = calls_made / max(1, elapsed)
            remaining = (total_calls - calls_made - len(completed)) / max(1, rate)
            logger.info("  %d/%d diseases, %d calls (%.0f/min), ~%.0f min left, %d errors",
                       si + 1, len(sample), calls_made, rate * 60, remaining / 60, errors)

    ckpt_f.close()

    # Filter out errors
    good = [r for r in all_results if "ERROR" not in str(r.get("llm_answer", ""))]
    logger.info("Total results: %d good, %d errors", len(good), errors)

    # Aggregate
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # acc[model][condition][age] = [0/1]

    for r in good:
        acc[r["model"]][r["condition"]][r["age"]].append(1 if r["correct"] else 0)

    # Print
    n_cands = sample[0]["pp_data"] and len(AGE_POINTS)  # Approximate
    random_pct = 100.0 / (N_DISTRACTORS + 1)

    print(f"\n{'=' * 90}")
    print("LLM-BASED TEMPORAL DIAGNOSTIC SIMULATION")
    print(f"{'=' * 90}")
    print(f"Diseases: {len(sample)}, Ages: {AGE_POINTS}, Models: {models}")
    print(f"Conditions: No KG | PrimeKG (static) | ChronoMedKG (temporal)")
    print(f"Random baseline: {random_pct:.0f}%")
    print(f"Ground truth: GA4GH Phenopackets (independent patient cases)")

    for model in models:
        print(f"\n--- {model} ---")
        print(f"{'Age':>5} | {'No KG':>8} | {'PrimeKG':>8} | {'TA':>8} | {'PK-NR':>8} | {'TA-PK':>8} | {'N':>5}")
        print("-" * 65)

        for age in AGE_POINTS:
            nr = acc[model]["no_retrieval"].get(age, [])
            pk = acc[model]["primekg_rag"].get(age, [])
            ta = acc[model]["chronomedkg_rag"].get(age, [])

            if not nr:
                continue

            nr_acc = 100 * sum(nr) / len(nr)
            pk_acc = 100 * sum(pk) / len(pk) if pk else 0
            ta_acc = 100 * sum(ta) / len(ta) if ta else 0
            pk_gain = pk_acc - nr_acc
            ta_gain = ta_acc - pk_acc  # KEY: TA vs PrimeKG (same diseases, temporal = only difference)
            n = len(nr)

            marker = " ***" if ta_gain > 3 else ""
            print(f"{age:>5} | {nr_acc:>7.1f}% | {pk_acc:>7.1f}% | {ta_acc:>7.1f}% | "
                  f"{pk_gain:>+7.1f}pp | {ta_gain:>+7.1f}pp | {n:>5}{marker}")

    # Overall headline
    print(f"\n{'=' * 90}")
    print("OVERALL HEADLINE")
    print(f"{'=' * 90}")

    for model in models:
        all_nr = [v for age_vals in acc[model]["no_retrieval"].values() for v in age_vals]
        all_pk = [v for age_vals in acc[model]["primekg_rag"].values() for v in age_vals]
        all_ta = [v for age_vals in acc[model]["chronomedkg_rag"].values() for v in age_vals]

        nr_o = 100 * sum(all_nr) / max(1, len(all_nr))
        pk_o = 100 * sum(all_pk) / max(1, len(all_pk))
        ta_o = 100 * sum(all_ta) / max(1, len(all_ta))

        print(f"\n  {model}:")
        print(f"    No KG:           {nr_o:.1f}%")
        print(f"    + PrimeKG:       {pk_o:.1f}% ({pk_o - nr_o:+.1f}pp vs no KG)")
        print(f"    + ChronoMedKG: {ta_o:.1f}% ({ta_o - pk_o:+.1f}pp vs PrimeKG)")
        print(f"    Temporal advantage: {ta_o - pk_o:+.1f}pp (THIS is the key number)")

    # Save
    output = {
        "experiment": "LLM-Based Temporal Diagnostic Simulation",
        "ground_truth": "GA4GH Phenopackets (independent patient cases)",
        "methodology": {
            "overlap_set": True,
            "models": models,
            "conditions": conditions,
            "age_points": AGE_POINTS,
            "n_diseases": len(sample),
            "n_distractors": N_DISTRACTORS,
        },
        "results_count": len(good),
        "errors": errors,
        "per_model_results": {
            model: {
                "by_age": {
                    age: {
                        cond: round(100 * sum(acc[model][cond].get(age, [])) / max(1, len(acc[model][cond].get(age, []))), 2)
                        for cond in conditions
                    }
                    for age in AGE_POINTS if acc[model]["no_retrieval"].get(age)
                },
                "overall": {
                    cond: round(100 * sum(v for vals in acc[model][cond].values() for v in vals) /
                               max(1, sum(len(vals) for vals in acc[model][cond].values())), 2)
                    for cond in conditions
                },
            }
            for model in models
        },
    }

    out_file = BENCHMARK_DIR / "temporal_diagnosis_llm.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
