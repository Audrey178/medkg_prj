#!/usr/bin/env python3
"""
Experiment 2 Final: Phenopacket Store Temporal Diagnostic Simulation
=====================================================================
Systematic LLM-based evaluation using 87 ID-matched diseases (343 cases)
from the GA4GH Phenopacket Store.

Design:
  - Each phenopacket case has phenotypes with onset ages (ground truth)
  - At each patient age, present the LLM with currently-visible phenotypes
  - MCQ: "Which disease is most likely?" (1 correct + 4 distractors)
  - 3 conditions: No KG / PrimeKG (static) / ChronoMedKG (temporal)

Non-circular:
  - Phenopacket Store = independent patient cases (not literature summaries)
  - TA was built from population-level literature extraction
  - PrimeKG from curated databases
  - Ground truth = confirmed genetic diagnosis in each phenopacket

Key metric: TA-temporal vs PrimeKG-static accuracy at each age point

Output:
  data/benchmark/phenopacket_store_dx.json
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("ppstore_dx")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
PRIMEKG_DIR = PROJECT_ROOT / "data" / "primekg"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"

# Load env
for line in open(PROJECT_ROOT / ".env"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        key, val = k.strip(), v.strip()
        if val and not os.environ.get(key):
            os.environ[key] = val

AGE_POINTS = [0, 1, 2, 5, 10, 15, 20, 30, 50]


def load_matched_cases():
    """Load ID-matched Phenopacket Store cases."""
    with open(PROJECT_ROOT / "data/validation_sources/phenopacket_store/matched_cases.json") as f:
        data = json.load(f)
    logger.info("Loaded %d matched diseases (%d cases)", data["total_matched"],
                sum(len(d["cases"]) for d in data["diseases"]))
    return data["diseases"]


def load_primekg_phenotypes():
    """Load PrimeKG disease→phenotypes by disease dir name."""
    # Build MONDO→phenotypes from PrimeKG using disease names
    import yaml
    # Map dir_name → disease_name from configs
    dir_to_name = {}
    for yf in (PROJECT_ROOT / "config/diseases").glob("*.yaml"):
        try:
            with open(yf) as f:
                cfg = yaml.safe_load(f)
            if cfg:
                dir_to_name[yf.stem] = cfg.get("disease_name", "").lower().strip()
        except:
            pass

    # Load PrimeKG phenotype index by disease name
    name_to_phenos = defaultdict(set)
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
                name_to_phenos[x_name].add(y_name)
            elif y_type == "disease" and x_type == "effect/phenotype":
                name_to_phenos[y_name].add(x_name)

    # Map dir_name → phenotypes
    dir_to_phenos = {}
    for dir_name, disease_name in dir_to_name.items():
        if disease_name in name_to_phenos:
            dir_to_phenos[dir_name] = name_to_phenos[disease_name]
        else:
            # Try substring
            for pk_name, phenos in name_to_phenos.items():
                if disease_name and (disease_name in pk_name or pk_name in disease_name) and len(disease_name) > 8:
                    dir_to_phenos[dir_name] = phenos
                    break

    logger.info("PrimeKG phenotypes mapped for %d disease dirs", len(dir_to_phenos))
    return dir_to_phenos


def load_ta_temporal(dir_names):
    """Load TA temporal phenotype data for specific disease dirs."""
    ta_data = {}
    for dir_name in dir_names:
        vf = EXTRACTED_DIR / dir_name / "validated_triples.jsonl"
        if not vf.exists():
            continue
        entries = []
        phenos = set()
        with open(vf) as f:
            for line in f:
                if not line.strip():
                    continue
                t = json.loads(line)
                if "phenotype" not in t.get("relation", ""):
                    continue
                phenotype = t.get("target_name", "").lower().strip()
                if not phenotype:
                    continue
                phenos.add(phenotype)
                temporal = t.get("temporal", {}) or {}
                omin = temporal.get("onset_age_min")
                if omin is not None:
                    try:
                        omin = float(omin)
                        omax = float(temporal.get("onset_age_max") or omin)
                        if 0 <= omin <= 120:
                            entries.append((phenotype, omin, omax))
                    except:
                        pass
        ta_data[dir_name] = {"temporal": entries, "binary": phenos}
    logger.info("Loaded TA data for %d disease dirs", len(ta_data))
    return ta_data


def build_primekg_context(candidates, pk_phenos):
    """Static phenotype context from PrimeKG."""
    lines = []
    for i, cand in enumerate(candidates):
        letter = chr(65 + i)
        phenos = pk_phenos.get(cand["dir_name"], set())
        pheno_list = sorted(phenos)[:15]
        name = cand["cases"][0]["disease_name"] if cand["cases"] else cand["pp_name"]
        if pheno_list:
            lines.append(f"{letter}. {name}: Associated phenotypes: {', '.join(pheno_list)}")
        else:
            lines.append(f"{letter}. {name}: No phenotype data available in this knowledge base")
    return "Disease-phenotype associations (static, no temporal information):\n" + "\n".join(lines)


def build_ta_context(candidates, ta_data):
    """Temporal phenotype context from ChronoMedKG."""
    lines = []
    for i, cand in enumerate(candidates):
        letter = chr(65 + i)
        name = cand["cases"][0]["disease_name"] if cand["cases"] else cand["pp_name"]
        data = ta_data.get(cand["dir_name"], {})
        temporal = data.get("temporal", [])

        if temporal:
            pheno_ages = defaultdict(list)
            for pheno, omin, omax in temporal:
                if omin == omax:
                    pheno_ages[pheno].append(f"{omin:.0f}y")
                else:
                    pheno_ages[pheno].append(f"{omin:.0f}-{omax:.0f}y")

            strs = []
            for pheno, ages in sorted(pheno_ages.items())[:12]:
                strs.append(f"{pheno} (onset: {ages[0]})")
            lines.append(f"{letter}. {name}: {', '.join(strs)}")
        else:
            binary = data.get("binary", set())
            if binary:
                lines.append(f"{letter}. {name}: Phenotypes: {', '.join(sorted(binary)[:12])} (no onset age data)")
            else:
                lines.append(f"{letter}. {name}: No data available")

    return "Disease-phenotype associations with onset ages (from ChronoMedKG):\n" + "\n".join(lines)


def build_prompt(patient_age, phenotypes, candidates, condition, context):
    options = "\n".join(
        f"  {chr(65+i)}. {c['cases'][0]['disease_name'] if c['cases'] else c['pp_name']}"
        for i, c in enumerate(candidates)
    )
    pheno_str = ", ".join(phenotypes[:10])
    if condition == "no_retrieval":
        return (f"A patient presents at age {patient_age:.1f} years with: {pheno_str}.\n\n"
                f"Which disease is most likely?\n{options}\n\n"
                f"Consider the patient's age and typical onset patterns. "
                f"Answer with ONLY the letter (A-E).")
    else:
        return (f"A patient presents at age {patient_age:.1f} years with: {pheno_str}.\n\n"
                f"{context}\n\n"
                f"Which disease is most likely?\n{options}\n\n"
                f"Use the disease information above and consider the patient's age. "
                f"Answer with ONLY the letter (A-E).")


def call_model(model, prompt):
    from openai import OpenAI
    if model == "deepseek-v3":
        client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        m = "deepseek-chat"
    else:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        m = model
    resp = client.chat.completions.create(
        model=m, messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=50)
    return resp.choices[0].message.content.strip()


def parse_answer(text, n=5):
    text = text.strip().upper()
    valid = [chr(65+i) for i in range(n)]
    if text and text[0] in valid:
        return text[0]
    for v in valid:
        if f"ANSWER IS {v}" in text or f"ANSWER: {v}" in text:
            return v
    for v in valid:
        if v in text:
            return v
    return None


def main():
    logger.info("=" * 75)
    logger.info("Phenopacket Store Temporal Diagnostic Simulation")
    logger.info("=" * 75)

    # Load data
    matched = load_matched_cases()
    pk_phenos = load_primekg_phenotypes()
    all_dirs = set(m["dir_name"] for m in matched)
    ta_data = load_ta_temporal(all_dirs)

    # Select diseases with >=3 cases and enough phenotype data
    eligible = [m for m in matched if len(m["cases"]) >= 2]
    logger.info("Eligible diseases (>=2 cases): %d", len(eligible))

    # For each disease, we need distractors from the SAME matched set
    models = ["deepseek-v3", "gpt-4o-mini"]
    conditions = ["no_retrieval", "primekg_rag", "chronomedkg_rag"]

    # Checkpoint
    results_dir = BENCHMARK_DIR / "ppstore_dx_results"
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
                    completed.add((r["case_id"], r["age"], r["model"], r["condition"]))
        logger.info("Resumed: %d completed", len(completed))

    ckpt_f = open(ckpt_file, "a")
    calls = 0
    errors = 0
    start = time.monotonic()

    # Process each case
    for di, disease in enumerate(eligible):
        disease_name = disease["cases"][0]["disease_name"]
        dir_name = disease["dir_name"]

        # Find distractors: other diseases in the matched set
        distractors = [d for d in eligible if d["dir_name"] != dir_name]
        # Sort by phenotype overlap (using TA binary phenotypes)
        target_phenos = ta_data.get(dir_name, {}).get("binary", set())
        scored_dist = []
        for d in distractors:
            other_phenos = ta_data.get(d["dir_name"], {}).get("binary", set())
            overlap = len(target_phenos & other_phenos)
            scored_dist.append((overlap, d))
        scored_dist.sort(key=lambda x: -x[0])
        distractors = [d[1] for d in scored_dist[:N_DISTRACTORS]]

        if len(distractors) < 3:
            continue

        for case in disease["cases"][:3]:  # Max 3 cases per disease
            case_id = case["case_id"]
            pheno_onsets = case.get("phenotypes_with_onset", [])
            if len(pheno_onsets) < 2:
                continue

            # Build candidates (randomized)
            candidates = [disease] + distractors
            random.seed(hash(case_id))
            random.shuffle(candidates)
            correct_idx = next(i for i, c in enumerate(candidates) if c["dir_name"] == dir_name)
            correct_letter = chr(65 + correct_idx)

            # Build contexts once
            pk_ctx = build_primekg_context(candidates, pk_phenos)
            ta_ctx = build_ta_context(candidates, ta_data)

            for age in AGE_POINTS:
                # Phenotypes visible at this age
                visible = []
                for po in pheno_onsets:
                    onset = po.get("onset_age")
                    if onset is not None and onset <= age:
                        visible.append(po["phenotype"])

                if len(visible) < 1:
                    continue

                for model in models:
                    for cond in conditions:
                        key = (case_id, age, model, cond)
                        if key in completed:
                            continue

                        ctx = "" if cond == "no_retrieval" else (pk_ctx if cond == "primekg_rag" else ta_ctx)
                        prompt = build_prompt(age, visible, candidates, cond, ctx)

                        try:
                            answer = call_model(model, prompt)
                            parsed = parse_answer(answer, len(candidates))
                            correct = parsed == correct_letter
                        except Exception as e:
                            answer = f"ERROR: {e}"
                            parsed = None
                            correct = False
                            errors += 1

                        result = {
                            "case_id": case_id,
                            "disease": disease_name,
                            "age": age,
                            "n_visible_phenotypes": len(visible),
                            "model": model,
                            "condition": cond,
                            "correct_letter": correct_letter,
                            "parsed": parsed,
                            "correct": correct,
                            "llm_answer": str(answer)[:80],
                        }
                        all_results.append(result)
                        ckpt_f.write(json.dumps(result) + "\n")
                        ckpt_f.flush()
                        calls += 1
                        time.sleep(0.12)

        if (di + 1) % 5 == 0:
            elapsed = time.monotonic() - start
            good = [r for r in all_results if "ERROR" not in str(r.get("llm_answer", ""))]
            rate = calls / max(1, elapsed) * 60
            logger.info("  %d/%d diseases, %d calls (%.0f/min), %d errors, %d good results",
                       di + 1, len(eligible), calls, rate, errors, len(good))

    ckpt_f.close()
    good = [r for r in all_results if "ERROR" not in str(r.get("llm_answer", ""))]
    logger.info("Done: %d good results, %d errors, %d API calls", len(good), errors, calls)

    # Aggregate
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for r in good:
        acc[r["model"]][r["condition"]][r["age"]].append(1 if r["correct"] else 0)

    rand = 100.0 / (N_DISTRACTORS + 1)

    print(f"\n{'=' * 90}")
    print("PHENOPACKET STORE — TEMPORAL DIAGNOSTIC SIMULATION")
    print(f"{'=' * 90}")
    print(f"Cases: {len(good)}, Diseases: {len(eligible)}, Models: {models}")
    print(f"Conditions: No KG / PrimeKG (static) / ChronoMedKG (temporal)")
    print(f"Random baseline: {rand:.0f}%")

    for model in models:
        print(f"\n--- {model} ---")
        print(f"{'Age':>5} | {'No KG':>8} | {'PrimeKG':>8} | {'TA':>8} | {'TA-PK':>8} | {'N':>5}")
        print("-" * 55)
        for age in AGE_POINTS:
            nr = acc[model]["no_retrieval"].get(age, [])
            pk = acc[model]["primekg_rag"].get(age, [])
            ta = acc[model]["chronomedkg_rag"].get(age, [])
            if not nr:
                continue
            nr_a = 100 * sum(nr) / len(nr)
            pk_a = 100 * sum(pk) / len(pk) if pk else 0
            ta_a = 100 * sum(ta) / len(ta) if ta else 0
            gain = ta_a - pk_a
            n = len(nr)
            print(f"{age:>5} | {nr_a:>7.1f}% | {pk_a:>7.1f}% | {ta_a:>7.1f}% | {gain:>+7.1f}pp | {n:>5}")

    # Overall
    print(f"\n{'=' * 90}")
    print("HEADLINE")
    print(f"{'=' * 90}")
    for model in models:
        all_nr = [v for vals in acc[model]["no_retrieval"].values() for v in vals]
        all_pk = [v for vals in acc[model]["primekg_rag"].values() for v in vals]
        all_ta = [v for vals in acc[model]["chronomedkg_rag"].values() for v in vals]
        nr_o = 100 * sum(all_nr) / max(1, len(all_nr))
        pk_o = 100 * sum(all_pk) / max(1, len(all_pk))
        ta_o = 100 * sum(all_ta) / max(1, len(all_ta))
        print(f"  {model}: NoKG={nr_o:.1f}%, PrimeKG={pk_o:.1f}%, TA={ta_o:.1f}% "
              f"(TA-PK={ta_o-pk_o:+.1f}pp)")

    # Save
    output = {
        "experiment": "Phenopacket Store Temporal Diagnostic Simulation",
        "ground_truth": "GA4GH Phenopacket Store (6,668 independent patient cases)",
        "n_results": len(good),
        "n_errors": errors,
        "models": models,
        "per_model": {
            model: {
                "overall": {
                    cond: round(100 * sum(v for vals in acc[model][cond].values() for v in vals) /
                               max(1, sum(len(v) for v in acc[model][cond].values())), 2)
                    for cond in conditions
                },
                "by_age": {
                    age: {cond: round(100 * sum(acc[model][cond].get(age, [])) /
                                     max(1, len(acc[model][cond].get(age, []))), 2)
                          for cond in conditions}
                    for age in AGE_POINTS if acc[model]["no_retrieval"].get(age)
                }
            }
            for model in models
        },
    }
    out_file = BENCHMARK_DIR / "phenopacket_store_dx.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Saved: {out_file}")


N_DISTRACTORS = 4

if __name__ == "__main__":
    main()
