#!/usr/bin/env python3
"""
LLM-as-Judge Calibration — Gold-Standard-Matched Diseases
==========================================================
Runs the same DeepSeek V3 judge on 100 diseases WHERE WE KNOW THE ANSWER
from Orphadata. This tells us how harsh/lenient the judge is, independent
of extraction quality.

Two conditions:
  A) Judge TA triples against their own evidence_text (same as novelty script)
  B) Judge TA triples against Orphadata gold standard (ground truth check)

This calibrates the 42.9% novelty result by showing the judge's baseline
on "known-correct" diseases.

Output:
  data/benchmark/llm_judge_calibration.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation_sources"
CONFIG_DIR = PROJECT_ROOT / "config" / "diseases"

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

# Improved judge prompt — addresses methodology issues from the novelty run:
# 1. Explicitly allows parametric knowledge, not just evidence snippet
# 2. Asks about disease-level onset, not phenotype timing
# 3. More nuanced scoring rubric
JUDGE_PROMPT = """You are a biomedical expert evaluating whether a temporal claim about disease onset is accurate.

Disease: {disease_name}
Claimed onset age range: {onset_min:.1f} to {onset_max:.1f} years

Evidence text from cited paper (may be partial):
"{evidence_text}"

PMID(s): {pmids}

TASK: Is the claimed onset age range for this disease accurate? Use BOTH the evidence text AND your biomedical knowledge. The evidence text may be a short excerpt — if you recognize the disease, use what you know about it.

Rate as one of:
- SUPPORTED: The claimed age range is clinically accurate for this disease
- PARTIALLY_SUPPORTED: The range overlaps with known onset but is too wide, too narrow, or slightly shifted
- NOT_SUPPORTED: The claimed range is clearly wrong for this disease
- UNVERIFIABLE: You don't know enough about this disease to judge

Respond with ONLY a JSON object:
{{"verdict": "SUPPORTED|PARTIALLY_SUPPORTED|NOT_SUPPORTED|UNVERIFIABLE", "reasoning": "one sentence explanation"}}"""


def load_gold_matched_diseases(n=100):
    """Load diseases that match between TA and Orphadata, with onset triples."""
    import yaml
    import statistics

    # Load crosswalk and Orphadata
    with open(VALIDATION_DIR / "mondo_crosswalk.json") as f:
        xwalk = json.load(f)
    with open(VALIDATION_DIR / "orphadata_with_ids.json") as f:
        orpha_data = json.load(f)

    mondo_to_orpha = xwalk["mondo_to_orpha"]
    orpha_by_id = orpha_data["by_orpha_id"]

    # Load disease configs for MONDO ID mapping
    configs = {}
    for yf in CONFIG_DIR.glob("*.yaml"):
        try:
            with open(yf) as f:
                cfg = yaml.safe_load(f)
            if cfg and cfg.get("mondo_id"):
                padded = f"MONDO:{cfg['mondo_id'].split(':')[1].zfill(7)}"
                configs[yf.stem] = {
                    "mondo_id": padded,
                    "disease_name": cfg.get("disease_name", ""),
                }
        except Exception:
            pass

    # Find diseases with onset triples that match Orphadata
    matched = []
    for disease_dir in sorted(EXTRACTED_DIR.iterdir()):
        if not disease_dir.is_dir():
            continue
        vf = disease_dir / "validated_triples.jsonl"
        if not vf.exists():
            continue

        dir_name = disease_dir.name
        cfg = configs.get(dir_name)
        if not cfg:
            continue

        mondo_id = cfg["mondo_id"]

        # Check Orphadata match
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
            continue  # Skip uninformative gold standard

        # Load onset triples with evidence text
        onset_triples = []
        with open(vf) as f:
            for line in f:
                if not line.strip():
                    continue
                t = json.loads(line.strip())
                temporal = t.get("temporal") or {}
                omin = temporal.get("onset_age_min")
                if omin is None:
                    continue
                omax = temporal.get("onset_age_max")
                if omax is None:
                    omax = omin
                try:
                    omin = float(omin)
                    omax = float(omax)
                except (ValueError, TypeError):
                    continue
                if omin > 120 or omax > 120:
                    continue

                ev = t.get("evidence") or {}
                evidence_text = ev.get("evidence_text", "")
                pmids = ev.get("source_ids", [])
                if not evidence_text or len(evidence_text) < 20:
                    continue

                onset_triples.append({
                    "onset_min": omin,
                    "onset_max": omax,
                    "evidence_text": evidence_text,
                    "pmids": pmids,
                })

        if not onset_triples:
            continue

        # Compute median onset
        all_mins = [t["onset_min"] for t in onset_triples]
        all_maxs = [t["onset_max"] for t in onset_triples]
        med_min = statistics.median(all_mins)
        med_max = statistics.median(all_maxs)

        # Pick the triple with longest evidence text for judging
        best_triple = max(onset_triples, key=lambda t: len(t["evidence_text"]))

        # Check if TA is correct vs gold standard (contained)
        is_contained = med_min >= ref_min and med_max <= ref_max

        matched.append({
            "disease_name": cfg["disease_name"],
            "mondo_id": mondo_id,
            "dir_name": dir_name,
            "ta_onset_min": med_min,
            "ta_onset_max": med_max,
            "gold_min": ref_min,
            "gold_max": ref_max,
            "is_contained": is_contained,
            "n_onset_triples": len(onset_triples),
            "evidence_text": best_triple["evidence_text"],
            "pmids": best_triple["pmids"],
        })

    logger.info(f"Found {len(matched)} gold-matched diseases with onset triples + evidence")
    logger.info(f"  Contained (TA correct vs Orphadata): {sum(1 for m in matched if m['is_contained'])}")

    # Sample 100: 50 correct + 50 incorrect (or all if fewer)
    correct = [m for m in matched if m["is_contained"]]
    incorrect = [m for m in matched if not m["is_contained"]]

    random.seed(42)
    n_correct = min(50, len(correct))
    n_incorrect = min(50, len(incorrect))
    # Fill remaining from whichever pool has more
    remaining = n - n_correct - n_incorrect
    if remaining > 0:
        if len(correct) > n_correct:
            n_correct += min(remaining, len(correct) - n_correct)
            remaining = n - n_correct - n_incorrect
        if remaining > 0 and len(incorrect) > n_incorrect:
            n_incorrect += min(remaining, len(incorrect) - n_incorrect)

    sample = random.sample(correct, n_correct) + random.sample(incorrect, n_incorrect)
    random.shuffle(sample)

    logger.info(f"  Sampled {len(sample)}: {n_correct} correct, {n_incorrect} incorrect vs gold standard")
    return sample


async def judge_single(client, entry, semaphore):
    """Judge a single disease's onset claim."""
    async with semaphore:
        prompt = JUDGE_PROMPT.format(
            disease_name=entry["disease_name"],
            onset_min=entry["ta_onset_min"],
            onset_max=entry["ta_onset_max"],
            evidence_text=entry["evidence_text"][:800],  # More context than before
            pmids=", ".join(entry["pmids"][:3]),
        )

        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
                timeout=30,
            )
            raw = response.choices[0].message.content.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            result = json.loads(raw)
            return {
                "disease": entry["disease_name"],
                "mondo_id": entry["mondo_id"],
                "ta_onset": [entry["ta_onset_min"], entry["ta_onset_max"]],
                "gold_onset": [entry["gold_min"], entry["gold_max"]],
                "is_contained": entry["is_contained"],
                "verdict": result.get("verdict", "PARSE_ERROR"),
                "reasoning": result.get("reasoning", ""),
                "error": None,
            }
        except json.JSONDecodeError:
            return {
                "disease": entry["disease_name"],
                "verdict": "PARSE_ERROR",
                "reasoning": f"Could not parse: {raw[:100]}",
                "error": "json_parse",
            }
        except Exception as e:
            return {
                "disease": entry["disease_name"],
                "verdict": "API_ERROR",
                "reasoning": str(e)[:100],
                "error": "api",
            }


async def run_all(samples):
    """Run LLM judge on all samples."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )

    semaphore = asyncio.Semaphore(5)
    tasks = [judge_single(client, s, semaphore) for s in samples]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    clean = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            clean.append({
                "disease": samples[i]["disease_name"],
                "verdict": "EXCEPTION",
                "reasoning": str(r)[:100],
                "error": "exception",
            })
        else:
            clean.append(r)
    return clean


def main():
    logger.info("=" * 75)
    logger.info("LLM-as-Judge Calibration — Gold-Standard-Matched Diseases")
    logger.info("=" * 75)

    if not DEEPSEEK_API_KEY:
        logger.error("DEEPSEEK_API_KEY not set in .env")
        return

    # Load gold-matched diseases
    samples = load_gold_matched_diseases(n=100)
    if not samples:
        logger.error("No gold-matched diseases found")
        return

    # Run judge
    logger.info(f"\n  Running DeepSeek V3 judge on {len(samples)} gold-matched diseases...")
    start = time.time()
    results = asyncio.run(run_all(samples))
    elapsed = time.time() - start
    logger.info(f"  Completed in {elapsed:.1f}s")

    # Tally verdicts
    verdicts = Counter(r["verdict"] for r in results)

    logger.info(f"\n{'=' * 75}")
    logger.info("OVERALL VERDICT DISTRIBUTION")
    logger.info(f"{'=' * 75}")
    total = len(results)
    for v in ["SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED", "UNVERIFIABLE"]:
        count = verdicts.get(v, 0)
        if count > 0:
            logger.info(f"  {v:<25} {count:>5} ({100*count/total:.1f}%)")

    supported = verdicts.get("SUPPORTED", 0) + verdicts.get("PARTIALLY_SUPPORTED", 0)
    verifiable = supported + verdicts.get("NOT_SUPPORTED", 0)
    accuracy = 100 * supported / max(1, verifiable)
    logger.info(f"\n  Overall accuracy (S+P / verifiable): {accuracy:.1f}%")

    # Break down by gold standard correctness
    logger.info(f"\n{'=' * 75}")
    logger.info("BREAKDOWN BY GOLD STANDARD CORRECTNESS")
    logger.info(f"{'=' * 75}")

    for label, condition in [("TA CORRECT (contained in gold)", True), ("TA INCORRECT (not contained)", False)]:
        subset = [r for r in results if r.get("is_contained") == condition]
        if not subset:
            continue
        sub_verdicts = Counter(r["verdict"] for r in subset)
        sub_total = len(subset)
        sub_supported = sub_verdicts.get("SUPPORTED", 0) + sub_verdicts.get("PARTIALLY_SUPPORTED", 0)
        sub_verifiable = sub_supported + sub_verdicts.get("NOT_SUPPORTED", 0)
        sub_acc = 100 * sub_supported / max(1, sub_verifiable)

        logger.info(f"\n  {label} (n={sub_total}):")
        for v in ["SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED", "UNVERIFIABLE"]:
            count = sub_verdicts.get(v, 0)
            if count > 0:
                logger.info(f"    {v:<25} {count:>5} ({100*count/sub_total:.1f}%)")
        logger.info(f"    Judge accuracy: {sub_acc:.1f}%")

    # Save
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "model": MODEL,
        "n_samples": total,
        "elapsed_seconds": round(elapsed, 1),
        "verdicts": dict(verdicts),
        "overall_accuracy": round(accuracy, 2),
        "breakdown": {
            "correct_vs_gold": {
                "n": sum(1 for r in results if r.get("is_contained")),
                "judge_supported": sum(1 for r in results if r.get("is_contained") and r["verdict"] in ("SUPPORTED", "PARTIALLY_SUPPORTED")),
            },
            "incorrect_vs_gold": {
                "n": sum(1 for r in results if r.get("is_contained") is False),
                "judge_supported": sum(1 for r in results if r.get("is_contained") is False and r["verdict"] in ("SUPPORTED", "PARTIALLY_SUPPORTED")),
            },
        },
        "results": results,
    }
    out_file = BENCHMARK_DIR / "llm_judge_calibration.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
