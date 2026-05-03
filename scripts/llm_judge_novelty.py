#!/usr/bin/env python3
"""
LLM-as-Judge — Novelty Verification
=====================================
Takes the 100 sampled novel diseases and asks DeepSeek V3 to judge:
"Does the cited evidence support the claimed onset age range?"

For each disease, sends the evidence text + PMID + claimed onset range,
and asks for a structured verdict: SUPPORTED / PARTIALLY_SUPPORTED /
NOT_SUPPORTED / UNVERIFIABLE.

Uses async calls for speed, with rate limiting to avoid 429s.

Output:
  data/benchmark/novelty_llm_judge.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"

# Load .env
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"  # DeepSeek V3

JUDGE_PROMPT = """You are a biomedical expert evaluating whether a temporal claim about disease onset is supported by the cited evidence.

Disease: {disease_name}
Claimed onset age range: {onset_min:.1f} to {onset_max:.1f} years
Temporal qualifier: {temporal_qualifier}
Relation: {relation}
Target entity: {target_name}

Evidence text from cited paper:
"{evidence_text}"

PMID(s): {pmids}

TASK: Does the evidence text support the claimed onset age range for this disease/phenotype?

Rate as one of:
- SUPPORTED: The evidence clearly states or strongly implies the claimed age range
- PARTIALLY_SUPPORTED: The evidence mentions onset timing but the extracted range is imprecise or slightly off
- NOT_SUPPORTED: The evidence does not support this onset claim, or discusses something unrelated
- UNVERIFIABLE: The evidence text is too vague or short to make a judgment

Respond with ONLY a JSON object:
{{"verdict": "SUPPORTED|PARTIALLY_SUPPORTED|NOT_SUPPORTED|UNVERIFIABLE", "reasoning": "one sentence explanation"}}"""


async def judge_single(client, disease_entry, semaphore):
    """Judge a single disease's onset claim."""
    async with semaphore:
        # Use the best onset detail (highest credibility, or first with evidence text)
        details = disease_entry.get("onset_details", [])
        if not details:
            return {
                "disease": disease_entry.get("disease_name", ""),
                "verdict": "UNVERIFIABLE",
                "reasoning": "No onset details available",
                "error": None,
            }

        # Pick the detail with the longest evidence text
        best = max(details, key=lambda d: len(d.get("evidence_text") or ""))
        evidence_text = best.get("evidence_text", "")
        if not evidence_text or len(evidence_text) < 10:
            return {
                "disease": disease_entry.get("disease_name", ""),
                "verdict": "UNVERIFIABLE",
                "reasoning": "Evidence text too short or missing",
                "error": None,
            }

        prompt = JUDGE_PROMPT.format(
            disease_name=disease_entry.get("disease_name", "unknown"),
            onset_min=disease_entry.get("onset_age_min_median", 0),
            onset_max=disease_entry.get("onset_age_max_median", 0),
            temporal_qualifier=best.get("temporal_qualifier", "none"),
            relation=best.get("relation", "unknown"),
            target_name=best.get("target_name", "unknown"),
            evidence_text=evidence_text[:500],  # Truncate long evidence
            pmids=", ".join(best.get("pmids", [])),
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

            # Parse JSON response
            # Handle cases where model wraps in markdown code block
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            result = json.loads(raw)
            return {
                "disease": disease_entry.get("disease_name", ""),
                "mondo_id": disease_entry.get("mondo_id", ""),
                "onset_range": [
                    disease_entry.get("onset_age_min_median", 0),
                    disease_entry.get("onset_age_max_median", 0),
                ],
                "verdict": result.get("verdict", "PARSE_ERROR"),
                "reasoning": result.get("reasoning", ""),
                "evidence_text_used": evidence_text[:200],
                "pmids_used": best.get("pmids", []),
                "error": None,
            }
        except json.JSONDecodeError:
            return {
                "disease": disease_entry.get("disease_name", ""),
                "verdict": "PARSE_ERROR",
                "reasoning": f"Could not parse LLM response: {raw[:100]}",
                "error": "json_parse",
            }
        except Exception as e:
            return {
                "disease": disease_entry.get("disease_name", ""),
                "verdict": "API_ERROR",
                "reasoning": str(e)[:100],
                "error": "api",
            }


async def run_all(samples):
    """Run LLM judge on all samples with concurrency control."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )

    # Limit concurrency to avoid rate limits
    semaphore = asyncio.Semaphore(5)

    tasks = [judge_single(client, s, semaphore) for s in samples]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions that slipped through
    clean_results = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            clean_results.append({
                "disease": samples[i].get("disease_name", ""),
                "verdict": "EXCEPTION",
                "reasoning": str(r)[:100],
                "error": "exception",
            })
        else:
            clean_results.append(r)

    return clean_results


def main():
    logger.info("=" * 75)
    logger.info("LLM-as-Judge — Novelty Verification (DeepSeek V3)")
    logger.info("=" * 75)

    if not DEEPSEEK_API_KEY:
        logger.error("DEEPSEEK_API_KEY not set in .env")
        return

    # Load novelty samples
    sample_file = BENCHMARK_DIR / "novelty_verification_sample.json"
    if not sample_file.exists():
        logger.error(f"Not found: {sample_file}")
        return

    with open(sample_file) as f:
        data = json.load(f)

    samples = data.get("samples", [])
    logger.info(f"  Loaded {len(samples)} samples")

    # Run LLM judge
    logger.info(f"  Running DeepSeek V3 judge (concurrency=5)...")
    start = time.time()
    results = asyncio.run(run_all(samples))
    elapsed = time.time() - start
    logger.info(f"  Completed in {elapsed:.1f}s")

    # Tally verdicts
    from collections import Counter
    verdicts = Counter(r["verdict"] for r in results)

    logger.info(f"\n{'=' * 75}")
    logger.info("VERDICT DISTRIBUTION")
    logger.info(f"{'=' * 75}")

    total = len(results)
    for verdict in ["SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED",
                    "UNVERIFIABLE", "PARSE_ERROR", "API_ERROR", "EXCEPTION"]:
        count = verdicts.get(verdict, 0)
        if count > 0:
            logger.info(f"  {verdict:<25} {count:>5} ({100*count/total:.1f}%)")

    supported = verdicts.get("SUPPORTED", 0)
    partial = verdicts.get("PARTIALLY_SUPPORTED", 0)
    not_supported = verdicts.get("NOT_SUPPORTED", 0)
    unverifiable = verdicts.get("UNVERIFIABLE", 0)
    errors = total - supported - partial - not_supported - unverifiable

    verifiable = supported + partial + not_supported
    if verifiable > 0:
        accuracy = 100 * (supported + partial) / verifiable
    else:
        accuracy = 0

    logger.info(f"\n  SUMMARY:")
    logger.info(f"  Verifiable samples: {verifiable}")
    logger.info(f"  Supported + Partial: {supported + partial} ({accuracy:.1f}% of verifiable)")
    logger.info(f"  Not supported: {not_supported}")
    logger.info(f"  Unverifiable/errors: {unverifiable + errors}")

    # Show NOT_SUPPORTED examples for audit
    not_supp = [r for r in results if r["verdict"] == "NOT_SUPPORTED"]
    if not_supp:
        logger.info(f"\n  NOT_SUPPORTED examples:")
        for r in not_supp[:5]:
            logger.info(f"    {r['disease'][:45]}")
            logger.info(f"      Range: {r.get('onset_range', '?')}")
            logger.info(f"      Reason: {r['reasoning'][:80]}")

    # Save
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "model": MODEL,
        "n_samples": total,
        "elapsed_seconds": round(elapsed, 1),
        "verdicts": dict(verdicts),
        "accuracy_of_verifiable": round(accuracy, 2),
        "supported_count": supported,
        "partial_count": partial,
        "not_supported_count": not_supported,
        "unverifiable_count": unverifiable,
        "results": results,
    }
    out_file = BENCHMARK_DIR / "novelty_llm_judge.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
