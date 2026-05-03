#!/usr/bin/env python3
"""
Multi-judge novelty re-verification (v2 protocol).

Re-scores the 100-sample novelty file with THREE LLM judges
(DeepSeek V3, GPT-4o-mini, Claude 3 Haiku) using a citation-aware
v2 prompt that requires the judge to first extract the onset-timing
claim from the evidence text before rendering a verdict.

Writes: data/benchmark/novelty_multi_judge.json
"""
from __future__ import annotations
import asyncio, json, logging, os, time
from pathlib import Path
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

JUDGE_PROMPT_V2 = """You are a biomedical evidence adjudicator. Your task is to decide whether a piece of cited evidence supports a specific claim about disease-onset age.

Disease: {disease_name}
Claimed onset age range: {onset_min:.2f} to {onset_max:.2f} years
Temporal qualifier: {temporal_qualifier}
Relation type: {relation}
Target entity: {target_name}

Evidence text (from PMID {pmids}):
\"\"\"{evidence_text}\"\"\"

Follow this procedure:

STEP 1 — Quote extraction: identify EXACTLY the clause(s) in the evidence text that mention timing, age, onset, or stage. Quote them verbatim. If the evidence does not mention timing at all, say "NO TIMING CLAUSE".

STEP 2 — Interpretation: translate the quoted clause(s) into an age range in years. Use these conventions:
  prenatal/antenatal = 0-0
  neonatal/at birth/congenital = 0-0.08
  infancy/infantile = 0.08-2
  early childhood = 1-5
  childhood = 1-11
  adolescence = 10-18
  adulthood = 18-65
If the clause is numeric (e.g. "2 to 10 years"), use the numbers directly.

STEP 3 — Comparison: compare the claimed range [{onset_min:.2f}, {onset_max:.2f}] to the evidence-implied range. Decide:
  - SUPPORTED: the ranges overlap meaningfully, or the evidence text numerically states the claimed range
  - PARTIALLY_SUPPORTED: the ranges are in the same clinical era but do not overlap (e.g. evidence says "childhood", claim is "5-10y")
  - NOT_SUPPORTED: the ranges are in different clinical eras, OR the evidence text does not mention timing
  - UNVERIFIABLE: evidence is too short/vague to extract any timing claim

Respond with ONLY this JSON object (no markdown, no prose outside it):
{{"quote": "<verbatim timing clause or NO TIMING CLAUSE>", "evidence_range_years": "<e.g. 0-2 or N/A>", "verdict": "SUPPORTED|PARTIALLY_SUPPORTED|NOT_SUPPORTED|UNVERIFIABLE", "reasoning": "<one sentence>"}}"""


TIMING_KEYWORDS = [
    "year", "month", "week", "day", "age", "onset", "birth", "congenital",
    "neonatal", "infancy", "infant", "childhood", "adolescen", "adult", "elderly",
    "prenatal", "antenatal", "juvenile", "toddler", "senior", "young",
]


def _has_timing(text: str) -> bool:
    if not text: return False
    low = text.lower()
    return any(kw in low for kw in TIMING_KEYWORDS)


def pick_best_detail(disease_entry: dict) -> dict | None:
    """Prefer details whose evidence text contains timing keywords; tiebreak on length."""
    details = disease_entry.get("onset_details") or []
    details = [d for d in details if d.get("evidence_text") and len(d["evidence_text"]) >= 10]
    if not details:
        return None
    # Prefer details with timing keyword AND numeric onset range (both min and max set)
    timing_details = [d for d in details if _has_timing(d.get("evidence_text", ""))]
    pool = timing_details if timing_details else details
    return max(pool, key=lambda d: len(d.get("evidence_text") or ""))


def format_prompt(disease_entry: dict, detail: dict) -> str:
    # Compare the triple's OWN onset range, not the disease-level aggregated median.
    return JUDGE_PROMPT_V2.format(
        disease_name=disease_entry.get("disease_name", "unknown"),
        onset_min=float(detail.get("onset_age_min", 0.0) or 0.0),
        onset_max=float(detail.get("onset_age_max", 0.0) or 0.0),
        temporal_qualifier=detail.get("temporal_qualifier", "none"),
        relation=detail.get("relation", "unknown"),
        target_name=detail.get("target_name", "unknown"),
        evidence_text=(detail.get("evidence_text") or "")[:800],
        pmids=", ".join(detail.get("pmids", [])),
    )


def _extract_json(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        # strip code fences
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()
    # tolerate leading prose: find first { and last }
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        t = t[start:end+1]
    return json.loads(t)


async def judge_deepseek(prompt: str) -> dict:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    r = await client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=400, timeout=60,
    )
    return _extract_json(r.choices[0].message.content)


async def judge_openai(prompt: str) -> dict:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    r = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=400, timeout=60,
    )
    return _extract_json(r.choices[0].message.content)


async def judge_anthropic(prompt: str) -> dict:
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    r = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_json(r.content[0].text)


JUDGES = [
    ("deepseek-v3", judge_deepseek),
    ("gpt-4o-mini", judge_openai),
    ("claude-haiku-4-5", judge_anthropic),
]


async def judge_one(disease_entry: dict, detail: dict, sem: asyncio.Semaphore) -> dict:
    results = {}
    prompt = format_prompt(disease_entry, detail)
    async with sem:
        for name, fn in JUDGES:
            try:
                results[name] = await fn(prompt)
            except json.JSONDecodeError as e:
                results[name] = {"verdict": "PARSE_ERROR", "reasoning": str(e)[:100]}
            except Exception as e:
                results[name] = {"verdict": "API_ERROR", "reasoning": str(e)[:200]}
    return {
        "mondo_id": disease_entry.get("mondo_id"),
        "disease": disease_entry.get("disease_name"),
        "claim_range": [disease_entry.get("onset_age_min_median", 0.0),
                        disease_entry.get("onset_age_max_median", 0.0)],
        "detail_used": {k: detail.get(k) for k in ("relation", "target_name", "temporal_qualifier", "pmids", "evidence_text", "credibility_score")},
        "judgments": results,
    }


def majority_vote(judgments: dict) -> str:
    votes = [j.get("verdict", "ERROR") for j in judgments.values()]
    c = Counter(votes)
    top, n = c.most_common(1)[0]
    if n >= 2:
        return top
    return "DISAGREE"


async def run(samples: list):
    sem = asyncio.Semaphore(4)
    tasks = []
    for s in samples:
        d = pick_best_detail(s)
        if d is None:
            tasks.append(asyncio.sleep(0, result={
                "mondo_id": s.get("mondo_id"),
                "disease": s.get("disease_name"),
                "judgments": {name: {"verdict": "UNVERIFIABLE", "reasoning": "no evidence text"} for name, _ in JUDGES},
                "claim_range": [s.get("onset_age_min_median", 0), s.get("onset_age_max_median", 0)],
                "detail_used": None,
            }))
        else:
            tasks.append(judge_one(s, d, sem))
    return await asyncio.gather(*tasks)


def main():
    if not DEEPSEEK_API_KEY or not OPENAI_API_KEY or not ANTHROPIC_API_KEY:
        logger.error("Missing API key(s); aborting")
        return
    with open(BENCHMARK_DIR / "novelty_verification_sample.json") as f:
        data = json.load(f)
    samples = data["samples"]
    logger.info(f"Loaded {len(samples)} samples; running 3 judges with v2 protocol")
    start = time.time()
    results = asyncio.run(run(samples))
    elapsed = time.time() - start
    logger.info(f"Finished in {elapsed:.1f}s")

    # Per-judge tallies
    per_judge = defaultdict(Counter)
    for r in results:
        for name, j in r["judgments"].items():
            per_judge[name][j.get("verdict", "ERROR")] += 1

    # Majority + agreement
    for r in results:
        r["majority_verdict"] = majority_vote(r["judgments"])

    maj = Counter(r["majority_verdict"] for r in results)
    # Agreement: fraction where all 3 judges issued the same verdict (among non-error rows)
    n_all_agree = 0
    n_2_of_3 = 0
    n_disagree = 0
    err_rows = 0
    for r in results:
        vs = [j.get("verdict") for j in r["judgments"].values()]
        if any(v in ("PARSE_ERROR", "API_ERROR") for v in vs):
            err_rows += 1
            continue
        c = Counter(vs)
        if len(c) == 1: n_all_agree += 1
        elif c.most_common(1)[0][1] == 2: n_2_of_3 += 1
        else: n_disagree += 1

    logger.info("\n=== PER-JUDGE VERDICTS ===")
    for name in [n for n, _ in JUDGES]:
        c = per_judge[name]
        logger.info(f"  {name}:")
        for v in ["SUPPORTED","PARTIALLY_SUPPORTED","NOT_SUPPORTED","UNVERIFIABLE","PARSE_ERROR","API_ERROR"]:
            if c.get(v): logger.info(f"    {v}: {c[v]}")

    logger.info("\n=== MAJORITY VERDICT (2-of-3) ===")
    for v in ["SUPPORTED","PARTIALLY_SUPPORTED","NOT_SUPPORTED","UNVERIFIABLE","DISAGREE"]:
        if maj.get(v): logger.info(f"  {v}: {maj[v]} ({maj[v]/len(results)*100:.1f}%)")

    logger.info("\n=== AGREEMENT ===")
    logger.info(f"  All 3 agree:  {n_all_agree}")
    logger.info(f"  2-of-3 agree: {n_2_of_3}")
    logger.info(f"  3-way split:  {n_disagree}")
    logger.info(f"  Rows w/ error:{err_rows}")

    supp = maj.get("SUPPORTED", 0) + maj.get("PARTIALLY_SUPPORTED", 0)
    total_verifiable = sum(maj[v] for v in ("SUPPORTED","PARTIALLY_SUPPORTED","NOT_SUPPORTED"))
    acc = supp / total_verifiable * 100 if total_verifiable else 0.0
    logger.info(f"\n  Supported+Partial / Verifiable = {supp}/{total_verifiable} = {acc:.1f}%")

    out = {
        "protocol_version": "v2_citation_aware_cot_per_triple",
        "judges": [n for n, _ in JUDGES],
        "n_samples": len(results),
        "elapsed_seconds": round(elapsed, 1),
        "per_judge_verdicts": {name: dict(c) for name, c in per_judge.items()},
        "majority_verdicts": dict(maj),
        "agreement": {"all_3": n_all_agree, "2_of_3": n_2_of_3, "3_way_split": n_disagree, "rows_with_error": err_rows},
        "accuracy_of_verifiable_majority": round(acc, 2),
        "results": results,
    }
    out_path = BENCHMARK_DIR / "novelty_multi_judge_v2.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
