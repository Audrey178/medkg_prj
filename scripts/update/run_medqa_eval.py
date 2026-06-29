#!/usr/bin/env python3
"""
MEDQA Evaluation — ChronoMedKG KG-RAG Pipeline
===============================================
Evaluates QAPipeline on MEDQA (USMLE-style 4/5-option MCQs).

MEDQA JSONL format expected
---------------------------
  {"question": "...", "options": {"A": "...", "B": "...", "C": "...", "D": "..."}, "answer": "B"}

  Also accepted:
  {"question": ..., "options": {...}, "answer_idx": "B"}
  {"question": ..., "options": {...}, "correct_answer": "B"}

Public dataset: https://github.com/jind11/MedQA  (data_clean/questions/US/)

Metrics
-------
  accuracy   : exact letter match (A/B/C/D/E)
  95% CI     : bootstrap confidence interval

Usage
-----
  python -m scripts.run_medqa_eval --data data/medqa/test.jsonl --n 100
  python -m scripts.run_medqa_eval --data data/medqa/test.jsonl --mode llm_only
  python -m scripts.run_medqa_eval --data data/medqa/test.jsonl --mode both --n 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_env = PROJECT_ROOT / ".env"
if _env.exists():
    for _line in open(_env):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            if _v.strip():
                os.environ[_k.strip()] = _v.strip()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("medqa_eval")

RESULTS_DIR = PROJECT_ROOT / "data" / "benchmark" / "medqa_results"
RANDOM_SEED = 42


# ============================================================================
# DATA LOADING
# ============================================================================

def load_medqa(path: Path, n: int | None, seed: int = RANDOM_SEED) -> list[dict]:
    """Load MEDQA questions from a JSONL file."""
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                q = json.loads(line)
            except json.JSONDecodeError:
                continue
            answer = q.get("answer_idx") or q.get("correct_answer") or q.get("answer", "")
            if not answer:
                continue
            answer = str(answer).strip().upper()
            if len(answer) > 1:
                answer = answer[0]
            if answer not in "ABCDE":
                continue
            questions.append({
                "question": q["question"],
                "options": q["options"],
                "answer": answer,
            })

    if n is not None and n < len(questions):
        rng = random.Random(seed)
        questions = rng.sample(questions, n)

    logger.info("Loaded %d MEDQA questions from %s", len(questions), path)
    return questions


# ============================================================================
# SCORING
# ============================================================================

def parse_answer(raw: str) -> str:
    """Extract answer letter from LLM output."""
    raw = raw.strip().upper()
    if raw and raw[0] in "ABCDE":
        return raw[0]
    m = re.search(r'\b([A-E])\b', raw)
    if m:
        return m.group(1)
    return raw[:1] if raw else "?"


def bootstrap_ci(scores: list[float], n_boot: int = 1000, seed: int = RANDOM_SEED):
    if not scores:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(seed)
    arr = np.array(scores)
    means = [np.mean(rng.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    return float(np.mean(arr)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def run_eval(args: argparse.Namespace) -> None:
    from agents.qa_inference import QAPipeline

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_medqa(Path(args.data), args.n)
    if not questions:
        logger.error("No questions loaded.")
        sys.exit(1)

    pipeline = QAPipeline()
    modes = [args.mode] if args.mode != "both" else ["llm_only", "kg_rag"]

    scores: dict[str, list[float]] = {m: [] for m in modes}
    kg_hits = 0
    details: list[dict] = []

    for i, q in enumerate(questions):
        q_details: dict = {
            "idx": i + 1,
            "question": q["question"],
            "options": q["options"],
            "ground_truth": q["answer"],
            "modes": {},
        }

        for mode in modes:
            try:
                raw = pipeline.run(
                    q["question"],
                    benchmark_type="medqa",
                    mode=mode,
                    options={"choices": q["options"]},
                )
                answer = raw.get("answer") or {}
                predicted_raw = answer.get("answer") or ""
                reasoning_ans = raw.get("reasoning_answer", "")
                explanation = answer.get("explanation", "")
                kg_coverage = raw.get("kg_coverage", False)
                if mode == "kg_rag" and kg_coverage:
                    kg_hits += 1
            except Exception as exc:
                logger.warning("[Q%d] error: %s", i, exc)
                predicted_raw = ""
                reasoning_ans= ""
                explanation = f"error: {exc}"
                kg_coverage = False

            predicted_letter = parse_answer(str(predicted_raw))
            correct = float(predicted_letter == q["answer"])
            scores[mode].append(correct)

            q_details["modes"][mode] = {
                "predicted": predicted_letter,
                "reasoning": reasoning_ans,
                "explanation": explanation,
                "kg_coverage": kg_coverage,
                "score": correct,
            }

            if args.verbose:
                logger.info(
                    "[Q%03d] mode=%-8s correct=%s pred=%s gold=%s\n"
                    "         Q: %s",
                    i + 1, mode, "OK" if correct else "WRONG",
                    predicted_letter, q["answer"],
                    q["question"][:120],
                )

        details.append(q_details)

        if (i + 1) % 10 == 0:
            logger.info("Progress: %d/%d", i + 1, len(questions))

    # ---- Report ----
    print("\n" + "=" * 70)
    print("MEDQA EVALUATION — ChronoMedKG KG-RAG Pipeline")
    print("=" * 70)
    print(f"Questions : {len(questions)}")
    print(f"Modes     : {', '.join(modes)}")
    print()

    output_results: dict = {}
    for mode in modes:
        sc_list = scores[mode]
        mean, lo, hi = bootstrap_ci(sc_list)
        print(f"Mode: {mode}")
        print(f"  {'Metric':<20} {'Score':>8}  {'95% CI':>20}  {'N':>5}")
        print(f"  {'-'*20} {'-'*8}  {'-'*20}  {'-'*5}")
        print(f"  {'Accuracy':<20} {mean:>8.3f}  [{lo:.3f} – {hi:.3f}]  {len(sc_list):>5}")
        print()
        output_results[mode] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(sc_list)}

    if "llm_only" in output_results and "kg_rag" in output_results:
        delta = output_results["kg_rag"]["mean"] - output_results["llm_only"]["mean"]
        print(f"Delta (kg_rag − llm_only): {delta:+.3f}")
        print()

    if "kg_rag" in modes:
        print(f"KG hit rate: {kg_hits / len(questions):.1%} ({kg_hits}/{len(questions)})")
        print(
            "  Note: ChronoMedKG covers rare genetic diseases. Low hit rate on MEDQA "
            "(USMLE) is expected."
        )

    out_path = RESULTS_DIR / f"medqa_eval_{len(questions)}q_{args.mode}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "n_questions": len(questions),
                "mode": args.mode,
                "kg_hit_rate": kg_hits / len(questions) if "kg_rag" in modes else None,
                "results": output_results,
                "details": details,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
        json.dump(details)
    logger.info("Results saved to %s", out_path)


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ChronoMedKG KG-RAG pipeline on MEDQA (USMLE-style MCQ)"
    )
    parser.add_argument("--data", required=True,
                        help="Path to MEDQA JSONL file")
    parser.add_argument("--n", type=int, default=None,
                        help="Questions to sample (default: all)")
    parser.add_argument("--mode", choices=["kg_rag", "llm_only", "kg_only", "both"],
                        default="both",
                        help="Pipeline mode (default: both llm_only and kg_rag)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-question predictions")
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
