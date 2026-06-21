#!/usr/bin/env python3
"""
BioASQ Evaluation — ChronoMedKG KG-RAG Pipeline
================================================
Evaluates QAPipeline on BioASQ question types: yes/no, factoid, list, summary.

BioASQ JSON format expected
---------------------------
  {
    "questions": [
      {
        "id": "...",
        "body": "Is X associated with Y?",
        "type": "yesno",          # yesno | factoid | list | summary
        "ideal_answer": ["Yes, ..."],
        "exact_answer": "yes"     # yesno: "yes"/"no"
                                  # factoid: [["answer", "synonym"]]
                                  # list: [["item1"], ["item2", "syn2"]]
                                  # summary: (no exact_answer, use ideal_answer)
      }
    ]
  }

Official dataset: http://participants-area.bioasq.org/

Metrics
-------
  yes/no   : accuracy
  factoid  : lenient accuracy (match any synonym, case-insensitive)
  list     : macro-averaged entity F1 per question
  summary  : ROUGE-2 F1 (no external dependencies)

Usage
-----
  python -m scripts.run_bioasq_eval --data data/bioasq/BioASQ-task12b-testset4.json --n 200
  python -m scripts.run_bioasq_eval --data data/bioasq/BioASQ-task12b-testset4.json --mode llm_only
  python -m scripts.run_bioasq_eval --data data/bioasq/BioASQ-task12b-testset4.json --type yesno factoid
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
from collections import defaultdict
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
logger = logging.getLogger("bioasq_eval")

RESULTS_DIR = PROJECT_ROOT / "data" / "benchmark" / "bioasq_results"
RANDOM_SEED = 42

_TYPE_MAP = {"yesno": "yes_no", "factoid": "factoid", "list": "list", "summary": "summary"}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_bioasq(path: Path, n: int | None, types: list[str], seed: int = RANDOM_SEED) -> list[dict]:
    with open(path) as f:
        raw = json.load(f)

    questions_raw = raw.get("questions", raw) if isinstance(raw, dict) else raw
    questions = []
    for q in questions_raw:
        qtype = q.get("type", "")
        if types and qtype not in types:
            continue
        questions.append({
            "id": q.get("id", ""),
            "body": q.get("body", q.get("question", "")),
            "type": qtype,
            "pipeline_type": _TYPE_MAP.get(qtype, "factoid"),
            "ideal_answer": q.get("ideal_answer", []),
            "exact_answer": q.get("exact_answer", ""),
        })

    if n is not None and n < len(questions):
        rng = random.Random(seed)
        questions = rng.sample(questions, n)

    dist = defaultdict(int)
    for q in questions:
        dist[q["type"]] += 1
    logger.info("Loaded %d questions: %s", len(questions), dict(dist))
    return questions


# ============================================================================
# SCORING
# ============================================================================

def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def score_yesno(predicted: str, gold: str) -> bool:
    return _normalize(predicted) in ("yes", "no") and \
           _normalize(predicted) == _normalize(gold)


def score_factoid(predicted: str, gold_synonyms: list[list[str]]) -> bool:
    """Lenient match: predicted matches any synonym of any gold answer."""
    pred_norm = _normalize(predicted)
    for synonym_group in gold_synonyms:
        for syn in synonym_group:
            if pred_norm == _normalize(syn):
                return True
            # Partial containment for long phrases
            if len(pred_norm) > 5 and (pred_norm in _normalize(syn) or _normalize(syn) in pred_norm):
                return True
    return False


def _token_f1(predicted_tokens: set[str], gold_tokens: set[str]) -> float:
    if not predicted_tokens or not gold_tokens:
        return 0.0
    precision = len(predicted_tokens & gold_tokens) / len(predicted_tokens)
    recall = len(predicted_tokens & gold_tokens) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_list(predicted: list[str], gold_synonym_groups: list[list[str]]) -> float:
    """
    BioASQ-style list F1: for each gold answer, check if any predicted item matches.
    Returns macro F1 averaged over gold answers.
    """
    if not gold_synonym_groups:
        return 0.0
    pred_norms = {_normalize(p) for p in predicted}

    gold_hits = 0
    for syn_group in gold_synonym_groups:
        gold_norms = {_normalize(s) for s in syn_group}
        if pred_norms & gold_norms:
            gold_hits += 1

    recall = gold_hits / len(gold_synonym_groups)
    precision = gold_hits / len(predicted) if predicted else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge2_f1(predicted: str, gold: str) -> float:
    """ROUGE-2 F1 (no external library)."""
    def bigrams(text: str) -> set[tuple]:
        tokens = text.lower().split()
        return set(zip(tokens, tokens[1:]))

    pred_bg = bigrams(predicted)
    gold_bg = bigrams(gold)
    if not pred_bg or not gold_bg:
        return 0.0
    overlap = len(pred_bg & gold_bg)
    precision = overlap / len(pred_bg)
    recall = overlap / len(gold_bg)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


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
    questions = load_bioasq(Path(args.data), args.n, args.type)
    if not questions:
        logger.error("No questions loaded.")
        sys.exit(1)

    pipeline = QAPipeline()
    modes = [args.mode] if args.mode != "both" else ["llm_only", "kg_rag"]

    # Per-type scores: {mode: {type: [scores]}}
    scores: dict[str, dict[str, list[float]]] = {
        m: defaultdict(list) for m in modes
    }
    kg_hits = 0

    for i, q in enumerate(questions):
        for mode in modes:
            try:
                raw = pipeline.run(
                    q["body"],
                    benchmark_type="bioasq",
                    mode=mode,
                )
                answer = raw.get("answer") or {}
                predicted_raw = answer.get("answer", "")
                if mode == "kg_rag" and raw.get("kg_coverage"):
                    kg_hits += 1
            except Exception as exc:
                logger.warning("[Q%d] error: %s", i, exc)
                predicted_raw = ""
                answer = {}

            qtype = q["type"]
            exact = q["exact_answer"]
            ideal = q["ideal_answer"]

            if qtype == "yesno":
                sc = float(score_yesno(str(predicted_raw), str(exact)))

            elif qtype == "factoid":
                gold_syns = exact if isinstance(exact, list) else [[str(exact)]]
                sc = float(score_factoid(str(predicted_raw), gold_syns))

            elif qtype == "list":
                pred_list = predicted_raw if isinstance(predicted_raw, list) else [str(predicted_raw)]
                gold_syns = exact if isinstance(exact, list) else [[str(exact)]]
                sc = score_list(pred_list, gold_syns)

            elif qtype == "summary":
                gold_text = ideal[0] if isinstance(ideal, list) and ideal else str(ideal)
                key_points = answer.get("key_points", [])
                pred_text = str(predicted_raw) + " " + " ".join(key_points)
                sc = rouge2_f1(pred_text, gold_text)

            else:
                sc = 0.0

            scores[mode][qtype].append(sc)

            if args.verbose:
                logger.info("[Q%03d] type=%s mode=%s score=%.3f", i + 1, qtype, mode, sc)

        if (i + 1) % 10 == 0:
            logger.info("Progress: %d/%d", i + 1, len(questions))

    # ---- Report ----
    print("\n" + "=" * 70)
    print("BIOASQ EVALUATION — ChronoMedKG KG-RAG Pipeline")
    print("=" * 70)
    print(f"Questions : {len(questions)}")
    print(f"Modes     : {', '.join(modes)}")
    print()

    metric_labels = {
        "yesno": "Accuracy",
        "factoid": "Lenient Accuracy",
        "list": "F1",
        "summary": "ROUGE-2 F1",
    }

    output_results: dict = {}
    for mode in modes:
        print(f"Mode: {mode}")
        print(f"  {'Type':<12} {'Metric':<20} {'Score':>8}  {'95% CI':>20}  {'N':>5}")
        print(f"  {'-'*12} {'-'*20} {'-'*8}  {'-'*20}  {'-'*5}")
        output_results[mode] = {}
        for qtype in ["yesno", "factoid", "list", "summary"]:
            sc_list = scores[mode].get(qtype, [])
            if not sc_list:
                continue
            mean, lo, hi = bootstrap_ci(sc_list)
            label = metric_labels.get(qtype, "Score")
            print(f"  {qtype:<12} {label:<20} {mean:>8.3f}  [{lo:.3f} – {hi:.3f}]  {len(sc_list):>5}")
            output_results[mode][qtype] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(sc_list)}

        all_scores = [s for sl in scores[mode].values() for s in sl]
        if all_scores:
            overall = np.mean(all_scores)
            print(f"  {'OVERALL':<12} {'(macro avg)':<20} {overall:>8.3f}")
        print()

    if "kg_rag" in modes:
        print(f"KG hit rate: {kg_hits / len(questions):.1%} ({kg_hits}/{len(questions)})")

    out_path = RESULTS_DIR / f"bioasq_eval_{len(questions)}q_{args.mode}.json"
    with open(out_path, "w") as f:
        json.dump({"n_questions": len(questions), "mode": args.mode,
                   "kg_hit_rate": kg_hits / len(questions) if "kg_rag" in modes else None,
                   "results": output_results}, f, indent=2)
    logger.info("Results saved to %s", out_path)


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ChronoMedKG KG-RAG pipeline on BioASQ"
    )
    parser.add_argument("--data", required=True,
                        help="Path to BioASQ JSON file")
    parser.add_argument("--n", type=int, default=None,
                        help="Questions to sample (default: all)")
    parser.add_argument("--type", nargs="+",
                        choices=["yesno", "factoid", "list", "summary"],
                        default=[],
                        help="Question types to evaluate (default: all)")
    parser.add_argument("--mode", choices=["kg_rag", "llm_only", "kg_only", "both"],
                        default="both",
                        help="Pipeline mode (default: both llm_only and kg_rag)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-question scores")
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
