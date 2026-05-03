#!/usr/bin/env python3
"""
TQA Benchmark Cleanup: v1 -> v2
================================
Filters garbage questions from ChronoMedKG-TQA v1:
1. Same min/max age ("Between age 5 and 5") -> drop
2. Non-temporal treatment answers -> drop
3. Tautological stage questions -> drop
4. Deduplication by (disease_id, type, target_entity)
5. Add hard difficulty tier

Usage:
    python3 scripts/cleanup_tqa_benchmark.py
    python3 scripts/cleanup_tqa_benchmark.py --strict   # Tighter filtering for 8-10K target
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"

# ──────────────────────────────────────────────────────────────────────
# Temporal keywords for treatment answer validation
# ──────────────────────────────────────────────────────────────────────

TEMPORAL_KEYWORDS = {
    "year", "month", "week", "day", "hour",
    "age", "onset", "early", "late", "first", "initial",
    "chronic", "acute", "stage", "phase",
    "duration", "period", "time", "timeline",
    "before", "after", "during", "since", "until",
    "19",  # catches 1990s, 1980s, etc
    "20",  # catches 2000s, 2010s, etc
    "neonatal", "infantile", "childhood", "adolescent",
    "adult", "elderly", "congenital", "prenatal", "postnatal",
    "progressive", "rapid", "slow", "gradual",
    "approved", "introduced", "discovered", "developed",
}


def has_temporal_content(answer: str) -> bool:
    """Check if an answer contains any temporal keywords."""
    answer_lower = answer.lower()
    return any(kw in answer_lower for kw in TEMPORAL_KEYWORDS)


def is_same_min_max_age(answer: str) -> bool:
    """Check if answer is 'Between age X and X.' where X == X."""
    m = re.search(r'[Bb]etween age ([\d.]+) and ([\d.]+?)\.?\s*$', answer)
    if m:
        try:
            return float(m.group(1)) == float(m.group(2))
        except ValueError:
            return m.group(1) == m.group(2)
    return False


def is_tautological_stage(question: str, answer: str) -> bool:
    """Check if a stage_phenotype answer is tautological (obvious from question)."""
    answer_lower = answer.lower().strip()

    # Short generic answers like "During the acute stage."
    if len(answer_lower) < 50 and "stage" in answer_lower:
        # Extract stage word from answer
        stage_match = re.search(r'(?:during|in|at) the (\w+) (?:stage|phase|period)', answer_lower)
        if stage_match:
            stage_word = stage_match.group(1)
            # If the stage word appears in the question, it's tautological
            if stage_word in question.lower():
                return True

    # Very short answers (< 40 chars) that just name a stage
    if len(answer_lower) < 40:
        return True

    return False


def assign_difficulty(q: dict, disease_triple_counts: dict) -> str:
    """Assign difficulty tier: easy, medium, hard."""
    qtype = q["type"]
    answer = q.get("answer", "")
    disease_id = q.get("disease_id", "")
    triple_count = disease_triple_counts.get(disease_id, 0)

    # Hard: diseases with sparse evidence (<5 triples) or wide age ranges
    if triple_count < 5:
        return "hard"

    if qtype == "onset_age":
        m = re.search(r'[Bb]etween age ([\d.]+) and ([\d.]+)', answer)
        if m:
            span = float(m.group(2)) - float(m.group(1))
            if span > 30:
                return "hard"
            elif span > 10:
                return "medium"
            else:
                return "easy"

    if qtype == "treatment_timeline":
        return "medium"

    if qtype == "milestone_timing":
        return "easy"

    if qtype == "stage_phenotype":
        return "medium"

    return q.get("difficulty", "easy")


def load_disease_triple_counts() -> dict:
    """Count triples per disease from extracted data."""
    extracted_dir = PROJECT_ROOT / "data" / "extracted"
    counts = {}
    if not extracted_dir.exists():
        return counts

    for disease_dir in extracted_dir.iterdir():
        if not disease_dir.is_dir():
            continue
        vt = disease_dir / "validated_triples.jsonl"
        if vt.exists():
            disease_id = disease_dir.name.replace("_", ":")
            try:
                with open(vt) as f:
                    counts[disease_id] = sum(1 for line in f if line.strip())
            except Exception:
                pass
    return counts


def main():
    parser = argparse.ArgumentParser(description="TQA Benchmark Cleanup v1 -> v2")
    parser.add_argument("--strict", action="store_true",
                        help="Apply stricter filtering (confidence threshold) to reach 8-10K target")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Minimum consensus_confidence (only with --strict, default 0.85)")
    args = parser.parse_args()

    if args.strict and args.min_confidence == 0.0:
        args.min_confidence = 0.85

    # Load v1 benchmark
    v1_file = BENCHMARK_DIR / "chronomedkg_tqa_v1.json"
    logger.info("Loading %s...", v1_file)
    with open(v1_file) as f:
        v1 = json.load(f)

    questions = v1["questions"]
    logger.info("Loaded %d questions (v1)", len(questions))

    # Load disease triple counts for difficulty assignment
    logger.info("Counting triples per disease...")
    disease_triple_counts = load_disease_triple_counts()
    logger.info("Found triple counts for %d diseases", len(disease_triple_counts))

    # ── Filter 1: Same min/max age ──
    filter_stats = defaultdict(int)
    kept = []
    for q in questions:
        if is_same_min_max_age(q.get("answer", "")):
            filter_stats["same_min_max_age"] += 1
            continue
        kept.append(q)
    questions = kept
    logger.info("After same-min/max-age filter: %d (removed %d)", len(questions), filter_stats["same_min_max_age"])

    # ── Filter 2: Non-temporal treatment answers ──
    kept = []
    for q in questions:
        if q["type"] == "treatment_timeline" and not has_temporal_content(q.get("answer", "")):
            filter_stats["non_temporal_treatment"] += 1
            continue
        kept.append(q)
    questions = kept
    logger.info("After non-temporal treatment filter: %d (removed %d)", len(questions), filter_stats["non_temporal_treatment"])

    # ── Filter 3: Tautological stage questions ──
    kept = []
    for q in questions:
        if q["type"] == "stage_phenotype" and is_tautological_stage(q["question"], q.get("answer", "")):
            filter_stats["tautological_stage"] += 1
            continue
        kept.append(q)
    questions = kept
    logger.info("After tautological stage filter: %d (removed %d)", len(questions), filter_stats["tautological_stage"])

    # ── Filter 4: Deduplication ──
    seen = {}
    kept = []
    for q in questions:
        dedup_key = (q.get("disease_id", ""), q["type"], q.get("target_entity", "").lower().strip())
        if dedup_key in seen:
            # Keep the one with higher confidence
            existing_idx = seen[dedup_key]
            if q.get("consensus_confidence", 0) > kept[existing_idx].get("consensus_confidence", 0):
                filter_stats["duplicates"] += 1
                kept[existing_idx] = q  # replace with better one
            else:
                filter_stats["duplicates"] += 1
            continue
        seen[dedup_key] = len(kept)
        kept.append(q)
    questions = kept
    logger.info("After dedup: %d (removed %d)", len(questions), filter_stats["duplicates"])

    # ── Filter 5 (strict mode): Confidence threshold ──
    if args.strict:
        kept = []
        for q in questions:
            if q.get("consensus_confidence", 0) < args.min_confidence:
                filter_stats["low_confidence"] += 1
                continue
            kept.append(q)
        questions = kept
        logger.info("After confidence filter (>= %.2f): %d (removed %d)",
                     args.min_confidence, len(questions), filter_stats["low_confidence"])

    # ── Reassign difficulty tiers ──
    difficulty_counts = defaultdict(int)
    for q in questions:
        q["difficulty"] = assign_difficulty(q, disease_triple_counts)
        difficulty_counts[q["difficulty"]] += 1

    # ── Renumber IDs ──
    for i, q in enumerate(questions, 1):
        q["id"] = f"TQA-{i:05d}"

    # ── Compute stats ──
    type_counts = defaultdict(int)
    disease_ids = set()
    for q in questions:
        type_counts[q["type"]] += 1
        disease_ids.add(q.get("disease_id", ""))

    # ── Build v2 ──
    v2 = {
        "name": "ChronoMedKG-TQA",
        "version": "2.0.0",
        "description": "Temporal QA Benchmark for Biomedical Knowledge Graphs -- cleaned from v1 (garbage removed, difficulty recalibrated)",
        "statistics": {
            "total_questions": len(questions),
            "diseases": len(disease_ids),
            "by_type": dict(type_counts),
            "by_difficulty": dict(difficulty_counts),
            "avg_consensus_confidence": sum(q.get("consensus_confidence", 0) for q in questions) / len(questions) if questions else 0,
        },
        "cleanup_from_v1": {
            "v1_total": v1["statistics"]["total_questions"],
            "v2_total": len(questions),
            "removed": v1["statistics"]["total_questions"] - len(questions),
            "removal_rate": (v1["statistics"]["total_questions"] - len(questions)) / v1["statistics"]["total_questions"] * 100,
            "filters_applied": dict(filter_stats),
            "strict_mode": args.strict,
        },
        "questions": questions,
    }

    # ── Save ──
    v2_file = BENCHMARK_DIR / "chronomedkg_tqa_v2.json"
    with open(v2_file, "w") as f:
        json.dump(v2, f, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("TQA BENCHMARK CLEANUP SUMMARY")
    logger.info("=" * 60)
    logger.info("v1 questions:          %d", v1["statistics"]["total_questions"])
    logger.info("v2 questions:          %d", len(questions))
    logger.info("Removed:               %d (%.1f%%)",
                v1["statistics"]["total_questions"] - len(questions),
                (v1["statistics"]["total_questions"] - len(questions)) / v1["statistics"]["total_questions"] * 100)
    logger.info("")
    logger.info("Filters applied:")
    for k, v in filter_stats.items():
        logger.info("  %-25s %d", k, v)
    logger.info("")
    logger.info("By type:")
    for t, c in sorted(type_counts.items()):
        logger.info("  %-25s %d", t, c)
    logger.info("")
    logger.info("By difficulty:")
    for d, c in sorted(difficulty_counts.items()):
        logger.info("  %-25s %d", d, c)
    logger.info("")
    logger.info("Diseases covered:      %d", len(disease_ids))
    logger.info("Saved to %s", v2_file)


if __name__ == "__main__":
    main()
