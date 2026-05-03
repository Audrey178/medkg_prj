#!/usr/bin/env python3
"""
Cross-KG Benchmark Evaluation for ChronoMedKG-TQA v2
=======================================================
Evaluates TQA questions against multiple sources:
1. ChronoMedKG (graph lookup against own cleaned triples)
2. Orphadata (onset_age only, from orpha_parsed.pkl)
3. HPO (onset_age only, from hpo_parsed.pkl)
4. PrimeKG / iKraph / Hetionet (0% -- no structured temporal data)

Usage:
    python3 scripts/compute_cross_kg_benchmark.py
    python3 scripts/compute_cross_kg_benchmark.py --tqa-file data/benchmark/chronomedkg_tqa_v2.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation_sources"


def parse_answer_age_range(answer: str) -> tuple[float | None, float | None]:
    """Extract age range from answer text."""
    # "Between age X and Y."
    m = re.search(r'[Bb]etween age ([\d.]+) and ([\d.]+)', answer)
    if m:
        return float(m.group(1)), float(m.group(2))

    # "Around age X."
    m = re.search(r'[Aa]round age ([\d.]+)', answer)
    if m:
        age = float(m.group(1))
        return age, age

    return None, None


def age_ranges_overlap(our_min: float, our_max: float, ref_min: float, ref_max: float) -> bool:
    """Check if two age ranges overlap."""
    return our_min <= ref_max and our_max >= ref_min


# ──────────────────────────────────────────────────────────────────────
# Source 1: ChronoMedKG self-evaluation
# ──────────────────────────────────────────────────────────────────────

def build_triple_index() -> dict:
    """Load all validated triples indexed by disease_id."""
    index = {}  # disease_id -> list of triples
    if not EXTRACTED_DIR.exists():
        return index

    for disease_dir in EXTRACTED_DIR.iterdir():
        if not disease_dir.is_dir():
            continue
        vt = disease_dir / "validated_triples.jsonl"
        if not vt.exists():
            continue

        disease_id = disease_dir.name.replace("_", ":")
        triples = []
        with open(vt) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        triples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if triples:
            index[disease_id] = triples
    return index


def eval_chronomedkg(questions: list[dict], triple_index: dict) -> dict:
    """Evaluate TQA against ChronoMedKG graph (self-referential)."""
    answerable = 0
    correct = 0

    for q in questions:
        disease_id = q.get("disease_id", "")
        triples = triple_index.get(disease_id, [])
        if not triples:
            continue

        # v3 format: evidence_edges contain the source triple
        # v1 format: target_entity field
        target = q.get("target_entity", "").lower().strip()
        if not target and q.get("evidence_edges"):
            edge = q["evidence_edges"][0]
            target = (edge.get("target_name") or edge.get("source_name") or "").lower().strip()

        qtype = q.get("type", q.get("task_type", "unknown"))

        # For v3 temporal_fact questions: the answer was generated from evidence_edges
        # If the question has evidence_edges with edge_id, check if that edge exists
        if q.get("evidence_edges"):
            edge_ids = {e.get("edge_id") for e in q["evidence_edges"] if e.get("edge_id")}
            for t in triples:
                if t.get("edge_id") in edge_ids:
                    answerable += 1
                    correct += 1
                    break
            else:
                # Fallback: fuzzy match by target name
                for t in triples:
                    t_target = (t.get("target_name") or "").lower().strip()
                    t_source = (t.get("source_name") or "").lower().strip()
                    if target and (target in t_target or target in t_source or t_target in target):
                        temporal = t.get("temporal", {})
                        if any(temporal.get(k) for k in ["onset_age_min", "temporal_qualifier", "milestone", "progression_stage", "duration", "discovery_date"]):
                            answerable += 1
                            correct += 1
                            break
        else:
            # v1 format fallback
            matched = False
            for t in triples:
                t_target = (t.get("target_name") or "").lower().strip()
                t_source = (t.get("source_name") or "").lower().strip()
                if target in t_target or target in t_source or t_target in target or t_source in target:
                    temporal = t.get("temporal", {})
                    if qtype == "onset_age" and temporal.get("onset_age_min") is not None:
                        matched = True
                        break
                    elif qtype == "treatment_timeline" and (temporal.get("temporal_qualifier") or temporal.get("discovery_date") or temporal.get("treatment_start_age") is not None):
                        matched = True
                        break
                    elif qtype == "stage_phenotype" and temporal.get("progression_stage"):
                        matched = True
                        break
                    elif qtype == "milestone_timing" and (temporal.get("milestone") or temporal.get("onset_age_min") is not None):
                        matched = True
                        break
            if matched:
                answerable += 1
                correct += 1

    return {
        "answerable": answerable,
        "correct": correct,
        "total": len(questions),
        "coverage": answerable / len(questions) if questions else 0,
        "accuracy": correct / answerable if answerable else 0,
    }


# ──────────────────────────────────────────────────────────────────────
# Source 2: Orphadata
# ──────────────────────────────────────────────────────────────────────

def eval_orphadata(questions: list[dict], orpha: dict) -> dict:
    """Evaluate onset_age questions against Orphadata."""
    answerable = 0
    correct = 0
    # v1: type=onset_age, v3: task_type=temporal_fact with onset in answer
    onset_qs = [q for q in questions if q.get("type", q.get("task_type", "")) == "onset_age"]
    if not onset_qs:
        # v3 format: filter temporal_fact questions that have age ranges in answers
        import re
        onset_qs = [q for q in questions if re.search(r'[Bb]etween age|onset.*age|[Aa]ge.*\d', q.get("answer", ""))]

    for q in onset_qs:
        disease_name = q.get("disease_name", "").lower().strip()
        orpha_entry = orpha.get(disease_name)
        if not orpha_entry:
            continue

        our_min, our_max = parse_answer_age_range(q.get("answer", ""))
        if our_min is None:
            continue

        answerable += 1
        ref_min = orpha_entry.get("min_age", 0)
        ref_max = orpha_entry.get("max_age", 120)

        if age_ranges_overlap(our_min, our_max, ref_min, ref_max):
            correct += 1

    return {
        "answerable": answerable,
        "correct": correct,
        "total": len(questions),
        "coverage": answerable / len(questions) if questions else 0,
        "accuracy": correct / answerable if answerable else 0,
        "note": "Only evaluates onset_age questions where disease name matches Orphadata",
    }


# ──────────────────────────────────────────────────────────────────────
# Source 3: HPO
# ──────────────────────────────────────────────────────────────────────

def eval_hpo(questions: list[dict], hpo: dict) -> dict:
    """Evaluate onset_age questions against HPO onset data."""
    answerable = 0
    correct = 0
    onset_qs = [q for q in questions if q.get("type", q.get("task_type", "")) == "onset_age"]
    if not onset_qs:
        import re
        onset_qs = [q for q in questions if re.search(r'[Bb]etween age|onset.*age|[Aa]ge.*\d', q.get("answer", ""))]

    for q in onset_qs:
        disease_name = q.get("disease_name", "").lower().strip()
        hpo_entry = hpo.get(disease_name)
        if not hpo_entry:
            continue

        our_min, our_max = parse_answer_age_range(q.get("answer", ""))
        if our_min is None:
            continue

        answerable += 1
        ref_min = hpo_entry.get("min_age", 0)
        ref_max = hpo_entry.get("max_age", 120)

        # HPO sometimes has -1 for antenatal; treat as 0
        if ref_min < 0:
            ref_min = 0

        if age_ranges_overlap(our_min, our_max, ref_min, ref_max):
            correct += 1

    return {
        "answerable": answerable,
        "correct": correct,
        "total": len(questions),
        "coverage": answerable / len(questions) if questions else 0,
        "accuracy": correct / answerable if answerable else 0,
        "note": "Only evaluates onset_age questions where disease name matches HPO",
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-KG Benchmark Evaluation")
    parser.add_argument("--tqa-file", type=str, default=str(BENCHMARK_DIR / "chronomedkg_tqa_v2.json"),
                        help="Path to TQA benchmark file")
    args = parser.parse_args()

    tqa_file = Path(args.tqa_file)
    if not tqa_file.exists():
        logger.error("TQA file not found: %s", tqa_file)
        return

    logger.info("Loading TQA from %s...", tqa_file)
    with open(tqa_file) as f:
        tqa = json.load(f)
    questions = tqa["questions"]
    logger.info("Loaded %d questions", len(questions))

    # Load validation sources
    logger.info("Loading Orphadata...")
    with open(VALIDATION_DIR / "orpha_parsed.pkl", "rb") as f:
        orpha = pickle.load(f)
    logger.info("Loaded %d Orphadata diseases", len(orpha))

    logger.info("Loading HPO...")
    with open(VALIDATION_DIR / "hpo_parsed.pkl", "rb") as f:
        hpo = pickle.load(f)
    logger.info("Loaded %d HPO diseases", len(hpo))

    # Build triple index
    logger.info("Building ChronoMedKG triple index...")
    triple_index = build_triple_index()
    logger.info("Indexed %d diseases with triples", len(triple_index))

    # Evaluate all sources
    logger.info("\nEvaluating ChronoMedKG (graph lookup)...")
    ta_result = eval_chronomedkg(questions, triple_index)

    logger.info("Evaluating Orphadata...")
    orpha_result = eval_orphadata(questions, orpha)

    logger.info("Evaluating HPO...")
    hpo_result = eval_hpo(questions, hpo)

    # Static KGs: no temporal data
    static_result = {
        "answerable": 0, "correct": 0, "total": len(questions),
        "coverage": 0.0, "accuracy": 0,
        "note": "No structured temporal metadata",
    }

    # Build results
    results = {
        "benchmark": f"ChronoMedKG-TQA {tqa.get('version', '?')} (cross-KG evaluation)",
        "total_questions": len(questions),
        "diseases": tqa.get("statistics", {}).get("diseases", "?"),
        "by_type": tqa.get("statistics", {}).get("by_type", {}),
        "results": {
            "ChronoMedKG": ta_result,
            "Orphadata": orpha_result,
            "HPO": hpo_result,
            "PrimeKG": static_result,
            "iKraph": static_result,
            "Hetionet": static_result,
        },
    }

    # Save
    version = tqa.get("version", "2.0.0").replace(".", "_")
    output_file = BENCHMARK_DIR / f"tqa_v2_all_sources_benchmark.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print
    logger.info("\n" + "=" * 60)
    logger.info("CROSS-KG BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info("Total questions: %d", len(questions))
    logger.info("")
    logger.info("%-15s %8s %8s %8s %8s", "Source", "Answer.", "Correct", "Cover.", "Accur.")
    logger.info("-" * 55)
    for name, r in results["results"].items():
        logger.info("%-15s %8d %8d %7.1f%% %7.1f%%",
                     name, r["answerable"], r["correct"],
                     r["coverage"] * 100, r["accuracy"] * 100 if r["answerable"] else 0)

    logger.info("\nSaved to %s", output_file)


if __name__ == "__main__":
    main()
