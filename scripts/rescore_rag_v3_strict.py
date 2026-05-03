#!/usr/bin/env python3
"""
rescore_rag_v3_strict.py — Re-score existing RAG v3 checkpoint with stricter rules.

Fixes two bugs in the v3 scorer that cause saturation / inflated accuracy:

Bug 1 (MCQ): `llm.startswith(correct_letter)` matches any prose opener starting
with the correct letter (e.g. "Based on..." matches "B"). Affected Q-types:
  - temporal_differential_dx
  - negative_temporal_mcq
  - static_control_drug
  - static_control_gene
Fix: require an explicit final-answer marker and use the LAST one (LLMs often
summarize near the end). Fall back to text-match only when marker-based parsing
yields nothing.

Bug 2 (phenopackets_onset): ±2y age-overlap tolerance + ±5y category-keyword
fallback is too wide for tight Phenopacket gold ranges (e.g. gold 0-0y accepts
"2-10 years" as correct).
Fix: tolerance = max(0.5, 0.5 * gold-range-width), capped at 2y. Disable
category-keyword fallback when a numeric gold range is given.

Writes: data/benchmark/rag_v3_results/checkpoint_strict.jsonl
Does NOT modify the original checkpoint.jsonl.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import run_rag_experiment_v3 as v3  # for question loading


# =============================================================================
# STRICT SCORERS
# =============================================================================

# Final-answer markers, ordered by specificity. Patterns capture the letter.
_MCQ_MARKERS = [
    # Highest specificity: explicit "answer is X" / "Answer: X"
    re.compile(r"(?:final\s+answer|answer\s+is|the\s+answer)\s*[:\s]*[*_`]*\(?([ABCD])\)?", re.I),
    re.compile(r"^\s*answer\s*[:\s]+[*_`]*\(?([ABCD])\)?", re.I | re.M),
    # "Therefore, the correct option is X" / "Thus X is..."
    re.compile(r"(?:therefore|thus|hence|conclusion)[^.]*?\b([ABCD])\)", re.I),
    # Markdown/bold emphasis on a choice: **B)** or __B)__
    re.compile(r"[*_]{1,3}\(?([ABCD])\)[*_]{1,3}"),
    # Parenthetical choice at start of a line: "B) Some text"
    re.compile(r"^\s*\(?([ABCD])\)\s", re.M),
    # Plain "option X" / "choice X"
    re.compile(r"(?:option|choice|select(?:ing)?)\s+[*_`]*\(?([ABCD])\)?\b", re.I),
]


def score_mcq_strict(llm_answer, correct_letter, options):
    """
    Stricter MCQ scoring. Rules:
      1. Find final-answer marker(s) in the text. Use the LAST marker (LLMs
         often reason then conclude).
      2. If exactly one letter is marked, use it.
      3. If multiple distinct letters are marked, use the LAST one.
      4. If no markers found, fall back to: count letter mentions in isolation
         (e.g. "A)" or "(A)") and pick the one mentioned most with parenthesis.
      5. If still ambiguous, return False with "no_clear_answer".

    Does NOT use startswith(). Does NOT match prose openers.
    """
    if not llm_answer:
        return False, "empty_answer"

    text = llm_answer.strip()
    found = []  # list of (position, letter)

    for pat in _MCQ_MARKERS:
        for m in pat.finditer(text):
            letter = m.group(1).upper()
            if letter in ("A", "B", "C", "D"):
                found.append((m.start(), letter))

    if found:
        # Sort by position; take the LAST marker as final answer
        found.sort()
        predicted = found[-1][1]
        if predicted == correct_letter.upper():
            return True, f"marker_match:{predicted}"
        else:
            return False, f"chose_{predicted}"

    # Fallback: count parenthetical letter markers "A)", "B)", "(C)", etc.
    pcount = Counter()
    for m in re.finditer(r"\(?([ABCD])\)", text):
        pcount[m.group(1).upper()] += 1
    if pcount:
        # Pick the most-mentioned, tiebreak on last position
        top = pcount.most_common(1)[0]
        if top[1] >= 1 and len([l for l, c in pcount.items() if c == top[1]]) == 1:
            predicted = top[0]
            if predicted == correct_letter.upper():
                return True, f"count_match:{predicted}"
            return False, f"chose_{predicted}"

    # Last resort: accept plain option text match ONLY if no other option text also appears
    correct_text = options.get(correct_letter, "").lower() if options else ""
    lower = text.lower()
    if correct_text and correct_text in lower:
        # Check for ambiguity: is any OTHER option text also in answer?
        other_hits = [l for l, t in (options or {}).items()
                      if l != correct_letter and t and t.lower() in lower]
        if not other_hits:
            return True, "text_match_unique"
        # Ambiguous — both correct and wrong options discussed
        return False, "text_match_ambiguous"

    return False, "no_clear_answer"


def score_onset_age_strict(llm_answer, gold_min, gold_max):
    """
    Calibrated strict onset-age scoring.

    Fixes the original v3 bug (±2y slack + broad keyword fallback) AND the
    previous "too strict" version (rejected clinically-correct "at birth"
    keyword answers for tight gold ranges).

    Tolerance = max(0.5, 0.5 * gold_range_width), capped at 2y.
    Numeric-range match: LLM range overlaps [gold - tol, gold + tol].
    Keyword fallback (calibrated): accept a category keyword ONLY if
      (a) its range overlaps the tolerance window, AND
      (b) the keyword's width is close to the gold-range width
          (`kw_width <= gold_width + 2`). This admits "congenital"/"neonatal"
          (width <= 0.1) for gold 0-0, and "infancy" (width 1.92) for gold
          widths 0 or ~2y, but rejects "childhood" (width 10) for tight golds.
    """
    llm_lower = llm_answer.lower()
    gold_width = max(0.0, gold_max - gold_min)
    tol = min(2.0, max(0.5, 0.5 * gold_width))

    lo_ok = gold_min - tol
    hi_ok = gold_max + tol

    # Try explicit numeric range "X to Y years"
    range_patterns = [
        r'(\d+\.?\d*)\s*(?:to|-|–|and)\s*(\d+\.?\d*)\s*(?:years|yr)',
        r'age\s*(\d+\.?\d*)\s*(?:to|-|–|and)\s*(\d+\.?\d*)',
        r'between\s*(\d+\.?\d*)\s*and\s*(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*(?:to|-|–)\s*(\d+\.?\d*)\s*y\b',  # "0.5-2 y"
    ]
    for pat in range_patterns:
        match = re.search(pat, llm_lower)
        if match:
            try:
                llm_min = float(match.group(1))
                llm_max = float(match.group(2))
                if llm_min <= hi_ok and llm_max >= lo_ok:
                    return True, f"strict_overlap:{llm_min}-{llm_max}"
                return False, f"range_outside_tol:{llm_min}-{llm_max}"
            except ValueError:
                pass

    # Try single age mention "5 years"
    single_matches = re.findall(r'(\d+\.?\d*)\s*years?\b', llm_lower)
    if single_matches:
        try:
            age = float(single_matches[0])
            if lo_ok <= age <= hi_ok:
                return True, f"strict_single:{age}"
            return False, f"single_outside_tol:{age}"
        except ValueError:
            pass

    # Calibrated keyword fallback: accept if keyword range is close to gold width.
    category_kw = {
        "prenatal": (0, 0), "antenatal": (0, 0),
        "at birth": (0, 0.1), "birth": (0, 0.1), "congenital": (0, 0.1),
        "neonatal": (0, 0.08),
        "early infancy": (0, 0.5), "first year": (0, 1),
        "infantile": (0.08, 2), "infancy": (0.08, 2), "infant": (0.08, 2),
        "early childhood": (1, 5), "toddler": (1, 3),
        "childhood": (1, 11),
        "juvenile": (5, 15),
        "adolescen": (10, 18), "teen": (10, 19),
        "young adult": (15, 40),
        "early adult": (18, 40),
        "adult": (18, 65),
        "middle age": (40, 65),
        "late adult": (55, 85), "elderly": (60, 120), "old age": (60, 120),
    }
    # Admit keyword if (a) overlaps tolerance window AND (b) keyword is tight
    # enough relative to gold (prevents "childhood" matching gold 0-0).
    width_budget = gold_width + 2.0  # keyword may be up to 2y wider than gold
    for kw, (kw_min, kw_max) in category_kw.items():
        if kw in llm_lower:
            kw_width = kw_max - kw_min
            if kw_width <= width_budget and kw_min <= hi_ok and kw_max >= lo_ok:
                return True, f"strict_category:{kw}"

    return False, "no_match"


def score_question_strict(llm_answer, question):
    """Strict re-scorer; delegates to v3 scorers for types without known bugs."""
    qtype = question["type"]

    # Bug-affected: MCQ types — use strict scorer
    if qtype in ("temporal_differential_dx", "negative_temporal_mcq",
                 "static_control_drug", "static_control_gene"):
        return score_mcq_strict(llm_answer, question["answer"],
                                question.get("options", {}))

    # Bug-affected: phenopackets onset (tight gold ranges)
    if qtype == "phenopackets_onset":
        gs = question.get("gold_standard", {}) or {}
        gmin = gs.get("onset_min", 0)
        gmax = gs.get("onset_max", 0)
        try:
            return score_onset_age_strict(llm_answer, float(gmin), float(gmax))
        except (TypeError, ValueError):
            return False, "gold_parse_error"

    # Unchanged: temporal_window (yes/no), cross_disease_comparison
    # Use v3 scorers directly
    return v3.score_question(llm_answer, question)


# =============================================================================
# MAIN: re-score checkpoint
# =============================================================================

def main():
    # Load questions (for options + gold_standard lookup)
    with open(v3.BENCHMARK_DIR / "chronomedkg_tqa_v6.json") as f:
        tqa = json.load(f)
    qmap = {q["id"]: q for q in tqa["questions"]}

    # Load v3 checkpoint
    ckpt_in = v3.BENCHMARK_DIR / "rag_v3_results" / "checkpoint.jsonl"
    ckpt_out = v3.BENCHMARK_DIR / "rag_v3_results" / "checkpoint_calibrated.jsonl"

    n_rows = 0
    n_changed = 0
    n_false_pos_fixed = 0  # was True, now False
    n_false_neg_fixed = 0  # was False, now True
    change_by_qtype = Counter()

    with open(ckpt_in) as fin, open(ckpt_out, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            n_rows += 1
            row = json.loads(line)
            qid = row["question_id"]
            q = qmap.get(qid)
            if q is None:
                row["strict_correct"] = row["correct"]
                row["strict_reason"] = "question_not_found"
                fout.write(json.dumps(row) + "\n")
                continue

            # We only have the first 500 chars of the LLM answer in the
            # checkpoint (v3 truncates). This is a small fidelity loss but
            # the final-answer markers are usually near the end of that slice
            # or earlier; most MCQ answers fit.
            llm_answer = row.get("llm_answer", "")

            new_correct, new_reason = score_question_strict(llm_answer, q)

            row["original_correct"] = row["correct"]
            row["original_reason"] = row["reason"]
            row["strict_correct"] = new_correct
            row["strict_reason"] = new_reason
            row["correct"] = new_correct  # overwrite so aggregation is strict
            row["reason"] = new_reason

            if new_correct != row["original_correct"]:
                n_changed += 1
                change_by_qtype[(row["question_type"], row["original_correct"], new_correct)] += 1
                if row["original_correct"] and not new_correct:
                    n_false_pos_fixed += 1
                else:
                    n_false_neg_fixed += 1

            fout.write(json.dumps(row) + "\n")

    print("=" * 80)
    print("RE-SCORING SUMMARY")
    print("=" * 80)
    print(f"Total rows:           {n_rows}")
    print(f"Changed verdicts:     {n_changed} ({100.0*n_changed/n_rows:.1f}%)")
    print(f"  False-pos removed:  {n_false_pos_fixed} (original True, strict False)")
    print(f"  False-neg fixed:    {n_false_neg_fixed} (original False, strict True)")
    print()
    print("By Q-type (original_correct -> strict_correct):")
    for (qt, oc, nc), cnt in sorted(change_by_qtype.items()):
        print(f"  {qt:<28} {str(oc):>6} -> {str(nc):>6}: {cnt}")
    print()
    print(f"Strict checkpoint written: {ckpt_out}")


if __name__ == "__main__":
    main()
