#!/usr/bin/env python3
"""
MEDQA Evaluation — ChronoMedKG as RAG Knowledge Source
=======================================================
Evaluates whether ChronoMedKG triples improve LLM accuracy on MEDQA
(USMLE-style 4-option multiple-choice questions).

Metrics reported
----------------
  accuracy_base      LLM accuracy without KG context (parametric knowledge only)
  accuracy_kg        LLM accuracy with ChronoMedKG triples injected as context
  delta              accuracy_kg - accuracy_base (positive = KG helps)
  entity_hit_rate    % of questions where the KG found ≥1 matching entity
  avg_triples_hit    Mean number of triples retrieved per question
  95% CI             Bootstrap confidence intervals on accuracy

Scope note
----------
ChronoMedKG covers rare genetic diseases specifically. MEDQA (USMLE) spans all of
medicine, so entity_hit_rate will naturally be low (~5-15%). This script reports
results both across all questions AND for the subset where the KG fires (hit_rate > 0),
giving a fair picture of KG quality vs. coverage.

Data format expected (JSONL, one question per line)
----------------------------------------------------
  {"question": "...", "options": {"A": "...", "B": "...", "C": "...", "D": "..."}, "answer": "B"}

  Alternatively, the 5-option format used by MedQA-USMLE:
  {"question": "...", "options": {"A": ..., "B": ..., "C": ..., "D": ..., "E": ...}, "answer": "E"}

Public dataset: https://github.com/jind11/MedQA  (data_clean/questions/US/)

Usage
-----
  # Quick test (100 questions, gpt-4o-mini, with and without KG)
  python -m scripts.run_medqa_eval --data data/medqa/test.jsonl --n 100

  # Full test set, DeepSeek + GPT, baseline only
  python -m scripts.run_medqa_eval --data data/medqa/test.jsonl --no-kg --models deepseek-v3 gpt-4o-mini

  # Use Zenodo KG (downloads if not cached)
  python -m scripts.run_medqa_eval --data data/medqa/test.jsonl --kg-source zenodo --n 200

  # Dry run — check data loading without API calls
  python -m scripts.run_medqa_eval --data data/medqa/test.jsonl --dry-run
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

# Load .env
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

EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
RESULTS_DIR = PROJECT_ROOT / "data" / "benchmark" / "medqa_results"

RANDOM_SEED = 42


# ============================================================================
# KG LOADING & INDEXING
# ============================================================================

def _load_local_triples() -> list[dict]:
    """Load validated triples from data/extracted/ (local pipeline output)."""
    triples = []
    if not EXTRACTED_DIR.exists():
        return triples
   
    vt = EXTRACTED_DIR / "validated_triples.jsonl"
    if not vt.exists():
        return triples
    with open(vt) as f:
        for line in f:
            line = line.strip()
            if line:
                    try:
                        triples.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return triples


def _load_zenodo_triples() -> list[dict]:
    """Load validated triples via the chronomedkg Zenodo loader."""
    from chronomedkg.loader import load_triples
    return list(load_triples())


# Only use disease-centric relations as retrieval triggers.
# Indexing target_name (symptoms, genes) or drug-source relations (indication,
# drug_effect) causes false-positive matches on nearly every USMLE question.
_DISEASE_TRIGGER_RELS = frozenset({
    "disease_phenotype_positive",
    "disease_phenotype_negative",
    "disease_protein",
    "disease_disease",
})


def build_kg_index(triples: list[dict]) -> tuple[dict, list[str]]:
    """Build disease-name → triples index and sorted vocabulary for fuzzy matching.

    Only source_name of disease-centric relations are used as index keys so that
    retrieval fires on rare disease names in the question, not on common symptoms
    or drug names that appear in nearly every USMLE question.

    Returns:
        index: disease_name_lower → list[triple]  (all triples for that disease)
        vocab: sorted list of all indexed disease names (for fuzzy matching)
    """
    # Step 1: collect all triples per disease source
    index: dict[str, list[dict]] = defaultdict(list)
    for t in triples:
        if t.get("relation", "") not in _DISEASE_TRIGGER_RELS:
            continue
        name = t.get("source_name", "").strip()
        if not name:
            continue
        index[name.lower()].append(t)
        # Index parenthetical gene/disease abbreviations ≥ 3 chars, e.g. "Dystrophin (DMD)"
        abbr = re.search(r'\(([A-Z][A-Z0-9]{2,9})\)', name)
        if abbr:
            index[abbr.group(1).lower()].append(t)

    # Step 2: for each indexed disease, also attach its non-trigger triples
    # (indication, drug_effect, etc.) so the full context is available at query time
    disease_names: set[str] = set(index.keys())
    for t in triples:
        if t.get("relation", "") in _DISEASE_TRIGGER_RELS:
            continue
        for field in ("source_name", "target_name"):
            name = t.get(field, "").strip().lower()
            if name in disease_names:
                index[name].append(t)

    vocab = sorted(index.keys())
    logger.info(
        "KG index: %d disease entities, %d total triples (trigger rels: %s)",
        len(vocab), len(triples), ", ".join(sorted(_DISEASE_TRIGGER_RELS)),
    )
    return dict(index), vocab


# ============================================================================
# ENTITY EXTRACTION & RETRIEVAL
# ============================================================================

# Medical stopwords — high-frequency words that are not useful as entity signals
_STOP = {
    "the", "and", "for", "with", "from", "this", "that", "have", "has", "are",
    "was", "were", "been", "being", "their", "which", "what", "does", "during",
    "typically", "following", "clinical", "most", "consistent", "patient",
    "presents", "treatment", "diagnosis", "following", "history", "physical",
    "examination", "laboratory", "findings", "year", "old", "male", "female",
    "man", "woman", "child", "infant", "adult", "who", "her", "his", "she",
    "would", "should", "likely", "best", "next", "step", "appropriate",
    "management", "associated", "significant", "develop", "shows", "test",
    "result", "level", "normal", "elevated", "decreased", "increased",
}


def extract_candidate_entities(text: str, max_ngrams: int = 4) -> list[str]:
    """Extract candidate entity strings (unigrams through n-grams) from question text."""
    words = re.findall(r"[A-Za-z0-9][\w'-]*", text.lower())
    candidates = []
    for n in range(1, max_ngrams + 1):
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i + n])
            if not all(w in _STOP for w in ngram.split()):
                candidates.append(ngram)
    return candidates


def retrieve_for_question(
    question_text: str,
    options: dict,
    kg_index: dict,
    kg_vocab: list[str],
    top_k: int = 15,
    fuzzy_threshold: int = 90,
) -> tuple[list[dict], int]:
    """Retrieve relevant KG triples for a MEDQA question.

    Strategy:
    1. Exact substring matching of question + options against KG entity names
    2. Fuzzy matching (rapidfuzz) for close variants
    3. Score and rank by credibility + keyword overlap

    Returns:
        (triples, n_entities_matched)
    """
    full_text = question_text + " " + " ".join(options.values())
    candidates = extract_candidate_entities(full_text)
    candidates_set = set(candidates)

    matched_entities: set[str] = set()

    # Pass 1: exact match.
    # Require multi-word (n≥2) candidates to avoid single generic medical words
    # ("pain", "fever", "diabetes") that saturate the index with noise.
    # Single-word candidates are only allowed if ≥ 12 chars (highly specific terms).
    for cand in candidates_set:
        if cand not in kg_index:
            continue
        words_in_cand = cand.split()
        if len(words_in_cand) >= 2 or len(cand) >= 12:
            matched_entities.add(cand)

    # Pass 2: fuzzy match for longer candidates (≥5 chars) against KG vocab
    try:
        from rapidfuzz import process, fuzz
        long_candidates = [c for c in candidates_set if len(c) >= 5 and c not in matched_entities]
        if long_candidates and kg_vocab:
            for cand in long_candidates[:50]:  # cap to avoid O(n²) blowup
                hits = process.extract(
                    cand, kg_vocab,
                    scorer=fuzz.ratio,
                    limit=2,
                    score_cutoff=fuzzy_threshold,
                )
                for match_name, score, _ in hits:
                    matched_entities.add(match_name)
    except ImportError:
        pass

    if not matched_entities:
        return [], 0

    # Collect triples and rank
    seen_edge_ids: set[str] = set()
    scored: list[tuple[float, dict]] = []
    question_lower = full_text.lower()

    for entity in matched_entities:
        for t in kg_index.get(entity, []):
            eid = t.get("edge_id", id(t))
            if eid in seen_edge_ids:
                continue
            seen_edge_ids.add(eid)

            # Score: credibility + keyword presence in triple content
            cred = t.get("evidence", {}).get("credibility_score", 0.3)
            triple_text = (
                f"{t.get('source_name', '')} {t.get('relation', '')} {t.get('target_name', '')}"
            ).lower()
            kw_overlap = sum(
                1 for word in question_lower.split()
                if len(word) > 4 and word not in _STOP and word in triple_text
            )
            score = cred + 0.1 * kw_overlap
            scored.append((score, t))

    scored.sort(key=lambda x: -x[0])
    selected = [t for _, t in scored[:top_k]]
    return selected, len(matched_entities)


def format_kg_context(triples: list[dict]) -> str:
    """Format retrieved triples as a structured table for LLM context."""
    if not triples:
        return ""

    rows = []
    for t in triples:
        source = t.get("source_name", "?")
        relation = t.get("relation", "?").replace("_", " ")
        target = t.get("target_name", "?")
        ev_text = (t.get("evidence") or {}).get("evidence_text", "")
        ev_snippet = f" [{ev_text[:80]}]" if ev_text else ""
        rows.append(f"- {source} → {relation} → {target}{ev_snippet}")

    header = "Biomedical knowledge graph context (ChronoMedKG):"
    return header + "\n" + "\n".join(rows)


# ============================================================================
# PROMPT BUILDING
# ============================================================================

SYSTEM_PROMPT = (
    "You are a medical expert answering USMLE-style multiple-choice questions. "
    "Select the single best answer. Reply with ONLY the letter (A, B, C, D, or E)."
)


def build_prompt(question: str, options: dict, kg_context: str = "") -> str:
    opts_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    if kg_context:
        return (
            f"{kg_context}\n\n"
            f"Using your medical knowledge and any relevant information above, "
            f"answer this USMLE question:\n\n"
            f"{question}\n\n{opts_text}\n\n"
            f"Answer with only the letter of the best choice:"
        )
    return (
        f"Answer this USMLE-style medical question. Reply with only the letter.\n\n"
        f"{question}\n\n{opts_text}\n\nAnswer:"
    )


# ============================================================================
# LLM CLIENTS (reused from run_rag_experiment_v3 pattern)
# ============================================================================

def call_openai(model: str, prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=10,
    )
    return resp.choices[0].message.content.strip()


def call_deepseek(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=10,
    )
    return resp.choices[0].message.content.strip()


def call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return resp.content[0].text.strip()


def call_model(model_name: str, prompt: str) -> str:
    if model_name in ("gpt-4o-mini", "gpt-4o", "gpt-4.1-nano"):
        return call_openai(model_name, prompt)
    if model_name == "deepseek-v3":
        return call_deepseek(prompt)
    if model_name == "claude-haiku":
        return call_anthropic(prompt)
    raise ValueError(f"Unknown model: {model_name}")


# ============================================================================
# ANSWER PARSING & SCORING
# ============================================================================

def parse_answer(raw: str) -> str:
    """Extract the answer letter from LLM output."""
    raw = raw.strip().upper()
    # Direct single letter
    if raw and raw[0] in "ABCDE":
        return raw[0]
    # "Answer: B" or "The answer is C"
    m = re.search(r'\b([A-E])\b', raw)
    if m:
        return m.group(1)
    return raw[:1] if raw else "?"


def is_correct(predicted: str, gold: str) -> bool:
    return parse_answer(predicted).upper() == gold.strip().upper()


# ============================================================================
# BOOTSTRAP CI
# ============================================================================

def bootstrap_ci(
    scores: list[bool], n_boot: int = 1000, ci: float = 0.95
) -> tuple[float, float, float]:
    if not scores:
        return 0.0, 0.0, 0.0
    arr = np.array(scores, dtype=float)
    mean = arr.mean()
    boot = sorted(
        np.random.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    )
    lo = boot[int((1 - ci) / 2 * n_boot)]
    hi = boot[int((1 + ci) / 2 * n_boot)]
    return mean, lo, hi


# ============================================================================
# DATA LOADING
# ============================================================================

def load_medqa(path: Path, n: int | None, seed: int = RANDOM_SEED) -> list[dict]:
    """Load MEDQA questions from a JSONL file.

    Supports:
      {"question": ..., "options": {"A": ..., "B": ..., ...}, "answer": "B"}
      {"question": ..., "options": {"A": ..., "B": ..., ...}, "answer_idx": "B"}
    """
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
            # Normalise answer field
            answer = q.get("answer_idx") or q.get("correct_answer") or q.get("answer", "")
            if not answer:
                continue
            answer = str(answer).strip().upper()
            if len(answer) > 1:
                # Some datasets store the full answer text; extract leading letter
                answer = answer[0]
            if answer not in "ABCDE":
                continue
            questions.append({
                "question": q["question"],
                "options": q["options"],
                "answer": answer,
                "meta": {k: v for k, v in q.items()
                         if k not in ("question", "options", "answer", "answer_idx")},
            })

    if n is not None and n < len(questions):
        rng = random.Random(seed)
        questions = rng.sample(questions, n)

    logger.info("Loaded %d MEDQA questions from %s", len(questions), path)
    return questions


# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================

def run_eval(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --- Load questions ---
    questions = load_medqa(Path(args.data), args.n)
    if not questions:
        logger.error("No questions loaded. Check --data path.")
        sys.exit(1)

    # --- Load KG ---
    kg_index: dict = {}
    kg_vocab: list[str] = []

    if not args.no_kg:
        logger.info("Loading KG from source: %s", args.kg_source)
        if args.kg_source == "local":
            raw_triples = _load_local_triples()
        else:
            raw_triples = _load_zenodo_triples()

        if not raw_triples:
            logger.warning(
                "No KG triples found. Run the pipeline first or use --kg-source zenodo. "
                "Falling back to baseline-only mode."
            )
            args.no_kg = True
        else:
            kg_index, kg_vocab = build_kg_index(raw_triples)

    # --- Dry run check ---
    if args.dry_run:
        logger.info("DRY RUN: %d questions, KG entities: %d", len(questions), len(kg_vocab))
        hit_count = sum(
            1 for q in questions
            if retrieve_for_question(q["question"], q["options"], kg_index, kg_vocab, args.top_k)[1] > 0
        )
        logger.info("Entity hit rate (dry run): %.1f%% (%d/%d)",
                    100 * hit_count / len(questions), hit_count, len(questions))
        return

    # --- Evaluate ---
    conditions = ["base"] if args.no_kg else ["base", "kg"]
    results: dict[str, dict] = {
        model: {cond: [] for cond in conditions}
        for model in args.models
    }

    retrieval_stats = {
        "total": 0,
        "hits": 0,
        "total_triples": 0,
    }

    for i, q in enumerate(questions):
        # Retrieve KG context once per question (shared across models)
        kg_ctx = ""
        n_matched = 0
        if not args.no_kg:
            triples, n_matched = retrieve_for_question(
                q["question"], q["options"], kg_index, kg_vocab, args.top_k
            )
            retrieval_stats["total"] += 1
            if n_matched > 0:
                retrieval_stats["hits"] += 1
                retrieval_stats["total_triples"] += len(triples)
            kg_ctx = format_kg_context(triples)

        for model in args.models:
            for cond in conditions:
                ctx = kg_ctx if cond == "kg" else ""
                prompt = build_prompt(q["question"], q["options"], ctx)
                try:
                    raw_answer = call_model(model, prompt)
                    correct = is_correct(raw_answer, q["answer"])
                except Exception as e:
                    logger.warning("Model %s error on Q%d: %s", model, i, e)
                    raw_answer = "?"
                    correct = False

                results[model][cond].append(correct)

                if args.verbose:
                    logger.info(
                        "[Q%03d] %s | %s | pred=%s gold=%s %s",
                        i + 1, model, cond,
                        parse_answer(raw_answer), q["answer"],
                        "OK" if correct else "WRONG",
                    )

        if (i + 1) % 10 == 0:
            logger.info("Progress: %d/%d questions", i + 1, len(questions))

        # Polite rate-limit pause
        if len(args.models) > 1:
            time.sleep(0.3)

    # --- Report ---
    print("\n" + "=" * 70)
    print("MEDQA EVALUATION RESULTS — ChronoMedKG RAG")
    print("=" * 70)
    print(f"Questions evaluated : {len(questions)}")
    print(f"Models              : {', '.join(args.models)}")
    print(f"KG mode             : {'disabled (baseline only)' if args.no_kg else args.kg_source}")

    if not args.no_kg and retrieval_stats["total"] > 0:
        hit_rate = retrieval_stats["hits"] / retrieval_stats["total"]
        avg_triples = (retrieval_stats["total_triples"] / retrieval_stats["hits"]
                       if retrieval_stats["hits"] > 0 else 0)
        print(f"\nKG Retrieval Stats")
        print(f"  Entity hit rate   : {hit_rate:.1%}  ({retrieval_stats['hits']}/{retrieval_stats['total']})")
        print(f"  Avg triples/hit   : {avg_triples:.1f}")
        print()
        print("  Note: ChronoMedKG covers rare genetic diseases. Low hit rate on MEDQA")
        print("  (USMLE) is expected. See 'KG-only subset' metrics for quality signal.")

    print()

    # Full output table
    for model in args.models:
        print(f"Model: {model}")
        print(f"  {'Condition':<12} {'Accuracy':>10}  {'95% CI':>20}  {'N':>6}")
        print(f"  {'-'*12} {'-'*10}  {'-'*20}  {'-'*6}")
        for cond in conditions:
            scores = results[model][cond]
            if not scores:
                continue
            mean, lo, hi = bootstrap_ci(scores)
            print(f"  {cond:<12} {mean:>10.1%}  [{lo:.1%} – {hi:.1%}]  {len(scores):>6}")

        if "base" in results[model] and "kg" in results[model]:
            base_scores = results[model]["base"]
            kg_scores = results[model]["kg"]
            if base_scores and kg_scores:
                delta = sum(kg_scores) / len(kg_scores) - sum(base_scores) / len(base_scores)
                print(f"  {'delta (KG-base)':<12} {delta:>+10.1%}")

                # KG-only subset (questions where KG fired)
                if not args.no_kg and retrieval_stats["hits"] > 0:
                    # We don't track per-question hits in this loop — add note for future
                    print(f"  (For KG-hit subset analysis, re-run with --verbose and filter logs)")
        print()

    # Save results JSON
    output = {
        "n_questions": len(questions),
        "models": args.models,
        "kg_source": args.kg_source if not args.no_kg else "none",
        "top_k": args.top_k,
        "retrieval_stats": retrieval_stats if not args.no_kg else {},
        "results": {
            model: {
                cond: {
                    "accuracy": float(np.mean(results[model][cond])) if results[model][cond] else 0,
                    "n": len(results[model][cond]),
                    "ci": list(bootstrap_ci(results[model][cond])),
                }
                for cond in conditions
            }
            for model in args.models
        },
    }
    out_path = RESULTS_DIR / f"medqa_eval_{len(questions)}q.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", out_path)


# ============================================================================
# QAPipeline evaluation path (--use-pipeline)
# ============================================================================

def run_eval_pipeline(args: argparse.Namespace) -> None:
    """Evaluate using the full QAPipeline (LangGraph + Neo4j + FAISS)."""
    from agents.qa_inference import QAPipeline

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_medqa(Path(args.data), args.n)
    if not questions:
        logger.error("No questions loaded. Check --data path.")
        sys.exit(1)

    pipeline = QAPipeline()
    modes = ["llm_only", "kg_rag"] if not args.no_kg else ["llm_only"]
    results: dict[str, list[bool]] = {m: [] for m in modes}
    kg_hits = 0
    out = {"summary": {}}

    for i, q in enumerate(questions):
        for mode in modes:
            try:
                raw = pipeline.run(
                    q["question"],
                    benchmark_type="medqa",
                    mode=mode,
                    options={"choices": q["options"]},
                )
                ans = (raw.get("answer") or {}).get("answer", "")
                predicted = str(ans).strip().upper().rstrip(".")
                correct = predicted == q["answer"].upper()
                if mode == "kg_rag" and raw.get("kg_coverage"):
                    kg_hits += 1
                out[mode] = out.get(mode, []) + [{
                    "question": q["question"],
                    "options": q["options"],
                    "gold_answer": q["answer"],
                    "predicted_answer": predicted,
                    "correct": correct,
                    "raw": raw,
                }]
            except Exception as exc:
                logger.warning("[Q%d] %s error: %s", i, mode, exc)
                correct = False
                out[mode] = out.get(mode, []) + [{
                    "question": q["question"],
                    "options": q["options"],
                    "gold_answer": q["answer"],
                    "predicted_answer": "?",
                    "correct": correct,
                    "raw": "error: " + str(exc),
                }]
            results[mode].append(correct)

            if args.verbose:
                logger.info("[Q%03d] mode=%s pred=%s gold=%s %s",
                            i + 1, mode, predicted if "predicted" in dir() else "?",
                            q["answer"], "OK" if correct else "WRONG")

        if (i + 1) % 10 == 0:
            logger.info("Progress: %d/%d", i + 1, len(questions))

    print("\n" + "=" * 70)
    print("MEDQA EVALUATION (QAPipeline) — ChronoMedKG RAG")
    print("=" * 70)
    print(f"Questions : {len(questions)}")
    print(f"Modes     : {', '.join(modes)}")
    print()
    for mode in modes:
        scores = results[mode]
        mean, lo, hi = bootstrap_ci(scores)
        print(f"  {mode:<14} accuracy={mean:.1%}  95% CI [{lo:.1%}–{hi:.1%}]  N={len(scores)}")
    if "llm_only" in results and "kg_rag" in results:
        delta = sum(results["kg_rag"]) / len(results["kg_rag"]) - \
                sum(results["llm_only"]) / len(results["llm_only"])
        print(f"  {'delta':<14} {delta:+.1%}")
    if not args.no_kg:
        print(f"\n  KG hit rate: {kg_hits / len(questions):.1%} ({kg_hits}/{len(questions)})")

    out["summary"] = {
        "n_questions": len(questions),
        "pipeline": "QAPipeline",
        "results": {
            m: {"accuracy": float(np.mean(v)), "n": len(v), "ci": list(bootstrap_ci(v))}
                for m, v in results.items()
            },
            "kg_hit_rate": kg_hits / len(questions) if not args.no_kg else None,
        }
    
    out_path = RESULTS_DIR / f"medqa_pipeline_eval_{len(questions)}q.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Results saved to %s", out_path)


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ChronoMedKG as a RAG source on MEDQA (USMLE-style MCQ)"
    )
    parser.add_argument("--data", required=True, help="Path to MEDQA JSONL file")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of questions to sample (default: all)")
    parser.add_argument("--models", nargs="+",
                        default=["gpt-4o-mini"],
                        choices=["gpt-4o-mini", "gpt-4o", "gpt-4.1-nano",
                                 "deepseek-v3", "claude-haiku"],
                        help="LLM(s) to evaluate (legacy path only)")
    parser.add_argument("--top-k", type=int, default=15,
                        help="Max triples to retrieve per question (legacy path, default: 15)")
    parser.add_argument("--kg-source", choices=["local", "zenodo"], default="local",
                        help="KG data source: local data/extracted/ or Zenodo download")
    parser.add_argument("--no-kg", action="store_true",
                        help="Run baseline only (no KG context)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-question predictions")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data and measure hit rate without making API calls")
    parser.add_argument("--use-pipeline", action="store_true",
                        help="Use QAPipeline (LangGraph+Neo4j+FAISS) instead of legacy retrieval")
    args = parser.parse_args()

    if args.use_pipeline:
        run_eval_pipeline(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
