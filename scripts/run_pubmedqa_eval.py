#!/usr/bin/env python3
"""
PUBMEDQA Evaluation — ChronoMedKG as RAG Knowledge Source
==========================================================
Evaluates whether ChronoMedKG triples improve LLM accuracy on PubMedQA
(yes / no / maybe biomedical research question answering).

PubMedQA asks: given a research question (and optionally the abstract context),
predict the final answer: "yes", "no", or "maybe".

Two variants are supported
--------------------------
  PQA-L (labeled)   12k expert-labeled Q&A pairs — the standard eval split
  PQA-U (unlabeled) 211k auto-generated pairs — not commonly used for eval

Metrics reported
----------------
  accuracy_base    LLM accuracy without KG context
  accuracy_kg      LLM accuracy with ChronoMedKG triples injected
  delta            accuracy_kg - accuracy_base
  entity_hit_rate  % of questions with ≥1 KG entity match
  macro_f1         Macro-averaged F1 across yes/no/maybe classes
  95% CI           Bootstrap confidence intervals

Data format expected (JSON, dict of pubid → record)
----------------------------------------------------
  {
    "12345678": {
      "QUESTION": "Does aspirin reduce colorectal cancer risk?",
      "CONTEXTS": ["Abstract sentence 1.", "Abstract sentence 2.", ...],
      "LABELS": ["METHODS", "RESULTS", ...],
      "MESHES": ["Aspirin", "Colorectal Neoplasms"],
      "final_decision": "yes"
    },
    ...
  }

  Also supports the flat list variant:
  [{"pubid": ..., "question": ..., "context": ..., "final_decision": ...}, ...]

Public dataset: https://github.com/pubmedqa/pubmedqa  (data/ori_pqal.json)

Usage
-----
  # Quick test on 200 questions
  python -m scripts.run_pubmedqa_eval --data data/pubmedqa/ori_pqal.json --n 200

  # No abstract context (pure KG vs parametric knowledge)
  python -m scripts.run_pubmedqa_eval --data data/pubmedqa/ori_pqal.json --no-abstract

  # Baseline only
  python -m scripts.run_pubmedqa_eval --data data/pubmedqa/ori_pqal.json --no-kg

  # Dry run — check entity hit rate without API calls
  python -m scripts.run_pubmedqa_eval --data data/pubmedqa/ori_pqal.json --dry-run
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
logger = logging.getLogger("pubmedqa_eval")

EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
RESULTS_DIR = PROJECT_ROOT / "data" / "benchmark" / "pubmedqa_results"

RANDOM_SEED = 42
VALID_LABELS = {"yes", "no", "maybe"}


# ============================================================================
# KG LOADING & INDEXING  (same pattern as run_medqa_eval.py)
# ============================================================================

def _load_local_triples() -> list[dict]:
    triples = []
    if not EXTRACTED_DIR.exists():
        return triples
    for disease_dir in EXTRACTED_DIR.iterdir():
        if not disease_dir.is_dir():
            continue
        vt = disease_dir / "validated_triples.jsonl"
        if not vt.exists():
            continue
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
    from chronomedkg.loader import load_triples
    return list(load_triples())


def build_kg_index(triples: list[dict]) -> tuple[dict, list[str]]:
    index: dict[str, list[dict]] = defaultdict(list)
    for t in triples:
        for field in ("source_name", "target_name"):
            name = t.get(field, "").strip()
            if name:
                index[name.lower()].append(t)
                abbr = re.search(r'\(([A-Z][A-Z0-9]{1,9})\)', name)
                if abbr:
                    index[abbr.group(1).lower()].append(t)
    vocab = sorted(index.keys())
    logger.info("KG index: %d unique entities, %d total triples", len(vocab), len(triples))
    return dict(index), vocab


# ============================================================================
# ENTITY EXTRACTION & RETRIEVAL
# ============================================================================

_STOP = {
    "the", "and", "for", "with", "from", "this", "that", "have", "has", "are",
    "was", "were", "been", "being", "their", "which", "what", "does", "during",
    "study", "results", "methods", "patients", "subjects", "controls", "group",
    "groups", "showed", "found", "suggest", "association", "significant",
    "between", "among", "compared", "using", "used", "data", "analysis",
    "compared", "objective", "background", "conclusions", "conclusions",
}


def extract_candidate_entities(text: str, max_ngrams: int = 4) -> list[str]:
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
    mesh_terms: list[str],
    kg_index: dict,
    kg_vocab: list[str],
    top_k: int = 15,
    fuzzy_threshold: int = 85,
) -> tuple[list[dict], int]:
    """Retrieve KG triples relevant to a PubMedQA question.

    MeSH terms are used as high-confidence entity seeds when available.
    """
    # MeSH terms are reliable entity signals — check them first
    matched_entities: set[str] = set()
    for mesh in mesh_terms:
        mesh_lower = mesh.lower()
        if mesh_lower in kg_index:
            matched_entities.add(mesh_lower)

    # Then candidate n-gram matching from question text
    candidates = set(extract_candidate_entities(question_text))
    for cand in candidates:
        if cand in kg_index:
            matched_entities.add(cand)

    # Fuzzy matching for long candidates not yet matched
    try:
        from rapidfuzz import process, fuzz
        unmatched = [c for c in candidates if len(c) >= 5 and c not in matched_entities]
        if unmatched and kg_vocab:
            for cand in unmatched[:50]:
                hits = process.extract(
                    cand, kg_vocab,
                    scorer=fuzz.token_sort_ratio,
                    limit=2,
                    score_cutoff=fuzzy_threshold,
                )
                for match_name, score, _ in hits:
                    matched_entities.add(match_name)
    except ImportError:
        pass

    if not matched_entities:
        return [], 0

    seen_edge_ids: set[str] = set()
    scored: list[tuple[float, dict]] = []
    question_lower = question_text.lower()

    for entity in matched_entities:
        for t in kg_index.get(entity, []):
            eid = t.get("edge_id", id(t))
            if eid in seen_edge_ids:
                continue
            seen_edge_ids.add(eid)
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
    return "Biomedical knowledge graph context (ChronoMedKG):\n" + "\n".join(rows)


# ============================================================================
# PROMPT BUILDING
# ============================================================================

SYSTEM_PROMPT = (
    "You are a biomedical expert. Given a research question, "
    "answer with exactly one word: yes, no, or maybe."
)


def build_prompt(
    question: str,
    abstract_sentences: list[str],
    kg_context: str = "",
    include_abstract: bool = True,
) -> str:
    parts = []

    if kg_context:
        parts.append(kg_context)

    if include_abstract and abstract_sentences:
        abstract_text = " ".join(abstract_sentences[:10])  # cap to avoid token bloat
        parts.append(f"Research abstract:\n{abstract_text}")

    parts.append(
        f"Question: {question}\n\n"
        f"Based on the information above and your biomedical knowledge, "
        f"answer with exactly one word — yes, no, or maybe:"
    )

    return "\n\n".join(parts)


# ============================================================================
# LLM CLIENTS
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
        max_tokens=5,
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
        max_tokens=5,
    )
    return resp.choices[0].message.content.strip()


def call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=5,
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
    """Extract yes / no / maybe from LLM output."""
    lower = raw.strip().lower()
    if lower.startswith("yes"):
        return "yes"
    if lower.startswith("no"):
        return "no"
    if "maybe" in lower or "uncertain" in lower or "unclear" in lower:
        return "maybe"
    # Fallback: check anywhere in response
    for label in ("yes", "no", "maybe"):
        if label in lower:
            return label
    return "maybe"  # default to maybe on ambiguous output


def macro_f1(predictions: list[str], gold_labels: list[str]) -> float:
    """Compute macro-averaged F1 across yes/no/maybe."""
    f1s = []
    for label in VALID_LABELS:
        tp = sum(1 for p, g in zip(predictions, gold_labels) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, gold_labels) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, gold_labels) if p != label and g == label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        f1s.append(f1)
    return sum(f1s) / len(f1s)


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

def load_pubmedqa(path: Path, n: int | None, seed: int = RANDOM_SEED) -> list[dict]:
    """Load PubMedQA questions.

    Supports the standard ori_pqal.json format (dict keyed by pubid)
    and flat list variants.
    """
    with open(path) as f:
        raw = json.load(f)

    questions = []

    if isinstance(raw, dict):
        # Standard PQA-L format: {pubid: {QUESTION, CONTEXTS, LABELS, MESHES, final_decision}}
        for pubid, record in raw.items():
            decision = record.get("final_decision", "").lower().strip()
            if decision not in VALID_LABELS:
                continue
            questions.append({
                "pubid": pubid,
                "question": record.get("QUESTION", record.get("question", "")),
                "contexts": record.get("CONTEXTS", record.get("context", [])),
                "mesh_terms": record.get("MESHES", record.get("meshes", [])),
                "answer": decision,
            })
    elif isinstance(raw, list):
        # Flat list variant
        for record in raw:
            decision = record.get("final_decision", record.get("answer", "")).lower().strip()
            if decision not in VALID_LABELS:
                continue
            ctx = record.get("context", record.get("CONTEXTS", []))
            if isinstance(ctx, str):
                ctx = [ctx]
            questions.append({
                "pubid": record.get("pubid", ""),
                "question": record.get("question", record.get("QUESTION", "")),
                "contexts": ctx,
                "mesh_terms": record.get("meshes", record.get("MESHES", [])),
                "answer": decision,
            })

    if n is not None and n < len(questions):
        rng = random.Random(seed)
        questions = rng.sample(questions, n)

    # Log label distribution
    dist = {label: sum(1 for q in questions if q["answer"] == label) for label in VALID_LABELS}
    logger.info(
        "Loaded %d PubMedQA questions: yes=%d, no=%d, maybe=%d",
        len(questions), dist["yes"], dist["no"], dist["maybe"],
    )
    return questions


# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================

def run_eval(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_SEED)

    questions = load_pubmedqa(Path(args.data), args.n)
    if not questions:
        logger.error("No questions loaded. Check --data path.")
        sys.exit(1)

    # --- Load KG ---
    kg_index: dict = {}
    kg_vocab: list[str] = []

    if not args.no_kg:
        logger.info("Loading KG from source: %s", args.kg_source)
        raw_triples = (
            _load_local_triples() if args.kg_source == "local"
            else _load_zenodo_triples()
        )
        if not raw_triples:
            logger.warning("No KG triples found. Falling back to baseline-only mode.")
            args.no_kg = True
        else:
            kg_index, kg_vocab = build_kg_index(raw_triples)

    # --- Dry run ---
    if args.dry_run:
        hit_count = sum(
            1 for q in questions
            if retrieve_for_question(
                q["question"], q["mesh_terms"], kg_index, kg_vocab, args.top_k
            )[1] > 0
        )
        logger.info(
            "DRY RUN: %d questions | Entity hit rate: %.1f%% (%d/%d)",
            len(questions), 100 * hit_count / len(questions),
            hit_count, len(questions),
        )
        # Label distribution breakdown
        for label in VALID_LABELS:
            n_label = sum(1 for q in questions if q["answer"] == label)
            logger.info("  %s: %d (%.1f%%)", label, n_label, 100 * n_label / len(questions))
        return

    # --- Evaluate ---
    conditions = ["base"]
    if not args.no_kg:
        if args.no_abstract:
            conditions = ["base", "kg_only"]
        else:
            conditions = ["base", "abstract_only", "kg+abstract"]

    results: dict[str, dict] = {
        model: {cond: {"correct": [], "predicted": [], "gold": []} for cond in conditions}
        for model in args.models
    }

    retrieval_stats = {"total": 0, "hits": 0, "total_triples": 0}

    for i, q in enumerate(questions):
        kg_ctx = ""
        n_matched = 0
        if not args.no_kg:
            triples, n_matched = retrieve_for_question(
                q["question"], q["mesh_terms"], kg_index, kg_vocab, args.top_k
            )
            retrieval_stats["total"] += 1
            if n_matched > 0:
                retrieval_stats["hits"] += 1
                retrieval_stats["total_triples"] += len(triples)
            kg_ctx = format_kg_context(triples)

        for model in args.models:
            for cond in conditions:
                if cond == "base":
                    prompt = build_prompt(q["question"], [], "", include_abstract=False)
                elif cond == "abstract_only":
                    prompt = build_prompt(q["question"], q["contexts"], "", include_abstract=True)
                elif cond == "kg_only":
                    prompt = build_prompt(q["question"], [], kg_ctx, include_abstract=False)
                else:  # kg+abstract
                    prompt = build_prompt(q["question"], q["contexts"], kg_ctx, include_abstract=True)

                try:
                    raw_answer = call_model(model, prompt)
                    predicted = parse_answer(raw_answer)
                    correct = predicted == q["answer"]
                except Exception as e:
                    logger.warning("Model %s error on Q%d: %s", model, i, e)
                    predicted = "maybe"
                    correct = predicted == q["answer"]

                results[model][cond]["correct"].append(correct)
                results[model][cond]["predicted"].append(predicted)
                results[model][cond]["gold"].append(q["answer"])

                if args.verbose:
                    logger.info(
                        "[Q%03d] %s | %s | pred=%s gold=%s %s",
                        i + 1, model, cond,
                        predicted, q["answer"], "OK" if correct else "WRONG",
                    )

        if (i + 1) % 10 == 0:
            logger.info("Progress: %d/%d", i + 1, len(questions))

        if len(args.models) > 1:
            time.sleep(0.3)

    # --- Report ---
    print("\n" + "=" * 70)
    print("PUBMEDQA EVALUATION RESULTS — ChronoMedKG RAG")
    print("=" * 70)
    print(f"Questions evaluated : {len(questions)}")
    print(f"Models              : {', '.join(args.models)}")
    print(f"KG mode             : {'disabled' if args.no_kg else args.kg_source}")
    print(f"Abstract included   : {'no (--no-abstract)' if args.no_abstract else 'yes (default)'}")

    if not args.no_kg and retrieval_stats["total"] > 0:
        hit_rate = retrieval_stats["hits"] / retrieval_stats["total"]
        avg_t = (retrieval_stats["total_triples"] / retrieval_stats["hits"]
                 if retrieval_stats["hits"] > 0 else 0)
        print(f"\nKG Retrieval")
        print(f"  Entity hit rate : {hit_rate:.1%} ({retrieval_stats['hits']}/{retrieval_stats['total']})")
        print(f"  Avg triples/hit : {avg_t:.1f}")

    print()

    for model in args.models:
        print(f"Model: {model}")
        print(f"  {'Condition':<16} {'Accuracy':>10}  {'95% CI':>20}  {'Macro-F1':>10}  {'N':>6}")
        print(f"  {'-'*16} {'-'*10}  {'-'*20}  {'-'*10}  {'-'*6}")
        for cond in conditions:
            r = results[model][cond]
            if not r["correct"]:
                continue
            mean, lo, hi = bootstrap_ci(r["correct"])
            mf1 = macro_f1(r["predicted"], r["gold"])
            print(f"  {cond:<16} {mean:>10.1%}  [{lo:.1%} – {hi:.1%}]  {mf1:>10.3f}  {len(r['correct']):>6}")

        # Delta: best KG condition vs base
        base_acc = (sum(results[model]["base"]["correct"]) / len(results[model]["base"]["correct"])
                    if results[model]["base"]["correct"] else 0)
        for cond in conditions:
            if cond == "base" or not results[model][cond]["correct"]:
                continue
            kg_acc = sum(results[model][cond]["correct"]) / len(results[model][cond]["correct"])
            delta = kg_acc - base_acc
            print(f"  delta ({cond} vs base): {delta:+.1%}")
        print()

    # Save
    output = {
        "n_questions": len(questions),
        "models": args.models,
        "conditions": conditions,
        "kg_source": args.kg_source if not args.no_kg else "none",
        "include_abstract": not args.no_abstract,
        "retrieval_stats": retrieval_stats if not args.no_kg else {},
        "results": {
            model: {
                cond: {
                    "accuracy": float(np.mean(results[model][cond]["correct"]))
                    if results[model][cond]["correct"] else 0,
                    "macro_f1": macro_f1(
                        results[model][cond]["predicted"],
                        results[model][cond]["gold"],
                    ),
                    "n": len(results[model][cond]["correct"]),
                    "ci": list(bootstrap_ci(results[model][cond]["correct"])),
                    "label_dist": {
                        label: results[model][cond]["predicted"].count(label)
                        for label in VALID_LABELS
                    },
                }
                for cond in conditions
            }
            for model in args.models
        },
    }
    out_path = RESULTS_DIR / f"pubmedqa_eval_{len(questions)}q.json"
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
    questions = load_pubmedqa(Path(args.data), args.n)
    if not questions:
        logger.error("No questions loaded. Check --data path.")
        sys.exit(1)

    pipeline = QAPipeline()
    modes = ["llm_only", "kg_rag"] if not args.no_kg else ["llm_only"]
    results: dict[str, list[str]] = {m: [] for m in modes}
    gold: list[str] = [q["answer"] for q in questions]
    kg_hits = 0

    for i, q in enumerate(questions):
        for mode in modes:
            try:
                raw = pipeline.run(
                    q["question"],
                    benchmark_type="pubmedqa",
                    mode=mode,
                )
                ans = (raw.get("answer") or {}).get("answer", "")
                predicted = str(ans).strip().lower()
                if predicted not in VALID_LABELS:
                    predicted = "maybe"
                if mode == "kg_rag" and raw.get("kg_coverage"):
                    kg_hits += 1
            except Exception as exc:
                logger.warning("[Q%d] %s error: %s", i, mode, exc)
                predicted = "maybe"
            results[mode].append(predicted)

            if args.verbose:
                logger.info("[Q%03d] mode=%s pred=%s gold=%s %s",
                            i + 1, mode, predicted, q["answer"],
                            "OK" if predicted == q["answer"] else "WRONG")

        if (i + 1) % 10 == 0:
            logger.info("Progress: %d/%d", i + 1, len(questions))

    print("\n" + "=" * 70)
    print("PUBMEDQA EVALUATION (QAPipeline) — ChronoMedKG RAG")
    print("=" * 70)
    print(f"Questions : {len(questions)}")
    print()
    for mode in modes:
        preds = results[mode]
        acc = sum(p == g for p, g in zip(preds, gold)) / len(gold)
        mf1 = macro_f1(preds, gold)
        _, lo, hi = bootstrap_ci([p == g for p, g in zip(preds, gold)])
        print(f"  {mode:<14} accuracy={acc:.1%}  macro-F1={mf1:.3f}  95% CI [{lo:.1%}–{hi:.1%}]")
    if not args.no_kg:
        print(f"\n  KG hit rate: {kg_hits / len(questions):.1%} ({kg_hits}/{len(questions)})")

    out = {
        "n_questions": len(questions),
        "pipeline": "QAPipeline",
        "results": {
            m: {
                "accuracy": sum(p == g for p, g in zip(results[m], gold)) / len(gold),
                "macro_f1": macro_f1(results[m], gold),
            }
            for m in modes
        },
    }
    out_path = RESULTS_DIR / f"pubmedqa_pipeline_eval_{len(questions)}q.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Results saved to %s", out_path)


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ChronoMedKG as a RAG source on PubMedQA (yes/no/maybe)"
    )
    parser.add_argument("--data", required=True, help="Path to PubMedQA JSON file (ori_pqal.json)")
    parser.add_argument("--n", type=int, default=None,
                        help="Questions to sample (default: all)")
    parser.add_argument("--models", nargs="+",
                        default=["gpt-4o-mini"],
                        choices=["gpt-4o-mini", "gpt-4o", "gpt-4.1-nano",
                                 "deepseek-v3", "claude-haiku"],
                        help="LLM(s) to evaluate (legacy path only)")
    parser.add_argument("--top-k", type=int, default=15,
                        help="Max triples retrieved per question (legacy, default: 15)")
    parser.add_argument("--kg-source", choices=["local", "zenodo"], default="local",
                        help="KG data source")
    parser.add_argument("--no-kg", action="store_true",
                        help="Baseline only, no KG context")
    parser.add_argument("--no-abstract", action="store_true",
                        help="Do not include the PubMed abstract in the prompt")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-question results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Measure hit rate without API calls")
    parser.add_argument("--use-pipeline", action="store_true",
                        help="Use QAPipeline (LangGraph+Neo4j+FAISS) instead of legacy retrieval")
    args = parser.parse_args()

    if args.use_pipeline:
        run_eval_pipeline(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
