#!/usr/bin/env python3
"""
Mine disease names from MedQA (Tầng A) + BioASQ (Tầng B) + HPO HPOA (Tầng C),
normalize to OMIM IDs, and output a TSV for the orchestrator.

Output: data/medqa/diseases_from_benchmarks.tsv
Format: disease_id<TAB>disease_name (compatible with --disease-file)

Usage:
    python scripts/update/build_disease_list_from_benchmarks.py
    python scripts/update/build_disease_list_from_benchmarks.py --batch-size 64 --min-freq 2
    python scripts/update/build_disease_list_from_benchmarks.py --cache-ner data/medqa/ner_cache.json
    python scripts/update/build_disease_list_from_benchmarks.py --bioasq-file data/bioasq/training14b.json
    python scripts/update/build_disease_list_from_benchmarks.py --skip-bioasq
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MEDQA_FILES = [
    PROJECT_ROOT / "data" / "medqa" / "train.jsonl",
    PROJECT_ROOT / "data" / "medqa" / "test.jsonl",
]
BIOASQ_PATH = PROJECT_ROOT / "data" / "bioasq" / "training14b.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "diseases" / "diseases_from_benchmarks.tsv"

# HPO evidence codes that are NOT reliable (inferred electronic annotation)
_UNRELIABLE_EVIDENCE = {"IEA"}

# Simple normalizer: lowercase, collapse whitespace, strip trailing punctuation
_PUNCT_RE = re.compile(r"[^\w\s-]")
_SPACE_RE = re.compile(r"\s+")


def _normalize(name: str) -> str:
    name = name.lower().strip()
    name = _PUNCT_RE.sub("", name)
    name = _SPACE_RE.sub(" ", name)
    return name.strip()


# ---------------------------------------------------------------------------
# Step 1 — Load MedQA
# ---------------------------------------------------------------------------

def load_medqa_texts(paths: list[Path]) -> list[str]:
    texts = []
    for path in paths:
        if not path.exists():
            print(f"  [WARN] MedQA file not found, skipping: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                q = json.loads(line)
                combined = q.get("question", "") + " " + " ".join(q.get("options", {}).values())
                texts.append(combined)
    print(f"  [Tầng A] Loaded {len(texts)} MedQA question+options strings")
    return texts


# ---------------------------------------------------------------------------
# Step 1b — Load BioASQ
# ---------------------------------------------------------------------------

def load_bioasq_texts(path: Path) -> list[str]:
    """Extract question body + ideal answers + snippet texts from BioASQ JSON."""
    if not path.exists():
        print(f"  [WARN] BioASQ file not found, skipping: {path}")
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    texts: list[str] = []
    for q in questions:
        parts: list[str] = []
        body = q.get("body", "")
        if body:
            parts.append(body)
        for ans in q.get("ideal_answer", []):
            if isinstance(ans, str) and ans:
                parts.append(ans)
        for snippet in q.get("snippets", []):
            text = snippet.get("text", "")
            if text:
                parts.append(text)
        if parts:
            texts.append(" ".join(parts))

    print(f"  [Tầng B] Loaded {len(texts)} BioASQ question+answer+snippet strings from {path.name}")
    return texts


# ---------------------------------------------------------------------------
# Step 2 — NER with openmed
# ---------------------------------------------------------------------------

def run_ner(texts: list[str], batch_size: int, cache_path: Path | None, tier_label: str = "Tầng A") -> Counter:
    """Return Counter of normalized disease name → frequency."""
    if cache_path and cache_path.exists():
        print(f"  [{tier_label}] Loading NER cache from {cache_path}")
        with open(cache_path, encoding="utf-8") as f:
            raw = json.load(f)
        return Counter(raw)

    from openmed import process_batch

    freq: Counter = Counter()
    total = len(texts)
    processed = 0

    print(f"  [{tier_label}] Running openmed NER on {total} texts (batch_size={batch_size})")
    print(f"  [{tier_label}] First run may take ~30-60s for model load...")

    for i in range(0, total, batch_size):
        chunk = texts[i : i + batch_size]
        result = process_batch(chunk, model_name="disease_detection_superclinical")
        for item in result.get_successful_results():
            if item.result is None:
                continue
            for entity in item.result.entities:
                if "DISEASE" in entity.label.upper() and entity.confidence >= 0.5:
                    norm = _normalize(entity.text)
                    if norm and len(norm) >= 3:
                        freq[norm] += 1
        processed += len(chunk)
        if processed % 500 == 0 or processed == total:
            print(f"  [{tier_label}] {processed}/{total} texts processed, {len(freq)} unique entities so far")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(dict(freq), f, ensure_ascii=False, indent=2)
        print(f"  [{tier_label}] NER cache saved → {cache_path}")

    return freq


# ---------------------------------------------------------------------------
# Step 3 — OMIM Normalization via biopython Entrez
# ---------------------------------------------------------------------------

def normalize_to_omim(
    disease_names: list[str],
    top_k: int,
) -> list[tuple[str, str]]:
    """Return list of (disease_id, disease_name). Uses OMIM: prefix or NAME: fallback."""
    from Bio import Entrez

    Entrez.email = "cuongdm.forwork@gmail.com"
    ncbi_key = os.environ.get("NCBI_API_KEY", "")
    if ncbi_key:
        Entrez.api_key = ncbi_key
    delay = 0.034 if ncbi_key else 0.11

    names_to_query = disease_names[:top_k]
    results = []
    omim_hits = 0
    name_fallbacks = 0

    print(f"  [Tầng A] Normalizing {len(names_to_query)} disease names via NCBI OMIM...")

    for i, name in enumerate(names_to_query):
        try:
            handle = Entrez.esearch(db="omim", term=f'"{name}"[TITL]', retmax=1)
            record = Entrez.read(handle)
            handle.close()
            ids = record.get("IdList", [])
            if ids:
                disease_id = f"OMIM:{ids[0]}"
                omim_hits += 1
            else:
                disease_id = f"NAME:{name.replace(' ', '_')}"
                name_fallbacks += 1
        except Exception:
            disease_id = f"NAME:{name.replace(' ', '_')}"
            name_fallbacks += 1

        results.append((disease_id, name))
        time.sleep(delay)

        if (i + 1) % 100 == 0:
            print(f"  [Tầng A] {i+1}/{len(names_to_query)} normalized — {omim_hits} OMIM, {name_fallbacks} NAME fallback")

    print(f"  [Tầng A] Normalization done: {omim_hits} OMIM IDs, {name_fallbacks} NAME fallbacks")
    return results


# ---------------------------------------------------------------------------
# Step 5 — Merge & output
# ---------------------------------------------------------------------------

def merge_and_write(
    tier_a: list[tuple[str, str]],
    output: Path,
) -> None:
    seen_ids: set[str] = set()
    rows: list[tuple[str, str]] = []

    # Tầng A first (MedQA — benchmark-derived, highest retrieval relevance)
    for did, name in tier_a:
        if did not in seen_ids:
            seen_ids.add(did)
            rows.append((did, name))

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for did, name in rows:
            f.write(f"{did}\t{name}\n")

    print(f"\n{'='*60}")
    print(f"[Tầng A] MedQA diseases      : {len(tier_a)}")
    print(f"[MERGED] Total unique diseases: {len(rows)}")
    print(f"[OUTPUT] {output}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build disease list from benchmarks for ChronoMedKG pipeline")
    p.add_argument("--batch-size", type=int, default=32, help="openmed NER chunk size (default: 32)")
    p.add_argument("--min-freq", type=int, default=0, help="Min benchmark occurrences to keep a disease (default: 2)")
    p.add_argument("--top-k", type=int, default=1000, help="Max diseases from Tầng A+B to normalize via OMIM (default: 1000)")
    p.add_argument("--bioasq-file", type=Path, default=BIOASQ_PATH, help="Path to BioASQ training JSON (default: data/bioasq/training14b.json)")
    p.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output TSV path")
    p.add_argument("--cache-ner", type=Path, default=None, help="Cache MedQA NER results to JSON (enables resume)")
    p.add_argument("--cache-ner-bioasq", type=Path, default=None, help="Cache BioASQ NER results to JSON (enables resume)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("\n=== ChronoMedKG Disease List Builder ===\n")

    # --- Tầng A: MedQA ---
    print("[Tầng A] MedQA disease mining")
    medqa_texts = load_medqa_texts(MEDQA_FILES)
    freq_a = run_ner(medqa_texts, batch_size=args.batch_size, cache_path=args.cache_ner, tier_label="Tầng A")

    # --- Tầng B: BioASQ ---
    freq_b: Counter = Counter()
    print("\n[Tầng B] BioASQ disease mining")
    bioasq_texts = load_bioasq_texts(args.bioasq_file)
    if bioasq_texts:
        freq_b = run_ner(bioasq_texts, batch_size=args.batch_size, cache_path=args.cache_ner_bioasq, tier_label="Tầng B")

    # Merge frequency counters from Tầng A + B, filter by min-freq, sort descending
    freq_combined = freq_a + freq_b
    filtered = [(name, count) for name, count in freq_combined.items() if count >= args.min_freq]
    filtered.sort(key=lambda x: x[1], reverse=True)
    a_unique = len(freq_a)
    b_unique = len(freq_b)
    print(f"\n  [A+B] MedQA unique entities: {a_unique}, BioASQ unique entities: {b_unique}")
    print(f"  [A+B] {len(filtered)} diseases with combined freq ≥ {args.min_freq} (from {len(freq_combined)} unique entities)")

    disease_names = [name for name, _ in filtered]
    tier_a = normalize_to_omim(disease_names, top_k=args.top_k)

    # --- Merge ---
    merge_and_write(tier_a, args.output)


if __name__ == "__main__":
    main()
