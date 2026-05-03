#!/usr/bin/env python3
"""SapBERT-based phenotype canonicalization for ChronoMedKG.

Goal: reduce ~108K free-text TA phenotype names to a smaller set of canonical
concepts by embedding with SapBERT and union-finding neighbors above a cosine
threshold. Output: mapping file (JSON) that later experiments can consume.

Run with the SapBERT venv:
    .venv-sapbert/bin/python scripts/experiment_sapbert_canonicalize.py

Audit checkpoints are logged throughout; the pipeline aborts on anomalies.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import re
import unicodedata

import numpy as np


def prenormalize(name: str) -> str:
    """Light string normalisation applied BEFORE SapBERT embedding.
    Strips punctuation differences, possessives, case, extra whitespace so that
    "Gowers sign" and "Gowers' sign" collapse to the same string. Does not attempt
    semantic normalisation — that is SapBERT's job."""
    n = unicodedata.normalize("NFKD", name).lower().strip()
    n = n.replace("'", "").replace("`", "").replace("\u2019", "")
    n = re.sub(r"\s+", " ", n)
    n = re.sub(r"[.,;:]$", "", n)
    return n

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sapbert")

ROOT = Path(__file__).resolve().parent.parent
TRIPLES = ROOT / "data" / "huggingface_upload" / "validated_triples.jsonl"
OUT_DIR = ROOT / "data" / "sapbert"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
THRESHOLD = 0.90      # cosine-similarity threshold for merging (raised from 0.85
                       # after pilot: 0.85 merged "neonatal presentation" with "adult
                       # presentation" due to shared word dominance)
BATCH_SIZE = 128
TOPK = 10             # k-NN per entity for cluster graph

# Phrases that are onset-stage meta-labels, not clinical phenotypes. We do NOT
# merge these across categories — treat each as its own canonical.
META_LABEL_PROTECTED_EXACT = {
    "symptom onset", "diagnosis", "death", "organ failure", "infant mortality",
    "age of onset", "onset age", "disease onset", "clinical presentation",
}
# Patterns that match onset-stage labels ("X presentation", "X-onset"). Any two
# names matching this pattern are prevented from merging across each other
# (e.g., "neonatal presentation" vs "adult presentation").
import re as _re
META_PATTERN = _re.compile(
    r"^(neonatal|congenital|prenatal|fetal|birth|newborn|infant(ile|cy|)?|"
    r"pediatric|childhood|early childhood|late childhood|juvenile|adolescent|"
    r"young adult|adult|elderly|late[- ]onset|early[- ]onset)\s+(presentation|onset)$",
    _re.IGNORECASE,
)

def is_protected_meta(name: str) -> bool:
    n = name.lower().strip()
    return n in META_LABEL_PROTECTED_EXACT or bool(META_PATTERN.match(n))

# --------------------------------------------------------------------------
# 1. Load phenotype vocabulary
# --------------------------------------------------------------------------
def load_phenotypes() -> list[tuple[str, int]]:
    """Load raw phenotype names with counts. Case / apostrophe / whitespace
    variants collapse via prenormalize() before the expensive SapBERT step."""
    log.info("Loading phenotypes from %s", TRIPLES)
    raw_counts: Counter[str] = Counter()
    total = 0
    with open(TRIPLES) as f:
        for line in f:
            t = json.loads(line)
            if t.get("target_type") == "phenotype":
                name = (t.get("target_name") or "").strip()
                if name:
                    raw_counts[name] += 1
                    total += 1
    # Collapse trivial variants with prenormalize (keeps the most-frequent
    # surface form as the key; counts aggregated)
    collapsed: Counter[str] = Counter()
    canonical_surface: dict[str, str] = {}  # prenorm -> most-frequent raw form
    for raw, c in raw_counts.items():
        key = prenormalize(raw)
        if not key:
            continue
        collapsed[key] += c
        if key not in canonical_surface or c > raw_counts.get(canonical_surface[key], 0):
            canonical_surface[key] = raw
    # Use the most-frequent raw form as the display name (preserves nice casing
    # for the canonical output).
    items = sorted(((canonical_surface[k], c) for k, c in collapsed.items()),
                   key=lambda x: -x[1])
    log.info("Loaded %d phenotype triples; %d raw names -> %d after pre-norm",
             total, len(raw_counts), len(items))
    return items


# --------------------------------------------------------------------------
# 2. SapBERT embedding
# --------------------------------------------------------------------------
def embed_with_sapbert(names: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading SapBERT (%s) on %s", MODEL_NAME, device)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    out = np.zeros((len(names), 768), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(names), BATCH_SIZE):
            batch = names[i : i + BATCH_SIZE]
            toks = tok(batch, padding=True, truncation=True, max_length=32, return_tensors="pt").to(device)
            # SapBERT convention: use [CLS] embedding (index 0)
            vecs = model(**toks).last_hidden_state[:, 0, :].cpu().numpy()
            out[i : i + len(batch)] = vecs
            if (i // BATCH_SIZE) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + len(batch)) / max(elapsed, 0.001)
                log.info("  embedded %d/%d (%.0f/s)", i + len(batch), len(names), rate)
    # L2 normalize -> dot product == cosine similarity
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1
    out /= norms

    # ---- Audit: check embeddings are non-degenerate ----
    nan_count = int(np.isnan(out).sum())
    zero_rows = int((np.linalg.norm(out, axis=1) < 1e-6).sum())
    assert nan_count == 0, f"AUDIT FAIL: {nan_count} NaN entries in embeddings"
    assert zero_rows == 0, f"AUDIT FAIL: {zero_rows} zero-norm embeddings"
    # Check variance: if all vectors are identical, normalization failed
    sample = out[: min(100, len(out))] @ out[0]
    log.info("Audit: embeddings OK (no NaN, no zero-norm, self=%.4f, mean@[0]=%.4f, std@[0]=%.4f)",
             float(out[0] @ out[0]), float(np.mean(sample)), float(np.std(sample)))
    assert float(np.std(sample)) > 1e-3, "AUDIT FAIL: embeddings have no variance"
    return out


# --------------------------------------------------------------------------
# 3. Build kNN graph + union-find clusters
# --------------------------------------------------------------------------
def cluster_by_similarity(
    names: list[str], counts: list[int], emb: np.ndarray, threshold: float, topk: int
) -> dict[str, str]:
    """Union-find clustering: for each entity, merge with its top-K neighbors
    whose cosine similarity exceeds threshold. Returns {original -> canonical}."""
    n = len(names)
    try:
        import faiss  # type: ignore
        log.info("Using FAISS for kNN (faiss=%s)", faiss.__version__)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        sims, idxs = index.search(emb, topk + 1)
    except ImportError:
        log.info("FAISS not available; using chunked matmul (slower)")
        chunk = 2048
        sims = np.zeros((n, topk + 1), dtype=np.float32)
        idxs = np.zeros((n, topk + 1), dtype=np.int64)
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            S = emb[a:b] @ emb.T  # [chunk, N]
            # top-k per row (descending)
            part = np.argpartition(-S, topk, axis=1)[:, : topk + 1]
            for r in range(b - a):
                order = part[r][np.argsort(-S[r, part[r]])]
                idxs[a + r] = order
                sims[a + r] = S[r, order]
            if (a // chunk) % 5 == 0:
                log.info("  kNN rows %d/%d", b, n)

    # Union-find
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Protected meta-labels: do not union across category, even if SapBERT says
    # they're close. These are onset-stage labels that linguistically share
    # "X presentation" so they embed near each other even though they're
    # semantically opposite.
    protected_idx = {i for i, n_ in enumerate(names) if is_protected_meta(n_)}
    log.info("  %d protected meta-label names excluded from clustering", len(protected_idx))

    # Build a mutual-kNN edge set: i~j only if i is in j's top-k AND j is in i's top-k
    # above threshold. Prevents transitive chaining through a dense graph that
    # would otherwise union ~22% of all phenotypes into a single "mega-cluster".
    knn_set: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for k in range(1, topk + 1):
            j = int(idxs[i, k])
            if j == i or sims[i, k] < threshold: continue
            knn_set[i].add(j)

    n_edges = 0
    mutual_pairs = 0
    nonmutual_skipped = 0
    for i in range(n):
        for j in knn_set[i]:
            if j <= i: continue           # process each pair once
            if i not in knn_set[j]:
                nonmutual_skipped += 1
                continue
            # Protected meta-labels: do not merge either side
            if i in protected_idx or j in protected_idx:
                continue
            union(i, j); n_edges += 1; mutual_pairs += 1
    log.info("  mutual-kNN: %d pairs merged, %d non-mutual skipped",
             mutual_pairs, nonmutual_skipped)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    log.info("Formed %d clusters from %d entities (%d union edges at sim >= %.2f)",
             len(groups), n, n_edges, threshold)

    # Pick canonical per group: most-frequent name in the cluster
    mapping: dict[str, str] = {}
    for root, members in groups.items():
        rep_idx = max(members, key=lambda m: counts[m])
        canon = names[rep_idx]
        for m in members:
            mapping[names[m]] = canon
    return mapping


# --------------------------------------------------------------------------
# 4. Audits
# --------------------------------------------------------------------------
def audit_mapping(mapping: dict[str, str], counts_by_name: dict[str, int]) -> None:
    log.info("=== AUDIT: canonicalisation mapping ===")
    uniq_before = len(mapping)
    uniq_after = len(set(mapping.values()))
    log.info("  unique names  before = %d", uniq_before)
    log.info("  unique canons after  = %d  (%.1fx reduction)",
             uniq_after, uniq_before / max(uniq_after, 1))

    # Cluster-size histogram
    clusters: dict[str, list[str]] = defaultdict(list)
    for orig, canon in mapping.items():
        clusters[canon].append(orig)
    sizes = [len(v) for v in clusters.values()]
    log.info("  cluster size: mean=%.2f median=%.0f max=%d", np.mean(sizes), np.median(sizes), max(sizes))
    bigs = sorted(clusters.items(), key=lambda kv: -len(kv[1]))[:3]
    for canon, members in bigs:
        sample = members[:6]
        log.info("  top cluster '%s' (n=%d): %s", canon, len(members), sample)

    # Triple-count preservation: sum of counts under each canon should equal original total
    orig_total = sum(counts_by_name.values())
    canon_total = sum(counts_by_name[orig] for orig in mapping)
    assert canon_total == orig_total, f"AUDIT FAIL: triple counts do not match ({canon_total} vs {orig_total})"
    log.info("  triple counts preserved: %d == %d (OK)", canon_total, orig_total)

    # Spot-check expected merges (sanity)
    expected_merges = [
        ("proximal weakness", "proximal muscle weakness"),
        ("delayed walking", "motor delay"),
        ("Gowers sign", "Gowers' sign"),
        ("cardiomyopathy", "dilated cardiomyopathy"),  # may or may not merge
    ]
    log.info("  spot-check merges (same canon means merged):")
    for a, b in expected_merges:
        ca = mapping.get(a, "(not-in-vocab)")
        cb = mapping.get(b, "(not-in-vocab)")
        same = "✓ MERGED" if ca == cb and ca != "(not-in-vocab)" else "✗ separate"
        log.info("    %-35s | %-35s -> %s", a, b, same)


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--limit", type=int, default=0,
                    help="limit to top-N most frequent names (0=all; useful for pilot runs)")
    args = ap.parse_args()

    items = load_phenotypes()
    if args.limit:
        items = items[: args.limit]
        log.info("PILOT mode: restricting to top-%d names", args.limit)
    names = [n for n, _ in items]
    counts = [c for _, c in items]
    counts_by_name = dict(items)

    log.info("Embedding %d phenotype names with SapBERT...", len(names))
    t0 = time.time()
    emb_cache = OUT_DIR / f"embeddings_top{args.limit or 'all'}.npy"
    names_cache = OUT_DIR / f"names_top{args.limit or 'all'}.json"
    if emb_cache.exists() and names_cache.exists():
        cached_names = json.loads(names_cache.read_text())
        if cached_names == names:
            log.info("Loading cached embeddings from %s", emb_cache)
            emb = np.load(emb_cache)
        else:
            emb = embed_with_sapbert(names)
            np.save(emb_cache, emb)
            names_cache.write_text(json.dumps(names))
    else:
        emb = embed_with_sapbert(names)
        np.save(emb_cache, emb)
        names_cache.write_text(json.dumps(names))
    log.info("Embedding done in %.1fs", time.time() - t0)

    log.info("Clustering at cosine >= %.2f (top-%d neighbours)...", args.threshold, args.topk)
    t0 = time.time()
    mapping = cluster_by_similarity(names, counts, emb, args.threshold, args.topk)
    log.info("Clustering done in %.1fs", time.time() - t0)

    audit_mapping(mapping, counts_by_name)

    tag = f"thr{args.threshold}_top{args.topk}"
    if args.limit:
        tag += f"_pilot{args.limit}"
    out_path = OUT_DIR / f"canonical_mapping_{tag}.json"
    out_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2))
    log.info("Wrote %s (%d entries)", out_path, len(mapping))


if __name__ == "__main__":
    main()
