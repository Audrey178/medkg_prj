"""
4-strategy entity matching cascade for KG-RAG pipeline.

Strategies (in order):
  1. Exact — string match against name/aliases/CUI (confidence 1.0)
  2. Abbreviation — regex detection + alias expansion (confidence 0.95)
  3. Fuzzy — Levenshtein ratio via rapidfuzz (confidence = ratio)
  4. Semantic — cosine similarity via injected FAISS search fn (confidence = cosine)
"""

import re
from typing import Callable

from rapidfuzz import fuzz

from ..state import MatchedNode


# ---------------------------------------------------------------------------
# Strategy 1 — Exact
# ---------------------------------------------------------------------------

def exact_match(entity: str, candidates: list[dict]) -> list[MatchedNode]:
    entity_lower = entity.strip().lower()
    results: list[MatchedNode] = []

    for c in candidates:
        names = [c.get("name", "")] + c.get("aliases", [])
        for name in names:
            if entity_lower == name.strip().lower():
                results.append(MatchedNode(cui=c["cui"], name=c["name"], confidence=1.0, strategy="exact"))
                break

    return results


# ---------------------------------------------------------------------------
# Strategy 2 — Abbreviation expansion
# ---------------------------------------------------------------------------

_ABBR_PATTERN = re.compile(r"^[A-Z][A-Z0-9]{1,7}$")


def _looks_like_abbreviation(entity: str) -> bool:
    return bool(_ABBR_PATTERN.match(entity.strip()))


def abbreviation_match(entity: str, candidates: list[dict]) -> list[MatchedNode]:
    """Match if entity looks like an abbreviation and appears in a candidate's alias list."""
    if not _looks_like_abbreviation(entity):
        return []

    entity_upper = entity.strip().upper()
    results: list[MatchedNode] = []

    for c in candidates:
        aliases = [a.upper() for a in c.get("aliases", [])]
        if entity_upper in aliases:
            results.append(MatchedNode(cui=c["cui"], name=c["name"], confidence=0.95, strategy="abbreviation"))

    return results


# ---------------------------------------------------------------------------
# Strategy 3 — Fuzzy (Levenshtein ratio via rapidfuzz)
# ---------------------------------------------------------------------------

def fuzzy_match(entity: str, candidates: list[dict], threshold: float = 0.8) -> list[MatchedNode]:
    entity_q = entity.strip().lower()
    results: list[MatchedNode] = []

    for c in candidates:
        names = [c.get("name", "")] + c.get("aliases", [])
        best_ratio = 0.0
        for name in names:
            ratio = fuzz.ratio(entity_q, name.strip().lower()) / 100.0
            # Also try token_sort_ratio for multi-word entities
            ratio_ts = fuzz.token_sort_ratio(entity_q, name.strip().lower()) / 100.0
            best = max(ratio, ratio_ts)
            if best > best_ratio:
                best_ratio = best

        if best_ratio >= threshold:
            results.append(MatchedNode(cui=c["cui"], name=c["name"], confidence=round(best_ratio, 4), strategy="fuzzy"))

    return results


# ---------------------------------------------------------------------------
# Strategy 4 — Semantic (cosine via injected FAISS search fn)
# ---------------------------------------------------------------------------

def semantic_match(
    entity: str,
    faiss_search_fn: Callable[[str], list[MatchedNode]],
    threshold: float = 0.7,
) -> list[MatchedNode]:
    """
    faiss_search_fn: built by FAISSIndex (TIP-003).
    Returns pre-filtered list with confidence = cosine similarity.
    """
    raw = faiss_search_fn(entity)
    return [n for n in raw if n["confidence"] >= threshold]


# ---------------------------------------------------------------------------
# Cascade orchestrator
# ---------------------------------------------------------------------------

def _deduplicate(nodes: list[MatchedNode]) -> list[MatchedNode]:
    """Keep highest-confidence result per CUI."""
    seen: dict[str, MatchedNode] = {}
    for n in nodes:
        if n["cui"] not in seen or n["confidence"] > seen[n["cui"]]["confidence"]:
            seen[n["cui"]] = n
    return sorted(seen.values(), key=lambda x: x["confidence"], reverse=True)


def cascade_match(
    entity: str,
    candidates: list[dict],
    faiss_search_fn: Callable[[str], list[MatchedNode]] | None = None,
    config: dict | None = None,
    max_results: int = 10,
) -> list[MatchedNode]:
    """
    Run all strategies in order, deduplicate by CUI, return top-k by confidence.
    Stops early at exact match (already confidence 1.0, nothing better exists).
    """
    cfg = config or {}
    fuzzy_threshold = cfg.get("fuzzy_threshold", 0.8)
    semantic_threshold = cfg.get("semantic_threshold", 0.7)

    collected: list[MatchedNode] = []

    # 1. Exact
    exact = exact_match(entity, candidates)
    if exact:
        # Exact hits are definitive — return immediately, no need to continue
        return _deduplicate(exact)[:max_results]

    # 2. Abbreviation
    abbr = abbreviation_match(entity, candidates)
    collected.extend(abbr)

    # 3. Fuzzy
    fuzzy = fuzzy_match(entity, candidates, threshold=fuzzy_threshold)
    collected.extend(fuzzy)

    # 4. Semantic (only if FAISS available)
    if faiss_search_fn is not None:
        sem = semantic_match(entity, faiss_search_fn, threshold=semantic_threshold)
        collected.extend(sem)

    return _deduplicate(collected)[:max_results]
