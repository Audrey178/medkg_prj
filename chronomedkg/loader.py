"""Stream + cache ChronoMedKG release files from Zenodo.

Dataset: https://doi.org/10.5281/zenodo.19697543 (v0.0.1, CC BY 4.0)

The loader has zero hard dependencies. Files are downloaded on first call
and cached locally; subsequent calls read from disk.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Iterator

ZENODO_VERSION_URL = "https://zenodo.org/records/19697543/files"

_DEFAULT_CACHE = Path.home() / ".cache" / "chronomedkg"


def cache_dir() -> Path:
    """Return the local cache directory, creating it if missing.

    Override the location with the ``CHRONOMEDKG_CACHE`` environment variable.
    """
    path = Path(os.environ.get("CHRONOMEDKG_CACHE", _DEFAULT_CACHE))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fetch(filename: str) -> Path:
    """Download ``filename`` from the Zenodo deposit if not already cached."""
    local = cache_dir() / filename
    if local.exists() and local.stat().st_size > 0:
        return local
    url = f"{ZENODO_VERSION_URL}/{filename}"
    sys.stderr.write(f"chronomedkg: fetching {url}\n  -> {local}\n")
    sys.stderr.flush()
    tmp = local.with_suffix(local.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(local)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return local


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _iter_jsonl(filename: str) -> Iterator[dict[str, Any]]:
    path = _fetch(filename)
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_triples() -> Iterator[dict[str, Any]]:
    """Yield 460,497 validated consensus triples (Gold tier, post-QC).

    Each record carries: subject/object IDs and names, PrimeKG relation,
    a ``temporal`` block (onset window, progression stage, milestone), an
    ``evidence`` block (PMID, verbatim quote, study type, six-signal
    credibility, multi-LLM consensus), and a ``quality_grade`` tag.
    """
    yield from _iter_jsonl("validated_triples.jsonl")


def load_consensus() -> Iterator[dict[str, Any]]:
    """Yield 443,114 pre-QC multi-LLM consensus rows (Silver tier, gzipped)."""
    yield from _iter_jsonl("consensus_triples.jsonl.gz")


def load_benchmark() -> dict[str, Any]:
    """Return the ChronoTQA benchmark (3,341 questions across 8 task types)."""
    path = _fetch("tqa_benchmark.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pmc_cases() -> dict[str, Any]:
    """Return the 31 diagnostic-odyssey PMC case studies."""
    path = _fetch("pmc_clinical_cases.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
