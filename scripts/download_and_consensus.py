#!/usr/bin/env python3
"""
Download Anthropic batch results for chunks 4-12, then run consensus+QC.
Also re-runs consensus on chunks 1-3 that already have cached results.

This merges Anthropic (batch) + real-time (DeepSeek/GPT-4o/Gemini) results
into 4-model consensus with QC validation.

Usage:
    python3 scripts/download_and_consensus.py
"""
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env
env_file = ROOT / ".env"
if env_file.exists():
    for line in open(env_file):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

from core.batch_llm import BatchLLMClient
from core.schema_alignment import PrimeKGIndex
from agents.knowledge_extractor import KnowledgeExtractor
from agents.quality_controller import QualityController
from agents.disease_profiler import DiseaseProfiler
from core.models import DiseaseProfile, ExtractionResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("download_consensus")

EXTRACTED_DIR = ROOT / "data" / "extracted"
BATCH_DIR = ROOT / "data" / "batches"


def find_chunk_dirs() -> list[Path]:
    """Find all chunk dirs with Anthropic batch IDs (chunks 1-12)."""
    dirs = []
    for chunk_dir in sorted(BATCH_DIR.glob("chunk_*")):
        meta_path = chunk_dir / "batch_metadata.json"
        if not meta_path.exists():
            continue
        meta = json.load(open(meta_path))
        batch_ids = meta.get("batch_ids", {})
        if "anthropic" in batch_ids and batch_ids["anthropic"]:
            dirs.append(chunk_dir)
    return dirs


def download_batch_results(chunk_dir: Path, batch_client: BatchLLMClient) -> dict:
    """Download and cache Anthropic batch results for a chunk."""
    cache_path = chunk_dir / "batch_results.json"

    if cache_path.exists():
        logger.info("  Loading cached results from %s", cache_path.name)
        with open(cache_path) as f:
            return json.load(f)

    meta = json.load(open(chunk_dir / "batch_metadata.json"))
    batch_ids = meta.get("batch_ids", {})

    if not batch_ids.get("anthropic"):
        return {}

    logger.info("  Downloading Anthropic results for %s (%d batches)...",
                chunk_dir.name, len(batch_ids["anthropic"]))

    # Poll (should be instant since all are "ended")
    statuses = batch_client.poll(batch_ids, poll_interval=5, max_wait=60)
    logger.info("  Statuses: %s", statuses)

    # Retrieve
    results = batch_client.retrieve(batch_ids)
    logger.info("  Retrieved %d results", len(results))

    # Cache
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


async def run_consensus_for_chunk(
    chunk_dir: Path,
    batch_results: dict,
    primekg_index: PrimeKGIndex,
) -> dict:
    """Run consensus + QC for diseases in a chunk."""
    meta = json.load(open(chunk_dir / "batch_metadata.json"))
    diseases = meta.get("diseases", [])
    batch_ts = meta.get("timestamp", "")
    chunk_id = chunk_dir.name.split("_20")[0]

    config = {"max_extraction_docs": 200}
    profiler = DiseaseProfiler(config)
    extractor = KnowledgeExtractor(config, primekg_index=primekg_index)
    quality = QualityController(config, primekg_index=primekg_index)

    successes = 0
    skipped = 0
    failed = 0
    total_triples = 0

    for d_idx, disease in enumerate(diseases):
        disease_id = disease["disease_id"]
        disease_name = disease["disease_name"]
        disease_id_safe = disease_id.replace(":", "_")
        disease_dir = EXTRACTED_DIR / disease_id_safe

        # Skip if already validated
        if (disease_dir / "validated_triples.jsonl").exists():
            skipped += 1
            continue

        # Need RT checkpoint
        import gzip
        rt_path = disease_dir / "realtime_checkpoint.json.gz"
        if not rt_path.exists():
            continue

        try:
            with gzip.open(rt_path, "rt") as f:
                rt_results = json.load(f)
            # Convert string keys back to int
            rt_results = {int(k): v for k, v in rt_results.items()}
        except Exception as e:
            logger.debug("Failed to load RT checkpoint for %s: %s", disease_name, e)
            continue

        # Profile disease
        try:
            profiler_result = await profiler.run_with_retry({
                "disease_id": disease_id,
                "disease_name": disease_name,
            })
            if profiler_result.status == "failed":
                continue
            profile = DiseaseProfile.from_dict(profiler_result.data["profile"])
        except Exception:
            continue

        # Load evidence to get documents
        ev_gz = disease_dir / "evidence_collection.json.gz"
        ev_json = disease_dir / "evidence_collection.json"
        if ev_gz.exists():
            with gzip.open(ev_gz, "rt") as f:
                ev_data = json.load(f)
        elif ev_json.exists():
            with open(ev_json) as f:
                ev_data = json.load(f)
        else:
            continue

        from scripts.presubmit_batches import parse_source_document
        documents = []
        for doc_data in ev_data.get("tier1_sources", []) + ev_data.get("tier2_sources", []):
            doc = parse_source_document(doc_data)
            if doc:
                documents.append(doc)
        if len(documents) > 200:
            documents = documents[:200]

        # Build prompts to get doc mapping
        prompts = []
        for doc in documents:
            try:
                prompt = extractor._build_prompt(profile, doc)
                prompts.append((prompt, doc))
            except Exception:
                pass

        if not prompts:
            continue

        # Merge batch + RT results per document
        result = ExtractionResult(disease_id=disease_id)

        for doc_idx, (prompt, doc) in enumerate(prompts):
            batch_doc_id = f"{disease_id_safe}__doc{doc_idx}"
            anthropic_key = BatchLLMClient._make_custom_id(
                f"{chunk_id}_{batch_ts}", batch_doc_id, "claudehaiku"
            )

            per_model_triples = {}

            # Anthropic batch results
            if anthropic_key in batch_results:
                try:
                    raw = batch_results[anthropic_key]["triples"]
                    parsed = [extractor._parse_triple(t, doc, "claude-haiku") for t in raw]
                    per_model_triples["claude-haiku"] = [t for t in parsed if t]
                except Exception:
                    pass

            # RT results
            rt_doc = rt_results.get(doc_idx, {})
            for model_name, triples in rt_doc.items():
                if triples:
                    parsed_rt = []
                    for t in triples:
                        if isinstance(t, dict):
                            parsed_rt.append(extractor._parse_triple(t, doc, model_name))
                        else:
                            parsed_rt.append(t)
                    per_model_triples[model_name] = [t for t in parsed_rt if t]

            for model_triples in per_model_triples.values():
                result.raw_triples.extend(model_triples)

            try:
                consensus = extractor._compute_consensus(per_model_triples)
                result.consensus_triples.extend(consensus)
            except Exception:
                pass

        if not result.consensus_triples:
            continue

        # Normalize + save
        try:
            extractor._normalize_triples(result.consensus_triples)
        except Exception:
            pass

        try:
            extractor._save_results(result, disease_dir)
        except Exception as e:
            logger.error("Save failed for %s: %s", disease_name, e)
            failed += 1
            continue

        # QC
        try:
            quality_result = await quality.run_with_retry({
                "profile": profile.to_dict(),
                "consensus_triples": [t.to_dict() for t in result.consensus_triples],
            })
        except Exception:
            pass

        n = len(result.consensus_triples)
        total_triples += n
        successes += 1

        if successes % 10 == 0:
            logger.info("  [%s] %d/%d done, %d skipped, %d triples so far",
                        chunk_id, successes, len(diseases), skipped, total_triples)

    return {
        "chunk": chunk_dir.name,
        "successes": successes,
        "skipped": skipped,
        "failed": failed,
        "total_triples": total_triples,
    }


async def main():
    logger.info("=" * 70)
    logger.info("DOWNLOAD BATCH RESULTS + RUN CONSENSUS")
    logger.info("=" * 70)

    # Find chunks with Anthropic batches
    chunk_dirs = find_chunk_dirs()
    logger.info("Found %d chunks with Anthropic batch IDs", len(chunk_dirs))

    # Load PrimeKG (SQLite)
    primekg_index = PrimeKGIndex()
    primekg_index.load()

    batch_client = BatchLLMClient()

    # Phase 1: Download all batch results
    logger.info("Phase 1: Downloading batch results...")
    all_results = {}
    for chunk_dir in chunk_dirs:
        results = download_batch_results(chunk_dir, batch_client)
        all_results[chunk_dir.name] = results
        logger.info("  %s: %d results", chunk_dir.name, len(results))

    # Phase 2: Run consensus for each chunk
    logger.info("Phase 2: Running consensus + QC...")
    summary = []
    for chunk_dir in chunk_dirs:
        results = all_results.get(chunk_dir.name, {})
        if not results:
            continue
        logger.info("Processing %s...", chunk_dir.name)
        stats = await run_consensus_for_chunk(chunk_dir, results, primekg_index)
        summary.append(stats)
        logger.info("  Done: %d succeeded, %d skipped, %d triples",
                    stats["successes"], stats["skipped"], stats["total_triples"])

    # Summary
    logger.info("=" * 70)
    total_s = sum(s["successes"] for s in summary)
    total_t = sum(s["total_triples"] for s in summary)
    logger.info("TOTAL: %d diseases with consensus, %d triples", total_s, total_t)
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
