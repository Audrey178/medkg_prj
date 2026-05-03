#!/usr/bin/env python3
"""
ChronoMedKG 17K Batch Extraction
====================================
Processes all 17,080 PrimeKG diseases using OpenAI + Anthropic batch APIs
(50% cost) plus real-time Gemini Flash. Designed for multi-day unattended runs.

Workflow per chunk of N diseases:
  1. Profile + Harvest evidence (sequential per disease)
  2. Submit all extraction prompts to OpenAI + Anthropic batch APIs
  3. Run Gemini Flash real-time while batches process
  4. Poll batch APIs until completion
  5. Retrieve results, compute consensus, run QC
  6. Save chunk metadata for resume

Usage:
    # Full run from scratch (processes in chunks of 500):
    python3 scripts/run_batch_17k.py

    # Custom chunk size and starting point:
    python3 scripts/run_batch_17k.py --chunk-size 200 --start 1000

    # Resume from a specific chunk:
    python3 scripts/run_batch_17k.py --resume-from 3

    # Limit documents per disease:
    python3 scripts/run_batch_17k.py --max-docs 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env using os.environ[k] = v (NOT setdefault)
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

from agents.disease_profiler import DiseaseProfiler
from agents.evidence_harvester import EvidenceHarvester, _load_evidence_json
from agents.knowledge_extractor import KnowledgeExtractor, LLMClient
from agents.quality_controller import QualityController
from core.batch_llm import BatchLLMClient
from core.models import (
    DiseaseProfile,
    EvidenceTier,
    ExtractionResult,
    RawTriple,
    SourceDocument,
    StudyType,
)
from core.schema_alignment import PrimeKGIndex

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "data" / "batch_17k.log"),
    ],
    force=True,
)
# Ensure stdout is unbuffered for nohup/redirect
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
logger = logging.getLogger("batch_17k")

EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
BATCH_DIR = PROJECT_ROOT / "data" / "batches"
STATE_FILE = PROJECT_ROOT / "data" / "batch_17k_state.json"


# ===========================================================================
# Helpers
# ===========================================================================

def load_all_diseases(filepath: str) -> list[dict]:
    """Load all diseases from config/all_diseases.tsv."""
    diseases = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                diseases.append({"disease_id": parts[0], "disease_name": parts[1]})
    logger.info("Loaded %d diseases from %s", len(diseases), filepath)
    return diseases


def is_disease_completed(disease_id: str) -> bool:
    """Check if a disease already has validated_triples.jsonl with >0 lines."""
    cache_dir = EXTRACTED_DIR / disease_id.replace(":", "_")
    validated = cache_dir / "validated_triples.jsonl"
    if not validated.exists():
        return False
    try:
        with open(validated) as f:
            for i, line in enumerate(f):
                if line.strip():
                    return True  # At least 1 non-empty line
        return False
    except Exception:
        return False


def filter_completed(diseases: list[dict]) -> tuple[list[dict], int]:
    """Remove already-completed diseases. Returns (pending, skipped_count)."""
    pending = []
    skipped = 0
    for d in diseases:
        if is_disease_completed(d["disease_id"]):
            skipped += 1
        else:
            pending.append(d)
    return pending, skipped


def chunk_list(lst: list, size: int) -> list[list]:
    """Split a list into chunks of the given size."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def load_state() -> dict:
    """Load batch processing state from disk."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "completed_chunks": [],
        "failed_diseases": [],
        "total_processed": 0,
        "total_triples": 0,
        "last_updated": None,
    }


def save_state(state: dict) -> None:
    """Persist batch processing state to disk."""
    state["last_updated"] = datetime.utcnow().isoformat()
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def parse_source_document(doc_data: dict) -> SourceDocument | None:
    """Parse a raw document dict into a SourceDocument, tolerating bad data."""
    try:
        valid_fields = {f.name for f in SourceDocument.__dataclass_fields__.values()}
        filtered = {k: v for k, v in doc_data.items() if k in valid_fields}
        if "tier" in filtered and isinstance(filtered["tier"], int):
            filtered["tier"] = EvidenceTier(filtered["tier"])
        if "study_type" in filtered and isinstance(filtered["study_type"], str):
            try:
                filtered["study_type"] = StudyType(filtered["study_type"])
            except ValueError:
                filtered["study_type"] = None
        if "sections" in filtered and isinstance(filtered["sections"], list):
            filtered["sections"] = None
        if "publication_date" in filtered and isinstance(filtered["publication_date"], str):
            from datetime import date
            try:
                filtered["publication_date"] = date.fromisoformat(filtered["publication_date"])
            except (ValueError, TypeError):
                filtered["publication_date"] = None
        return SourceDocument(**filtered)
    except Exception:
        return None


# ===========================================================================
# Core pipeline stages
# ===========================================================================

def is_disease_harvested(disease_id: str) -> bool:
    """Check if a disease already has evidence_collection on disk (harvested but not yet extracted)."""
    cache_dir = EXTRACTED_DIR / disease_id.replace(":", "_")
    return (cache_dir / "evidence_collection.json.gz").exists() or (cache_dir / "evidence_collection.json").exists()


async def prepare_disease(
    disease: dict,
    profiler: DiseaseProfiler,
    harvester: EvidenceHarvester,
    max_docs: int,
    semaphore: asyncio.Semaphore | None = None,
    worker_id: int = 0,
) -> tuple[DiseaseProfile | None, list[SourceDocument]]:
    """Run profiler + harvester for one disease. Returns (profile, documents).

    If semaphore is provided, uses it to limit concurrency (multi-worker mode).
    """
    disease_id = disease["disease_id"]
    disease_name = disease["disease_name"]

    async def _inner():
        try:
            # Check if already harvested — skip profiling + harvesting, just load from disk
            cache_dir = EXTRACTED_DIR / disease_id.replace(":", "_")
            if is_disease_harvested(disease_id):
                # Still need a profile for prompt building
                profiler_result = await profiler.run_with_retry({
                    "disease_id": disease_id,
                    "disease_name": disease_name,
                })
                if profiler_result.status == "failed":
                    logger.error("[W%d] Profiler FAILED: %s", worker_id, disease_name)
                    return None, []
                profile = DiseaseProfile.from_dict(profiler_result.data["profile"])
                logger.info("[W%d] %s — using cached evidence", worker_id, disease_name)
            else:
                profiler_result = await profiler.run_with_retry({
                    "disease_id": disease_id,
                    "disease_name": disease_name,
                })
                if profiler_result.status == "failed":
                    logger.error("[W%d] Profiler FAILED: %s", worker_id, disease_name)
                    return None, []

                profile = DiseaseProfile.from_dict(profiler_result.data["profile"])

                harvester_result = await harvester.run_with_retry({
                    "profile": profiler_result.data["profile"],
                    "max_docs": max_docs,
                })
                if harvester_result.status == "failed":
                    logger.error("[W%d] Harvester FAILED: %s", worker_id, disease_name)
                    return None, []

            # Load harvested documents from cache
            ev_data = _load_evidence_json(cache_dir)
            if not ev_data:
                logger.warning("[W%d] No evidence data for %s", worker_id, disease_name)
                return profile, []

            documents = []
            for doc_data in ev_data.get("tier1_sources", []) + ev_data.get("tier2_sources", []):
                doc = parse_source_document(doc_data)
                if doc:
                    documents.append(doc)

            if len(documents) > max_docs:
                documents = documents[:max_docs]

            return profile, documents

        except Exception as e:
            logger.error("[W%d] prepare_disease EXCEPTION for %s: %s", worker_id, disease_name, e)
            return None, []

    if semaphore is not None:
        async with semaphore:
            return await _inner()
    else:
        return await _inner()


def build_prompts(
    extractor: KnowledgeExtractor,
    profile: DiseaseProfile,
    documents: list[SourceDocument],
) -> list[tuple[str, SourceDocument]]:
    """Build extraction prompts for all documents."""
    prompts = []
    for doc in documents:
        try:
            prompt = extractor._build_prompt(profile, doc)
            prompts.append((prompt, doc))
        except Exception as e:
            logger.debug("Prompt build failed for doc: %s", e)
    return prompts


async def _extract_with_retry(
    llm_client: LLMClient,
    model_name: str,
    prompt: str,
    doc,
    extractor: KnowledgeExtractor,
    max_retries: int = 4,
) -> list[RawTriple]:
    """Extract triples from a single doc with one LLM. Exponential backoff on 429s."""
    for attempt in range(max_retries + 1):
        try:
            raw = await llm_client.extract_async(model_name, prompt)
            return [t for t in (extractor._parse_triple(t, doc, model_name) for t in raw) if t is not None]
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "429" in err_str or "resource_exhausted" in err_str or "rate" in err_str
            if is_rate_limit and attempt < max_retries:
                backoff = min(5 * (2 ** attempt), 60)  # 5, 10, 20, 40, 60 seconds
                await asyncio.sleep(backoff)
                continue
            if not is_rate_limit:
                logger.debug("%s failed: %s", model_name, e)
            return []


async def run_realtime_extraction(
    llm_client: LLMClient,
    prompts: list[tuple[str, SourceDocument]],
    extractor: KnowledgeExtractor,
    max_concurrent: int = 10,
    disable_gemini: bool = False,
) -> dict[int, dict[str, list[RawTriple]]]:
    """Run dual primary extractors concurrently with automatic Gemini disable.

    Mode 1 (Gemini available): Gemini + DeepSeek primary, GPT-4o-mini fallback
    Mode 2 (Gemini disabled/429'd): DeepSeek + GPT-4o-mini dual primary (no fallback wait)

    Auto-detects: if Gemini fails 5 consecutive times, disables it for this run.

    Returns: {doc_idx: {"gemini-flash": [...], "deepseek-v3": [...], "gpt-4o-mini": [...]}}
    """
    results: dict[int, dict[str, list[RawTriple]]] = {}
    stats = {"gemini_ok": 0, "deepseek_ok": 0, "gpt4o_ok": 0, "gemini_disabled": disable_gemini}
    consecutive_gemini_fails = 0
    GEMINI_FAIL_THRESHOLD = 5  # Auto-disable after 5 consecutive failures
    sem = asyncio.Semaphore(max_concurrent)
    gemini_sem = asyncio.Semaphore(min(max_concurrent, 5))
    lock = asyncio.Lock()

    async def extract_one(i: int, prompt: str, doc):
        nonlocal consecutive_gemini_fails
        async with sem:
            doc_results: dict[str, list[RawTriple]] = {}
            tasks = {}

            # DeepSeek always runs as primary
            tasks["deepseek-v3"] = _extract_with_retry(
                llm_client, "deepseek-v3", prompt, doc, extractor
            )

            # GPT-4o-mini as second primary (same price as Gemini, no daily quota issues)
            if "gpt-4o-mini" in llm_client.available_models:
                tasks["gpt-4o-mini"] = _extract_with_retry(
                    llm_client, "gpt-4o-mini", prompt, doc, extractor
                )

            # Run all tasks in parallel
            task_names = list(tasks.keys())
            task_results = await asyncio.gather(*tasks.values(), return_exceptions=True)

            for name, result in zip(task_names, task_results):
                if isinstance(result, Exception):
                    result = []
                if result:
                    doc_results[name] = result

            async with lock:
                if "deepseek-v3" in doc_results:
                    stats["deepseek_ok"] += 1
                if "gpt-4o-mini" in doc_results:
                    stats["gpt4o_ok"] += 1
                if "gemini-flash" in doc_results:
                    stats["gemini_ok"] += 1

                # Fallback: if only 1 model succeeded, try Gemini as fallback
                if len(doc_results) < 2:
                    use_gemini = not stats["gemini_disabled"] and "gemini-flash" in llm_client.available_models
                    if use_gemini and "gemini-flash" not in doc_results:
                        try:
                            async with gemini_sem:
                                gemini_triples = await _extract_with_retry(
                                    llm_client, "gemini-flash", prompt, doc, extractor
                                )
                            if gemini_triples:
                                doc_results["gemini-flash"] = gemini_triples
                                stats["gemini_ok"] += 1
                            else:
                                consecutive_gemini_fails += 1
                                if consecutive_gemini_fails >= GEMINI_FAIL_THRESHOLD:
                                    stats["gemini_disabled"] = True
                                    logger.warning(
                                        "  ⚡ Gemini auto-disabled after %d consecutive failures.",
                                        GEMINI_FAIL_THRESHOLD,
                                    )
                        except Exception:
                            consecutive_gemini_fails += 1

                results[i] = doc_results

    # Process in waves for progress logging
    batch_size = max_concurrent * 5  # 50 docs per wave
    total = len(prompts)
    for wave_start in range(0, total, batch_size):
        wave_end = min(wave_start + batch_size, total)
        wave_tasks = [
            extract_one(i, prompt, doc)
            for i, (prompt, doc) in enumerate(prompts[wave_start:wave_end], start=wave_start)
        ]
        await asyncio.gather(*wave_tasks)
        gemini_status = f"(fallback: {stats['gemini_ok']})" if stats["gemini_ok"] else "(fallback)"
        logger.info("  Real-time progress: %d/%d docs | DeepSeek: %d ok | GPT-4o: %d ok | Gemini %s",
                     wave_end, total, stats["deepseek_ok"], stats["gpt4o_ok"], gemini_status)

    logger.info("  Real-time complete: DeepSeek %d/%d, GPT-4o %d/%d, Gemini fallback: %d",
                 stats["deepseek_ok"], total, stats["gpt4o_ok"], total, stats["gemini_ok"])
    return results


async def process_chunk(
    chunk_idx: int,
    diseases: list[dict],
    max_docs: int,
    state: dict,
    workers: int = 1,
) -> dict:
    """Process a single chunk of diseases through the full batch pipeline."""
    chunk_start = time.monotonic()
    chunk_id = f"chunk_{chunk_idx:04d}"

    logger.info("=" * 70)
    logger.info("CHUNK %d: %d diseases (max_docs=%d)", chunk_idx, len(diseases), max_docs)
    logger.info("=" * 70)

    # Initialize shared resources
    primekg_index = PrimeKGIndex()
    primekg_index.load()

    config = {"max_extraction_docs": max_docs}
    profiler = DiseaseProfiler(config)
    harvester = EvidenceHarvester(config)
    extractor = KnowledgeExtractor(config, primekg_index=primekg_index)
    quality = QualityController(config, primekg_index=primekg_index)
    llm_client = LLMClient()
    batch_client = BatchLLMClient()

    batch_ts = time.strftime("%Y%m%d_%H%M%S")
    batch_dir = BATCH_DIR / f"{chunk_id}_{batch_ts}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: Profile + Harvest all diseases in this chunk (CONCURRENT)
    # ------------------------------------------------------------------
    logger.info("Phase 1: Profiling + Harvesting (%d workers)...", workers)
    all_prepared = []
    failed_diseases = []

    if workers > 1:
        # Concurrent harvesting with semaphore to limit NCBI load
        semaphore = asyncio.Semaphore(workers)
        harvest_counter = {"done": 0, "total": len(diseases)}

        async def _harvest_one(idx: int, disease: dict) -> dict | None:
            """Harvest a single disease, return prepared dict or None."""
            profile, documents = await prepare_disease(
                disease, profiler, harvester, max_docs,
                semaphore=semaphore, worker_id=idx % workers,
            )
            harvest_counter["done"] += 1
            done = harvest_counter["done"]
            total = harvest_counter["total"]

            if profile and documents:
                prompts = build_prompts(extractor, profile, documents)
                if prompts:
                    logger.info("  [%d/%d] %s -> %d docs, %d prompts",
                                done, total, disease["disease_name"], len(documents), len(prompts))
                    return {
                        "disease": disease,
                        "profile": profile,
                        "documents": documents,
                        "prompts": prompts,
                    }
                else:
                    logger.warning("  [%d/%d] %s -> No prompts", done, total, disease["disease_name"])
                    return {"_failed": True, "disease_id": disease["disease_id"],
                            "disease_name": disease["disease_name"], "reason": "no_prompts"}
            else:
                reason = "no_profile" if not profile else "no_documents"
                logger.warning("  [%d/%d] %s -> Skipped: %s",
                               done, total, disease["disease_name"], reason)
                return {"_failed": True, "disease_id": disease["disease_id"],
                        "disease_name": disease["disease_name"], "reason": reason}

        # Launch all concurrently, semaphore limits actual parallelism
        tasks = [_harvest_one(i, d) for i, d in enumerate(diseases)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                logger.error("  Unexpected harvest exception: %s", r)
            elif r is None:
                pass
            elif r.get("_failed"):
                failed_diseases.append({k: v for k, v in r.items() if k != "_failed"})
            else:
                all_prepared.append(r)
    else:
        # Sequential fallback (original behavior)
        for i, disease in enumerate(diseases):
            logger.info("  [%d/%d] %s", i + 1, len(diseases), disease["disease_name"])
            profile, documents = await prepare_disease(disease, profiler, harvester, max_docs)
            if profile and documents:
                prompts = build_prompts(extractor, profile, documents)
                if prompts:
                    all_prepared.append({
                        "disease": disease,
                        "profile": profile,
                        "documents": documents,
                        "prompts": prompts,
                    })
                    logger.info("    -> %d docs, %d prompts", len(documents), len(prompts))
                else:
                    logger.warning("    -> No prompts generated")
                    failed_diseases.append({
                        "disease_id": disease["disease_id"],
                        "disease_name": disease["disease_name"],
                        "reason": "no_prompts",
                    })
            else:
                reason = "no_profile" if not profile else "no_documents"
                logger.warning("    -> Skipped: %s", reason)
                failed_diseases.append({
                    "disease_id": disease["disease_id"],
                    "disease_name": disease["disease_name"],
                    "reason": reason,
                })

    if not all_prepared:
        logger.warning("Chunk %d: No diseases to extract. All failed profiling/harvesting.", chunk_idx)
        return {
            "chunk_idx": chunk_idx,
            "status": "empty",
            "diseases_attempted": len(diseases),
            "diseases_prepared": 0,
            "failed_diseases": failed_diseases,
        }

    # ------------------------------------------------------------------
    # Phase 2: Submit prompts to Anthropic batch API only
    # (OpenAI skipped: 2M token enqueue limit; DeepSeek runs real-time in Phase 3)
    # ------------------------------------------------------------------
    logger.info("Phase 2: Submitting %d diseases to Anthropic batch API...", len(all_prepared))

    all_prompts = []
    all_doc_ids = []
    prompt_to_disease = {}  # global_idx -> (d_idx, doc_idx)

    for d_idx, prep in enumerate(all_prepared):
        disease_id = prep["disease"]["disease_id"].replace(":", "_")
        for doc_idx, (prompt, doc) in enumerate(prep["prompts"]):
            global_idx = len(all_prompts)
            all_prompts.append(prompt)
            all_doc_ids.append(f"{disease_id}__doc{doc_idx}")
            prompt_to_disease[global_idx] = (d_idx, doc_idx)

    total_prompts = len(all_prompts)
    logger.info("Total prompts to submit: %d across %d diseases", total_prompts, len(all_prepared))

    # Submit to batch APIs
    batch_ids = {}
    try:
        batch_ids = batch_client.submit(
            prompts=all_prompts,
            disease_id=f"{chunk_id}_{batch_ts}",
            doc_ids=all_doc_ids,
            providers=["anthropic"],  # Skip OpenAI (2M token limit); GPT-4o-mini used as real-time fallback
        )
        logger.info("Batch IDs: %s", batch_ids)
    except Exception as e:
        logger.error("Batch submission failed: %s", e)
        logger.info("Falling back to Gemini-only extraction for this chunk.")

    # Save batch metadata for resume
    metadata = {
        "chunk_idx": chunk_idx,
        "batch_ids": batch_ids,
        "disease_count": len(all_prepared),
        "prompt_count": total_prompts,
        "diseases": [p["disease"] for p in all_prepared],
        "timestamp": batch_ts,
        "batch_dir": str(batch_dir),
    }
    with open(batch_dir / "batch_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # ------------------------------------------------------------------
    # Phase 3: Run Gemini 2.5 Flash + DeepSeek V3 real-time (10-concurrent)
    # GPT-4o-mini fires as fallback only when either returns 0 triples
    # ------------------------------------------------------------------
    logger.info("Phase 3: Running Gemini 2.5 Flash + DeepSeek V3 real-time (concurrent)...")
    realtime_results_by_disease = {}
    for d_idx, prep in enumerate(all_prepared):
        disease_name = prep["disease"]["disease_name"]
        logger.info("  Extracting [%d/%d]: %s (%d docs)",
                     d_idx + 1, len(all_prepared), disease_name, len(prep["prompts"]))
        rt_results = await run_realtime_extraction(llm_client, prep["prompts"], extractor)
        realtime_results_by_disease[d_idx] = rt_results

    # ------------------------------------------------------------------
    # Phase 4: Poll batch APIs until completion
    # ------------------------------------------------------------------
    batch_results = {}
    if batch_ids:
        logger.info("Phase 4: Polling batch APIs (max 4h)...")
        statuses = batch_client.poll(batch_ids, poll_interval=60, max_wait=14400)
        logger.info("Batch statuses: %s", statuses)

        # v2 poll returns {provider: {batch_id: status}} or v1 {provider: status}
        # Check for failures — handle both formats
        has_completed = False
        for provider, status_val in statuses.items():
            if isinstance(status_val, dict):
                # v2 format: {batch_id: status}
                for bid, s in status_val.items():
                    if s != "completed":
                        logger.error("Batch %s/%s ended with status: %s", provider, bid[:16], s)
                    else:
                        has_completed = True
            else:
                # v1 format: status string
                if status_val != "completed":
                    logger.error("Batch %s ended with status: %s", provider, status_val)
                else:
                    has_completed = True

        # Retrieve results from all batches (retrieve handles both formats)
        if has_completed:
            logger.info("Phase 4b: Retrieving batch results...")
            batch_results = batch_client.retrieve(batch_ids)
            logger.info("Retrieved %d batch results", len(batch_results))

            # Log cost summary
            logger.info(batch_client.cost_summary)

            # Persist raw results
            with open(batch_dir / "batch_results.json", "w") as f:
                json.dump(batch_results, f, indent=2, default=str)
    else:
        logger.info("Phase 4: No batch IDs — skipping poll.")

    # ------------------------------------------------------------------
    # Phase 5: Consensus + QC per disease
    # ------------------------------------------------------------------
    logger.info("Phase 5: Computing consensus + QC for %d diseases...", len(all_prepared))
    chunk_total_triples = 0
    chunk_successes = 0

    for d_idx, prep in enumerate(all_prepared):
        disease = prep["disease"]
        profile = prep["profile"]
        documents = prep["documents"]
        disease_id_safe = disease["disease_id"].replace(":", "_")
        disease_name = disease["disease_name"]

        logger.info("  Consensus [%d/%d]: %s", d_idx + 1, len(all_prepared), disease_name)

        result = ExtractionResult(disease_id=disease["disease_id"])

        for doc_idx, (prompt, doc) in enumerate(prep["prompts"]):
            # Build lookup keys using the SAME logic as batch submission
            # to guarantee key match (both sides use _make_custom_id)
            batch_disease_id = f"{chunk_id}_{batch_ts}"
            batch_doc_id = f"{disease_id_safe}__doc{doc_idx}"
            openai_key = BatchLLMClient._make_custom_id(batch_disease_id, batch_doc_id, "gpt4omini")
            anthropic_key = BatchLLMClient._make_custom_id(batch_disease_id, batch_doc_id, "claudehaiku")

            per_model_triples = {}

            # Anthropic results from batch
            if anthropic_key in batch_results:
                try:
                    raw = batch_results[anthropic_key]["triples"]
                    parsed = [extractor._parse_triple(t, doc, "claude-haiku") for t in raw]
                    per_model_triples["claude-haiku"] = [t for t in parsed if t]
                except Exception as e:
                    logger.debug("Anthropic parse error: %s", e)

            # OpenAI results from batch (if available — may be empty)
            if openai_key in batch_results:
                try:
                    raw = batch_results[openai_key]["triples"]
                    parsed = [extractor._parse_triple(t, doc, "gpt-4o-mini") for t in raw]
                    per_model_triples["gpt-4o-mini"] = [t for t in parsed if t]
                except Exception as e:
                    logger.debug("OpenAI parse error: %s", e)

            # Real-time results (Gemini + DeepSeek + GPT-4o-mini fallback)
            rt_doc = realtime_results_by_disease.get(d_idx, {}).get(doc_idx, {})
            for model_name, triples in rt_doc.items():
                if triples:
                    per_model_triples[model_name] = triples

            # Aggregate raw triples
            for model_triples in per_model_triples.values():
                result.raw_triples.extend(model_triples)

            # Per-document consensus
            try:
                consensus = extractor._compute_consensus(per_model_triples)
                result.consensus_triples.extend(consensus)
            except Exception as e:
                logger.debug("Consensus computation error: %s", e)

        # Normalize entities
        try:
            extractor._normalize_triples(result.consensus_triples)
        except Exception as e:
            logger.debug("Normalization error for %s: %s", disease_name, e)

        # Save extraction results
        cache_dir = EXTRACTED_DIR / disease["disease_id"].replace(":", "_")
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            extractor._save_results(result, cache_dir)
        except Exception as e:
            logger.error("Failed to save results for %s: %s", disease_name, e)
            failed_diseases.append({
                "disease_id": disease["disease_id"],
                "disease_name": disease_name,
                "reason": f"save_failed: {e}",
            })
            continue

        # Quality control
        try:
            quality_result = await quality.run_with_retry({
                "profile": profile.to_dict(),
                "consensus_triples": [t.to_dict() for t in result.consensus_triples],
            })
            status = "SUCCESS" if quality_result.status == "success" else "PARTIAL"
        except Exception as e:
            logger.error("QC failed for %s: %s", disease_name, e)
            status = "QC_FAILED"

        n_triples = len(result.consensus_triples)
        chunk_total_triples += n_triples
        chunk_successes += 1
        logger.info("    %s: %d consensus triples", status, n_triples)

    # ------------------------------------------------------------------
    # Update global state
    # ------------------------------------------------------------------
    elapsed = time.monotonic() - chunk_start
    chunk_result = {
        "chunk_idx": chunk_idx,
        "status": "completed",
        "diseases_attempted": len(diseases),
        "diseases_prepared": len(all_prepared),
        "diseases_succeeded": chunk_successes,
        "failed_diseases": failed_diseases,
        "total_triples": chunk_total_triples,
        "total_prompts": total_prompts,
        "batch_ids": batch_ids,
        "batch_dir": str(batch_dir),
        "elapsed_seconds": round(elapsed, 1),
    }

    state["completed_chunks"].append(chunk_result)
    state["total_processed"] += chunk_successes
    state["total_triples"] += chunk_total_triples
    state["failed_diseases"].extend(failed_diseases)
    save_state(state)

    logger.info("CHUNK %d DONE: %d/%d succeeded, %d triples, %.0fs",
                chunk_idx, chunk_successes, len(all_prepared), chunk_total_triples, elapsed)
    logger.info(batch_client.cost_summary)

    return chunk_result


# ===========================================================================
# Main entry point
# ===========================================================================

async def run(args: argparse.Namespace) -> None:
    """Main orchestration loop: load diseases, chunk, process."""
    # Ensure output directories exist
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    # Load all diseases
    disease_file = str(PROJECT_ROOT / args.disease_file)
    all_diseases = load_all_diseases(disease_file)

    # Apply --start offset
    if args.start > 0:
        all_diseases = all_diseases[args.start:]
        logger.info("Starting from index %d (%d diseases remaining)", args.start, len(all_diseases))

    # Filter already-completed diseases
    pending, skipped = filter_completed(all_diseases)
    logger.info("Skipping %d already-completed diseases. %d remaining.", skipped, len(pending))

    if not pending:
        logger.info("All diseases already processed. Nothing to do.")
        return

    # Split into chunks
    chunks = chunk_list(pending, args.chunk_size)
    logger.info("Split %d pending diseases into %d chunks of up to %d",
                len(pending), len(chunks), args.chunk_size)

    # Load state for resume
    state = load_state()

    # Determine starting chunk
    start_chunk = 0
    if args.resume_from is not None:
        start_chunk = args.resume_from
        logger.info("Resuming from chunk %d", start_chunk)

    # Process each chunk
    for chunk_idx in range(start_chunk, len(chunks)):
        chunk = chunks[chunk_idx]

        # Re-filter completed within chunk (in case earlier chunks produced results
        # for diseases that appear later, or manual runs completed some)
        chunk_pending = [d for d in chunk if not is_disease_completed(d["disease_id"])]
        if not chunk_pending:
            logger.info("Chunk %d: all %d diseases already completed, skipping.", chunk_idx, len(chunk))
            continue

        logger.info("")
        logger.info("*" * 70)
        logger.info("STARTING CHUNK %d/%d (%d diseases)", chunk_idx + 1, len(chunks), len(chunk_pending))
        logger.info("*" * 70)

        try:
            chunk_result = await process_chunk(chunk_idx, chunk_pending, args.max_docs, state, workers=args.workers)
        except Exception as e:
            logger.error("CHUNK %d FAILED with exception: %s", chunk_idx, e)
            logger.error(traceback.format_exc())
            state["completed_chunks"].append({
                "chunk_idx": chunk_idx,
                "status": "error",
                "error": str(e),
            })
            save_state(state)
            # Continue to next chunk rather than aborting the whole run
            continue

    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("BATCH 17K COMPLETE")
    logger.info("=" * 70)
    logger.info("Total processed: %d", state["total_processed"])
    logger.info("Total triples: %d", state["total_triples"])
    logger.info("Total failed: %d", len(state["failed_diseases"]))
    logger.info("State saved to: %s", STATE_FILE)


def main():
    parser = argparse.ArgumentParser(
        description="ChronoMedKG 17K batch extraction — processes all PrimeKG diseases "
                    "using OpenAI + Anthropic batch APIs (50%% cost) + Gemini Flash real-time.",
    )
    parser.add_argument(
        "--disease-file", type=str, default="config/all_diseases.tsv",
        help="Path to disease TSV file (relative to project root). Default: config/all_diseases.tsv",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500,
        help="Number of diseases per batch chunk. Default: 500",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Start index in the disease file (skip first N diseases). Default: 0",
    )
    parser.add_argument(
        "--max-docs", type=int, default=200,
        help="Max documents per disease. Default: 200",
    )
    parser.add_argument(
        "--resume-from", type=int, default=None,
        help="Resume from a specific chunk index (0-based). Skips earlier chunks.",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of concurrent disease harvesters in Phase 1. Default: 4",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
