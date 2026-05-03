#!/usr/bin/env python3
"""
Batch Extraction for ChronoMedKG
=====================================
Submits extraction prompts to OpenAI + Anthropic batch APIs (50% cost, no rate limits).
Gemini runs real-time (fast enough). DeepSeek runs as fallback for GPT-4o-mini zeros.

Workflow:
1. For each disease: harvest evidence → generate all prompts
2. Submit prompts to OpenAI + Anthropic batch APIs
3. While batches process: run Gemini real-time
4. Retrieve batch results → compute consensus → quality control

Usage:
    # Submit batch for a chunk of diseases:
    python3 scripts/run_batch_extraction.py --disease-file config/all_diseases.tsv \
        --start 0 --count 500 --max-docs 200

    # Resume: retrieve results and process consensus:
    python3 scripts/run_batch_extraction.py --retrieve --batch-dir data/batches/batch_001

    # Full auto: submit, poll, retrieve, consensus:
    python3 scripts/run_batch_extraction.py --disease-file config/all_diseases.tsv \
        --start 0 --count 500 --max-docs 200 --auto
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file
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
from agents.knowledge_extractor import KnowledgeExtractor, LLMClient, EXTRACTION_PROMPT
from agents.quality_controller import QualityController
from core.batch_llm import BatchLLMClient
from core.models import (
    DiseaseProfile, SourceDocument, EvidenceTier, StudyType,
    ExtractionResult, RawTriple,
)
from core.schema_alignment import PrimeKGIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("batch_extraction")

EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
BATCH_DIR = PROJECT_ROOT / "data" / "batches"


def load_diseases(filepath: str, start: int = 0, count: int | None = None) -> list[dict]:
    """Load diseases from TSV file."""
    diseases = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                diseases.append({"disease_id": parts[0], "disease_name": parts[1]})

    diseases = diseases[start:]
    if count:
        diseases = diseases[:count]
    return diseases


async def prepare_disease(
    disease: dict,
    profiler: DiseaseProfiler,
    harvester: EvidenceHarvester,
    max_docs: int,
) -> tuple[DiseaseProfile | None, list[SourceDocument]]:
    """Run profiler + harvester for a single disease. Returns (profile, documents)."""
    disease_id = disease["disease_id"]
    disease_name = disease["disease_name"]
    cache_dir = EXTRACTED_DIR / disease_id.replace(":", "_")

    # Check if already completed
    validated = cache_dir / "validated_triples.jsonl"
    consensus = cache_dir / "consensus_triples.jsonl"
    if validated.exists() and consensus.exists():
        line_count = sum(1 for _ in open(consensus))
        if line_count > 10:
            logger.info("SKIP (already completed): %s", disease_name)
            return None, []

    # Profile
    profiler_result = await profiler.run_with_retry({
        "disease_id": disease_id,
        "disease_name": disease_name,
    })
    if profiler_result.status == "failed":
        logger.error("Profiler failed for %s", disease_name)
        return None, []

    profile = DiseaseProfile.from_dict(profiler_result.data["profile"])

    # Harvest evidence
    harvester_result = await harvester.run_with_retry({
        "profile": profiler_result.data["profile"],
        "max_docs": max_docs,
    })
    if harvester_result.status == "failed":
        logger.error("Harvester failed for %s", disease_name)
        return None, []

    # Load documents
    ev_data = _load_evidence_json(cache_dir)
    if not ev_data:
        return profile, []

    documents = []
    for doc_data in ev_data.get("tier1_sources", []) + ev_data.get("tier2_sources", []):
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
            documents.append(SourceDocument(**filtered))
        except Exception as e:
            logger.debug("Failed to parse document: %s", e)

    # Limit docs
    if len(documents) > max_docs:
        documents = documents[:max_docs]

    return profile, documents


def build_prompts(
    extractor: KnowledgeExtractor,
    profile: DiseaseProfile,
    documents: list[SourceDocument],
) -> list[tuple[str, SourceDocument]]:
    """Build extraction prompts for all documents."""
    prompts = []
    for doc in documents:
        prompt = extractor._build_prompt(profile, doc)
        prompts.append((prompt, doc))
    return prompts


async def run_gemini_realtime(
    llm_client: LLMClient,
    prompts: list[tuple[str, SourceDocument]],
    extractor: KnowledgeExtractor,
) -> dict[int, list[RawTriple]]:
    """Run Gemini Flash in real-time (fast enough, no batch API needed)."""
    results = {}
    for i, (prompt, doc) in enumerate(prompts):
        try:
            raw = await llm_client.extract_async("gemini-flash", prompt)
            parsed = [extractor._parse_triple(t, doc, "gemini-flash") for t in raw]
            results[i] = [t for t in parsed if t is not None]
        except Exception as e:
            logger.debug("Gemini extraction failed for doc %d: %s", i, e)
            results[i] = []

        if (i + 1) % 20 == 0:
            logger.info("  Gemini progress: %d/%d docs", i + 1, len(prompts))

    return results


async def run_deepseek_fallback(
    llm_client: LLMClient,
    prompts: list[tuple[str, SourceDocument]],
    extractor: KnowledgeExtractor,
    gpt_zero_indices: list[int],
) -> dict[int, list[RawTriple]]:
    """Run DeepSeek V3 as fallback for GPT-4o-mini zero-triple docs."""
    results = {}
    if "deepseek-v3" not in llm_client.available_models:
        return results

    for i in gpt_zero_indices:
        prompt, doc = prompts[i]
        try:
            raw = await llm_client.extract_async("deepseek-v3", prompt)
            parsed = [extractor._parse_triple(t, doc, "deepseek-v3") for t in raw]
            results[i] = [t for t in parsed if t is not None]
        except Exception as e:
            logger.debug("DeepSeek fallback failed for doc %d: %s", i, e)
            results[i] = []

    return results


async def process_batch(
    diseases: list[dict],
    max_docs: int,
    auto: bool = False,
):
    """Main batch processing flow."""
    config = {}
    primekg_index = PrimeKGIndex()
    primekg_index.load()

    profiler = DiseaseProfiler(config)
    harvester = EvidenceHarvester(config)
    extractor = KnowledgeExtractor(config, primekg_index=primekg_index)
    quality = QualityController(config, primekg_index=primekg_index)
    llm_client = LLMClient()
    batch_client = BatchLLMClient()

    # Create batch output directory
    batch_ts = time.strftime("%Y%m%d_%H%M%S")
    batch_dir = BATCH_DIR / f"batch_{batch_ts}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BATCH EXTRACTION: %d diseases, max_docs=%d", len(diseases), max_docs)
    logger.info("Batch dir: %s", batch_dir)
    logger.info("=" * 60)

    # Phase 1: Prepare all diseases (profile + harvest)
    all_prepared = []
    for i, disease in enumerate(diseases):
        logger.info("[%d/%d] Preparing: %s", i + 1, len(diseases), disease["disease_name"])
        profile, documents = await prepare_disease(disease, profiler, harvester, max_docs)
        if profile and documents:
            prompts = build_prompts(extractor, profile, documents)
            all_prepared.append({
                "disease": disease,
                "profile": profile,
                "documents": documents,
                "prompts": prompts,
            })
            logger.info("  → %d documents, %d prompts", len(documents), len(prompts))
        else:
            logger.info("  → Skipped (no profile or documents)")

    if not all_prepared:
        logger.info("No diseases to process.")
        return

    # Phase 2: Submit ALL prompts to batch APIs
    # Flatten all prompts with tracking info
    all_prompts = []
    all_doc_ids = []
    prompt_to_disease = {}  # Maps global prompt index → (disease_idx, doc_idx)

    for d_idx, prep in enumerate(all_prepared):
        disease_id = prep["disease"]["disease_id"].replace(":", "_")
        for doc_idx, (prompt, doc) in enumerate(prep["prompts"]):
            global_idx = len(all_prompts)
            all_prompts.append(prompt)
            all_doc_ids.append(f"{disease_id}__doc{doc_idx}")
            prompt_to_disease[global_idx] = (d_idx, doc_idx)

    logger.info("Total prompts to submit: %d", len(all_prompts))

    # Submit to OpenAI + Anthropic batch APIs
    batch_ids = batch_client.submit(
        prompts=all_prompts,
        disease_id=f"batch_{batch_ts}",
        doc_ids=all_doc_ids,
    )

    # Save batch metadata
    metadata = {
        "batch_ids": batch_ids,
        "disease_count": len(all_prepared),
        "prompt_count": len(all_prompts),
        "diseases": [p["disease"] for p in all_prepared],
        "timestamp": batch_ts,
    }
    with open(batch_dir / "batch_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Batches submitted: %s", batch_ids)

    # Phase 3: Run Gemini real-time while batches process
    logger.info("Running Gemini Flash real-time extraction...")
    gemini_results_by_disease = {}
    for d_idx, prep in enumerate(all_prepared):
        disease_name = prep["disease"]["disease_name"]
        logger.info("  Gemini: %s (%d docs)", disease_name, len(prep["prompts"]))
        gemini_results = await run_gemini_realtime(llm_client, prep["prompts"], extractor)
        gemini_results_by_disease[d_idx] = gemini_results

    # Phase 4: Poll batch APIs
    if auto and batch_ids:
        logger.info("Polling batch APIs (max 2h)...")
        statuses = batch_client.poll(batch_ids, poll_interval=30, max_wait=7200)
        logger.info("Batch statuses: %s", statuses)

        # Phase 5: Retrieve results
        batch_results = batch_client.retrieve(batch_ids)
        logger.info("Retrieved %d batch results", len(batch_results))

        # Save raw batch results
        with open(batch_dir / "batch_results.json", "w") as f:
            json.dump(batch_results, f, indent=2, default=str)

        # Phase 6: Process each disease — consensus + QC
        for d_idx, prep in enumerate(all_prepared):
            disease = prep["disease"]
            profile = prep["profile"]
            documents = prep["documents"]
            disease_id = disease["disease_id"].replace(":", "_")

            logger.info("Processing consensus: %s", disease["disease_name"])

            result = ExtractionResult(disease_id=disease["disease_id"])

            # Collect per-document results from all models
            for doc_idx, (prompt, doc) in enumerate(prep["prompts"]):
                custom_id_base = f"batch_{batch_ts}__{disease_id}__doc{doc_idx}"

                per_model_triples = {}

                # OpenAI results
                openai_key = f"{custom_id_base}__gpt4omini"
                if openai_key in batch_results:
                    raw = batch_results[openai_key]["triples"]
                    parsed = [extractor._parse_triple(t, doc, "gpt-4o-mini") for t in raw]
                    per_model_triples["gpt-4o-mini"] = [t for t in parsed if t]

                # Anthropic results
                anthropic_key = f"{custom_id_base}__claudehaiku"
                if anthropic_key in batch_results:
                    raw = batch_results[anthropic_key]["triples"]
                    parsed = [extractor._parse_triple(t, doc, "claude-haiku") for t in raw]
                    per_model_triples["claude-haiku"] = [t for t in parsed if t]

                # Gemini results (from real-time)
                gemini = gemini_results_by_disease.get(d_idx, {}).get(doc_idx, [])
                if gemini:
                    per_model_triples["gemini-flash"] = gemini

                # DeepSeek fallback for GPT-4o-mini zeros
                gpt_triples = per_model_triples.get("gpt-4o-mini", [])
                if len(gpt_triples) == 0 and "deepseek-v3" in llm_client.available_models:
                    raw = await llm_client.extract_async("deepseek-v3", prompt)
                    parsed = [extractor._parse_triple(t, doc, "deepseek-v3") for t in raw]
                    per_model_triples["gpt-4o-mini"] = [t for t in parsed if t]

                # Add to raw triples
                for model_triples in per_model_triples.values():
                    result.raw_triples.extend(model_triples)

                # Per-document consensus
                consensus = extractor._compute_consensus(per_model_triples)
                result.consensus_triples.extend(consensus)

            # Normalize entities
            extractor._normalize_triples(result.consensus_triples)

            # Save extraction results
            cache_dir = EXTRACTED_DIR / disease["disease_id"].replace(":", "_")
            cache_dir.mkdir(parents=True, exist_ok=True)
            extractor._save_results(result, cache_dir)

            # Quality control
            quality_result = await quality.run_with_retry({
                "profile": profile.to_dict(),
                "consensus_triples": [t.to_dict() for t in result.consensus_triples],
            })

            status = "SUCCESS" if quality_result.status == "success" else "PARTIAL"
            logger.info("PIPELINE %s for %s (%d consensus triples)",
                        status, disease["disease_name"], len(result.consensus_triples))

    else:
        logger.info("Batches submitted. Run with --retrieve to process results.")
        logger.info("Batch IDs saved to: %s", batch_dir / "batch_metadata.json")


def main():
    parser = argparse.ArgumentParser(description="Batch extraction for ChronoMedKG")
    parser.add_argument("--disease-file", type=str, default="config/all_diseases.tsv")
    parser.add_argument("--start", type=int, default=0, help="Start index in disease file")
    parser.add_argument("--count", type=int, default=500, help="Number of diseases to process")
    parser.add_argument("--max-docs", type=int, default=200, help="Max docs per disease")
    parser.add_argument("--auto", action="store_true", help="Auto poll and process results")
    parser.add_argument("--retrieve", action="store_true", help="Retrieve existing batch results")
    parser.add_argument("--batch-dir", type=str, help="Batch directory to retrieve from")
    args = parser.parse_args()

    diseases = load_diseases(args.disease_file, args.start, args.count)
    logger.info("Loaded %d diseases (start=%d, count=%d)", len(diseases), args.start, args.count)

    asyncio.run(process_batch(diseases, args.max_docs, auto=args.auto))


if __name__ == "__main__":
    main()
