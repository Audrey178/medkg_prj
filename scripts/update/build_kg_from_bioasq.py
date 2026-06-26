"""
Build ChronoMedKG knowledge graph from BioASQ Task B data.

Hai chế độ extraction:

  [DEFAULT] Real-time mode  — cascading LLM như orchestrator hiện tại
    Agent 1→2→3→4 per question, kết quả ngay lập tức.
    Dùng khi test hoặc số sample nhỏ (< 50).

  [--batch] Batch mode      — OpenAI Batch API (50% cost) + DeepSeek real-time
    Phase 1 : Harvest tất cả profiles (Agent 1 + Agent 2)
    Phase 2 : Submit toàn bộ prompts lên OpenAI Batch API
    Phase 3 : Chạy DeepSeek real-time song song (làm second model trong khi đợi)
    Phase 4 : Poll + retrieve OpenAI batch results
    Phase 5 : Compute consensus (OpenAI gpt-4o-mini + DeepSeek) → Agent 4 QC
    Dùng khi chạy toàn bộ dataset (5K+ samples, tiết kiệm ~50% chi phí LLM).

Output: data/extracted/BIOASQ_{question_id}/ (cùng format với disease-driven pipeline)

Usage:
    # Test nhanh 5 sample (real-time)
    python scripts/update/build_kg_from_bioasq.py --max-samples 5

    # Một question cụ thể
    python scripts/update/build_kg_from_bioasq.py --question-id 55031181e9bde69634000014

    # Full run real-time với 3 workers
    python scripts/update/build_kg_from_bioasq.py --workers 3

    # Batch mode: submit OpenAI + DeepSeek parallel, poll 24h
    python scripts/update/build_kg_from_bioasq.py --batch --max-samples 500

    # Batch mode: chỉ submit (không poll, retrieve thủ công sau)
    python scripts/update/build_kg_from_bioasq.py --batch --no-auto --max-samples 500

    # Retrieve từ batch đã submit trước đó
    python scripts/update/build_kg_from_bioasq.py --batch --retrieve --batch-meta data/batches/bioasq_batch_20250625_120000/batch_meta.json
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import gzip
import json
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

from agents.disease_profiler import DiseaseProfiler
from agents.evidence_harvester import EvidenceHarvester
from agents.knowledge_extractor import KnowledgeExtractor, LLMClient
from agents.quality_controller import QualityController
from core.batch_llm import BatchLLMClient
from core.models import (
    BioASQProfile, EvidenceCollection, ExtractionResult,
    SourceDocument, EvidenceTier, StudyType,
)
from core.schema_alignment import PrimeKGIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("build_kg_from_bioasq")

BIOASQ_PATH = PROJECT_ROOT / "data" / "bioasq" / "training14b.json"
OUTPUT_DIR  = PROJECT_ROOT / "data" / "extracted"
BATCH_DIR   = PROJECT_ROOT / "data" / "batches"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _full_doc_to_dict(doc) -> dict:
    """Serialize SourceDocument with FULL text (to_dict() truncates at 500 chars)."""
    return {
        "source_id": doc.source_id,
        "source_type": doc.source_type,
        "tier": doc.tier.value,
        "title": doc.title,
        "text": doc.text,
        "sections": None,
        "publication_date": doc.publication_date.isoformat() if doc.publication_date else None,
        "journal": doc.journal,
        "credibility_score": doc.credibility_score,
        "study_type": doc.study_type.value if doc.study_type else None,
        "citation_count": doc.citation_count,
        "is_retracted": doc.is_retracted,
    }


def _reconstruct_source_doc(doc_dict: dict) -> SourceDocument:
    """Reconstruct SourceDocument from full dict (handles enum fields)."""
    valid = {f.name for f in SourceDocument.__dataclass_fields__.values()}
    filtered = {k: v for k, v in doc_dict.items() if k in valid}
    if "tier" in filtered and isinstance(filtered["tier"], int):
        filtered["tier"] = EvidenceTier(filtered["tier"])
    if "tier" in filtered and isinstance(filtered["tier"], str):
        filtered["tier"] = EvidenceTier(int(filtered["tier"]))
    if "study_type" in filtered and isinstance(filtered["study_type"], str):
        try:
            filtered["study_type"] = StudyType(filtered["study_type"])
        except ValueError:
            filtered["study_type"] = None
    if "sections" in filtered and isinstance(filtered["sections"], list):
        filtered["sections"] = None
    if "publication_date" in filtered and isinstance(filtered["publication_date"], str):
        from datetime import date as date_cls
        try:
            filtered["publication_date"] = date_cls.fromisoformat(filtered["publication_date"])
        except (ValueError, TypeError):
            filtered["publication_date"] = None
    return SourceDocument(**filtered)


def _save_collection_to_cache(collection: EvidenceCollection, cache_dir: Path) -> None:
    """Save evidence collection to disk so QualityController can load credibility metadata."""
    data = {
        "disease_id": collection.disease_id,
        "tier1_count": len(collection.tier1_documents),
        "tier2_count": len(collection.tier2_documents),
        "tier1_sources": [_full_doc_to_dict(d) for d in collection.tier1_documents],
        "tier2_sources": [_full_doc_to_dict(d) for d in collection.tier2_documents],
    }
    json_bytes = json.dumps(data, indent=2, default=str).encode("utf-8")
    with gzip.open(cache_dir / "evidence_collection.json.gz", "wb") as f:
        f.write(json_bytes)


def _make_adapter_profile(bioasq_profile: BioASQProfile) -> dict:
    """Create minimal DiseaseProfile-compatible dict from a BioASQProfile.

    KnowledgeExtractor and QualityController call DiseaseProfile.from_dict()
    on this — only disease_id, disease_name, coverage_flag are actually used.
    """
    question_preview = bioasq_profile.question_body[:200].replace("\n", " ")
    return {
        "disease_id": f"BIOASQ:{bioasq_profile.bioasq_id}",
        "disease_name": question_preview,
        "coverage_flag": "sparse",
    }


def _is_already_done(disease_id: str) -> bool:
    cache_dir = OUTPUT_DIR / disease_id.replace(":", "_")
    return (cache_dir / "validated_triples.jsonl").exists() and \
           (cache_dir / "consensus_triples.jsonl").exists()


# ---------------------------------------------------------------------------
# Harvest helper — shared between real-time and batch modes
# ---------------------------------------------------------------------------

def _harvest_profile(
    bioasq_profile: BioASQProfile,
    harvester: EvidenceHarvester,
) -> tuple[EvidenceCollection, list[dict]]:
    """Run Agent 2 for one BioASQProfile. Returns (collection, doc_dicts).

    Tier 1: UniProt entity summaries (background knowledge per topic entity).
    Tier 2: BioASQ gold snippets (question-specific evidence from PubMed).
    Both tiers are passed to the extractor — Tier 1 first so the LLM has
    entity context before reading the snippets.
    """
    disease_id = f"BIOASQ:{bioasq_profile.bioasq_id}"
    collection = EvidenceCollection(disease_id=disease_id)
    harvester._harvest_tier1_bioasq(bioasq_profile, collection)
    harvester._harvest_from_bioasq_gold(bioasq_profile, collection)

    cache_dir = OUTPUT_DIR / disease_id.replace(":", "_")
    cache_dir.mkdir(parents=True, exist_ok=True)
    if collection.total_sources:
        _save_collection_to_cache(collection, cache_dir)

    # Tier 1 first so extractor sees background knowledge before snippets
    docs = (
        [_full_doc_to_dict(d) for d in collection.tier1_documents]
        + [_full_doc_to_dict(d) for d in collection.tier2_documents]
    )
    return collection, docs


# ---------------------------------------------------------------------------
# Mode A — Real-time (default)
# ---------------------------------------------------------------------------

async def run_single_realtime(
    bioasq_profile: BioASQProfile,
    harvester: EvidenceHarvester,
    extractor: KnowledgeExtractor,
    quality: QualityController,
    config: dict,
    resume: bool = True,
) -> dict:
    """Run Agent 2→3→4 for one BioASQProfile in real-time mode."""
    disease_id = f"BIOASQ:{bioasq_profile.bioasq_id}"
    cache_dir = OUTPUT_DIR / disease_id.replace(":", "_")

    result = {
        "bioasq_id": bioasq_profile.bioasq_id,
        "disease_id": disease_id,
        "question_type": bioasq_profile.question_type,
        "status": "pending",
        "agents": {},
    }

    if resume and _is_already_done(disease_id):
        result["status"] = "skipped_resume"
        return result

    if not bioasq_profile.pmids_with_snippet:
        result["status"] = "skipped_no_snippets"
        return result

    try:
        adapter_profile = _make_adapter_profile(bioasq_profile)
        collection, documents = _harvest_profile(bioasq_profile, harvester)

        if not documents:
            result["status"] = "partial"
            result["reason"] = "no_documents"
            return result

        result["agents"]["harvester"] = {
            "status": "success",
            "metrics": {
                "tier1_count": len(collection.tier1_documents),
                "tier2_count": len(collection.tier2_documents),
            },
        }

        # Use bioasq_profile (not adapter_profile) so KnowledgeExtractor picks up
        # BIOASQ_EXTRACTION_PROMPT with ideal_answer anchor + topic_entities
        extractor_result = await extractor.run_with_retry({
            "bioasq_profile": bioasq_profile.to_dict(),
            "documents": documents,
        })
        result["agents"]["extractor"] = {
            "status": extractor_result.status,
            "metrics": extractor_result.metrics,
        }

        if extractor_result.status == "failed":
            result["status"] = "failed"
            result["reason"] = "extractor_failed"
            return result

        consensus_file = cache_dir / "consensus_triples.jsonl"
        consensus_triples = []
        if consensus_file.exists():
            with open(consensus_file) as f:
                for line in f:
                    if line.strip():
                        consensus_triples.append(json.loads(line))

        quality_result = await quality.run_with_retry({
            "profile": adapter_profile,
            "consensus_triples": consensus_triples,
        })
        result["agents"]["quality"] = {
            "status": quality_result.status,
            "metrics": quality_result.metrics,
        }
        result["status"] = "success"
        logger.info("DONE: %s — %d triples validated",
                    disease_id, quality_result.metrics.get("validated_triples", 0))

    except Exception as exc:
        logger.error("Pipeline failed for %s: %s", disease_id, exc, exc_info=True)
        result["status"] = "failed"
        result["error"] = str(exc)

    return result


def _run_sequential(profiles, config, primekg_index, resume):
    harvester = EvidenceHarvester(config=config)
    extractor = KnowledgeExtractor(config=config, primekg_index=primekg_index)
    quality   = QualityController(config=config, primekg_index=primekg_index)

    loop = asyncio.new_event_loop()
    results = []
    try:
        for i, p in enumerate(profiles):
            logger.info("Question %d/%d: %s", i + 1, len(profiles), p.bioasq_id)
            r = loop.run_until_complete(
                run_single_realtime(p, harvester, extractor, quality, config, resume)
            )
            results.append(r)
    finally:
        loop.close()
    return results


def _run_parallel(profiles, config, primekg_index, resume, workers):
    lock = threading.Lock()
    all_results = []

    def _worker(chunk, idx):
        h = EvidenceHarvester(config=config)
        e = KnowledgeExtractor(config=config, primekg_index=primekg_index)
        q = QualityController(config=config, primekg_index=primekg_index)
        loop = asyncio.new_event_loop()
        chunk_results = []
        try:
            for i, p in enumerate(chunk):
                logger.info("[W%d] %d/%d: %s", idx, i + 1, len(chunk), p.bioasq_id)
                r = loop.run_until_complete(
                    run_single_realtime(p, h, e, q, config, resume)
                )
                chunk_results.append(r)
        finally:
            loop.close()
        return chunk_results

    chunk_size = max(1, len(profiles) // workers)
    chunks = [profiles[i:i + chunk_size] for i in range(0, len(profiles), chunk_size)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, c, i): i for i, c in enumerate(chunks)}
        for future in concurrent.futures.as_completed(futures):
            try:
                with lock:
                    all_results.extend(future.result())
            except Exception as exc:
                logger.error("Worker failed: %s", exc, exc_info=True)

    return all_results


# ---------------------------------------------------------------------------
# Mode B — Batch (OpenAI Batch API + DeepSeek real-time)
# ---------------------------------------------------------------------------

async def _run_deepseek_realtime(
    llm_client: LLMClient,
    extractor: KnowledgeExtractor,
    prepared: list[dict],
) -> dict[str, dict[int, list]]:
    """Run DeepSeek real-time for all docs while OpenAI batch processes.

    Returns {disease_id: {doc_idx: list[RawTriple]}}.
    """
    
    deepseek_model = "deepseek-v4" if "deepseek-v4" in llm_client.available_models else None
    if not deepseek_model:
        logger.warning("DeepSeek not available — skipping real-time second model")
        return {}

    results_by_disease: dict[str, dict[int, list]] = {}
    total_docs = sum(len(p["prompts"]) for p in prepared)
    done = 0

    for item in prepared:
        disease_id = item["disease_id"]
        results_by_disease[disease_id] = {}
        for doc_idx, (prompt, doc) in enumerate(item["prompts"]):
            try:
                raw = await llm_client.extract_async(deepseek_model, prompt)
                parsed = [extractor._parse_triple(t, doc, deepseek_model) for t in raw]
                results_by_disease[disease_id][doc_idx] = [t for t in parsed if t]
            except Exception as exc:
                logger.debug("DeepSeek failed doc %s/%d: %s", disease_id, doc_idx, exc)
                results_by_disease[disease_id][doc_idx] = []
            done += 1
            if done % 50 == 0:
                logger.info("DeepSeek real-time: %d/%d docs processed", done, total_docs)

    logger.info("DeepSeek real-time done: %d docs", done)
    return results_by_disease


async def _run_batch_mode(
    profiles: list[BioASQProfile],
    config: dict,
    primekg_index: PrimeKGIndex,
    resume: bool,
    auto: bool,
    batch_meta_path: str | None,
) -> list[dict]:
    """Batch mode: OpenAI Batch API + DeepSeek real-time → consensus → QC."""

    harvester = EvidenceHarvester(config=config)
    extractor = KnowledgeExtractor(config=config, primekg_index=primekg_index)
    quality   = QualityController(config=config, primekg_index=primekg_index)
    llm_client = LLMClient()
    batch_client = BatchLLMClient()

    results = []

    # ── Phase 1: Harvest all profiles ──────────────────────────────────────────
    if not batch_meta_path:
        logger.info("=== Phase 1: Harvesting %d BioASQ profiles ===", len(profiles))
        prepared = []
        from core.models import DiseaseProfile, SourceDocument

        for i, p in enumerate(profiles):
            disease_id = f"BIOASQ:{p.bioasq_id}"
            if resume and _is_already_done(disease_id):
                logger.info("SKIP (done): %s", disease_id)
                results.append({"bioasq_id": p.bioasq_id, "disease_id": disease_id,
                                 "status": "skipped_resume", "agents": {}})
                continue
            if not p.pmids_with_snippet:
                results.append({"bioasq_id": p.bioasq_id, "disease_id": disease_id,
                                 "status": "skipped_no_snippets", "agents": {}})
                continue

            logger.info("[%d/%d] Harvesting: %s", i + 1, len(profiles), disease_id)
            _, doc_dicts = _harvest_profile(p, harvester)
            if not doc_dicts:
                results.append({"bioasq_id": p.bioasq_id, "disease_id": disease_id,
                                 "status": "partial", "reason": "no_documents", "agents": {}})
                continue

            adapter = _make_adapter_profile(p)
            source_docs = [_reconstruct_source_doc(d) for d in doc_dicts]
            # Use BioASQ prompt (ideal_answer anchor) instead of disease prompt
            prompts = [(extractor._build_bioasq_prompt(p, doc), doc)
                       for doc in source_docs]

            prepared.append({
                "bioasq_id": p.bioasq_id,
                "disease_id": disease_id,
                "question_type": p.question_type,
                "adapter_profile": adapter,
                "bioasq_profile_dict": p.to_dict(),   # stored for --retrieve rebuild
                "prompts": prompts,                    # list[(prompt_str, SourceDocument)]
                "doc_dicts": doc_dicts,
            })
            logger.info("  → %d docs, %d prompts", len(doc_dicts), len(prompts))

        logger.info("Phase 1 done: %d profiles prepared, %d skipped",
                    len(prepared), len(results))

        if not prepared:
            logger.info("Nothing to submit.")
            return results

        # ── Phase 2: Submit to OpenAI Batch API ──────────────────────────────
        logger.info("=== Phase 2: Submitting %d prompts to OpenAI Batch API ===",
                    sum(len(p["prompts"]) for p in prepared))

        all_prompts: list[str] = []
        all_doc_ids: list[str] = []
        prompt_map: list[tuple[int, int]] = []  # (prepared_idx, doc_idx)

        for prep_idx, item in enumerate(prepared):
            did_safe = item["disease_id"].replace(":", "_")
            for doc_idx, (prompt, _) in enumerate(item["prompts"]):
                all_prompts.append(prompt)
                all_doc_ids.append(f"{did_safe}__doc{doc_idx}")
                prompt_map.append((prep_idx, doc_idx))

        batch_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        batch_dir = BATCH_DIR / f"bioasq_batch_{batch_ts}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        # Submit OpenAI only (DeepSeek is real-time; Anthropic optional)
        batch_ids = batch_client.submit(
            prompts=all_prompts,
            disease_id=f"bioasq_{batch_ts}",
            doc_ids=all_doc_ids,
            providers=["openai"],  # skip Anthropic — DeepSeek covers second model cheaper
        )
        logger.info("OpenAI batch submitted: %s", batch_ids)

        # Save batch metadata for --retrieve resume
        meta = {
            "batch_ts": batch_ts,
            "batch_ids": batch_ids,
            "prompt_count": len(all_prompts),
            "prompt_map": prompt_map,
            "prepared": [
                {
                    "bioasq_id": item["bioasq_id"],
                    "disease_id": item["disease_id"],
                    "question_type": item["question_type"],
                    "adapter_profile": item["adapter_profile"],
                    "bioasq_profile_dict": item.get("bioasq_profile_dict"),
                    "doc_dicts": item["doc_dicts"],
                }
                for item in prepared
            ],
        }
        batch_meta_file = batch_dir / "batch_meta.json"
        with open(batch_meta_file, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        logger.info("Batch metadata saved: %s", batch_meta_file)
        logger.info("To retrieve later: --batch --retrieve --batch-meta %s", batch_meta_file)

        if not auto:
            logger.info("Batch submitted. Run with --auto or --retrieve to process results.")
            return results

    else:
        # ── Load from saved batch meta ──────────────────────────────────────
        logger.info("=== Loading batch metadata from %s ===", batch_meta_path)
        with open(batch_meta_path) as f:
            meta = json.load(f)
        batch_ids = meta["batch_ids"]
        prompt_map = [tuple(x) for x in meta["prompt_map"]]
        batch_dir = Path(batch_meta_path).parent
        batch_ts = meta["batch_ts"]

        # Reconstruct prepared (without prompts — will rebuild from doc_dicts)
        prepared = []
        for item in meta["prepared"]:
            source_docs = [_reconstruct_source_doc(d) for d in item["doc_dicts"]]
            # Prefer BioASQ prompt if profile dict was saved; fall back to disease prompt
            if "bioasq_profile_dict" in item:
                bp = BioASQProfile.from_dict(item["bioasq_profile_dict"])
                prompts = [(extractor._build_bioasq_prompt(bp, doc), doc)
                           for doc in source_docs]
            else:
                from core.models import DiseaseProfile
                dp = DiseaseProfile.from_dict(dict(item["adapter_profile"]))
                prompts = [(extractor._build_prompt(dp, doc), doc) for doc in source_docs]
            prepared.append({
                **item,
                "prompts": prompts,
            })
        logger.info("Loaded %d prepared profiles", len(prepared))
        auto = True  # Force auto when --retrieve

    # ── Phase 3: Run DeepSeek real-time ──────────────────────────────────────
    logger.info("=== Phase 3: Running DeepSeek real-time ===")
    deepseek_results = await _run_deepseek_realtime(llm_client, extractor, prepared)
    

    # ── Phase 4: Poll + retrieve OpenAI batch ────────────────────────────────
    if batch_ids:
        logger.info("=== Phase 4: Polling OpenAI Batch API ===")
        statuses = batch_client.poll(
            batch_ids, poll_interval=60, max_wait=86400  # max 24h
        )
        logger.info("Batch statuses: %s", statuses)

        logger.info("=== Phase 5: Retrieving batch results ===")
        batch_results = batch_client.retrieve(batch_ids)
        logger.info("Retrieved %d batch results", len(batch_results))

        with open(batch_dir / "batch_results_raw.json", "w") as f:
            json.dump(batch_results, f, indent=2, default=str)
    else:
        batch_results = {}
        logger.warning("No batch IDs to retrieve — processing with DeepSeek only")

    # ── Phase 5 (or 6): Consensus + QC per profile ──────────────────────────
    logger.info("=== Phase 6: Computing consensus + QC ===")

    for prep_idx, item in enumerate(prepared):
        disease_id = item["disease_id"]
        adapter_profile = item["adapter_profile"]
        cache_dir = OUTPUT_DIR / disease_id.replace(":", "_")
        cache_dir.mkdir(parents=True, exist_ok=True)

        ext_result = ExtractionResult(disease_id=disease_id)

        for doc_idx, (prompt, doc) in enumerate(item["prompts"]):
            did_safe = disease_id.replace(":", "_")
            doc_id = f"{did_safe}__doc{doc_idx}"
            # custom_id MUST be rebuilt with the SAME disease_id used at submit
            # time (Phase 2 passes the batch-wide `bioasq_{batch_ts}`), otherwise
            # the clean_d prefix differs and the OpenAI result key never matches.
            custom_id = BatchLLMClient._make_custom_id(
                f"bioasq_{batch_ts}", doc_id, "gpt4omini"
            )

            per_model: dict[str, list] = {}

            # OpenAI result
            openai_key = f"{custom_id}"
            if openai_key in batch_results:
                raw = batch_results[openai_key].get("triples", [])
                parsed = [extractor._parse_triple(t, doc, "gpt-4o-mini") for t in raw]
                per_model["gpt-4o-mini"] = [t for t in parsed if t]

            # DeepSeek real-time result
            ds = deepseek_results.get(disease_id, {}).get(doc_idx, [])
            if ds:
                per_model["deepseek-v4"] = ds

            for model_triples in per_model.values():
                ext_result.raw_triples.extend(model_triples)

            consensus = extractor._compute_consensus(per_model)
            ext_result.consensus_triples.extend(consensus)

        extractor._normalize_triples(ext_result.consensus_triples)
        extractor._save_results(ext_result, cache_dir)

        # Agent 4: QC
        consensus_triples_raw = []
        consensus_file = cache_dir / "consensus_triples.jsonl"
        if consensus_file.exists():
            with open(consensus_file) as f:
                for line in f:
                    if line.strip():
                        consensus_triples_raw.append(json.loads(line))

        quality_result = await quality.run_with_retry({
            "profile": adapter_profile,
            "consensus_triples": consensus_triples_raw,
        })
        validated = quality_result.metrics.get("validated_triples", 0)
        logger.info("DONE: %s — %d consensus, %d validated",
                    disease_id, len(ext_result.consensus_triples), validated)

        results.append({
            "bioasq_id": item["bioasq_id"],
            "disease_id": disease_id,
            "question_type": item["question_type"],
            "status": "success" if quality_result.status == "success" else "partial",
            "agents": {
                "extractor": {"status": "success",
                               "metrics": {"consensus_count": len(ext_result.consensus_triples)}},
                "quality": {"status": quality_result.status,
                             "metrics": quality_result.metrics},
            },
        })

    # Print cost summary
    logger.info("\n%s", batch_client.cost_summary)
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict], elapsed: float) -> None:
    counts: dict[str, int] = {}
    total_triples = 0
    for r in results:
        s = r.get("status", "failed")
        counts[s] = counts.get(s, 0) + 1
        total_triples += r.get("agents", {}).get("quality", {}).get("metrics", {}).get("validated_triples", 0)

    print(f"\n{'='*60}")
    print("BUILD KG FROM BIOASQ — SUMMARY")
    print(f"{'='*60}")
    print(f"  Total processed   : {len(results)}")
    for status, count in sorted(counts.items()):
        print(f"  {status:<22}: {count}")
    print(f"  Total triples     : {total_triples}")
    print(f"  Elapsed           : {elapsed:.1f}s")
    if counts.get("success", 0) > 0:
        print(f"  Avg time/success  : {elapsed / counts['success']:.1f}s")
    print(f"  Output dir        : {OUTPUT_DIR}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ChronoMedKG from BioASQ Task B"
    )
    parser.add_argument("--bioasq", type=Path, default=BIOASQ_PATH)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--question-id", type=str, default=None)
    parser.add_argument("--question-type",
                        choices=["summary", "yesno", "factoid", "list"], default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (real-time mode only, default: 1)")
    parser.add_argument("--max-docs", type=int, default=200)
    parser.add_argument("--skip-primekg", action="store_true",
                        help="Skip PrimeKG loading (no schema alignment/confirmation). "
                             "Use when kg_index.db not yet built and RAM < 8GB. "
                             "Run scripts/update/build_primekg_sqlite.py first, then remove this flag.")

    # Batch mode
    parser.add_argument("--batch", action="store_true",
                        help="Use OpenAI Batch API + DeepSeek real-time (50%% cost, ~24h)")
    parser.add_argument("--no-auto", action="store_true",
                        help="(batch mode) Submit only, do not poll/retrieve")
    parser.add_argument("--retrieve", action="store_true",
                        help="(batch mode) Retrieve from a previously submitted batch")
    parser.add_argument("--batch-meta", type=str, default=None,
                        help="(batch --retrieve) Path to batch_meta.json from prior submission")

    args = parser.parse_args()
    config = {"max_extraction_docs": args.max_docs}

    # ── Agent 1 ──────────────────────────────────────────────────────────────
    logger.info("=== Build KG from BioASQ | mode=%s ===",
                "batch" if args.batch else "real-time")

    if not (args.batch and args.retrieve):
        logger.info("[1/4] DiseaseProfiler (bioasq_data mode)...")
        profiler = DiseaseProfiler(config=config)
        loop = asyncio.new_event_loop()
        try:
            r = loop.run_until_complete(profiler.run({
                "mode": "bioasq_data",
                "file_path": str(args.bioasq),
            }))
        finally:
            loop.close()

        if r.status == "failed":
            logger.error("DiseaseProfiler failed")
            sys.exit(1)

        profiles = [BioASQProfile.from_dict(p) for p in r.data["profiles"]]
        logger.info("Agent 1: %d profiles (skipped %d invalid)",
                    len(profiles), r.metrics.get("skipped_invalid_item", 0))

        if args.question_id:
            profiles = [p for p in profiles if p.bioasq_id == args.question_id]
            if not profiles:
                logger.error("Question ID not found: %s", args.question_id)
                sys.exit(1)
        if args.question_type:
            before = len(profiles)
            profiles = [p for p in profiles if p.question_type == args.question_type]
            logger.info("Filtered by type=%s: %d → %d", args.question_type, before, len(profiles))
        if args.max_samples is not None:
            profiles = profiles[:args.max_samples]
            logger.info("Limited to %d (--max-samples)", len(profiles))

        logger.info("Processing %d profiles...", len(profiles))
    else:
        profiles = []  # Not needed for --retrieve mode

    if args.skip_primekg:
        logger.warning("--skip-primekg: PrimeKG NOT loaded. QC will skip schema alignment.")
        logger.warning("  Run 'python scripts/update/build_primekg_sqlite.py' to fix this.")
        primekg_index = PrimeKGIndex.__new__(PrimeKGIndex)
        primekg_index.__dict__.update({
            "kg_path": Path(""),
            "name_to_nodes": {},
            "disease_edges": {},
            "edge_lookup": {},
            "relation_pairs": {},
            "disease_nodes": {},
            "_edge_lookup_lite": {},
            "_lite_mode": True,
            "_sqlite_mode": False,
            "_db": None,
            "_loaded": True,   # mark loaded so QC doesn't try to load
        })
    else:
        logger.info("Loading PrimeKG index...")
        primekg_index = PrimeKGIndex()

    start = time.monotonic()

    if args.batch:
        results = asyncio.run(_run_batch_mode(
            profiles=profiles,
            config=config,
            primekg_index=primekg_index,
            resume=not args.no_resume,
            auto=not args.no_auto,
            batch_meta_path=args.batch_meta if args.retrieve else None,
        ))
    elif args.workers > 1 and len(profiles) > 1:
        results = _run_parallel(profiles, config, primekg_index,
                                resume=not args.no_resume, workers=args.workers)
    else:
        results = _run_sequential(profiles, config, primekg_index,
                                  resume=not args.no_resume)

    elapsed = time.monotonic() - start

    checkpoint_path = OUTPUT_DIR / "bioasq_pipeline_checkpoint.json"
    with open(checkpoint_path, "w") as f:
        json.dump({"timestamp": datetime.utcnow().isoformat(),
                   "total": len(results), "results": results},
                  f, indent=2, default=str)
    logger.info("Checkpoint: %s", checkpoint_path)

    _print_summary(results, elapsed)


if __name__ == "__main__":
    main()
