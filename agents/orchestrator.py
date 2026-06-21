"""
Orchestrator
============
Main pipeline controller. Takes a disease list, dispatches agents
sequentially per disease (Agent 1→2→3→4), handles failures gracefully.

Usage:
    # Single disease
    python -m agents.orchestrator --disease-id "OMIM:310200" --disease-name "Duchenne muscular dystrophy"

    # Multiple diseases from file
    python -m agents.orchestrator --disease-file diseases.txt

    # All diseases with GeneReviews (Phase 3)
    python -m agents.orchestrator --source genereviews
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from project root
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

from agents.disease_profiler import DiseaseProfiler
from agents.evidence_harvester import EvidenceHarvester, _load_evidence_json
from agents.knowledge_extractor import KnowledgeExtractor
from agents.quality_controller import QualityController
from core.models import AgentResult, DiseaseProfile
from core.schema_alignment import PrimeKGIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("orchestrator")


@dataclass
class PipelineProgress:
    total_diseases: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    current_disease: str = ""
    current_agent: str = ""
    start_time: float = field(default_factory=time.monotonic)
    results: list[dict] = field(default_factory=list)


class Orchestrator:
    """Main pipeline controller."""

    def __init__(self, config: dict | None = None,
                 primekg_index: PrimeKGIndex | None = None):
        self.config = config or {}
        self.progress = PipelineProgress()

        # PrimeKG index (shared across agents, loaded once)
        # Accept pre-loaded index to avoid duplicating hundreds of MB per worker
        self.primekg_index = primekg_index or PrimeKGIndex()

        # Initialize agents
        self.profiler = DiseaseProfiler(config=self.config)
        self.harvester = EvidenceHarvester(config=self.config)
        self.extractor = KnowledgeExtractor(config=self.config,
                                            primekg_index=self.primekg_index)
        self.quality = QualityController(config=self.config, primekg_index=self.primekg_index)

        self._output_dir = PROJECT_ROOT / "data" / "extracted"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def run_single(self, disease_id: str, disease_name: str) -> dict:
        """Run full pipeline for a single disease."""
        logger.info("=" * 60)
        logger.info("PIPELINE START: %s (%s)", disease_name, disease_id)
        logger.info("=" * 60)

        self.progress.current_disease = disease_name
        disease_start = time.monotonic()
        pipeline_result = {
            "disease_id": disease_id,
            "disease_name": disease_name,
            "status": "pending",
            "agents": {},
        }

        try:
            # Agent 1: Disease Profiler
            self.progress.current_agent = "DiseaseProfiler"
            logger.info("[1/4] Disease Profiler...")
            profiler_result = await self.profiler.run_with_retry({
                "disease_id": disease_id,
                "disease_name": disease_name,
            })
            pipeline_result["agents"]["profiler"] = {
                "status": profiler_result.status,
                "metrics": profiler_result.metrics,
            }

            if profiler_result.status == "failed":
                pipeline_result["status"] = "failed"
                pipeline_result["error"] = "Profiler failed"
                return pipeline_result

            profile = profiler_result.data["profile"]

            # Check if sufficient sources
            if not DiseaseProfile.from_dict(profile).has_sufficient_sources():
                logger.warning("Insufficient sources for %s — skipping", disease_name)
                pipeline_result["status"] = "skipped"
                pipeline_result["reason"] = "insufficient_sources"
                self.progress.skipped += 1
                return pipeline_result

            # Agent 2: Evidence Harvester
            self.progress.current_agent = "EvidenceHarvester"
            logger.info("[2/4] Evidence Harvester...")
            harvester_result = await self.harvester.run_with_retry({
                "profile": profile,
            })
            pipeline_result["agents"]["harvester"] = {
                "status": harvester_result.status,
                "metrics": harvester_result.metrics,
            }

            if harvester_result.status == "failed":
                pipeline_result["status"] = "failed"
                pipeline_result["error"] = "Harvester failed"
                return pipeline_result

            # Get documents for extraction
            # Load from cache (harvester saved full docs as .json.gz)
            cache_dir = self._output_dir / disease_id.replace(":", "_")
            documents = []
            ev_data = _load_evidence_json(cache_dir)
            if ev_data:
                documents = ev_data.get("tier1_sources", []) + ev_data.get("tier2_sources", [])

            if not documents:
                logger.warning("No documents harvested for %s", disease_name)
                pipeline_result["status"] = "partial"
                pipeline_result["reason"] = "no_documents"
                return pipeline_result

            # Literature-adaptive tiering: adjust max docs based on
            # disease literature availability (coverage_flag from profiler)
            configured_max = self.config.get("max_extraction_docs", 200) # số lượng tài liệu tối đa được cấu hình
            coverage = profile.get("coverage_flag", "moderate")
            pubmed_count = profile.get("pubmed_article_count",
                                       profile.get("pubmed_count", 0))

            if coverage == "rich": # Large tier: limitation = 200
                # Deep tier: GeneReviews + >500 PubMed — full 200 docs
                adaptive_max = min(configured_max, 200)
                tier_label = "Deep"
            elif pubmed_count >= 100:
                # Standard tier: >100 PubMed — 75-150 mixed docs
                adaptive_max = min(configured_max, 150)
                tier_label = "Standard"
            elif pubmed_count >= 20:
                # Light tier: 20-100 PubMed — ALL available papers
                adaptive_max = max(len(documents), configured_max)
                tier_label = "Light (all available)"
            else:
                # Minimal tier: <20 PubMed — ALL available papers
                adaptive_max = max(len(documents), configured_max)
                tier_label = "Minimal (all available)"

            logger.info("Adaptive tier: %s | PubMed=%d | max_docs=%d | available=%d",
                        tier_label, pubmed_count, adaptive_max, len(documents))

            if len(documents) > adaptive_max:
                # Prioritize: Tier 1 first, then Tier 2 sorted by credibility
                tier1 = [d for d in documents if d.get("tier") == 1]
                tier2 = sorted(
                    [d for d in documents if d.get("tier") != 1],
                    key=lambda d: d.get("credibility_score", 0),
                    reverse=True,
                )
                documents = tier1 + tier2[:adaptive_max - len(tier1)]
                logger.info("Limited to %d docs for extraction (was %d)",
                            len(documents), len(tier1) + len(tier2))

            # Agent 3: Knowledge Extractor
            self.progress.current_agent = "KnowledgeExtractor"
            logger.info("[3/4] Knowledge Extractor (%d documents)...", len(documents))
            extractor_result = await self.extractor.run_with_retry({
                "profile": profile,
                "documents": documents,
            })
            pipeline_result["agents"]["extractor"] = {
                "status": extractor_result.status,
                "metrics": extractor_result.metrics,
            }

            # Load consensus triples for quality control
            consensus_file = cache_dir / "consensus_triples.jsonl"
            consensus_triples = []
            if consensus_file.exists():
                with open(consensus_file) as f:
                    for line in f:
                        if line.strip():
                            consensus_triples.append(json.loads(line))

            # Agent 4: Quality Controller
            self.progress.current_agent = "QualityController"
            logger.info("[4/4] Quality Controller (%d consensus triples)...", len(consensus_triples))
            quality_result = await self.quality.run_with_retry({
                "profile": profile,
                "consensus_triples": consensus_triples,
            })
            pipeline_result["agents"]["quality"] = {
                "status": quality_result.status,
                "metrics": quality_result.metrics,
            }

            pipeline_result["status"] = "success"
            self.progress.completed += 1

        except Exception as e:
            logger.error("Pipeline failed for %s: %s", disease_name, e, exc_info=True)
            pipeline_result["status"] = "failed"
            pipeline_result["error"] = str(e)
            self.progress.failed += 1

        elapsed = time.monotonic() - disease_start
        pipeline_result["elapsed_seconds"] = round(elapsed, 1)

        logger.info("PIPELINE %s for %s (%.1fs)",
                     pipeline_result["status"].upper(), disease_name, elapsed)
        logger.info("")

        self.progress.results.append(pipeline_result)
        return pipeline_result

    async def run_batch(self, diseases: list[dict], resume: bool = True,
                        workers: int = 1) -> list[dict]:
        """Run pipeline for a batch of diseases.

        Args:
            diseases: list of {"disease_id": ..., "disease_name": ...}
            resume: skip diseases that already have validated_triples.jsonl
            workers: number of concurrent disease pipelines (1 = sequential)
        """
        self.progress.total_diseases = len(diseases)

        # Find already-completed diseases for resume
        completed_ids = set()
        if resume:
            for d in diseases:
                did = d["disease_id"]
                cache_dir = self._output_dir / did.replace(":", "_")
                validated = cache_dir / "validated_triples.jsonl"
                consensus = cache_dir / "consensus_triples.jsonl"
                if validated.exists() and consensus.exists():
                    line_count = sum(1 for _ in open(consensus))
                    if line_count > 10:
                        completed_ids.add(did)
            if completed_ids:
                logger.info("Resuming: skipping %d already-completed diseases", len(completed_ids))

        # Separate skip vs pending
        results = []
        pending = []
        for i, disease in enumerate(diseases):
            did = disease["disease_id"]
            if did in completed_ids:
                logger.info("SKIP (already completed): %s (%s)", disease["disease_name"], did)
                self.progress.skipped += 1
                results.append({
                    "disease_id": did,
                    "disease_name": disease["disease_name"],
                    "status": "skipped_resume",
                })
            else:
                pending.append(disease)

        if workers <= 1 or len(pending) <= 1:
            # Sequential mode (original behavior)
            for i, disease in enumerate(pending):
                logger.info("Disease %d/%d (sequential)", i + 1, len(pending))
                result = await self.run_single(disease["disease_id"], disease["disease_name"])
                results.append(result)
                self._save_checkpoint(results)
        else:
            # PARALLEL mode: run N diseases concurrently
            logger.info("=" * 60)
            logger.info("PARALLEL MODE: %d workers for %d diseases", workers, len(pending))
            logger.info("=" * 60)
            await self._run_parallel(pending, results, workers)

        self._print_summary(results)
        return results

    async def _run_parallel(self, pending: list[dict], results: list[dict],
                            workers: int) -> None:
        """Process diseases with N concurrent workers using threads.

        The agents use synchronous I/O (urllib, time.sleep) which blocks
        the asyncio event loop. We use a ThreadPoolExecutor so each disease
        pipeline runs in its own thread with its own Orchestrator instance.
        """
        import concurrent.futures
        import threading

        lock = threading.Lock()

        # Ensure PrimeKG index is loaded ONCE on the parent before spawning workers
        if not self.primekg_index.is_loaded:
            self.primekg_index.load()

        def _run_disease_sync(disease: dict, idx: int) -> dict:
            """Run a complete disease pipeline in a dedicated thread."""
            logger.info("Disease %d/%d (thread): %s",
                        idx + 1, len(pending), disease["disease_name"])
            # Each thread gets its own Orchestrator (own agents, own state)
            # but SHARES the parent's PrimeKG index to avoid OOM with many workers
            worker_orch = Orchestrator(config=self.config,
                                       primekg_index=self.primekg_index)
            # Run the async pipeline in a new event loop for this thread
            thread_loop = asyncio.new_event_loop()
            try:
                result = thread_loop.run_until_complete(
                    worker_orch.run_single(disease["disease_id"], disease["disease_name"])
                )
            finally:
                thread_loop.close()
            return result

        def _submit_and_collect():
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_disease = {
                    executor.submit(_run_disease_sync, d, i): d
                    for i, d in enumerate(pending)
                }

                for future in concurrent.futures.as_completed(future_to_disease):
                    disease = future_to_disease[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        logger.error("Worker failed for %s: %s", disease["disease_name"], e)
                        result = {
                            "disease_id": disease["disease_id"],
                            "disease_name": disease["disease_name"],
                            "status": "failed",
                            "error": str(e),
                        }
                        self.progress.failed += 1

                    with lock:
                        results.append(result)
                        if result.get("status") == "success":
                            self.progress.completed += 1
                        self._save_checkpoint(results)

        # Run the thread pool in an executor so we don't block the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _submit_and_collect)

    def _save_checkpoint(self, results: list[dict]) -> None:
        """Save pipeline progress to disk."""
        checkpoint = {
            "timestamp": datetime.utcnow().isoformat(),
            "progress": {
                "total": self.progress.total_diseases,
                "completed": self.progress.completed,
                "failed": self.progress.failed,
                "skipped": self.progress.skipped,
            },
            "results": results,
        }
        with open(self._output_dir / "pipeline_checkpoint.json", "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)

    def _print_summary(self, results: list[dict]) -> None:
        """Print pipeline summary."""
        elapsed = time.monotonic() - self.progress.start_time
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Total diseases: {len(results)}")
        print(f"Completed: {self.progress.completed}")
        print(f"Failed: {self.progress.failed}")
        print(f"Skipped: {self.progress.skipped}")
        print(f"Total time: {elapsed:.1f}s")
        if self.progress.completed > 0:
            print(f"Avg time/disease: {elapsed / self.progress.completed:.1f}s")

        for r in results:
            status_icon = {"success": "+", "failed": "X", "skipped": "-", "partial": "~"}.get(r["status"], "?")
            print(f"  [{status_icon}] {r['disease_name']} ({r['disease_id']}): {r['status']}")

            # Print agent stats
            for agent_name, agent_data in r.get("agents", {}).items():
                metrics = agent_data.get("metrics", {})
                key_metrics = {k: v for k, v in metrics.items()
                               if k in ("pubmed_count", "tier1_count", "tier2_count",
                                        "raw_count", "consensus_count", "validated_triples",
                                        "quality_grade", "temporal_coverage")}
                if key_metrics:
                    print(f"    {agent_name}: {key_metrics}")
        print()


    async def rerun_quality_control(self, disease_id: str, disease_name: str) -> dict:
        """
        Re-run Quality Controller on existing extraction data.
        Useful after updating QC logic (schema alignment, credibility, temporal reasoning)
        without re-running the expensive LLM extraction step.
        """
        logger.info("Re-running QC for %s (%s)", disease_name, disease_id)

        cache_dir = self._output_dir / disease_id.replace(":", "_")

        # Load disease profile from config
        config_file = PROJECT_ROOT / "config" / "diseases" / f"{disease_id.replace(':', '_')}.yaml"
        if config_file.exists():
            import yaml
            with open(config_file) as f:
                profile = yaml.safe_load(f)
        else:
            logger.error("No disease config found: %s", config_file)
            return {"status": "failed", "error": "no_config"}

        # Load consensus triples
        consensus_file = cache_dir / "consensus_triples.jsonl"
        if not consensus_file.exists():
            logger.error("No consensus triples found for %s", disease_id)
            return {"status": "failed", "error": "no_consensus_triples"}

        consensus_triples = []
        with open(consensus_file) as f:
            for line in f:
                if line.strip():
                    consensus_triples.append(json.loads(line))

        logger.info("Loaded %d consensus triples for QC", len(consensus_triples))

        quality_result = await self.quality.run_with_retry({
            "profile": profile,
            "consensus_triples": consensus_triples,
        })

        logger.info("QC result: %s", quality_result.metrics)
        return {
            "disease_id": disease_id,
            "disease_name": disease_name,
            "status": quality_result.status,
            "metrics": quality_result.metrics,
        }


async def main():
    parser = argparse.ArgumentParser(description="ChronoMedKG Pipeline Orchestrator")
    parser.add_argument("--disease-id", type=str, help="Single disease ID (e.g., OMIM:310200)")
    parser.add_argument("--disease-name", type=str, help="Disease name")
    parser.add_argument("--disease-file", type=str, help="File with disease IDs and names (TSV)")
    parser.add_argument("--max-docs", type=int, default=200,
                        help="Max documents per disease (default: 200; adaptive tiering adjusts per disease)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Don't skip already-completed diseases (re-run everything)")
    parser.add_argument("--workers", type=int, default=3,
                        help="Number of concurrent disease pipelines (default: 3)")
    parser.add_argument("--qc-only", action="store_true",
                        help="Re-run Quality Controller only on existing extraction data")
    args = parser.parse_args()

    config = {"max_extraction_docs": args.max_docs}
    orchestrator = Orchestrator(config=config)

    if args.qc_only:
        # Re-run QC on all diseases with existing extraction data
        extracted_dir = PROJECT_ROOT / "data" / "extracted"
        diseases_to_qc = []
        for d in sorted(extracted_dir.iterdir()):
            if d.is_dir() and (d / "consensus_triples.jsonl").exists():
                config_file = PROJECT_ROOT / "config" / "diseases" / f"{d.name}.yaml"
                if config_file.exists():
                    import yaml
                    with open(config_file) as f:
                        profile = yaml.safe_load(f)
                    diseases_to_qc.append({
                        "disease_id": profile.get("disease_id", d.name.replace("_", ":")),
                        "disease_name": profile.get("disease_name", d.name),
                    })

        if args.disease_id:
            diseases_to_qc = [d for d in diseases_to_qc if d["disease_id"] == args.disease_id]

        logger.info("Re-running QC on %d diseases", len(diseases_to_qc))
        for disease in diseases_to_qc:
            await orchestrator.rerun_quality_control(disease["disease_id"], disease["disease_name"])
        return

    if args.disease_id and args.disease_name:
        await orchestrator.run_single(args.disease_id, args.disease_name)
    elif args.disease_file:
        diseases = []
        with open(args.disease_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    diseases.append({"disease_id": parts[0], "disease_name": parts[1]})
        await orchestrator.run_batch(diseases, resume=not args.no_resume,
                                     workers=args.workers)
    else:
        # Default: run on Paper 1 diseases for validation
        diseases = [
            {"disease_id": "OMIM:310200", "disease_name": "Duchenne muscular dystrophy"},
            {"disease_id": "OMIM:300376", "disease_name": "Becker muscular dystrophy"},
            {"disease_id": "OMIM:254200", "disease_name": "Myasthenia gravis"},
        ]
        await orchestrator.run_batch(diseases, resume=not args.no_resume,
                                     workers=args.workers)


if __name__ == "__main__":
    asyncio.run(main())
