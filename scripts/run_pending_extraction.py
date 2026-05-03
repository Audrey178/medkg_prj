"""
Run extraction + QC on diseases that have evidence but no consensus triples.
Useful for completing partially-run pipelines without re-harvesting.

Usage:
    python -m scripts.run_pending_extraction
    python -m scripts.run_pending_extraction --disease-id "OMIM:139393"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

from agents.evidence_harvester import _load_evidence_json
from agents.knowledge_extractor import KnowledgeExtractor
from agents.quality_controller import QualityController
from core.models import DiseaseProfile
from core.schema_alignment import PrimeKGIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("pending_extraction")


def find_pending_diseases(extracted_dir: Path, config_dir: Path) -> list[dict]:
    """Find diseases with evidence but no extraction results."""
    pending = []
    for d in sorted(extracted_dir.iterdir()):
        if not d.is_dir():
            continue
        has_evidence = (d / "evidence_collection.json.gz").exists() or (d / "evidence_collection.json").exists()
        has_consensus = (d / "consensus_triples.jsonl").exists()
        config_file = config_dir / f"{d.name}.yaml"
        has_config = config_file.exists()

        if has_evidence and not has_consensus and has_config:
            import yaml
            with open(config_file) as f:
                profile = yaml.safe_load(f)
            pending.append({
                "disease_id": profile.get("disease_id", d.name.replace("_", ":")),
                "disease_name": profile.get("disease_name", d.name),
                "dir": d,
                "config_file": config_file,
            })
    return pending


async def run_extraction_and_qc(disease: dict, config: dict) -> dict:
    """Run extraction + QC for a single disease."""
    disease_id = disease["disease_id"]
    disease_name = disease["disease_name"]
    cache_dir = disease["dir"]

    logger.info("=" * 60)
    logger.info("EXTRACTION: %s (%s)", disease_name, disease_id)
    logger.info("=" * 60)

    # Load profile
    import yaml
    with open(disease["config_file"]) as f:
        profile = yaml.safe_load(f)

    # Load evidence documents (prefer .json.gz, fall back to .json)
    ev_data = _load_evidence_json(cache_dir)
    if ev_data is None:
        logger.error("No evidence collection found in %s", cache_dir)
        return {"disease_id": disease_id, "status": "failed", "error": "no_evidence"}
    documents = ev_data.get("tier1_sources", []) + ev_data.get("tier2_sources", [])

    # Limit documents
    max_docs = config.get("max_extraction_docs", 100)
    if len(documents) > max_docs:
        tier1 = [d for d in documents if d.get("tier") == 1]
        tier2 = sorted(
            [d for d in documents if d.get("tier") != 1],
            key=lambda d: d.get("credibility_score", 0),
            reverse=True,
        )
        documents = tier1 + tier2[:max_docs - len(tier1)]

    logger.info("Extracting from %d documents", len(documents))

    # Run extraction
    extractor = KnowledgeExtractor(config=config)
    extractor_result = await extractor.run_with_retry({
        "profile": profile,
        "documents": documents,
    })
    logger.info("Extraction: %s — %s", extractor_result.status, extractor_result.metrics)

    # Load consensus triples
    consensus_file = cache_dir / "consensus_triples.jsonl"
    consensus_triples = []
    if consensus_file.exists():
        with open(consensus_file) as f:
            for line in f:
                if line.strip():
                    consensus_triples.append(json.loads(line))

    # Run QC
    primekg_index = PrimeKGIndex()
    quality = QualityController(config=config, primekg_index=primekg_index)
    quality_result = await quality.run_with_retry({
        "profile": profile,
        "consensus_triples": consensus_triples,
    })
    logger.info("QC: %s — %s", quality_result.status, quality_result.metrics)

    return {
        "disease_id": disease_id,
        "disease_name": disease_name,
        "extraction": extractor_result.metrics,
        "quality": quality_result.metrics,
    }


async def main():
    parser = argparse.ArgumentParser(description="Run pending extraction + QC")
    parser.add_argument("--disease-id", type=str, help="Run for specific disease ID only")
    args = parser.parse_args()

    extracted_dir = PROJECT_ROOT / "data" / "extracted"
    config_dir = PROJECT_ROOT / "config" / "diseases"

    pending = find_pending_diseases(extracted_dir, config_dir)
    if args.disease_id:
        pending = [d for d in pending if d["disease_id"] == args.disease_id]

    if not pending:
        logger.info("No pending diseases found.")
        return

    logger.info("Found %d pending diseases:", len(pending))
    for d in pending:
        logger.info("  %s (%s)", d["disease_name"], d["disease_id"])

    config = {}
    for disease in pending:
        result = await run_extraction_and_qc(disease, config)
        logger.info("Done: %s", result)


if __name__ == "__main__":
    asyncio.run(main())
