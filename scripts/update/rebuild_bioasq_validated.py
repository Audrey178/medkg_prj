"""
Rebuild validated_triples.jsonl from consensus_triples.jsonl for BioASQ folders.

Skips the expensive LLM extraction step — only re-runs Agent 4 (QualityController).
Works without a disease YAML config by constructing a minimal DiseaseProfile.

Usage:
    # All BIOASQ_* folders in data/extracted/
    python -m scripts.update.rebuild_bioasq_validated

    # Single folder by disease_id
    python -m scripts.update.rebuild_bioasq_validated --disease-id BIOASQ:5a3e8683966455904c000007

    # Single folder by folder name
    python -m scripts.update.rebuild_bioasq_validated --folder BIOASQ_5a3e8683966455904c000007
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.quality_controller import QualityController
from core.schema_alignment import PrimeKGIndex

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("rebuild_bioasq_validated")

EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"


def _minimal_profile(disease_id: str) -> dict:
    """Minimal DiseaseProfile dict — only fields QualityController actually reads."""
    return {
        "disease_id": disease_id,
        "disease_name": disease_id,
        "synonyms": [],
        "disease_category": "",
        "inheritance_pattern": "",
        "disease_type": "",
        "differential_diseases": [],
        "tier1_relations": [],
        "guideline_records": [],
        "primekg_node_id": None,
        "primekg_neighbor_count": 0,
        "primekg_edge_types": [],
        "has_genereviews": False,
        "has_omim": False,
        "has_orphanet": False,
        "has_clinical_guidelines": [],
        "pubmed_article_count": 0,
        "pmc_oa_count": 0,
        "key_genes": [],
        "key_phenotypes": [],
        "mesh_terms": [],
        "has_uniprot": False,
        "uniprot_accession": {},
    }


def _load_consensus(folder: Path) -> list[dict]:
    triples = []
    consensus_file = folder / "consensus_triples.jsonl"
    if not consensus_file.exists():
        return triples
    with open(consensus_file) as f:
        for line in f:
            line = line.strip()
            if line:
                triples.append(json.loads(line))
    return triples


def _disease_id_from_folder(folder: Path) -> str:
    """Try quality_report.json first, then derive from folder name."""
    report_file = folder / "quality_report.json"
    if report_file.exists():
        try:
            report = json.loads(report_file.read_text())
            if report.get("disease_id"):
                return report["disease_id"]
        except Exception:
            pass
    # e.g. BIOASQ_5a3e8683966455904c000007 -> BIOASQ:5a3e8683966455904c000007
    return folder.name.replace("_", ":", 1)


async def rebuild_folder(folder: Path, qc: QualityController) -> dict:
    disease_id = _disease_id_from_folder(folder)
    consensus_triples = _load_consensus(folder)

    if not consensus_triples:
        logger.warning("%s — no consensus_triples.jsonl, skipping", disease_id)
        return {"disease_id": disease_id, "status": "skipped", "reason": "no_consensus"}

    logger.info("%s — %d triples → running QC", disease_id, len(consensus_triples))

    result = await qc.run_with_retry({
        "profile": _minimal_profile(disease_id),
        "consensus_triples": consensus_triples,
    })

    logger.info(
        "%s — done: validated=%s rejected=%s grade=%s",
        disease_id,
        result.metrics.get("validated_triples"),
        result.metrics.get("rejected_triples"),
        result.metrics.get("quality_grade"),
    )
    return {"disease_id": disease_id, "status": result.status, "metrics": result.metrics}


async def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild validated_triples from consensus_triples for BioASQ folders")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--disease-id", help="Single disease ID, e.g. BIOASQ:5a3e8683966455904c000007")
    group.add_argument("--folder", help="Single folder name under data/extracted/, e.g. BIOASQ_5a3e8683966455904c000007")
    args = parser.parse_args()

    # Resolve target folders
    if args.disease_id:
        folder_name = args.disease_id.replace(":", "_")
        folders = [EXTRACTED_DIR / folder_name]
    elif args.folder:
        folders = [EXTRACTED_DIR / args.folder]
    else:
        folders = sorted(p for p in EXTRACTED_DIR.iterdir() if p.is_dir() and p.name.startswith("BIOASQ_"))

    if not folders:
        logger.error("No BIOASQ folders found in %s", EXTRACTED_DIR)
        sys.exit(1)

    missing = [f for f in folders if not f.exists()]
    if missing:
        logger.error("Folders not found: %s", missing)
        sys.exit(1)

    logger.info("Loading PrimeKG index…")
    primekg = PrimeKGIndex()
    primekg.load()

    config: dict = {}
    qc = QualityController(config=config, primekg_index=primekg)

    results = []
    for folder in folders:
        result = await rebuild_folder(folder, qc)
        results.append(result)

    print("\n=== Summary ===")
    for r in results:
        print(f"  {r['disease_id']}: {r['status']}", end="")
        if r.get("metrics"):
            m = r["metrics"]
            print(f"  validated={m.get('validated_triples')} rejected={m.get('rejected_triples')} grade={m.get('quality_grade')}", end="")
        print()


if __name__ == "__main__":
    asyncio.run(main())
