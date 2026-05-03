"""
Re-compute consensus triples from existing raw triples using semantic matching,
then re-run QC. No LLM API calls needed — uses cached raw_triples.jsonl.

Usage:
    .venv-sapbert/bin/python -m scripts.recompute_consensus
    .venv-sapbert/bin/python -m scripts.recompute_consensus --disease-id "OMIM:310200"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

from agents.knowledge_extractor import KnowledgeExtractor
from agents.quality_controller import QualityController
from core.models import RawTriple
from core.schema_alignment import PrimeKGIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("recompute")


async def recompute_disease(disease_dir: Path, config_dir: Path, primekg_index: PrimeKGIndex) -> dict | None:
    """Re-compute consensus + QC for a single disease."""
    disease_id_str = disease_dir.name  # e.g., "OMIM_310200"

    raw_file = disease_dir / "raw_triples.jsonl"
    config_file = config_dir / f"{disease_id_str}.yaml"

    if not raw_file.exists() or not config_file.exists():
        return None

    import yaml
    with open(config_file) as f:
        profile = yaml.safe_load(f)

    disease_id = profile.get("disease_id", disease_id_str.replace("_", ":"))
    disease_name = profile.get("disease_name", disease_id_str)

    # Load raw triples
    with open(raw_file) as f:
        raw_dicts = [json.loads(l) for l in f if l.strip()]

    if not raw_dicts:
        return None

    # Group raw triples PER DOCUMENT, then per model within each document
    # This matches the original extraction pipeline which computes consensus per-doc
    per_doc_per_model: dict[str, dict[str, list[RawTriple]]] = defaultdict(lambda: defaultdict(list))
    for d in raw_dicts:
        t = RawTriple(**{k: v for k, v in d.items() if k in RawTriple.__dataclass_fields__})
        doc_id = t.source_id or "unknown"
        per_doc_per_model[doc_id][t.extraction_model].append(t)

    n_models = len({t.extraction_model for d in raw_dicts for t in [type('', (), d)]})
    model_names = set()
    for doc_models in per_doc_per_model.values():
        model_names.update(doc_models.keys())

    logger.info("%s (%s): %d raw triples from %d models across %d documents",
                disease_name, disease_id, len(raw_dicts), len(model_names), len(per_doc_per_model))

    # Re-compute consensus per document (matching original pipeline)
    extractor = KnowledgeExtractor.__new__(KnowledgeExtractor)
    extractor.consensus_threshold = 2
    consensus = []
    for doc_id, per_model in per_doc_per_model.items():
        doc_consensus = extractor._compute_consensus(per_model)
        consensus.extend(doc_consensus)

    # Save new consensus triples
    with open(disease_dir / "consensus_triples.jsonl", "w") as f:
        for t in consensus:
            f.write(json.dumps(t.to_dict(), default=str) + "\n")

    # Save updated extraction stats
    old_stats_file = disease_dir / "extraction_stats.json"
    old_stats = {}
    if old_stats_file.exists():
        with open(old_stats_file) as f:
            old_stats = json.load(f)

    old_stats["model_agreement_stats"] = {
        "total_raw": len(raw_dicts),
        "total_consensus": len(consensus),
        "consensus_rate": len(consensus) / max(1, len(raw_dicts)),
        "models_used": list(per_model.keys()),
        "consensus_method": "semantic_fuzzy_80",
    }
    with open(old_stats_file, "w") as f:
        json.dump(old_stats, f, indent=2, default=str)

    logger.info("  Semantic consensus: %d triples (%.1f%% rate)",
                len(consensus), 100 * len(consensus) / max(1, len(raw_dicts)))

    # Re-run QC
    quality = QualityController(config={}, primekg_index=primekg_index)
    consensus_dicts = [t.to_dict() for t in consensus]
    quality_result = await quality.run_with_retry({
        "profile": profile,
        "consensus_triples": consensus_dicts,
    })

    metrics = quality_result.metrics
    logger.info("  QC: confirmed=%d, novel=%d, temporal=%.0f%%, credibility=%.3f",
                metrics.get("confirmations_with_primekg", 0),
                metrics.get("novel_triples", 0),
                100 * metrics.get("temporal_coverage", 0),
                metrics.get("avg_credibility_score", 0))

    return {
        "disease_id": disease_id,
        "disease_name": disease_name,
        "raw_triples": len(raw_dicts),
        "consensus_triples": len(consensus),
        "consensus_rate": len(consensus) / max(1, len(raw_dicts)),
        **{k: metrics.get(k) for k in [
            "confirmations_with_primekg", "novel_triples",
            "temporal_coverage", "avg_credibility_score", "quality_grade",
        ]},
    }


async def main():
    parser = argparse.ArgumentParser(description="Re-compute consensus with semantic matching")
    parser.add_argument("--disease-id", type=str, help="Specific disease ID")
    args = parser.parse_args()

    extracted_dir = PROJECT_ROOT / "data" / "extracted"
    config_dir = PROJECT_ROOT / "config" / "diseases"

    # Load PrimeKG index once
    primekg_index = PrimeKGIndex()
    primekg_index.load()

    results = []
    for d in sorted(extracted_dir.iterdir()):
        if not d.is_dir() or not (d / "raw_triples.jsonl").exists():
            continue
        if args.disease_id:
            disease_id = d.name.replace("_", ":")
            if disease_id != args.disease_id:
                continue

        result = await recompute_disease(d, config_dir, primekg_index)
        if result:
            results.append(result)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Disease':<40} {'Raw':>5} {'Cons':>5} {'Rate':>6} {'PrKG':>5} {'Novel':>6} {'Temp':>5} {'Cred':>5}")
    print("=" * 90)
    for r in results:
        print(f"{r['disease_name']:<40} {r['raw_triples']:>5} {r['consensus_triples']:>5} "
              f"{r['consensus_rate']:>5.1%} {r.get('confirmations_with_primekg',0):>5} "
              f"{r.get('novel_triples',0):>6} {r.get('temporal_coverage',0):>4.0%} "
              f"{r.get('avg_credibility_score',0):>5.3f}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
