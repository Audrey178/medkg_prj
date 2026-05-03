#!/usr/bin/env python3
"""
Experiment: KG Link Prediction with Temporal Features
======================================================
Standard KG evaluation using PyKEEN. Tests whether temporal features
improve link prediction performance.

3 conditions (same graph structure, different edge features):
  1. STRUCTURE: plain (head, relation, tail) triples
  2. TEMPORAL_STAGE: relation augmented with onset age bin
     (e.g., disease_phenotype_positive -> disease_phenotype_positive__childhood)
  3. TEMPORAL_FULL: temporal stage + evidence recency bin

Metrics: Hits@1, Hits@3, Hits@10, MRR (standard KG evaluation)

Models: TransE, RotatE (standard baselines)

Output:
  data/benchmark/link_prediction_results.json
"""

from __future__ import annotations

import json
import logging
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("link_pred")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
RESULTS_DIR = BENCHMARK_DIR / "link_prediction"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def bin_onset_age(age_min, age_max):
    """Bin onset age into discrete stage."""
    if age_min is None:
        return None
    mid = (age_min + (age_max or age_min)) / 2
    if mid < 0.1:
        return "neonatal"
    elif mid < 2:
        return "infantile"
    elif mid < 12:
        return "childhood"
    elif mid < 18:
        return "juvenile"
    elif mid < 40:
        return "adult"
    elif mid < 65:
        return "middle_age"
    else:
        return "late_onset"


def bin_evidence_year(year):
    """Bin evidence publication year."""
    if year is None:
        return None
    if year >= 2020:
        return "recent"
    elif year >= 2010:
        return "modern"
    elif year >= 2000:
        return "older"
    else:
        return "historical"


def load_triples():
    """Load TA validated triples with metadata."""
    triples_struct = []       # (h, r, t) — structure only
    triples_temporal = []     # (h, r_with_stage, t) — temporal stage added
    triples_full = []         # (h, r_with_stage_evidence, t) — + evidence bin

    logger.info("Loading TA triples...")
    count = 0
    with_temporal = 0

    # Load PMID→year index for evidence binning
    pmid_year = {}
    import gzip
    logger.info("Building PMID-year index...")
    for d in EXTRACTED_DIR.iterdir():
        ec = d / "evidence_collection.json.gz"
        if not ec.exists():
            continue
        try:
            with gzip.open(ec, "rt") as f:
                data = json.load(f)
            for src in data.get("tier2_sources", []) + data.get("tier1_sources", []):
                sid = src.get("source_id", "")
                pd = src.get("publication_date")
                if sid and pd:
                    try:
                        pmid_year[sid] = int(str(pd)[:4])
                    except:
                        pass
        except:
            pass
    logger.info(f"PMID-year index: {len(pmid_year):,} entries")

    for d in sorted(EXTRACTED_DIR.iterdir()):
        vf = d / "validated_triples.jsonl"
        if not vf.exists():
            continue
        with open(vf) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    t = json.loads(line)
                except:
                    continue

                # Use entity names (normalized) as IDs
                h = t.get("source_name", "").lower().strip()
                r = t.get("relation", "").strip()
                tail = t.get("target_name", "").lower().strip()

                if not h or not r or not tail or h == tail:
                    continue

                count += 1

                # Basic structure triple
                triples_struct.append((h, r, tail))

                # Get temporal stage
                temporal = t.get("temporal", {}) or {}
                stage = bin_onset_age(temporal.get("onset_age_min"), temporal.get("onset_age_max"))

                if stage:
                    with_temporal += 1
                    r_temp = f"{r}__{stage}"
                    triples_temporal.append((h, r_temp, tail))

                    # Get evidence year
                    evidence = t.get("evidence", {}) or {}
                    src_ids = evidence.get("source_ids", [])
                    year = None
                    for sid in src_ids:
                        if sid in pmid_year:
                            year = pmid_year[sid]
                            break
                    ev_bin = bin_evidence_year(year)

                    if ev_bin:
                        r_full = f"{r}__{stage}__{ev_bin}"
                        triples_full.append((h, r_full, tail))
                    else:
                        triples_full.append((h, r_temp, tail))
                else:
                    # Keep original relation for triples without temporal info
                    triples_temporal.append((h, r, tail))
                    triples_full.append((h, r, tail))

    logger.info(f"Total triples: {count:,}, with temporal info: {with_temporal:,} ({100*with_temporal/count:.1f}%)")
    logger.info(f"Structure triples: {len(triples_struct):,}")
    logger.info(f"Temporal triples: {len(triples_temporal):,}")
    logger.info(f"Full triples: {len(triples_full):,}")

    return triples_struct, triples_temporal, triples_full


def deduplicate(triples):
    """Remove duplicate triples."""
    unique = list(set(triples))
    return unique


def build_triples_factory(triples, entity_to_id=None, relation_to_id=None):
    """Create PyKEEN TriplesFactory."""
    from pykeen.triples import TriplesFactory
    import numpy as np

    # Build ID mappings if not provided
    if entity_to_id is None:
        entities = sorted(set([h for h, _, _ in triples] + [t for _, _, t in triples]))
        entity_to_id = {e: i for i, e in enumerate(entities)}
    if relation_to_id is None:
        relations = sorted(set(r for _, r, _ in triples))
        relation_to_id = {r: i for i, r in enumerate(relations)}

    # Convert to numpy array
    arr = np.array([
        [entity_to_id[h], relation_to_id[r], entity_to_id[t]]
        for h, r, t in triples
        if h in entity_to_id and t in entity_to_id and r in relation_to_id
    ])

    factory = TriplesFactory(
        mapped_triples=torch.LongTensor(arr),
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )
    return factory, entity_to_id, relation_to_id


def split_triples(triples, train_frac=0.8, val_frac=0.1, seed=42):
    """Random split into train/val/test."""
    random.seed(seed)
    shuffled = triples.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return shuffled[:n_train], shuffled[n_train:n_train+n_val], shuffled[n_train+n_val:]


def train_and_evaluate(train_triples, val_triples, test_triples,
                       model_name="TransE", embedding_dim=50, num_epochs=30):
    """Train a KG embedding model and evaluate on test set."""
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory
    import numpy as np

    logger.info(f"Training {model_name} (dim={embedding_dim}, epochs={num_epochs})")
    logger.info(f"  Train: {len(train_triples):,}, Val: {len(val_triples):,}, Test: {len(test_triples):,}")

    # Combine to get full entity/relation vocabulary
    all_triples = train_triples + val_triples + test_triples
    entities = sorted(set([h for h, _, _ in all_triples] + [t for _, _, t in all_triples]))
    relations = sorted(set(r for _, r, _ in all_triples))
    entity_to_id = {e: i for i, e in enumerate(entities)}
    relation_to_id = {r: i for i, r in enumerate(relations)}

    def to_array(tps):
        arr = np.array([
            [entity_to_id[h], relation_to_id[r], entity_to_id[t]]
            for h, r, t in tps
        ])
        return torch.LongTensor(arr)

    train_arr = to_array(train_triples)
    val_arr = to_array(val_triples)
    test_arr = to_array(test_triples)

    # Check for MPS (Apple Silicon GPU)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"  Device: {device}")

    train_factory = TriplesFactory(
        mapped_triples=train_arr,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )
    val_factory = TriplesFactory(
        mapped_triples=val_arr,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )
    test_factory = TriplesFactory(
        mapped_triples=test_arr,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )

    result = pipeline(
        training=train_factory,
        validation=val_factory,
        testing=test_factory,
        model=model_name,
        model_kwargs={"embedding_dim": embedding_dim},
        training_kwargs={"num_epochs": num_epochs, "batch_size": 512},
        optimizer_kwargs={"lr": 0.01},
        random_seed=42,
        device=device,
    )

    metrics = result.metric_results.to_dict()
    # Extract key metrics
    # PyKEEN metrics are hierarchical: side > head/tail/both, type > realistic/optimistic/pessimistic, metric
    hits_at_1 = result.metric_results.get_metric("hits_at_1")
    hits_at_3 = result.metric_results.get_metric("hits_at_3")
    hits_at_10 = result.metric_results.get_metric("hits_at_10")
    mrr = result.metric_results.get_metric("mean_reciprocal_rank")

    return {
        "model": model_name,
        "n_entities": len(entities),
        "n_relations": len(relations),
        "n_train": len(train_triples),
        "n_val": len(val_triples),
        "n_test": len(test_triples),
        "hits_at_1": hits_at_1,
        "hits_at_3": hits_at_3,
        "hits_at_10": hits_at_10,
        "mrr": mrr,
    }


def main():
    logger.info("=" * 75)
    logger.info("KG Link Prediction with Temporal Features")
    logger.info("=" * 75)

    # Load triples
    triples_struct, triples_temporal, triples_full = load_triples()

    # Deduplicate
    triples_struct = deduplicate(triples_struct)
    triples_temporal = deduplicate(triples_temporal)
    triples_full = deduplicate(triples_full)

    logger.info(f"\nAfter deduplication:")
    logger.info(f"  Structure: {len(triples_struct):,}")
    logger.info(f"  Temporal:  {len(triples_temporal):,}")
    logger.info(f"  Full:      {len(triples_full):,}")

    # Relation count per condition
    for name, tps in [("Structure", triples_struct), ("Temporal", triples_temporal), ("Full", triples_full)]:
        rels = set(r for _, r, _ in tps)
        logger.info(f"  {name}: {len(rels)} unique relations")

    # Run experiments
    results = {}
    for name, triples in [
        ("structure", triples_struct),
        ("temporal_stage", triples_temporal),
        ("temporal_full", triples_full),
    ]:
        logger.info(f"\n{'='*70}")
        logger.info(f"Condition: {name}")
        logger.info(f"{'='*70}")

        train, val, test = split_triples(triples)
        logger.info(f"Split: {len(train):,} / {len(val):,} / {len(test):,}")

        # Run TransE (faster baseline)
        try:
            result = train_and_evaluate(train, val, test, model_name="TransE",
                                       embedding_dim=50, num_epochs=20)
            results[name] = result
            logger.info(f"  TransE: Hits@1={result['hits_at_1']:.3f}, Hits@10={result['hits_at_10']:.3f}, MRR={result['mrr']:.3f}")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            results[name] = {"error": str(e)}

    # Print summary
    print(f"\n{'=' * 75}")
    print("LINK PREDICTION RESULTS (TransE)")
    print(f"{'=' * 75}")
    print(f"{'Condition':<20} {'Hits@1':>10} {'Hits@3':>10} {'Hits@10':>10} {'MRR':>10}")
    print("-" * 70)
    for name, res in results.items():
        if "error" in res:
            print(f"{name:<20} ERROR: {res['error'][:50]}")
            continue
        print(f"{name:<20} {res['hits_at_1']:>10.4f} {res['hits_at_3']:>10.4f} "
              f"{res['hits_at_10']:>10.4f} {res['mrr']:>10.4f}")

    # Save
    out_file = RESULTS_DIR / "link_prediction_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
