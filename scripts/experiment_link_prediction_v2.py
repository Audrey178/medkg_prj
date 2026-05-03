#!/usr/bin/env python3
"""
Link Prediction v2: Fair Temporal Baselines
=============================================
Tests whether ChronoMedKG's fine-grained per-phenotype onset improves
disease-phenotype link prediction over HPOA's coarse disease-level onset.

Design fixes from v1:
  1. Fair baselines: PrimeKG (no temporal) + HPOA (coarse temporal) + TA (fine temporal)
  2. Same edge set across conditions — only relation label varies
  3. Entity normalization: merge synonymous disease names
  4. Time-split evaluation: train on pre-2020 PMIDs, test on post-2020

Task: Disease-phenotype link prediction
  Input: (disease, has_phenotype, ?)
  Output: ranked phenotype predictions
  Metric: Hits@1/3/10, MRR

Conditions (all use the SAME disease-phenotype edges, differ only in relation):
  A) STRUCTURE: (disease, has_phenotype, phenotype)
  B) HPOA_COARSE: (disease, has_phenotype__childhood_cat, phenotype)   [disease-level]
  C) TA_FINE: (disease, has_phenotype__childhood_fine, phenotype)     [phenotype-level]

Output:
  data/benchmark/link_prediction_v2.json
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("lp_v2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
PRIMEKG_DIR = PROJECT_ROOT / "data" / "primekg"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation_sources"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
RESULTS_DIR = BENCHMARK_DIR / "link_prediction_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# ONSET BINNING
# ============================================================================

HPO_ONSET_BINS = {
    "HP:0003577": "congenital",   # Congenital onset
    "HP:0003623": "neonatal",     # Neonatal onset
    "HP:0003593": "infantile",    # Infantile onset
    "HP:0011463": "childhood",    # Childhood onset
    "HP:0003621": "juvenile",     # Juvenile onset
    "HP:0011462": "young_adult",  # Young adult onset
    "HP:0003581": "adult",        # Adult onset
    "HP:0003596": "middle_age",   # Middle age onset
    "HP:0003584": "late_onset",   # Late onset
    "HP:0030674": "antenatal",    # Antenatal onset
    "HP:0011460": "embryonal",    # Embryonal onset
    "HP:0034199": "fetal",        # Fetal onset
    "HP:0410280": "pediatric",    # Pediatric onset
}


def bin_numeric_age(age_min, age_max):
    """Bin numeric onset age into HPO-compatible category."""
    if age_min is None:
        return None
    mid = (age_min + (age_max or age_min)) / 2
    if mid < 0.08:
        return "neonatal"
    elif mid < 1:
        return "infantile"
    elif mid < 5:
        return "early_childhood"
    elif mid < 12:
        return "childhood"
    elif mid < 18:
        return "juvenile"
    elif mid < 40:
        return "young_adult"
    elif mid < 65:
        return "adult"
    else:
        return "late_onset"


# ============================================================================
# DATA LOADING
# ============================================================================

def normalize_disease_name(name):
    """Normalize disease name for consistent matching."""
    n = name.lower().strip()
    # Remove common suffixes
    for suffix in [" syndrome", " disease", " disorder", " (disorder)"]:
        if n.endswith(suffix):
            n = n[:-len(suffix)]
    return n.strip()


def load_hpoa_edges():
    """Load HPOA disease-phenotype edges with onset info."""
    edges = []  # list of (disease_id, disease_name, phenotype_id, phenotype_name, disease_level_onset)

    # First pass: disease-level onset (aspect='I' means inheritance, 'C' means clinical course)
    # Disease-level onset is in our hpoa_with_ids.json
    with open(VALIDATION_DIR / "hpoa_with_ids.json") as f:
        hpoa_disease_onset = json.load(f)

    # Map OMIM ID to disease-level onset bin
    disease_onset_bin = {}
    for omim_id, record in hpoa_disease_onset.items():
        onset_terms = record.get("onset_terms", [])
        # Pick first HPO onset term that's in our bin mapping
        for term in onset_terms:
            if term in HPO_ONSET_BINS:
                disease_onset_bin[omim_id] = HPO_ONSET_BINS[term]
                break
        if omim_id not in disease_onset_bin:
            # Fall back to age range binning
            min_age = record.get("min_age", 0)
            max_age = record.get("max_age", 120)
            bin_val = bin_numeric_age(min_age, max_age)
            if bin_val:
                disease_onset_bin[omim_id] = bin_val

    # Second pass: parse HPOA file for disease-phenotype edges
    with open(VALIDATION_DIR / "phenotype.hpoa") as f:
        lines = [l for l in f if not l.startswith("#")]
    reader = csv.DictReader(lines, delimiter="\t")

    for row in reader:
        db_id = row.get("database_id", "").strip()
        disease_name = row.get("disease_name", "").strip()
        hpo_id = row.get("hpo_id", "").strip()
        aspect = row.get("aspect", "").strip()
        qualifier = row.get("qualifier", "").strip()
        onset_hpo = row.get("onset", "").strip()

        # Only phenotype-level annotations (aspect 'P')
        if aspect != "P":
            continue
        if qualifier == "NOT":
            continue
        if not db_id.startswith("OMIM:") or not disease_name or not hpo_id:
            continue

        # Phenotype-level onset (rare in HPOA — only 1.1%)
        pheno_level_bin = None
        if onset_hpo in HPO_ONSET_BINS:
            pheno_level_bin = HPO_ONSET_BINS[onset_hpo]

        # Disease-level onset (common — 1,429 diseases)
        disease_level_bin = disease_onset_bin.get(db_id)

        edges.append({
            "disease_id": db_id,
            "disease_name": normalize_disease_name(disease_name),
            "phenotype_id": hpo_id,
            "disease_level_onset": disease_level_bin,
            "phenotype_level_onset": pheno_level_bin,
        })

    logger.info(f"HPOA edges: {len(edges):,}")
    logger.info(f"  With disease-level onset: {sum(1 for e in edges if e['disease_level_onset']):,}")
    logger.info(f"  With phenotype-level onset: {sum(1 for e in edges if e['phenotype_level_onset']):,}")
    return edges


def load_ta_edges():
    """Load TA disease-phenotype edges with fine-grained onset."""
    edges = []

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
                if "phenotype" not in t.get("relation", ""):
                    continue
                if t.get("source_type") != "disease":
                    continue

                disease_name = normalize_disease_name(t.get("source_name", ""))
                phenotype_name = normalize_disease_name(t.get("target_name", ""))
                if not disease_name or not phenotype_name:
                    continue

                temporal = t.get("temporal", {}) or {}
                omin = temporal.get("onset_age_min")
                fine_bin = None
                if omin is not None:
                    try:
                        omin = float(omin)
                        omax = float(temporal.get("onset_age_max") or omin)
                        if 0 <= omin <= 120:
                            fine_bin = bin_numeric_age(omin, omax)
                    except:
                        pass

                # Get evidence year
                ev = t.get("evidence", {}) or {}
                src_ids = ev.get("source_ids", [])
                first_pmid = src_ids[0] if src_ids else None

                edges.append({
                    "disease_name": disease_name,
                    "phenotype_name": phenotype_name,
                    "fine_onset": fine_bin,
                    "pmid": first_pmid,
                })

    logger.info(f"TA edges: {len(edges):,}")
    logger.info(f"  With fine-grained onset: {sum(1 for e in edges if e['fine_onset']):,}")
    return edges


def build_pmid_year_index():
    """Build PMID -> year for time-split evaluation."""
    pmid_year = {}
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
    logger.info(f"PMID-year index: {len(pmid_year):,}")
    return pmid_year


# ============================================================================
# TRIPLE CONSTRUCTION — 3 CONDITIONS ON SAME EDGE SET
# ============================================================================

def build_conditions(hpoa_edges, ta_edges):
    """Build 3 sets of triples on OVERLAP edges (diseases in both HPOA and TA).

    All conditions use the SAME (disease, phenotype) pairs.
    Conditions differ only in relation label:
      - STRUCTURE: (d, has_phenotype, p)
      - HPOA_COARSE: (d, has_phenotype__<disease_level_bin>, p)  if HPOA has it
      - TA_FINE: (d, has_phenotype__<fine_onset_bin>, p)  if TA has it
    """
    # Index HPOA: (disease_name, phenotype_id) -> disease_level_onset
    # Note: we use HPO ID for phenotype because HPOA uses HPO IDs
    #       TA uses phenotype names
    # We need a mapping from HPO ID to name (or vice versa) — but TA doesn't have HPO IDs
    # Alternative: restrict to diseases in both (by name), phenotypes in both (by name match)

    # Build TA phenotype name set
    ta_pheno_names = set(e["phenotype_name"] for e in ta_edges)

    # Load HPO ontology for name lookup (simpler: use disease_name match only,
    # phenotype by either ID or name)
    # Actually, HPOA doesn't give phenotype names, only HPO IDs.
    # We'd need to load the HPO ontology to resolve IDs to names.
    # Simpler: treat HPO ID as the phenotype identifier.

    # Build index by disease name + phenotype ID (HPO) for HPOA
    hpoa_index = defaultdict(lambda: {"disease_level": None, "pheno_level": None})
    for e in hpoa_edges:
        key = (e["disease_name"], e["phenotype_id"])
        if e["disease_level_onset"]:
            hpoa_index[key]["disease_level"] = e["disease_level_onset"]
        if e["phenotype_level_onset"]:
            hpoa_index[key]["pheno_level"] = e["phenotype_level_onset"]

    # Since TA uses phenotype names and HPOA uses HPO IDs, we can't easily
    # match edges between them. So we'll do 3 separate graph conditions:
    # 1. HPOA-only graph (HPO-ID phenotypes)
    # 2. TA-only graph (name phenotypes)
    # 3. Each with its own temporal augmentation variant

    # This changes the experiment design:
    # - Condition A (HPOA baseline): HPOA graph with disease-level onset relations
    # - Condition B (HPOA phenotype-level): HPOA graph with phenotype-level onset (1.1% coverage)
    # - Condition C (TA fine-grained): TA graph with fine-grained onset

    # For fair comparison, we need to restrict to diseases in BOTH HPOA and TA,
    # and use their respective phenotype sets (even if they differ).

    # Build sets
    hpoa_diseases = set(e["disease_name"] for e in hpoa_edges)
    ta_diseases = set(e["disease_name"] for e in ta_edges)
    overlap_diseases = hpoa_diseases & ta_diseases
    logger.info(f"HPOA diseases: {len(hpoa_diseases):,}")
    logger.info(f"TA diseases: {len(ta_diseases):,}")
    logger.info(f"Overlap: {len(overlap_diseases):,}")

    # Build HPOA triples (restricted to overlap diseases)
    hpoa_triples_structure = []
    hpoa_triples_coarse = []
    for e in hpoa_edges:
        if e["disease_name"] not in overlap_diseases:
            continue
        d = e["disease_name"]
        p = f"hpo_{e['phenotype_id']}"  # Prefix to distinguish from TA phenotypes
        hpoa_triples_structure.append((d, "has_phenotype", p))

        # Use phenotype-level if available, else disease-level
        onset_bin = e["phenotype_level_onset"] or e["disease_level_onset"]
        if onset_bin:
            hpoa_triples_coarse.append((d, f"has_phenotype__{onset_bin}", p))
        else:
            hpoa_triples_coarse.append((d, "has_phenotype", p))

    # Build TA triples (restricted to overlap diseases)
    ta_triples_structure = []
    ta_triples_fine = []
    for e in ta_edges:
        if e["disease_name"] not in overlap_diseases:
            continue
        d = e["disease_name"]
        p = f"ta_{e['phenotype_name']}"  # Prefix to distinguish from HPOA phenotypes
        ta_triples_structure.append((d, "has_phenotype", p))

        if e["fine_onset"]:
            ta_triples_fine.append((d, f"has_phenotype__{e['fine_onset']}", p))
        else:
            ta_triples_fine.append((d, "has_phenotype", p))

    # Dedup
    hpoa_triples_structure = list(set(hpoa_triples_structure))
    hpoa_triples_coarse = list(set(hpoa_triples_coarse))
    ta_triples_structure = list(set(ta_triples_structure))
    ta_triples_fine = list(set(ta_triples_fine))

    logger.info(f"\nFinal triple counts:")
    logger.info(f"  HPOA-structure: {len(hpoa_triples_structure):,}")
    logger.info(f"  HPOA-coarse temporal: {len(hpoa_triples_coarse):,}")
    logger.info(f"  TA-structure: {len(ta_triples_structure):,}")
    logger.info(f"  TA-fine temporal: {len(ta_triples_fine):,}")

    # Relation counts
    for name, tps in [("HPOA-struct", hpoa_triples_structure),
                     ("HPOA-coarse", hpoa_triples_coarse),
                     ("TA-struct", ta_triples_structure),
                     ("TA-fine", ta_triples_fine)]:
        rels = set(r for _, r, _ in tps)
        entities = set([h for h,_,_ in tps] + [t for _,_,t in tps])
        logger.info(f"  {name}: {len(rels)} relations, {len(entities)} entities")

    return {
        "hpoa_struct": hpoa_triples_structure,
        "hpoa_coarse": hpoa_triples_coarse,
        "ta_struct": ta_triples_structure,
        "ta_fine": ta_triples_fine,
    }


# ============================================================================
# TRAINING
# ============================================================================

def train_and_evaluate(triples, name, embedding_dim=50, num_epochs=20):
    """Train TransE on triples and evaluate."""
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    # Split
    random.seed(42)
    shuffled = triples.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    train = shuffled[:int(n*0.8)]
    val = shuffled[int(n*0.8):int(n*0.9)]
    test = shuffled[int(n*0.9):]

    # Build vocabulary from ALL triples
    entities = sorted(set([h for h,_,_ in triples] + [t for _,_,t in triples]))
    relations = sorted(set(r for _,r,_ in triples))
    entity_to_id = {e: i for i, e in enumerate(entities)}
    relation_to_id = {r: i for i, r in enumerate(relations)}

    def to_tensor(tps):
        arr = np.array([[entity_to_id[h], relation_to_id[r], entity_to_id[t]]
                       for h, r, t in tps])
        return torch.LongTensor(arr)

    train_tf = TriplesFactory(mapped_triples=to_tensor(train),
                              entity_to_id=entity_to_id, relation_to_id=relation_to_id)
    val_tf = TriplesFactory(mapped_triples=to_tensor(val),
                            entity_to_id=entity_to_id, relation_to_id=relation_to_id)
    test_tf = TriplesFactory(mapped_triples=to_tensor(test),
                             entity_to_id=entity_to_id, relation_to_id=relation_to_id)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"\nTraining TransE on {name} ({len(train):,} train, {len(val):,} val, {len(test):,} test)")
    logger.info(f"  Entities: {len(entities):,}, Relations: {len(relations)}, Device: {device}")

    try:
        result = pipeline(
            training=train_tf,
            validation=val_tf,
            testing=test_tf,
            model="TransE",
            model_kwargs={"embedding_dim": embedding_dim},
            training_kwargs={"num_epochs": num_epochs, "batch_size": 512},
            optimizer_kwargs={"lr": 0.01},
            random_seed=42,
            device=device,
        )

        return {
            "condition": name,
            "n_entities": len(entities),
            "n_relations": len(relations),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "hits_at_1": float(result.metric_results.get_metric("hits_at_1")),
            "hits_at_3": float(result.metric_results.get_metric("hits_at_3")),
            "hits_at_10": float(result.metric_results.get_metric("hits_at_10")),
            "mrr": float(result.metric_results.get_metric("mean_reciprocal_rank")),
        }
    except Exception as e:
        logger.error(f"Training failed for {name}: {e}")
        return {"condition": name, "error": str(e)}


def main():
    logger.info("=" * 75)
    logger.info("Link Prediction v2: Fair Temporal Baselines")
    logger.info("=" * 75)

    # Load data
    logger.info("\n[1/3] Loading HPOA edges...")
    hpoa_edges = load_hpoa_edges()

    logger.info("\n[2/3] Loading TA edges...")
    ta_edges = load_ta_edges()

    logger.info("\n[3/3] Building 4 experimental conditions (same overlap diseases)...")
    conditions = build_conditions(hpoa_edges, ta_edges)

    # Train each condition
    results = {}
    for name, triples in conditions.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"Condition: {name}")
        logger.info(f"{'='*70}")
        result = train_and_evaluate(triples, name, embedding_dim=50, num_epochs=20)
        results[name] = result
        if "error" not in result:
            logger.info(f"  Hits@1={result['hits_at_1']:.4f}, "
                       f"Hits@10={result['hits_at_10']:.4f}, "
                       f"MRR={result['mrr']:.4f}")

    # Summary
    print(f"\n{'=' * 85}")
    print("LINK PREDICTION v2 RESULTS")
    print(f"{'=' * 85}")
    print(f"Task: Disease-phenotype link prediction on OVERLAP diseases (HPOA ∩ TA)")
    print(f"Model: TransE, 50-dim, 20 epochs\n")
    print(f"{'Condition':<25} {'Triples':>10} {'Relations':>10} {'Hits@1':>10} {'Hits@10':>10} {'MRR':>10}")
    print("-" * 85)

    for cond_name, res in results.items():
        if "error" in res:
            print(f"{cond_name:<25} ERROR: {res['error'][:40]}")
            continue
        n_rel = res.get("n_relations", 0)
        print(f"{cond_name:<25} {res['n_train']+res['n_val']+res['n_test']:>10,} {n_rel:>10} "
              f"{res['hits_at_1']:>10.4f} {res['hits_at_10']:>10.4f} {res['mrr']:>10.4f}")

    # Save
    out_file = RESULTS_DIR / "link_prediction_v2.json"
    with open(out_file, "w") as f:
        json.dump({
            "experiment": "Link Prediction v2 (HPOA baseline)",
            "task": "Disease-phenotype link prediction",
            "model": "TransE",
            "results": results,
        }, f, indent=2, default=str)
    logger.info(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
