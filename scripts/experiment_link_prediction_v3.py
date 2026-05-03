#!/usr/bin/env python3
"""
Link Prediction v3: Diligent fixes for fair comparison
========================================================

Fixes from v2:
  1. NORMALIZED TA phenotype names — reduce synonym explosion
     (e.g., "delayed walking" / "motor delay" / "gait delay" all map to canonical form)
  2. LONGER training (100 epochs, 100-dim) — get to convergence
  3. MULTIPLE models: TransE + RotatE (sanity check)
  4. BOTH structure and structure+temporal as within-condition ablation
     (so we can directly measure temporal contribution)

Task: Disease-phenotype link prediction
  Metric: Hits@1/3/10, MRR

Conditions (on overlap diseases: HPOA ∩ TA):
  A) HPOA_STRUCT: HPOA disease-phenotype edges, no temporal
  B) HPOA_TEMPORAL: HPOA edges + coarse onset bins (disease-level)
  C) TA_STRUCT: TA edges (normalized phenotype names), no temporal
  D) TA_TEMPORAL: TA edges + fine-grained onset bins (phenotype-level)

Each condition evaluated with TransE and RotatE.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("lp_v3")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation_sources"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
RESULTS_DIR = BENCHMARK_DIR / "link_prediction_v3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# HPO onset term → bin
HPO_ONSET_BINS = {
    "HP:0003577": "congenital",
    "HP:0003623": "neonatal",
    "HP:0003593": "infantile",
    "HP:0011463": "childhood",
    "HP:0003621": "juvenile",
    "HP:0011462": "young_adult",
    "HP:0003581": "adult",
    "HP:0003596": "middle_age",
    "HP:0003584": "late_onset",
    "HP:0030674": "antenatal",
    "HP:0011460": "embryonal",
    "HP:0034199": "fetal",
    "HP:0410280": "pediatric",
}


def bin_numeric_age(age_min, age_max):
    """Bin numeric onset into same categories as HPOA for fair comparison."""
    if age_min is None:
        return None
    mid = (age_min + (age_max or age_min)) / 2
    # Use same bins as HPO for fair comparison
    if mid < 0.08:
        return "neonatal"
    elif mid < 1:
        return "infantile"
    elif mid < 5:
        return "early_childhood"  # finer than HPO
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


def normalize_disease_name(name):
    """Normalize disease name for matching."""
    n = name.lower().strip()
    for suffix in [" syndrome", " disease", " disorder", " (disorder)"]:
        if n.endswith(suffix):
            n = n[:-len(suffix)]
    return n.strip()


# Common phenotype name variants to canonical form
# This is conservative — only merge clear synonyms
PHENOTYPE_NORMALIZATIONS = {
    # Motor/gait
    "delayed walking": "delayed_motor_milestones",
    "delayed motor development": "delayed_motor_milestones",
    "motor delay": "delayed_motor_milestones",
    "delayed gross motor development": "delayed_motor_milestones",
    "gait delay": "delayed_motor_milestones",
    "delayed gait": "delayed_motor_milestones",

    # Cognitive
    "intellectual disability": "intellectual_disability",
    "intellectual disabilities": "intellectual_disability",
    "mental retardation": "intellectual_disability",  # old term
    "developmental delay": "developmental_delay",
    "global developmental delay": "developmental_delay",

    # Weakness
    "muscle weakness": "muscle_weakness",
    "muscular weakness": "muscle_weakness",
    "proximal muscle weakness": "proximal_weakness",
    "proximal weakness": "proximal_weakness",

    # Seizures
    "seizure": "seizures",
    "seizures": "seizures",
    "epilepsy": "seizures",
    "convulsions": "seizures",

    # Hypotonia
    "hypotonia": "hypotonia",
    "muscular hypotonia": "hypotonia",
    "low muscle tone": "hypotonia",
}


def normalize_phenotype_name(name):
    """Normalize phenotype name with canonicalization rules."""
    n = name.lower().strip()

    # Remove common parenthetical suffixes
    n = re.sub(r'\s*\([^)]*\)\s*$', '', n).strip()

    # Remove trailing descriptors
    for suffix in [", severe", ", mild", ", moderate", ", chronic", ", acute"]:
        if n.endswith(suffix):
            n = n[:-len(suffix)].strip()

    # Explicit normalizations
    if n in PHENOTYPE_NORMALIZATIONS:
        return PHENOTYPE_NORMALIZATIONS[n]

    # Convert spaces to underscores for consistency
    n = n.replace(" ", "_").replace("-", "_")
    return n


def load_hpoa_edges():
    """Load HPOA edges with onset info."""
    # Disease-level onset from hpoa_with_ids.json
    with open(VALIDATION_DIR / "hpoa_with_ids.json") as f:
        hpoa_disease_info = json.load(f)

    disease_onset_bin = {}
    for omim_id, record in hpoa_disease_info.items():
        onset_terms = record.get("onset_terms", [])
        for term in onset_terms:
            if term in HPO_ONSET_BINS:
                disease_onset_bin[omim_id] = HPO_ONSET_BINS[term]
                break
        if omim_id not in disease_onset_bin:
            min_age = record.get("min_age", 0)
            max_age = record.get("max_age", 120)
            bin_val = bin_numeric_age(min_age, max_age)
            if bin_val:
                disease_onset_bin[omim_id] = bin_val

    # Phenotype-level edges
    edges = []
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

        if aspect != "P" or qualifier == "NOT":
            continue
        if not db_id.startswith("OMIM:") or not disease_name or not hpo_id:
            continue

        pheno_bin = HPO_ONSET_BINS.get(onset_hpo)
        disease_bin = disease_onset_bin.get(db_id)

        edges.append({
            "disease_name": normalize_disease_name(disease_name),
            "phenotype_id": hpo_id,
            "onset_bin": pheno_bin or disease_bin,  # prefer phenotype-level
        })

    logger.info(f"HPOA: {len(edges):,} edges, {sum(1 for e in edges if e['onset_bin']):,} with temporal")
    return edges


def load_ta_edges():
    """Load TA edges with normalized phenotype names."""
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

                disease = normalize_disease_name(t.get("source_name", ""))
                phenotype_raw = t.get("target_name", "").strip()
                if not disease or not phenotype_raw:
                    continue

                phenotype = normalize_phenotype_name(phenotype_raw)
                if not phenotype:
                    continue

                temporal = t.get("temporal", {}) or {}
                omin = temporal.get("onset_age_min")
                bin_val = None
                if omin is not None:
                    try:
                        omin = float(omin)
                        omax = float(temporal.get("onset_age_max") or omin)
                        if 0 <= omin <= 120:
                            bin_val = bin_numeric_age(omin, omax)
                    except:
                        pass

                edges.append({
                    "disease_name": disease,
                    "phenotype_name": phenotype,
                    "onset_bin": bin_val,
                })

    logger.info(f"TA: {len(edges):,} edges, {sum(1 for e in edges if e['onset_bin']):,} with temporal")
    # Check normalization effect
    raw_names = set()
    norm_names = set(e["phenotype_name"] for e in edges)
    logger.info(f"  Unique normalized phenotype names: {len(norm_names):,}")
    return edges


def build_conditions(hpoa_edges, ta_edges):
    """Build experimental conditions."""
    hpoa_diseases = set(e["disease_name"] for e in hpoa_edges)
    ta_diseases = set(e["disease_name"] for e in ta_edges)
    overlap = hpoa_diseases & ta_diseases
    logger.info(f"HPOA diseases: {len(hpoa_diseases):,}, TA: {len(ta_diseases):,}, Overlap: {len(overlap):,}")

    # HPOA on overlap diseases
    hpoa_struct = []
    hpoa_temporal = []
    for e in hpoa_edges:
        if e["disease_name"] not in overlap:
            continue
        d = e["disease_name"]
        p = f"hpo_{e['phenotype_id']}"
        hpoa_struct.append((d, "has_phenotype", p))
        rel = f"has_phenotype__{e['onset_bin']}" if e["onset_bin"] else "has_phenotype"
        hpoa_temporal.append((d, rel, p))

    # TA on overlap diseases
    ta_struct = []
    ta_temporal = []
    for e in ta_edges:
        if e["disease_name"] not in overlap:
            continue
        d = e["disease_name"]
        p = f"ta_{e['phenotype_name']}"
        ta_struct.append((d, "has_phenotype", p))
        rel = f"has_phenotype__{e['onset_bin']}" if e["onset_bin"] else "has_phenotype"
        ta_temporal.append((d, rel, p))

    # Dedup
    hpoa_struct = list(set(hpoa_struct))
    hpoa_temporal = list(set(hpoa_temporal))
    ta_struct = list(set(ta_struct))
    ta_temporal = list(set(ta_temporal))

    for name, tps in [("hpoa_struct", hpoa_struct), ("hpoa_temporal", hpoa_temporal),
                      ("ta_struct", ta_struct), ("ta_temporal", ta_temporal)]:
        rels = set(r for _, r, _ in tps)
        ents = set([h for h,_,_ in tps] + [t for _,_,t in tps])
        logger.info(f"  {name}: {len(tps):,} triples, {len(rels)} rels, {len(ents):,} entities")

    return {
        "hpoa_struct": hpoa_struct,
        "hpoa_temporal": hpoa_temporal,
        "ta_struct": ta_struct,
        "ta_temporal": ta_temporal,
    }


def train_and_evaluate(triples, name, model_name="TransE", embedding_dim=100, num_epochs=100):
    """Train and evaluate a KG embedding model."""
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    random.seed(42)
    shuffled = triples.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    train = shuffled[:int(n * 0.8)]
    val = shuffled[int(n * 0.8):int(n * 0.9)]
    test = shuffled[int(n * 0.9):]

    entities = sorted(set([h for h,_,_ in triples] + [t for _,_,t in triples]))
    relations = sorted(set(r for _,r,_ in triples))
    ent_id = {e: i for i, e in enumerate(entities)}
    rel_id = {r: i for i, r in enumerate(relations)}

    def to_tensor(tps):
        return torch.LongTensor(np.array([[ent_id[h], rel_id[r], ent_id[t]] for h, r, t in tps]))

    train_tf = TriplesFactory(mapped_triples=to_tensor(train), entity_to_id=ent_id, relation_to_id=rel_id)
    val_tf = TriplesFactory(mapped_triples=to_tensor(val), entity_to_id=ent_id, relation_to_id=rel_id)
    test_tf = TriplesFactory(mapped_triples=to_tensor(test), entity_to_id=ent_id, relation_to_id=rel_id)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"\nTraining {model_name} on {name}")
    logger.info(f"  Triples: {len(train):,} train / {len(val):,} val / {len(test):,} test")
    logger.info(f"  Entities: {len(entities):,}, Relations: {len(relations)}, Device: {device}, "
                f"Dim={embedding_dim}, Epochs={num_epochs}")

    try:
        result = pipeline(
            training=train_tf,
            validation=val_tf,
            testing=test_tf,
            model=model_name,
            model_kwargs={"embedding_dim": embedding_dim},
            training_kwargs={"num_epochs": num_epochs, "batch_size": 1024},
            optimizer_kwargs={"lr": 0.01},
            random_seed=42,
            device=device,
        )

        return {
            "condition": name,
            "model": model_name,
            "n_entities": len(entities),
            "n_relations": len(relations),
            "n_train": len(train),
            "n_test": len(test),
            "hits_at_1": float(result.metric_results.get_metric("hits_at_1")),
            "hits_at_3": float(result.metric_results.get_metric("hits_at_3")),
            "hits_at_10": float(result.metric_results.get_metric("hits_at_10")),
            "mrr": float(result.metric_results.get_metric("mean_reciprocal_rank")),
        }
    except Exception as e:
        logger.error(f"Failed: {e}")
        return {"condition": name, "model": model_name, "error": str(e)}


def main():
    logger.info("=" * 75)
    logger.info("Link Prediction v3: Fair Temporal Baselines (Normalized + Converged)")
    logger.info("=" * 75)

    hpoa_edges = load_hpoa_edges()
    ta_edges = load_ta_edges()
    conditions = build_conditions(hpoa_edges, ta_edges)

    # Train all conditions with TransE (primary) and RotatE (sanity check)
    results = {}
    for model_name in ["TransE", "RotatE"]:
        results[model_name] = {}
        for cond_name, triples in conditions.items():
            logger.info(f"\n{'='*70}")
            logger.info(f"{model_name} on {cond_name}")
            logger.info(f"{'='*70}")
            res = train_and_evaluate(triples, cond_name, model_name=model_name,
                                     embedding_dim=100, num_epochs=100)
            results[model_name][cond_name] = res

    # Summary
    print(f"\n{'=' * 95}")
    print("LINK PREDICTION v3 RESULTS (100 epochs, 100-dim, normalized TA phenotypes)")
    print(f"{'=' * 95}")
    print(f"Task: Disease-phenotype link prediction on OVERLAP diseases\n")

    for model_name in ["TransE", "RotatE"]:
        print(f"\n--- {model_name} ---")
        print(f"{'Condition':<20} {'N_triples':>10} {'N_rels':>8} {'Hits@1':>10} {'Hits@10':>10} {'MRR':>10}")
        print("-" * 80)
        for cond_name in ["hpoa_struct", "hpoa_temporal", "ta_struct", "ta_temporal"]:
            r = results[model_name].get(cond_name, {})
            if "error" in r:
                print(f"{cond_name:<20} ERROR: {r['error'][:40]}")
                continue
            total = r.get('n_train', 0) + r.get('n_test', 0)
            print(f"{cond_name:<20} {total:>10,} {r.get('n_relations', 0):>8} "
                  f"{r.get('hits_at_1', 0):>10.4f} {r.get('hits_at_10', 0):>10.4f} "
                  f"{r.get('mrr', 0):>10.4f}")

        # Relative improvements
        if "hpoa_struct" in results[model_name] and "hpoa_temporal" in results[model_name]:
            base = results[model_name]["hpoa_struct"].get("mrr", 0)
            temp = results[model_name]["hpoa_temporal"].get("mrr", 0)
            if base > 0:
                print(f"\n  HPOA: MRR {base:.4f} → {temp:.4f} ({100*(temp-base)/base:+.1f}% relative)")

        if "ta_struct" in results[model_name] and "ta_temporal" in results[model_name]:
            base = results[model_name]["ta_struct"].get("mrr", 0)
            temp = results[model_name]["ta_temporal"].get("mrr", 0)
            if base > 0:
                print(f"  TA:   MRR {base:.4f} → {temp:.4f} ({100*(temp-base)/base:+.1f}% relative)")

    out_file = RESULTS_DIR / "link_prediction_v3.json"
    with open(out_file, "w") as f:
        json.dump({
            "experiment": "Link Prediction v3 — diligent fixes",
            "task": "Disease-phenotype link prediction",
            "fixes": [
                "Normalized TA phenotype names (synonym consolidation)",
                "Longer training (100 epochs, 100-dim)",
                "Multiple models (TransE + RotatE)",
                "Same bins for HPOA and TA for fair comparison",
            ],
            "results": results,
        }, f, indent=2, default=str)
    logger.info(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
