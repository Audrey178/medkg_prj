#!/usr/bin/env python3
"""B3 — Coarsen-TA link prediction sensitivity.

Tests whether ChronoMedKG's +89.8% MRR gain over TA-struct (reported in
§6.5 of the paper) comes from *having* temporal bins, or from having
*fine-grained* bins. We coarsen TA's 8-bin onset categories into HPOA's
conceptual 5-bin granularity and retrain.

Conditions (TransE, 3 seeds each):
  - ta_struct           (from v3; re-runs for timing parity) — no temporal
  - ta_temporal_coarse  (NEW, 5 bins, this script)           — coarse temporal
  - ta_temporal_fine    (from v3; 8 bins)                    — fine temporal

Plus a small DistMult sensitivity on ta_temporal_coarse vs ta_temporal_fine
(1 seed each) to check whether model choice interacts with bin granularity.

Outputs: data/benchmark/link_prediction_b3/results_b3.json
"""
from __future__ import annotations
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("lp_b3")

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import experiment_link_prediction_v3 as v3

RESULTS_DIR = ROOT / "data" / "benchmark" / "link_prediction_b3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# HPOA-style 5-bin coarsening (maps v3.bin_numeric_age's 8 bins -> 5).
COARSE_MAP = {
    "neonatal":        "antenatal_infantile",
    "infantile":       "antenatal_infantile",
    "early_childhood": "childhood",
    "childhood":       "childhood",
    "juvenile":        "juvenile",
    "young_adult":     "adult",
    "adult":           "adult",
    "late_onset":      "late_onset",
}


def coarsen(bin_val):
    if bin_val is None:
        return None
    return COARSE_MAP.get(bin_val, bin_val)


def build_ta_conditions(ta_edges, overlap_diseases):
    """Build ta_struct / ta_temporal_fine / ta_temporal_coarse."""
    ta_struct, fine, coarse = [], [], []
    for e in ta_edges:
        if e["disease_name"] not in overlap_diseases:
            continue
        d = e["disease_name"]
        p = f"ta_{e['phenotype_name']}"
        ta_struct.append((d, "has_phenotype", p))
        bf = e["onset_bin"]
        bc = coarsen(bf)
        rel_fine  = f"has_phenotype__{bf}" if bf else "has_phenotype"
        rel_coarse = f"has_phenotype__{bc}" if bc else "has_phenotype"
        fine.append((d, rel_fine, p))
        coarse.append((d, rel_coarse, p))
    return {
        "ta_struct":          list(set(ta_struct)),
        "ta_temporal_coarse": list(set(coarse)),
        "ta_temporal_fine":   list(set(fine)),
    }


def train(triples, seed, model_name="TransE", epochs=100, embed_dim=100):
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory
    random.seed(seed)
    shuf = triples.copy(); random.shuffle(shuf)
    n = len(shuf)
    tr = shuf[:int(n * 0.8)]; va = shuf[int(n * 0.8):int(n * 0.9)]; te = shuf[int(n * 0.9):]
    ents = sorted(set([h for h, _, _ in triples] + [t for _, _, t in triples]))
    rels = sorted(set(r for _, r, _ in triples))
    ent_id = {e: i for i, e in enumerate(ents)}
    rel_id = {r: i for i, r in enumerate(rels)}

    def _t(tps):
        return torch.LongTensor(np.array([[ent_id[h], rel_id[r], ent_id[t]] for h, r, t in tps]))
    tf = TriplesFactory(mapped_triples=_t(tr), entity_to_id=ent_id, relation_to_id=rel_id)
    vf = TriplesFactory(mapped_triples=_t(va), entity_to_id=ent_id, relation_to_id=rel_id)
    sf = TriplesFactory(mapped_triples=_t(te), entity_to_id=ent_id, relation_to_id=rel_id)

    t0 = time.time()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    res = pipeline(
        training=tf, validation=vf, testing=sf, model=model_name,
        model_kwargs={"embedding_dim": embed_dim},
        training_kwargs={"num_epochs": epochs, "batch_size": 1024},
        optimizer_kwargs={"lr": 0.01},
        random_seed=seed, device=device,
    )
    dt = time.time() - t0
    return {
        "seed": seed, "model": model_name, "elapsed_sec": round(dt, 1),
        "n_entities": len(ents), "n_relations": len(rels),
        "n_train": len(tr), "n_test": len(te),
        "hits_at_1":  float(res.metric_results.get_metric("hits_at_1")),
        "hits_at_3":  float(res.metric_results.get_metric("hits_at_3")),
        "hits_at_10": float(res.metric_results.get_metric("hits_at_10")),
        "mrr":        float(res.metric_results.get_metric("mrr")),
    }


def main():
    log.info("=== B3: Coarsen-TA link-prediction sensitivity ===")
    log.info("Loading HPOA + TA edges via v3 infrastructure...")
    hpoa_edges = v3.load_hpoa_edges()
    ta_edges = v3.load_ta_edges()

    hpoa_d = set(e["disease_name"] for e in hpoa_edges)
    ta_d = set(e["disease_name"] for e in ta_edges)
    overlap = hpoa_d & ta_d
    log.info("Overlap diseases: %d", len(overlap))

    ta_conditions = build_ta_conditions(ta_edges, overlap)

    # Bin distributions to sanity-check the coarsening
    log.info("\n=== Bin distributions ===")
    from collections import Counter
    fine_rels = Counter(r for _, r, _ in ta_conditions["ta_temporal_fine"]  if "__" in r)
    coar_rels = Counter(r for _, r, _ in ta_conditions["ta_temporal_coarse"] if "__" in r)
    log.info("FINE (%d): %s", len(fine_rels), dict(fine_rels.most_common()))
    log.info("COARSE (%d): %s", len(coar_rels), dict(coar_rels.most_common()))
    log.info("")

    for name, tps in ta_conditions.items():
        rels = set(r for _, r, _ in tps)
        ents = set([h for h, _, _ in tps] + [t for _, _, t in tps])
        log.info("  %-22s %d triples | %d rels | %d ents", name, len(tps), len(rels), len(ents))

    seeds = [42, 7, 123]

    # Primary sweep: TransE × 3 conditions × 3 seeds
    transe_results = {}
    for cname, tps in ta_conditions.items():
        log.info("\n### TransE | %s", cname)
        transe_results[cname] = []
        for seed in seeds:
            r = train(tps, seed, "TransE")
            transe_results[cname].append(r)
            log.info("  seed=%d  MRR=%.4f  H@10=%.4f  t=%.1fs",
                     seed, r["mrr"], r["hits_at_10"], r["elapsed_sec"])

    # DistMult mini-sweep: coarse vs fine, 3 seeds, for model-sensitivity
    distmult_results = {}
    for cname in ("ta_temporal_coarse", "ta_temporal_fine"):
        log.info("\n### DistMult | %s", cname)
        distmult_results[cname] = []
        for seed in seeds:
            r = train(ta_conditions[cname], seed, "DistMult")
            distmult_results[cname].append(r)
            log.info("  seed=%d  MRR=%.4f  H@10=%.4f  t=%.1fs",
                     seed, r["mrr"], r["hits_at_10"], r["elapsed_sec"])

    # Aggregate
    def _agg(runs):
        mrrs = [r["mrr"] for r in runs]
        h10s = [r["hits_at_10"] for r in runs]
        return {
            "mrr_mean": float(np.mean(mrrs)),
            "mrr_std":  float(np.std(mrrs, ddof=1)) if len(mrrs) > 1 else 0.0,
            "hits10_mean": float(np.mean(h10s)),
            "hits10_std":  float(np.std(h10s, ddof=1)) if len(h10s) > 1 else 0.0,
            "per_seed": runs,
        }

    summary = {
        "experiment": "B3 coarsen-TA link-prediction sensitivity",
        "seeds": seeds,
        "bins": {
            "fine_definition": "v3.bin_numeric_age: neonatal(<0.08) / infantile(<1) / early_childhood(<5) / childhood(<12) / juvenile(<18) / young_adult(<40) / adult(<65) / late_onset(>=65)",
            "coarse_definition": "HPOA-style 5-bin collapse: antenatal_infantile / childhood / juvenile / adult / late_onset",
            "mapping": COARSE_MAP,
        },
        "overlap_diseases": len(overlap),
        "TransE": {c: _agg(rs) for c, rs in transe_results.items()},
        "DistMult": {c: _agg(rs) for c, rs in distmult_results.items()},
    }

    # Paired t-test: coarse vs fine (TransE across 3 seeds)
    from scipy.stats import ttest_rel
    coarse_mrrs = [r["mrr"] for r in transe_results["ta_temporal_coarse"]]
    fine_mrrs   = [r["mrr"] for r in transe_results["ta_temporal_fine"]]
    tstat, pval = ttest_rel(fine_mrrs, coarse_mrrs)
    summary["coarse_vs_fine_transe_paired_t"] = {
        "coarse_mrrs": coarse_mrrs, "fine_mrrs": fine_mrrs,
        "t": float(tstat), "p": float(pval),
        "fine_minus_coarse_mean": float(np.mean(fine_mrrs) - np.mean(coarse_mrrs)),
    }

    out = RESULTS_DIR / "results_b3.json"
    out.write_text(json.dumps(summary, indent=2))
    log.info("\nWrote %s", out)

    # Pretty print final table
    log.info("\n=== SUMMARY (MRR mean ± std across %d seeds) ===", len(seeds))
    for model_name, rmap in (("TransE", transe_results), ("DistMult", distmult_results)):
        for cname, runs in rmap.items():
            agg = _agg(runs)
            log.info("  %-10s %-22s MRR=%.4f ± %.4f  H@10=%.4f",
                     model_name, cname, agg["mrr_mean"], agg["mrr_std"], agg["hits10_mean"])
    t = summary["coarse_vs_fine_transe_paired_t"]
    log.info("\n  TransE: fine - coarse MRR = %+.4f (paired t=%.3f, p=%.4f)",
             t["fine_minus_coarse_mean"], t["t"], t["p"])


if __name__ == "__main__":
    main()
