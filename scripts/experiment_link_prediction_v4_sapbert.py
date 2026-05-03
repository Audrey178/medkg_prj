#!/usr/bin/env python3
"""Link prediction v4: TA phenotypes canonicalised via SapBERT.

Identical to v3 in every respect EXCEPT the phenotype-name normalisation step,
which now uses the SapBERT cosine-similarity clusters from
`data/sapbert/canonical_mapping_thr0.9_top5.json` (mutual-kNN, 45 meta-labels
protected) instead of v3's hand-curated synonym dictionary.

v3 is NOT modified. We compare the two result files and update the paper ONLY
if v4 shows a net improvement over v3 that's statistically meaningful (paired
t-test per condition). Written per user request: do SapBERT diligently, audit
aggressively, update paper only if it helps.

Runs with the main anaconda Python (same environment as v3).
"""
from __future__ import annotations
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("lp_v4")

# Re-use v3 infrastructure by import so behaviour is identical
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import experiment_link_prediction_v3 as v3  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CANON_MAP_PATH = ROOT / "data" / "sapbert" / "canonical_mapping_thr0.9_top5.json"
RESULTS_DIR = ROOT / "data" / "benchmark" / "link_prediction_v4_sapbert"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# SapBERT canonical lookup  (keys are the raw TA surface forms, pre-normalised
# to lowercase/apostrophe-stripped; values are the canonical representative)
# --------------------------------------------------------------------------
def load_canonical_map(path: Path) -> dict[str, str]:
    log.info("Loading SapBERT canonical map from %s", path)
    raw = json.loads(path.read_text())
    # Pre-normalise keys the SAME WAY the canonicalise script did, so lookup
    # survives capitalisation/apostrophe variants.
    import re, unicodedata
    def pre(s: str) -> str:
        n = unicodedata.normalize("NFKD", s).lower().strip()
        n = n.replace("'", "").replace("`", "").replace("\u2019", "")
        n = re.sub(r"\s+", " ", n)
        n = re.sub(r"[.,;:]$", "", n)
        return n
    out = {}
    for orig, canon in raw.items():
        out[pre(orig)] = canon
    log.info("  %d entries", len(out))
    return out


def sapbert_normalize_phenotype(name: str, canon_map: dict[str, str]) -> str:
    """Replaces v3.normalize_phenotype_name. If SapBERT has a canonical for the
    pre-normalised name, use it (lowercased + underscored for KG-triple hygiene).
    Otherwise fall back to v3's manual rules (conservative)."""
    import re, unicodedata
    pre = unicodedata.normalize("NFKD", name).lower().strip().replace("'", "")
    pre = re.sub(r"\s+", " ", pre)
    pre = re.sub(r"[.,;:]$", "", pre)
    canon = canon_map.get(pre)
    if canon is None:
        return v3.normalize_phenotype_name(name)
    return canon.lower().replace(" ", "_").replace("-", "_")


def load_ta_edges_sapbert(canon_map: dict[str, str]):
    """Mirror of v3.load_ta_edges but using the SapBERT-mapped phenotype name."""
    edges = []
    raw_seen = set()
    canon_seen = set()

    for d in sorted(v3.EXTRACTED_DIR.iterdir()):
        vf = d / "validated_triples.jsonl"
        if not vf.exists():
            continue
        with open(vf) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    t = json.loads(line)
                except Exception:
                    continue
                if "phenotype" not in t.get("relation", ""):
                    continue
                if t.get("source_type") != "disease":
                    continue
                disease = v3.normalize_disease_name(t.get("source_name", ""))
                phenotype_raw = t.get("target_name", "").strip()
                if not disease or not phenotype_raw:
                    continue
                raw_seen.add(phenotype_raw)
                phenotype = sapbert_normalize_phenotype(phenotype_raw, canon_map)
                canon_seen.add(phenotype)

                temporal = t.get("temporal", {}) or {}
                omin = temporal.get("onset_age_min")
                bin_val = None
                if omin is not None:
                    try:
                        omin = float(omin)
                        omax = float(temporal.get("onset_age_max") or omin)
                        if 0 <= omin <= 120:
                            bin_val = v3.bin_numeric_age(omin, omax)
                    except Exception:
                        pass
                edges.append({
                    "disease_name": disease,
                    "phenotype_name": phenotype,
                    "onset_bin": bin_val,
                })

    log.info("TA (SapBERT-canonicalised): %d edges; %d raw phenotype surface forms -> %d canonical names (%.2fx reduction)",
             len(edges), len(raw_seen), len(canon_seen), len(raw_seen) / max(len(canon_seen), 1))
    return edges


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    canon_map = load_canonical_map(CANON_MAP_PATH)

    log.info("Loading HPOA edges (unchanged from v3)...")
    hpoa_edges = v3.load_hpoa_edges()

    log.info("Loading TA edges with SapBERT canonicalisation...")
    ta_edges = load_ta_edges_sapbert(canon_map)

    # Build same 4 conditions (HPOA_STRUCT, HPOA_TEMPORAL, TA_STRUCT, TA_TEMPORAL)
    conditions = v3.build_conditions(hpoa_edges, ta_edges)

    # ---- AUDIT: edge-count parity between v3 and v4 (should match; we only
    # changed entity names, not edges)
    log.info("=== AUDIT: edge counts per condition ===")
    for cond_name, triples in conditions.items():
        log.info("  %-20s %d triples", cond_name, len(triples))

    # Train + evaluate each condition with 3 seeds, mirroring v3 seeds script
    # (critical: must pass random_seed to pipeline() — otherwise PyKEEN hardcodes
    # one internally and all 3 "seeds" give identical results).
    import torch, random
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    def train_with_seed(triples, name, seed):
        random.seed(seed)
        shuffled = triples.copy(); random.shuffle(shuffled)
        n = len(shuffled)
        train = shuffled[:int(n*0.8)]
        val = shuffled[int(n*0.8):int(n*0.9)]
        test = shuffled[int(n*0.9):]
        entities = sorted(set([h for h,_,_ in triples] + [t for _,_,t in triples]))
        relations = sorted(set(r for _,r,_ in triples))
        ent_id = {e: i for i, e in enumerate(entities)}
        rel_id = {r: i for i, r in enumerate(relations)}
        def to_t(tps):
            return torch.LongTensor(np.array([[ent_id[h], rel_id[r], ent_id[t]] for h,r,t in tps]))
        tf = TriplesFactory(mapped_triples=to_t(train), entity_to_id=ent_id, relation_to_id=rel_id)
        vf = TriplesFactory(mapped_triples=to_t(val),   entity_to_id=ent_id, relation_to_id=rel_id)
        sf = TriplesFactory(mapped_triples=to_t(test),  entity_to_id=ent_id, relation_to_id=rel_id)
        res = pipeline(
            training=tf, validation=vf, testing=sf, model="TransE",
            model_kwargs={"embedding_dim": 100},
            training_kwargs={"num_epochs": 100, "batch_size": 1024},
            optimizer_kwargs={"lr": 0.01},
            random_seed=seed, device="mps",
        )
        return {
            "seed": seed,
            "hits_at_1":  float(res.metric_results.get_metric("hits_at_1")),
            "hits_at_3":  float(res.metric_results.get_metric("hits_at_3")),
            "hits_at_10": float(res.metric_results.get_metric("hits_at_10")),
            "mrr":        float(res.metric_results.get_metric("mrr")),
        }

    results: dict[str, list[dict]] = {}
    seeds = [42, 7, 123]
    for cond_name, triples in conditions.items():
        log.info(f"\n{'='*60}\nCondition: {cond_name}\n{'='*60}")
        results[cond_name] = []
        for seed in seeds:
            r = train_with_seed(triples, f"{cond_name}_seed{seed}", seed)
            results[cond_name].append(r)
            log.info("  seed=%d  MRR=%.4f  Hits@10=%.4f", seed, r["mrr"], r["hits_at_10"])

    # Aggregate + save
    summary = {"conditions": {}, "config": {
        "variant": "v4_sapbert",
        "canonical_map": str(CANON_MAP_PATH),
        "seeds": seeds,
        "model": "TransE", "embedding_dim": 100, "epochs": 100,
    }}
    for cond_name, runs in results.items():
        mrrs = [r["mrr"] for r in runs]
        hits10 = [r["hits_at_10"] for r in runs]
        summary["conditions"][cond_name] = {
            "mrr_mean": float(np.mean(mrrs)),
            "mrr_std":  float(np.std(mrrs, ddof=1)) if len(mrrs) > 1 else 0.0,
            "hits10_mean": float(np.mean(hits10)),
            "hits10_std":  float(np.std(hits10, ddof=1)) if len(hits10) > 1 else 0.0,
            "per_seed": runs,
        }
        log.info("%-20s MRR=%.4f±%.4f  Hits@10=%.4f±%.4f",
                 cond_name, summary["conditions"][cond_name]["mrr_mean"],
                 summary["conditions"][cond_name]["mrr_std"],
                 summary["conditions"][cond_name]["hits10_mean"],
                 summary["conditions"][cond_name]["hits10_std"])

    out = RESULTS_DIR / "results_v4_sapbert.json"
    out.write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
