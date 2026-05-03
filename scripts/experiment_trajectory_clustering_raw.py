#!/usr/bin/env python3
"""
Experiment 3b: Disease Trajectory Clustering on Raw (13M) Triples
==================================================================
Same clustering as experiment 3a but on ALL raw triples (before consensus
filtering). This gives:
  1. More data per disease → more reliable temporal signatures
  2. Comparison: does consensus filtering change the cluster structure?
  3. Coverage: how many MORE diseases have onset data in raw vs validated?

Raw triples: 13,045,687 across 12,887 diseases
Validated triples: 460,497 across 10,852 diseases

Usage:
  .venv-sapbert/bin/python scripts/experiment_trajectory_clustering_raw.py
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
import yaml
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("trajectory_raw")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
CONFIG_DIR = PROJECT_ROOT / "config" / "diseases"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"


def extract_disease_features_raw():
    """Extract temporal features per disease from RAW triples (pre-consensus)."""
    diseases = {}
    total_raw_triples = 0
    total_diseases = 0

    for d in sorted(EXTRACTED_DIR.iterdir()):
        if not d.is_dir():
            continue
        rf = d / "raw_triples.jsonl"
        if not rf.exists() or rf.stat().st_size == 0:
            continue

        total_diseases += 1
        dir_name = d.name

        # Load disease name from config
        config_file = CONFIG_DIR / f"{dir_name}.yaml"
        disease_name = dir_name.replace("_", ":")
        disease_category = ""
        if config_file.exists():
            try:
                with open(config_file) as f:
                    cfg = yaml.safe_load(f)
                disease_name = cfg.get("disease_name", disease_name)
                disease_category = cfg.get("disease_category", "")
            except Exception:
                pass

        onset_ages = []
        stages = set()
        milestones = set()
        total_triples = 0
        onset_triples = 0

        with open(rf) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    t = json.loads(line)
                except Exception:
                    continue

                total_triples += 1
                total_raw_triples += 1
                tc = t.get("temporal_context")
                if not tc or not isinstance(tc, dict):
                    continue

                omin = tc.get("onset_age_min")
                omax = tc.get("onset_age_max")
                if omin is not None:
                    try:
                        omin = float(omin)
                        omax = float(omax) if omax is not None else omin
                        if 0 <= omin <= 120 and 0 <= omax <= 120:
                            onset_ages.append((omin, omax))
                            onset_triples += 1
                    except (ValueError, TypeError):
                        pass

                stage = tc.get("progression_stage")
                if stage and isinstance(stage, str) and stage.lower() not in ("unknown", "null", "none", ""):
                    stages.add(stage.lower().strip())

                milestone = tc.get("milestone")
                if milestone and isinstance(milestone, str) and milestone.lower() not in ("unknown", "null", "none", ""):
                    milestones.add(milestone.lower().strip())

        if total_triples == 0:
            continue

        features = {
            "disease_name": disease_name,
            "disease_category": disease_category,
            "dir_name": dir_name,
            "total_triples": total_triples,
            "onset_triples": onset_triples,
            "onset_fraction": onset_triples / total_triples,
            "n_stages": len(stages),
            "n_milestones": len(milestones),
            "has_milestones": 1 if milestones else 0,
            "stages": sorted(stages),
        }

        if onset_ages:
            all_mins = [a[0] for a in onset_ages]
            all_maxs = [a[1] for a in onset_ages]
            features["median_onset"] = statistics.median(all_mins)
            features["mean_onset"] = round(statistics.mean(all_mins), 2)
            features["onset_spread"] = max(all_maxs) - min(all_mins)
            features["earliest_onset"] = min(all_mins)
            features["latest_onset"] = max(all_maxs)
            features["onset_iqr"] = (
                float(np.percentile(all_mins, 75) - np.percentile(all_mins, 25))
                if len(all_mins) >= 4 else features["onset_spread"]
            )
        else:
            features["median_onset"] = None
            features["mean_onset"] = None
            features["onset_spread"] = None
            features["earliest_onset"] = None
            features["latest_onset"] = None
            features["onset_iqr"] = None

        diseases[dir_name] = features

    logger.info("Extracted features from %d diseases (%d raw triples)", total_diseases, total_raw_triples)
    logger.info("Diseases with features: %d", len(diseases))
    return diseases, total_raw_triples


def cluster_diseases(diseases, n_clusters_range=(4, 9)):
    """Cluster diseases by temporal features."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    with_onset = {k: v for k, v in diseases.items() if v["median_onset"] is not None}
    logger.info("Diseases with onset data for clustering: %d", len(with_onset))

    feature_names = [
        "median_onset", "onset_spread", "onset_iqr", "n_stages",
        "onset_fraction", "has_milestones", "earliest_onset", "latest_onset",
    ]

    disease_ids = list(with_onset.keys())
    X = np.array([
        [with_onset[d].get(f, 0) or 0 for f in feature_names]
        for d in disease_ids
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}
    for k in range(n_clusters_range[0], n_clusters_range[1]):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        results[k] = {"labels": labels, "silhouette": sil}
        logger.info("  k=%d: silhouette=%.3f", k, sil)

    best_k = max(results, key=lambda k: results[k]["silhouette"])
    logger.info("Best k=%d (silhouette=%.3f)", best_k, results[best_k]["silhouette"])

    labels = results[best_k]["labels"]
    for i, did in enumerate(disease_ids):
        with_onset[did]["cluster"] = int(labels[i])

    # t-SNE
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled) - 1))
        embedding = tsne.fit_transform(X_scaled)
        for i, did in enumerate(disease_ids):
            with_onset[did]["tsne_x"] = float(embedding[i, 0])
            with_onset[did]["tsne_y"] = float(embedding[i, 1])
    except Exception as e:
        logger.warning("t-SNE failed: %s", e)

    return with_onset, best_k, results


def characterize_clusters(diseases_clustered, n_clusters):
    """Characterize each cluster."""
    profiles = {}
    for c in range(n_clusters):
        members = [d for d in diseases_clustered.values() if d.get("cluster") == c]
        if not members:
            continue

        onsets = [d["median_onset"] for d in members if d.get("median_onset") is not None]
        spreads = [d["onset_spread"] for d in members if d.get("onset_spread") is not None]
        stages = [d["n_stages"] for d in members]

        med_onset = statistics.median(onsets) if onsets else 0
        med_spread = statistics.median(spreads) if spreads else 0
        mean_stages = statistics.mean(stages)

        # Auto-label
        if med_onset < 1:
            archetype = "Congenital/Neonatal"
        elif med_onset < 5:
            archetype = "Early Childhood Onset"
        elif med_onset < 15 and med_spread > 20:
            archetype = "Progressive Childhood"
        elif med_onset < 15:
            archetype = "Childhood/Juvenile"
        elif med_onset < 30 and med_spread > 30:
            archetype = "Variable Onset"
        elif med_onset < 30:
            archetype = "Young Adult Onset"
        elif med_onset < 50:
            archetype = "Adult Onset"
        else:
            archetype = "Late-Onset"

        if mean_stages >= 3 and med_spread > 20:
            archetype += " — Progressive"

        profiles[c] = {
            "archetype": archetype,
            "n_diseases": len(members),
            "median_onset_age": round(med_onset, 1),
            "onset_range": [round(min(onsets), 1), round(max(onsets), 1)] if onsets else None,
            "median_spread": round(med_spread, 1),
            "mean_stages": round(mean_stages, 1),
            "examples": [d["disease_name"] for d in sorted(members, key=lambda x: -x["total_triples"])[:8]],
        }
    return profiles


def compare_with_validated(raw_clusters, validated_file):
    """Compare raw vs validated clustering."""
    if not validated_file.exists():
        return None

    with open(validated_file) as f:
        val_data = json.load(f)

    val_diseases = val_data.get("diseases", {})
    comparison = {
        "raw_diseases_clustered": len(raw_clusters),
        "validated_diseases_clustered": len(val_diseases),
        "additional_diseases_in_raw": len(set(raw_clusters.keys()) - set(val_diseases.keys())),
        "note": "Raw triples provide temporal features for more diseases since consensus filtering removes some."
    }
    return comparison


def main():
    logger.info("=" * 75)
    logger.info("Experiment 3b: Trajectory Clustering on Raw (13M) Triples")
    logger.info("=" * 75)

    logger.info("\n[1/3] Extracting features from raw triples...")
    diseases, total_raw = extract_disease_features_raw()

    with_onset_count = sum(1 for d in diseases.values() if d["median_onset"] is not None)
    logger.info("Diseases with onset data: %d / %d (%.1f%%)",
                with_onset_count, len(diseases), 100 * with_onset_count / max(1, len(diseases)))

    logger.info("\n[2/3] Clustering...")
    diseases_clustered, best_k, cluster_results = cluster_diseases(diseases)

    logger.info("\n[3/3] Characterizing clusters...")
    profiles = characterize_clusters(diseases_clustered, best_k)

    # Compare with validated
    val_file = BENCHMARK_DIR / "trajectory_clustering.json"
    comparison = compare_with_validated(diseases_clustered, val_file)

    # Print
    print(f"\n{'=' * 75}")
    print("RAW TRIPLE CLUSTERING RESULTS")
    print(f"{'=' * 75}")
    print(f"Total raw triples processed: {total_raw:,}")
    print(f"Diseases with features: {len(diseases):,}")
    print(f"Diseases with onset data: {with_onset_count:,}")
    print(f"Optimal clusters: {best_k} (silhouette: {cluster_results[best_k]['silhouette']:.3f})")

    if comparison:
        print(f"\nComparison with validated-triple clustering:")
        print(f"  Raw: {comparison['raw_diseases_clustered']:,} diseases clustered")
        print(f"  Validated: {comparison['validated_diseases_clustered']:,} diseases clustered")
        print(f"  Additional from raw: {comparison['additional_diseases_in_raw']:,}")

    print(f"\n{'=' * 75}")
    print("CLUSTER ARCHETYPES (RAW)")
    print(f"{'=' * 75}")
    for c in sorted(profiles):
        p = profiles[c]
        print(f"\n  Cluster {c}: {p['archetype']}")
        print(f"    Diseases: {p['n_diseases']:,}")
        print(f"    Median onset: {p['median_onset_age']}y, Spread: {p['median_spread']}y")
        print(f"    Examples: {', '.join(p['examples'][:5])}")

    # Save
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "Disease Trajectory Clustering (Raw Triples)",
        "total_raw_triples": total_raw,
        "total_diseases": len(diseases),
        "diseases_with_onset": with_onset_count,
        "optimal_k": best_k,
        "silhouette_scores": {k: round(v["silhouette"], 4) for k, v in cluster_results.items()},
        "cluster_profiles": profiles,
        "comparison_with_validated": comparison,
        "diseases": {
            did: {
                "disease_name": d["disease_name"],
                "cluster": d.get("cluster"),
                "median_onset": d.get("median_onset"),
                "onset_spread": d.get("onset_spread"),
                "n_stages": d.get("n_stages"),
                "total_triples": d["total_triples"],
                "tsne_x": d.get("tsne_x"),
                "tsne_y": d.get("tsne_y"),
            }
            for did, d in diseases_clustered.items()
        },
    }

    out_file = BENCHMARK_DIR / "trajectory_clustering_raw.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
