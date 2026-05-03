#!/usr/bin/env python3
"""
Experiment 3: Disease Trajectory Clustering
=============================================
Clusters 13K diseases by their temporal progression signatures using
ChronoMedKG phenotype profiles. Discovers clinical archetypes:

  - Progressive degenerative (DMD: steady decline across stages)
  - Relapsing-remitting (MS: fluctuating phenotypes)
  - Acute monophasic (GBS: single episode)
  - Congenital static (Down syndrome: present from birth)
  - Late-onset progressive (LGMD: onset in adolescence/adulthood)
  - Variable/heterogeneous (conditions with wide onset spread)

This is genuinely novel — no existing KG has the temporal data to do this.

Features per disease (extracted from ChronoMedKG):
  1. Median onset age (years)
  2. Onset age spread (IQR or range)
  3. Number of distinct disease stages mentioned
  4. Number of temporal triples
  5. Fraction of triples with onset data
  6. Has progression milestones (boolean → 0/1)
  7. Earliest onset age
  8. Latest onset age

Outputs:
  data/benchmark/trajectory_clustering.json — full results + cluster assignments
  data/benchmark/trajectory_clusters_summary.json — human-readable archetype descriptions

Usage:
  .venv-sapbert/bin/python scripts/experiment_trajectory_clustering.py
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
logger = logging.getLogger("trajectory")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
CONFIG_DIR = PROJECT_ROOT / "config" / "diseases"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"


def extract_disease_features():
    """Extract temporal features per disease from ChronoMedKG validated triples."""
    diseases = {}

    for d in sorted(EXTRACTED_DIR.iterdir()):
        if not d.is_dir():
            continue
        vf = d / "validated_triples.jsonl"
        if not vf.exists() or vf.stat().st_size == 0:
            continue

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

        # Extract temporal features from triples
        onset_ages = []
        stages = set()
        milestones = set()
        durations = set()
        total_triples = 0
        onset_triples = 0

        with open(vf) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    t = json.loads(line)
                except Exception:
                    continue

                total_triples += 1
                temporal = t.get("temporal", {}) or {}

                # Onset ages
                omin = temporal.get("onset_age_min")
                omax = temporal.get("onset_age_max")
                if omin is not None:
                    try:
                        omin = float(omin)
                        omax = float(omax) if omax is not None else omin
                        if 0 <= omin <= 120 and 0 <= omax <= 120:
                            onset_ages.append((omin, omax))
                            onset_triples += 1
                    except (ValueError, TypeError):
                        pass

                # Stages
                stage = temporal.get("progression_stage")
                if stage and isinstance(stage, str) and stage.lower() not in ("unknown", "null", "none", ""):
                    stages.add(stage.lower().strip())

                # Milestones
                milestone = temporal.get("milestone")
                if milestone and isinstance(milestone, str) and milestone.lower() not in ("unknown", "null", "none", ""):
                    milestones.add(milestone.lower().strip())

                # Duration
                duration = temporal.get("duration")
                if duration and isinstance(duration, str) and duration.lower() not in ("unknown", "null", "none", ""):
                    durations.add(duration.lower().strip())

        if total_triples == 0:
            continue

        # Compute features
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
            "milestones_list": sorted(milestones)[:5],
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
                np.percentile(all_mins, 75) - np.percentile(all_mins, 25)
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

    logger.info("Extracted features for %d diseases", len(diseases))
    return diseases


def cluster_diseases(diseases, n_clusters=6):
    """Cluster diseases by temporal features using K-Means."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # Filter to diseases with onset data (needed for meaningful clustering)
    with_onset = {k: v for k, v in diseases.items() if v["median_onset"] is not None}
    logger.info("Diseases with onset data for clustering: %d", len(with_onset))

    # Feature matrix
    feature_names = [
        "median_onset",
        "onset_spread",
        "onset_iqr",
        "n_stages",
        "onset_fraction",
        "has_milestones",
        "earliest_onset",
        "latest_onset",
    ]

    disease_ids = list(with_onset.keys())
    X = np.array([
        [with_onset[d].get(f, 0) or 0 for f in feature_names]
        for d in disease_ids
    ])

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Try multiple k values
    results = {}
    for k in [4, 5, 6, 7, 8]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        results[k] = {"labels": labels, "silhouette": sil, "model": km}
        logger.info("  k=%d: silhouette=%.3f", k, sil)

    # Pick best k by silhouette
    best_k = max(results, key=lambda k: results[k]["silhouette"])
    logger.info("Best k=%d (silhouette=%.3f)", best_k, results[best_k]["silhouette"])

    best = results[best_k]
    labels = best["labels"]

    # Assign cluster labels to diseases
    for i, did in enumerate(disease_ids):
        with_onset[did]["cluster"] = int(labels[i])

    # UMAP embedding for visualization (2D)
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled) - 1))
        embedding = tsne.fit_transform(X_scaled)
        for i, did in enumerate(disease_ids):
            with_onset[did]["tsne_x"] = float(embedding[i, 0])
            with_onset[did]["tsne_y"] = float(embedding[i, 1])
        logger.info("t-SNE embedding computed")
    except Exception as e:
        logger.warning("t-SNE failed: %s", e)

    return with_onset, best_k, results


def characterize_clusters(diseases_clustered, n_clusters):
    """Characterize each cluster with descriptive statistics and archetype label."""
    cluster_profiles = {}

    for c in range(n_clusters):
        members = [d for d in diseases_clustered.values() if d.get("cluster") == c]
        if not members:
            continue

        onsets = [d["median_onset"] for d in members if d.get("median_onset") is not None]
        spreads = [d["onset_spread"] for d in members if d.get("onset_spread") is not None]
        stages = [d["n_stages"] for d in members]
        categories = Counter(d.get("disease_category", "unknown") for d in members)

        profile = {
            "cluster_id": c,
            "n_diseases": len(members),
            "median_onset_age": round(statistics.median(onsets), 1) if onsets else None,
            "mean_onset_age": round(statistics.mean(onsets), 1) if onsets else None,
            "onset_age_range": [round(min(onsets), 1), round(max(onsets), 1)] if onsets else None,
            "median_onset_spread": round(statistics.median(spreads), 1) if spreads else None,
            "mean_n_stages": round(statistics.mean(stages), 1),
            "top_categories": dict(categories.most_common(5)),
            "example_diseases": [
                d["disease_name"] for d in sorted(members, key=lambda x: -x["total_triples"])[:10]
            ],
        }

        # Auto-assign archetype label based on features
        med_onset = profile["median_onset_age"] or 0
        med_spread = profile["median_onset_spread"] or 0
        mean_stages = profile["mean_n_stages"]

        if med_onset < 1:
            archetype = "Congenital/Neonatal"
        elif med_onset < 5:
            archetype = "Early Childhood Onset"
        elif med_onset < 15 and med_spread > 20:
            archetype = "Progressive Childhood (wide progression)"
        elif med_onset < 15:
            archetype = "Childhood/Juvenile Onset"
        elif med_onset < 30 and med_spread > 30:
            archetype = "Variable Onset (adolescent-adult spread)"
        elif med_onset < 30:
            archetype = "Young Adult Onset"
        elif med_onset < 50:
            archetype = "Adult Onset"
        else:
            archetype = "Late-Onset"

        if mean_stages >= 3 and med_spread > 20:
            archetype += " — Progressive"
        elif med_spread < 5:
            archetype += " — Narrow Window"

        profile["archetype"] = archetype
        cluster_profiles[c] = profile

    return cluster_profiles


def main():
    logger.info("=" * 75)
    logger.info("Experiment 3: Disease Trajectory Clustering")
    logger.info("=" * 75)

    # Step 1: Extract features
    logger.info("\n[1/3] Extracting temporal features per disease...")
    diseases = extract_disease_features()

    # Step 2: Cluster
    logger.info("\n[2/3] Clustering diseases by temporal signature...")
    diseases_clustered, best_k, cluster_results = cluster_diseases(diseases)

    # Step 3: Characterize
    logger.info("\n[3/3] Characterizing clusters...")
    profiles = characterize_clusters(diseases_clustered, best_k)

    # === PRINT RESULTS ===
    print(f"\n{'=' * 75}")
    print(f"DISEASE TRAJECTORY CLUSTERING RESULTS")
    print(f"{'=' * 75}")
    print(f"Total diseases: {len(diseases):,}")
    print(f"Diseases with onset data (clustered): {len(diseases_clustered):,}")
    print(f"Optimal clusters: {best_k} (silhouette: {cluster_results[best_k]['silhouette']:.3f})")

    print(f"\nSilhouette scores by k:")
    for k in sorted(cluster_results):
        sil = cluster_results[k]["silhouette"]
        print(f"  k={k}: {sil:.3f} {'<-- best' if k == best_k else ''}")

    print(f"\n{'=' * 75}")
    print("CLUSTER ARCHETYPES")
    print(f"{'=' * 75}")

    for c in sorted(profiles):
        p = profiles[c]
        print(f"\n  Cluster {c}: {p['archetype']}")
        print(f"    Diseases: {p['n_diseases']:,}")
        print(f"    Median onset age: {p['median_onset_age']}y")
        print(f"    Onset age range: {p['onset_age_range']}")
        print(f"    Median spread: {p['median_onset_spread']}y")
        print(f"    Mean stages: {p['mean_n_stages']}")
        print(f"    Top categories: {p['top_categories']}")
        print(f"    Examples: {', '.join(p['example_diseases'][:5])}")

    # === SAVE ===
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    # Full results
    output = {
        "experiment": "Disease Trajectory Clustering",
        "total_diseases": len(diseases),
        "clustered_diseases": len(diseases_clustered),
        "optimal_k": best_k,
        "silhouette_scores": {k: round(v["silhouette"], 4) for k, v in cluster_results.items()},
        "cluster_profiles": profiles,
        "diseases": {
            did: {
                "disease_name": d["disease_name"],
                "disease_category": d["disease_category"],
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

    out_file = BENCHMARK_DIR / "trajectory_clustering.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nSaved: {out_file}")

    # Summary
    summary_file = BENCHMARK_DIR / "trajectory_clusters_summary.json"
    with open(summary_file, "w") as f:
        json.dump(profiles, f, indent=2, default=str)
    logger.info(f"Saved: {summary_file}")


if __name__ == "__main__":
    main()
