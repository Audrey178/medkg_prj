#!/usr/bin/env python3
"""HDBSCAN sensitivity check on trajectory clustering.

Same features as the published k-means result (§6.3). Density-based clustering
tests whether the "4 archetype" answer is an artefact of k-means enforcing
spherical clusters, or whether structure of that shape actually exists.

Does NOT modify the paper. Writes a report at
data/experiments/hdbscan_sensitivity.json that we can decide to cite (as a
supplementary ablation) only if it broadly agrees with k-means.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("hdbscan")

ROOT = Path(__file__).resolve().parent.parent
CLUSTER_FILE = ROOT / "data" / "benchmark" / "trajectory_clustering.json"
OUT_DIR = ROOT / "data" / "experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_features():
    """Reuse the same features as the published k-means pass."""
    data = json.loads(CLUSTER_FILE.read_text())
    diseases = data["diseases"]
    log.info("Loaded features for %d diseases from trajectory_clustering.json", len(diseases))
    return diseases


def main():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))

    diseases = load_features()
    with_onset = {k: v for k, v in diseases.items() if v.get("median_onset") is not None}
    log.info("Diseases with onset data: %d", len(with_onset))

    # Only the features actually stored in trajectory_clustering.json
    feature_names = ["median_onset", "onset_spread", "n_stages", "total_triples"]
    disease_ids = list(with_onset.keys())
    X = np.array([[with_onset[d].get(f, 0) or 0 for f in feature_names] for d in disease_ids])

    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)

    try:
        import hdbscan  # type: ignore
    except ImportError:
        import subprocess
        subprocess.check_call(["/opt/anaconda3/bin/pip", "install", "hdbscan"])
        import hdbscan

    results = {}
    # Sweep a few min_cluster_size values — HDBSCAN's main hyperparameter
    for mcs in [50, 100, 200, 500, 1000]:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, metric="euclidean",
                                     cluster_selection_method="eom")
        labels = clusterer.fit_predict(Xs)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        dbcv = float(clusterer.relative_validity_) if hasattr(clusterer, "relative_validity_") else None
        from sklearn.metrics import silhouette_score
        # silhouette only on non-noise points
        mask = labels != -1
        sil = None
        if mask.sum() > 10 and len(set(labels[mask])) > 1:
            sil = float(silhouette_score(Xs[mask], labels[mask]))
        # cluster sizes
        sizes = {int(c): int(np.sum(labels == c)) for c in set(labels) if c != -1}
        results[f"mcs={mcs}"] = {
            "min_cluster_size": mcs,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_fraction": n_noise / len(labels),
            "dbcv": dbcv,
            "silhouette_nonNoise": sil,
            "cluster_sizes": sizes,
        }
        log.info("  min_cluster_size=%4d  clusters=%2d  noise=%d (%.0f%%)  DBCV=%s  sil=%s",
                 mcs, n_clusters, n_noise, 100 * n_noise / len(labels),
                 f"{dbcv:.3f}" if dbcv is not None else "n/a",
                 f"{sil:.3f}" if sil is not None else "n/a")

    # Compare with published k=4 k-means result
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(Xs)
    kmeans_sil = float(silhouette_score(Xs, km.labels_))
    log.info("Reference k-means k=4 silhouette: %.3f", kmeans_sil)

    report = {
        "n_diseases": len(disease_ids),
        "features": feature_names,
        "kmeans_k4_silhouette": kmeans_sil,
        "hdbscan_sweep": results,
    }
    out = OUT_DIR / "hdbscan_sensitivity.json"
    out.write_text(json.dumps(report, indent=2))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
