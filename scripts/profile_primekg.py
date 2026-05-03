"""
Profile PrimeKG
===============
Step 0 of Paper 2: Analyze PrimeKG disease coverage.

Outputs:
  data/primekg/disease_coverage.csv — per-disease coverage analysis
  data/primekg/profile_summary.json — aggregate statistics

Usage:
  python -m primekg_t.scripts.profile_primekg
  python -m primekg_t.scripts.profile_primekg --check-genereviews  # also query NCBI for GeneReviews entries
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PRIMEKG_DATA_DIR = PROJECT_ROOT / "data" / "primekg"
PRIMEKG_CSV = PRIMEKG_DATA_DIR / "kg.csv"


def load_primekg(path: Path) -> pd.DataFrame:
    """Load PrimeKG CSV into a DataFrame."""
    logger.info("Loading PrimeKG from %s ...", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %d edges, columns: %s", len(df), list(df.columns))
    return df


def profile_diseases(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-disease profile from PrimeKG edges."""
    logger.info("Profiling diseases...")

    # Collect per-disease stats using dict-based approach (avoid pandas join issues)
    disease_info: dict[str, dict] = {}  # disease_id -> {name, source, edges, rel_types, neighbor_types}

    # Process x-side diseases
    x_diseases = df[df["x_type"] == "disease"]
    for disease_id, grp in x_diseases.groupby("x_id"):
        if disease_id not in disease_info:
            first = grp.iloc[0]
            disease_info[disease_id] = {
                "disease_name": first["x_name"],
                "source": first["x_source"],
                "edge_count": 0,
                "relation_types": set(),
                "neighbor_types": Counter(),
            }
        info = disease_info[disease_id]
        info["edge_count"] += len(grp)
        info["relation_types"].update(grp["relation"].unique())
        info["neighbor_types"].update(grp["y_type"])

    # Process y-side diseases
    y_diseases = df[df["y_type"] == "disease"]
    for disease_id, grp in y_diseases.groupby("y_id"):
        if disease_id not in disease_info:
            first = grp.iloc[0]
            disease_info[disease_id] = {
                "disease_name": first["y_name"],
                "source": first["y_source"],
                "edge_count": 0,
                "relation_types": set(),
                "neighbor_types": Counter(),
            }
        info = disease_info[disease_id]
        info["edge_count"] += len(grp)
        info["relation_types"].update(grp["relation"].unique())
        info["neighbor_types"].update(grp["x_type"])

    logger.info("Found %d unique disease nodes", len(disease_info))

    # Convert to DataFrame
    rows = []
    for disease_id, info in disease_info.items():
        rows.append({
            "disease_id": disease_id,
            "disease_name": info["disease_name"],
            "source": info["source"],
            "edge_count": info["edge_count"],
            "relation_types": sorted(info["relation_types"]),
            "neighbor_types": dict(info["neighbor_types"]),
        })

    profile = pd.DataFrame(rows).sort_values("edge_count", ascending=False).reset_index(drop=True)

    logger.info("Disease profiles computed. Top 10 by edge count:")
    for _, row in profile.head(10).iterrows():
        logger.info("  %s (%s): %d edges", row["disease_name"][:50], row["disease_id"], row["edge_count"])

    return profile


def check_genereviews_batch(disease_names: list[str], batch_size: int = 50) -> dict[str, str | None]:
    """
    Check which diseases have GeneReviews entries via NCBI E-utilities.
    Returns {disease_name: genereviews_id_or_None}.
    """
    try:
        from Bio import Entrez
    except ImportError:
        logger.warning("Biopython not installed. Skipping GeneReviews check. Install: pip install biopython")
        return {}

    api_key = os.environ.get("NCBI_API_KEY", "")
    Entrez.email = "chronomedkg@example.com"
    if api_key:
        Entrez.api_key = api_key

    results = {}
    total = len(disease_names)

    for i in range(0, total, batch_size):
        batch = disease_names[i:i + batch_size]
        logger.info("Checking GeneReviews %d-%d / %d ...", i + 1, min(i + batch_size, total), total)

        for name in batch:
            try:
                query = f'"{name}"[Title] AND "GeneReviews"[Book]'
                handle = Entrez.esearch(db="books", term=query, retmax=1)
                record = Entrez.read(handle)
                handle.close()

                if int(record.get("Count", 0)) > 0:
                    results[name] = record["IdList"][0]
                else:
                    results[name] = None

                # Rate limit
                time.sleep(0.15 if api_key else 0.4)

            except Exception as e:
                logger.debug("GeneReviews check failed for '%s': %s", name, e)
                results[name] = None

    found = sum(1 for v in results.values() if v is not None)
    logger.info("GeneReviews check complete: %d / %d diseases have entries", found, len(results))
    return results


def generate_summary(profile: pd.DataFrame, genereviews: dict | None = None) -> dict:
    """Generate aggregate statistics."""
    summary = {
        "total_diseases": len(profile),
        "edge_count_stats": {
            "mean": round(profile["edge_count"].mean(), 1),
            "median": round(profile["edge_count"].median(), 1),
            "min": int(profile["edge_count"].min()),
            "max": int(profile["edge_count"].max()),
            "total": int(profile["edge_count"].sum()),
        },
        "diseases_with_100plus_edges": int((profile["edge_count"] >= 100).sum()),
        "diseases_with_10plus_edges": int((profile["edge_count"] >= 10).sum()),
        "diseases_with_1plus_edges": int((profile["edge_count"] >= 1).sum()),
        "top_20_diseases": [
            {"id": row["disease_id"], "name": row["disease_name"], "edges": int(row["edge_count"])}
            for _, row in profile.head(20).iterrows()
        ],
    }

    # Source distribution
    if "source" in profile.columns:
        summary["source_distribution"] = profile["source"].value_counts().to_dict()

    # GeneReviews coverage
    if genereviews:
        gr_count = sum(1 for v in genereviews.values() if v is not None)
        summary["genereviews_coverage"] = {
            "checked": len(genereviews),
            "has_entry": gr_count,
            "coverage_pct": round(100.0 * gr_count / len(genereviews), 1) if genereviews else 0,
        }

    # Neighbor type distribution (aggregate)
    all_neighbor_types = Counter()
    for _, row in profile.iterrows():
        nt = row.get("neighbor_types")
        if isinstance(nt, dict):
            for k, v in nt.items():
                all_neighbor_types[k] += v
    summary["neighbor_type_distribution"] = dict(all_neighbor_types.most_common())

    # Relation type distribution
    all_rels = Counter()
    for _, row in profile.iterrows():
        rt = row.get("relation_types")
        if isinstance(rt, list):
            for r in rt:
                all_rels[r] += 1
    summary["relation_type_distribution"] = dict(all_rels.most_common())

    return summary


def main():
    parser = argparse.ArgumentParser(description="Profile PrimeKG disease coverage")
    parser.add_argument("--check-genereviews", action="store_true",
                        help="Query NCBI for GeneReviews entries (slow, requires API)")
    parser.add_argument("--primekg-csv", type=str, default=str(PRIMEKG_CSV),
                        help="Path to PrimeKG kg.csv")
    args = parser.parse_args()

    csv_path = Path(args.primekg_csv)
    if not csv_path.exists():
        logger.error("PrimeKG CSV not found at %s. Download first.", csv_path)
        logger.error("Run: python -c \"import urllib.request; urllib.request.urlretrieve("
                      "'https://dataverse.harvard.edu/api/access/datafile/6180620', '%s')\"", csv_path)
        sys.exit(1)

    df = load_primekg(csv_path)
    profile = profile_diseases(df)

    # Optional: check GeneReviews
    genereviews = None
    if args.check_genereviews:
        disease_names = profile["disease_name"].tolist()
        genereviews = check_genereviews_batch(disease_names)
        profile["has_genereviews"] = profile["disease_name"].map(
            lambda n: genereviews.get(n) is not None
        )
        profile["genereviews_id"] = profile["disease_name"].map(
            lambda n: genereviews.get(n)
        )

    # Save outputs
    output_dir = PRIMEKG_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV (without complex dict columns)
    csv_cols = ["disease_id", "disease_name", "source", "edge_count"]
    if "has_genereviews" in profile.columns:
        csv_cols.extend(["has_genereviews", "genereviews_id"])
    profile[csv_cols].to_csv(output_dir / "disease_coverage.csv", index=False)
    logger.info("Saved disease_coverage.csv with %d rows", len(profile))

    # Save full profile as JSON (includes nested dicts)
    profile_records = profile.to_dict(orient="records")
    with open(output_dir / "disease_profiles.json", "w") as f:
        json.dump(profile_records, f, indent=2, default=str)

    # Save summary
    summary = generate_summary(profile, genereviews)
    with open(output_dir / "profile_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved profile_summary.json")

    # Print key stats
    print("\n" + "=" * 60)
    print("PrimeKG Disease Profile Summary")
    print("=" * 60)
    print(f"Total diseases: {summary['total_diseases']:,}")
    print(f"Total edges: {summary['edge_count_stats']['total']:,}")
    print(f"Avg edges/disease: {summary['edge_count_stats']['mean']:.1f}")
    print(f"Diseases with 100+ edges: {summary['diseases_with_100plus_edges']:,}")
    print(f"Diseases with 10+ edges: {summary['diseases_with_10plus_edges']:,}")
    if genereviews:
        gr = summary["genereviews_coverage"]
        print(f"GeneReviews entries: {gr['has_entry']:,} / {gr['checked']:,} ({gr['coverage_pct']:.1f}%)")
    print("\nNeighbor types:")
    for ntype, count in summary["neighbor_type_distribution"].items():
        print(f"  {ntype}: {count:,}")
    print("\nRelation types:")
    for rtype, count in list(summary["relation_type_distribution"].items())[:15]:
        print(f"  {rtype}: {count:,}")
    print()


if __name__ == "__main__":
    main()
