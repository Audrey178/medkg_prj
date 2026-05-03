"""
Quick test: Run Disease Profiler on DMD to validate it works.

Usage:
  python -m primekg_t.scripts.test_profiler
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.disease_profiler import DiseaseProfiler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


async def main():
    config = {"max_retries": 2, "retry_delay_seconds": 3}

    profiler = DiseaseProfiler(config=config)

    # Test on 3 known diseases from Paper 1
    test_diseases = [
        {"disease_id": "OMIM:310200", "disease_name": "Duchenne muscular dystrophy"},
        {"disease_id": "OMIM:254200", "disease_name": "Myasthenia gravis"},
        {"disease_id": "ORPHA:93930", "disease_name": "Chronic inflammatory demyelinating polyneuropathy"},
    ]

    for disease in test_diseases:
        print(f"\n{'='*60}")
        print(f"Profiling: {disease['disease_name']}")
        print(f"{'='*60}")

        result = await profiler.run_with_retry(disease)

        print(f"Status: {result.status}")
        print(f"Metrics: {json.dumps(result.metrics, indent=2)}")

        if result.errors:
            print(f"Errors: {result.errors}")

        profile_data = result.data.get("profile", {})
        print(f"Config saved to: {result.data.get('config_path')}")
        print(f"Synonyms: {profile_data.get('synonyms', [])[:5]}")
        print(f"Key genes: {profile_data.get('key_genes', [])}")
        print(f"Tier 1 sources: {profile_data.get('tier1_sources', [])}")
        print(f"PubMed articles: {profile_data.get('pubmed_article_count', 0)}")
        print(f"Coverage: {profile_data.get('coverage_flag')}")


if __name__ == "__main__":
    asyncio.run(main())
