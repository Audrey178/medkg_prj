#!/usr/bin/env python3
"""
Tier 0: Export evidence-as-node representation for QA retrieval.

Reads all validated_triples.jsonl files under data/extracted/<disease>/,
calls TemporalEdge.to_evidence_grounded_dict() on each edge, and writes
data/exports/evidence_grounded.jsonl.

Usage:
    python scripts/export_evidence_grounded.py [--data-dir data/extracted] [--out data/exports/evidence_grounded.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models import TemporalEdge


def main() -> None:
    parser = argparse.ArgumentParser(description="Export evidence-grounded edge representation")
    parser.add_argument("--data-dir", default="data/extracted",
                        help="Root directory containing per-disease extracted/ subdirs")
    parser.add_argument("--out", default="data/exports/evidence_grounded.jsonl",
                        help="Output JSONL file path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with_pmid = 0
    diseases_found = 0

    with open(out_path, "w") as out_f:
        for vt_file in sorted(data_dir.glob("**/validated_triples.jsonl")):
            disease_dir = vt_file.parent
            diseases_found += 1
            edges_this_disease = 0

            with open(vt_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                        edge = TemporalEdge.from_dict(raw)
                        record = edge.to_evidence_grounded_dict()
                        out_f.write(json.dumps(record, default=str) + "\n")
                        total += 1
                        edges_this_disease += 1
                        if record["evidence_node"]["pmid"] is not None:
                            with_pmid += 1
                    except Exception as e:
                        print(f"  WARN: skipping line in {vt_file}: {e}", file=sys.stderr)

            print(f"  {disease_dir.name}: {edges_this_disease} edges")

    print(f"\nDone: {diseases_found} diseases, {total} edges written to {out_path}")
    if total > 0:
        print(f"  PMID coverage: {with_pmid}/{total} = {with_pmid/total:.1%}")
    else:
        print("  No edges found — run the pipeline first to generate validated_triples.jsonl")


if __name__ == "__main__":
    main()
