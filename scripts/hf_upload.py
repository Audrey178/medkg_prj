#!/usr/bin/env python3
"""
Upload the ChronoMedKG release artifacts to Hugging Face Hub as a gated dataset.

This script is a *stub* documenting the exact release procedure. The actual
upload requires an HF token with write access on the destination repo; set
HF_TOKEN in your environment before running.

Artifacts uploaded:
  - validated_triples.jsonl       (502 MB, primary KG)
  - tqa_benchmark.json            (3.0 MB, TQA benchmark)
  - pmc_clinical_cases.json       (62 KB,  31 case studies)
  - novelty_multi_judge_v2.json   (164 KB, 3-judge audit)
  - croissant.json                (16 KB,  Croissant metadata w/ 8 RAI fields)
  - README.md                     (data card w/ gated-access prompt)

Gating behaviour: the dataset is created as `gated="manual"` during the
NeurIPS review period. Requests are approved by the maintainer; after
acceptance the gate is released.

Usage:
    export HF_TOKEN=...
    python3 scripts/hf_upload.py --repo-id ORG/temporal-atlas --dry-run
    python3 scripts/hf_upload.py --repo-id ORG/temporal-atlas
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Artefact paths relative to repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]

ARTEFACTS: list[tuple[Path, str]] = [
    (REPO_ROOT / "data/huggingface_upload/validated_triples.jsonl", "validated_triples.jsonl"),
    (REPO_ROOT / "data/huggingface_upload/tqa_benchmark.json",      "tqa_benchmark.json"),
    (REPO_ROOT / "data/huggingface_upload/pmc_clinical_cases.json", "pmc_clinical_cases.json"),
    (REPO_ROOT / "data/benchmark/novelty_multi_judge_v2.json",      "novelty_multi_judge_v2.json"),
    (REPO_ROOT / "data/primekg_t/croissant.json",                   "croissant.json"),
    (REPO_ROOT / "data/huggingface_upload/README.md",               "README.md"),
    (REPO_ROOT / "LICENSE-DATA",                                    "LICENSE-DATA"),
    (REPO_ROOT / "NOTICE",                                          "NOTICE"),
]


def check_artefacts() -> list[tuple[Path, str, int]]:
    """Return list of (local_path, remote_path, size_bytes). Exit if anything missing."""
    out: list[tuple[Path, str, int]] = []
    missing: list[Path] = []
    for local, remote in ARTEFACTS:
        if not local.exists():
            missing.append(local)
            continue
        out.append((local, remote, local.stat().st_size))
    if missing:
        print("ERROR: missing artefacts:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        sys.exit(2)
    return out


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="e.g. ORG/temporal-atlas")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--gated", choices=["auto", "manual", "none"], default="manual")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify artefacts exist and print plan; do not upload.")
    args = parser.parse_args()

    artefacts = check_artefacts()
    total = sum(size for _, _, size in artefacts)
    print(f"Artefacts ({len(artefacts)} files, {human_bytes(total)} total):")
    for local, remote, size in artefacts:
        print(f"  {human_bytes(size):>10s}  {remote:<40s}  <- {local}")

    if args.dry_run:
        print("\n[dry-run] Would upload to", f"hf://{args.repo_id}", f"(gated={args.gated})")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set in environment", file=sys.stderr)
        sys.exit(2)

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: pip install huggingface_hub==0.26.5", file=sys.stderr)
        sys.exit(2)

    api = HfApi(token=token)

    # Create (or update) the dataset repo with the desired gating behaviour.
    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=False,
        token=token,
    )
    if args.gated != "none":
        api.update_repo_settings(
            repo_id=args.repo_id,
            repo_type="dataset",
            gated=args.gated,
            token=token,
        )

    # Upload each artefact.
    for local, remote, _ in artefacts:
        print(f"uploading {remote} ...", flush=True)
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            token=token,
        )

    print("\nRelease complete.")
    print(f"Landing page: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
