"""
Filter validated_triples.jsonl to only contain triples whose disease_profile_id
maps to a disease in disease_list.tsv, using fuzzy name matching.

Output: data/extracted/validated_triples_filtered.jsonl
"""

import argparse
import json
import re
from pathlib import Path

from rapidfuzz import fuzz, process

REPO_ROOT = Path(__file__).parent.parent
TRIPLES_FILE = REPO_ROOT / "data/extracted/validated_triples.jsonl"
DISEASE_LIST_FILE = REPO_ROOT / "data/disease_list.tsv"
OUTPUT_FILE = REPO_ROOT / "data/extracted/validated_triples_filtered.jsonl"

FUZZY_THRESHOLD = 75


def normalize(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"[,;'\"]", "", name)
    return name


def load_disease_list(path: Path) -> dict[str, str]:
    """Returns {normalized_name: cui_id}."""
    diseases = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                cui, name = parts[0], parts[1]
                diseases[normalize(name)] = cui
    return diseases


def build_profile_name_map(triples_path: Path) -> dict[str, str]:
    """Single pass: build {disease_profile_id → best_disease_name}."""
    profile_names: dict[str, str] = {}
    all_pids: set[str] = set()
    print("Pass 1: building disease_profile_id → name map ...", flush=True)
    with open(triples_path) as f:
        for i, line in enumerate(f):
            if i % 100_000 == 0:
                print(f"  {i:,} lines scanned", flush=True)
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = t.get("disease_profile_id")
            if not pid:
                continue
            all_pids.add(pid)
            if pid in profile_names:
                continue
            src_type = t.get("source_type", "")
            tgt_type = t.get("target_type", "")
            if src_type == "disease":
                name = t.get("source_name", "")
                if name:
                    profile_names[pid] = name
            elif tgt_type == "disease":
                name = t.get("target_name", "")
                if name:
                    profile_names[pid] = name
    nameless_count = len(all_pids) - len(profile_names)
    if nameless_count:
        print(
            f"  WARNING: {nameless_count:,} profiles had no disease entity — will be dropped",
            flush=True,
        )
    print(f"  Found names for {len(profile_names):,} disease profiles", flush=True)
    return profile_names


def match_profiles(
    profile_names: dict[str, str],
    tsv_diseases: dict[str, str],
    threshold: int = FUZZY_THRESHOLD,
) -> set[str]:
    """Return set of disease_profile_ids that fuzzy-match any TSV disease."""
    tsv_norm_names = list(tsv_diseases.keys())
    matched_ids: set[str] = set()
    unmatched: list[tuple[str, str]] = []

    print("Matching profile names against disease list ...", flush=True)
    for pid, raw_name in profile_names.items():
        norm = normalize(raw_name)
        if norm in tsv_diseases:
            matched_ids.add(pid)
        else:
            unmatched.append((pid, norm))

    print(
        f"  Exact matches: {len(matched_ids):,} / {len(profile_names):,}", flush=True
    )

    if unmatched:
        print(f"  Fuzzy matching {len(unmatched):,} unmatched profiles ...", flush=True)
        for pid, norm in unmatched:
            result = process.extractOne(
                norm,
                tsv_norm_names,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=threshold,
            )
            if result:
                matched_ids.add(pid)

    return matched_ids


def filter_triples(
    triples_path: Path,
    output_path: Path,
    matched_ids: set[str],
) -> tuple[int, int, int]:
    """Pass 2: write matching triples. Returns (written, malformed, filtered)."""
    print(f"Pass 2: filtering triples → {output_path} ...", flush=True)
    written = malformed = filtered = 0
    with open(triples_path) as fin, open(output_path, "w") as fout:
        for i, line in enumerate(fin):
            if i % 100_000 == 0:
                print(f"  {i:,} lines processed", flush=True)
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if t.get("disease_profile_id") in matched_ids:
                fout.write(line)
                written += 1
            else:
                filtered += 1
    return written, malformed, filtered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triples", type=Path, default=TRIPLES_FILE)
    parser.add_argument("--disease-list", type=Path, default=DISEASE_LIST_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument(
        "--threshold",
        type=int,
        default=FUZZY_THRESHOLD,
        help="Fuzzy match threshold (0-100)",
    )
    args = parser.parse_args()

    tsv_diseases = load_disease_list(args.disease_list)
    print(f"Loaded {len(tsv_diseases):,} diseases from TSV", flush=True)

    profile_names = build_profile_name_map(args.triples)

    matched_ids = match_profiles(profile_names, tsv_diseases, args.threshold)
    print(
        f"\nMatched {len(matched_ids):,} / {len(profile_names):,} disease profiles "
        f"({100 * len(matched_ids) / max(len(profile_names), 1):.1f}%)",
        flush=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written, malformed, filtered = filter_triples(args.triples, args.output, matched_ids)

    total = written + malformed + filtered
    print(
        f"\n=== DONE ===\n"
        f"Input triples:   {total:,}\n"
        f"Written:         {written:,}\n"
        f"Filtered out:    {filtered:,}\n"
        f"Malformed lines: {malformed:,}\n"
        f"Retention rate:  {100 * written / max(total, 1):.1f}%\n"
        f"Output:          {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
