#!/usr/bin/env python3
"""Parse Phenopackets JSON files into a benchmark-ready pickle format.

Reads all phenopacket JSON files from data/validation_sources/phenopackets/0.1.26/
and produces data/validation_sources/phenopackets_parsed.pkl with per-disease
phenotype onset information.
"""

import json
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHENOPACKETS_DIR = PROJECT_ROOT / "data" / "validation_sources" / "phenopackets" / "0.1.26"
OUTPUT_PATH = PROJECT_ROOT / "data" / "validation_sources" / "phenopackets_parsed.pkl"

# HPO onset term -> (min_age_years, max_age_years)
HPO_ONSET_MAP = {
    "HP:0030674": (0, 0),        # Antenatal
    "HP:0003577": (0, 0),        # Congenital
    "HP:0003623": (0, 0.08),     # Neonatal
    "HP:0003593": (0.08, 2),     # Infantile
    "HP:0011463": (1, 11),       # Childhood
    "HP:0003621": (5, 15),       # Juvenile
    "HP:0011462": (15, 40),      # Young adult
    "HP:0003584": (15, 120),     # Late onset
    "HP:0003581": (40, 120),     # Adult
    "HP:0025708": (40, 60),      # Middle age
}


def parse_iso8601_duration(duration_str: str) -> float | None:
    """Parse ISO 8601 duration string to years.

    Examples: P5Y -> 5.0, P6M -> 0.5, P30D -> ~0.08, P1Y6M -> 1.5
    """
    if not duration_str:
        return None
    match = re.match(r"P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)W)?(?:(\d+)D)?", duration_str)
    if not match:
        return None
    years = int(match.group(1) or 0)
    months = int(match.group(2) or 0)
    weeks = int(match.group(3) or 0)
    days = int(match.group(4) or 0)
    total_years = years + months / 12.0 + weeks * 7 / 365.25 + days / 365.25
    return round(total_years, 4)


def extract_onset(onset_obj: dict | None) -> tuple[float, float] | None:
    """Extract (min_age, max_age) from a phenopacket onset object."""
    if onset_obj is None:
        return None

    # HPO ontology class onset
    if "ontologyClass" in onset_obj:
        hpo_id = onset_obj["ontologyClass"].get("id", "")
        if hpo_id in HPO_ONSET_MAP:
            return HPO_ONSET_MAP[hpo_id]

    # ISO 8601 age onset
    if "age" in onset_obj:
        age_str = onset_obj["age"].get("iso8601duration", "")
        age_years = parse_iso8601_duration(age_str)
        if age_years is not None:
            return (age_years, age_years)

    # Age range
    if "ageRange" in onset_obj:
        start = parse_iso8601_duration(
            onset_obj["ageRange"].get("start", {}).get("iso8601duration", "")
        )
        end = parse_iso8601_duration(
            onset_obj["ageRange"].get("end", {}).get("iso8601duration", "")
        )
        if start is not None and end is not None:
            return (start, end)

    return None


def parse_phenopacket(filepath: Path) -> dict | None:
    """Parse a single phenopacket JSON file.

    Returns dict with disease_ids, phenotype_onsets, disease_onset, subject info.
    """
    try:
        with open(filepath) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARNING: Failed to parse {filepath.name}: {e}", file=sys.stderr)
        return None

    # Extract diseases
    diseases = []
    disease_onset = None
    for d in data.get("diseases", []):
        term = d.get("term", {})
        disease_id = term.get("id", "")
        disease_label = term.get("label", "")
        if disease_id:
            diseases.append((disease_id, disease_label))
        # Disease-level onset
        onset = extract_onset(d.get("onset"))
        if onset is not None:
            disease_onset = onset

    # Also check interpretations for disease info
    for interp in data.get("interpretations", []):
        diag = interp.get("diagnosis", {})
        dis = diag.get("disease", {})
        did = dis.get("id", "")
        dlabel = dis.get("label", "")
        if did and not any(x[0] == did for x in diseases):
            diseases.append((did, dlabel))

    if not diseases:
        return None

    # Extract phenotypic features with onset
    phenotype_onsets = {}
    for pf in data.get("phenotypicFeatures", []):
        pf_type = pf.get("type", {})
        hpo_label = pf_type.get("label", "").strip()
        if not hpo_label:
            continue

        # Skip excluded/negated phenotypes
        if pf.get("excluded", False) or pf.get("negated", False):
            continue

        onset = extract_onset(pf.get("onset"))
        label_lower = hpo_label.lower()
        if label_lower not in phenotype_onsets:
            phenotype_onsets[label_lower] = []
        phenotype_onsets[label_lower].append(onset)  # None if no onset

    # Subject info
    subject = data.get("subject", {})
    age_at_encounter = None
    time_obj = subject.get("timeAtLastEncounter", {})
    if "age" in time_obj:
        age_at_encounter = parse_iso8601_duration(
            time_obj["age"].get("iso8601duration", "")
        )
    sex = subject.get("sex", "UNKNOWN")

    return {
        "diseases": diseases,
        "phenotype_onsets": phenotype_onsets,
        "disease_onset": disease_onset,
        "age_at_encounter": age_at_encounter,
        "sex": sex,
    }


def main():
    if not PHENOPACKETS_DIR.exists():
        print(f"ERROR: Phenopackets directory not found: {PHENOPACKETS_DIR}")
        sys.exit(1)

    # Collect all JSON files
    json_files = sorted(PHENOPACKETS_DIR.rglob("*.json"))
    print(f"Found {len(json_files)} phenopacket JSON files in {PHENOPACKETS_DIR}")

    # Parse all files
    parsed_count = 0
    skip_count = 0
    disease_data = defaultdict(lambda: {
        "disease_ids": set(),
        "cases": 0,
        "phenotype_onsets": defaultdict(list),
        "disease_onsets": [],
        "ages_at_encounter": [],
    })

    for filepath in json_files:
        result = parse_phenopacket(filepath)
        if result is None:
            skip_count += 1
            continue
        parsed_count += 1

        for disease_id, disease_label in result["diseases"]:
            key = disease_label.lower().strip()
            if not key:
                key = disease_id.lower()

            entry = disease_data[key]
            entry["disease_ids"].add(disease_id)
            entry["cases"] += 1

            # Merge phenotype onsets
            for pheno_label, onsets in result["phenotype_onsets"].items():
                entry["phenotype_onsets"][pheno_label].extend(onsets)

            # Disease onset
            if result["disease_onset"] is not None:
                entry["disease_onsets"].append(result["disease_onset"])

            # Age at encounter
            if result["age_at_encounter"] is not None:
                entry["ages_at_encounter"].append(result["age_at_encounter"])

    # Convert to final format
    final_data = {}
    total_onsets_with_timing = 0
    total_onsets_without_timing = 0
    total_phenotypes = 0

    for disease_name, entry in disease_data.items():
        # Compute disease onset range from all cases
        disease_onset_min = None
        disease_onset_max = None
        if entry["disease_onsets"]:
            mins = [o[0] for o in entry["disease_onsets"]]
            maxs = [o[1] for o in entry["disease_onsets"]]
            disease_onset_min = min(mins)
            disease_onset_max = max(maxs)

        # Filter phenotype onsets: keep only those with at least one timing
        phenotype_onsets_final = {}
        for pheno, onsets in entry["phenotype_onsets"].items():
            total_phenotypes += 1
            timed = [o for o in onsets if o is not None]
            if timed:
                phenotype_onsets_final[pheno] = timed
                total_onsets_with_timing += 1
            else:
                total_onsets_without_timing += 1

        final_data[disease_name] = {
            "disease_ids": sorted(entry["disease_ids"]),
            "cases": entry["cases"],
            "phenotype_onsets": phenotype_onsets_final,
            "disease_onset_min": disease_onset_min,
            "disease_onset_max": disease_onset_max,
        }

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(final_data, f)

    # Summary stats
    diseases_with_onsets = sum(
        1 for d in final_data.values() if d["phenotype_onsets"]
    )
    diseases_with_disease_onset = sum(
        1 for d in final_data.values() if d["disease_onset_min"] is not None
    )
    total_cases = sum(d["cases"] for d in final_data.values())
    total_phenotype_onset_pairs = sum(
        len(v) for d in final_data.values() for v in d["phenotype_onsets"].values()
    )

    print(f"\n{'='*60}")
    print(f"PHENOPACKETS PARSING SUMMARY")
    print(f"{'='*60}")
    print(f"JSON files found:              {len(json_files)}")
    print(f"Successfully parsed:           {parsed_count}")
    print(f"Skipped (no disease/parse err): {skip_count}")
    print(f"{'='*60}")
    print(f"Unique diseases:               {len(final_data)}")
    print(f"Total cases (disease-patient):  {total_cases}")
    print(f"Diseases with phenotype onsets: {diseases_with_onsets}")
    print(f"Diseases with disease onset:    {diseases_with_disease_onset}")
    print(f"{'='*60}")
    print(f"Total unique phenotypes:        {total_phenotypes}")
    print(f"Phenotypes with onset timing:   {total_onsets_with_timing}")
    print(f"Phenotypes without timing:      {total_onsets_without_timing}")
    print(f"Total onset data points:        {total_phenotype_onset_pairs}")
    print(f"{'='*60}")
    print(f"Output saved to: {OUTPUT_PATH}")

    # Show top diseases by case count
    top_diseases = sorted(final_data.items(), key=lambda x: -x[1]["cases"])[:15]
    print(f"\nTop 15 diseases by case count:")
    for name, d in top_diseases:
        onset_str = f"  onset_range={d['disease_onset_min']}-{d['disease_onset_max']}" if d["disease_onset_min"] is not None else ""
        print(f"  {d['cases']:4d} cases | {len(d['phenotype_onsets']):3d} pheno w/onset | {name[:60]}{onset_str}")


if __name__ == "__main__":
    main()
