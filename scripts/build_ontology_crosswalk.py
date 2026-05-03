#!/usr/bin/env python3
"""
Build Ontology Crosswalk — MONDO ↔ Orphanet ↔ OMIM
====================================================
Parses Mondo JSON-LD to extract cross-references between MONDO, Orphanet,
and OMIM identifiers. Also extracts synonyms for improved entity matching.

Outputs:
  data/validation_sources/mondo_crosswalk.json
    {
      "mondo_to_orpha": {"MONDO:0000001": ["Orphanet:377788"], ...},
      "mondo_to_omim": {"MONDO:0005148": ["OMIM:125853"], ...},
      "orpha_to_mondo": {"Orphanet:377788": "MONDO:0000001", ...},
      "omim_to_mondo": {"OMIM:125853": "MONDO:0005148", ...},
      "mondo_names": {"MONDO:0000001": "disease", ...},
      "mondo_synonyms": {"MONDO:0000001": ["condition", ...], ...},
      "stats": { counts }
    }

  data/validation_sources/orphadata_with_ids.json
    Re-parsed Orphadata XML preserving OrphaCode → onset mapping
"""

import json
import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation_sources"

# Orphadata categorical onset → age range (years)
ORPHA_ONSET_MAP = {
    "Antenatal": (0, 0),
    "Neonatal": (0, 0.08),
    "Infancy": (0.08, 1),
    "Childhood": (1, 5),      # Note: Orphadata uses 1-5 for Childhood
    "Adolescent": (10, 18),
    "Adult": (15, 120),
    "Elderly": (60, 120),
    "All ages": (0, 120),
    "No data available": None,
}


def build_mondo_crosswalk():
    """Parse Mondo JSON-LD and extract MONDO ↔ Orphanet ↔ OMIM mappings."""
    mondo_file = VALIDATION_DIR / "mondo.json"
    if not mondo_file.exists():
        logger.error(f"Mondo file not found: {mondo_file}")
        return None

    logger.info(f"Loading Mondo ontology ({mondo_file.stat().st_size / 1024 / 1024:.0f} MB)...")
    with open(mondo_file) as f:
        mondo = json.load(f)

    graphs = mondo.get("graphs", [])
    if not graphs:
        logger.error("No graphs in Mondo JSON")
        return None

    nodes = graphs[0].get("nodes", [])
    logger.info(f"  Total nodes: {len(nodes)}")

    # Build crosswalk
    mondo_to_orpha = defaultdict(list)
    mondo_to_omim = defaultdict(list)
    orpha_to_mondo = {}
    omim_to_mondo = {}
    mondo_names = {}
    mondo_synonyms = {}

    mondo_count = 0
    for node in nodes:
        node_id = node.get("id", "")

        # Extract MONDO ID from URL: http://purl.obolibrary.org/obo/MONDO_0000001 → MONDO:0000001
        if "MONDO_" not in node_id:
            continue

        mondo_id = node_id.split("/")[-1].replace("_", ":")
        mondo_count += 1

        # Name
        label = node.get("lbl")
        if label:
            mondo_names[mondo_id] = label.lower().strip()

        # Synonyms
        meta = node.get("meta", {})
        syns = meta.get("synonyms", [])
        if syns:
            syn_list = [s.get("val", "").lower().strip() for s in syns if s.get("val")]
            if syn_list:
                mondo_synonyms[mondo_id] = syn_list

        # Cross-references
        xrefs = meta.get("xrefs", [])
        for xref in xrefs:
            val = xref.get("val", "")

            if val.startswith("Orphanet:"):
                mondo_to_orpha[mondo_id].append(val)
                # Prefer first mapping if multiple MONDO map to same Orphanet
                if val not in orpha_to_mondo:
                    orpha_to_mondo[val] = mondo_id

            elif val.startswith("OMIM:"):
                mondo_to_omim[mondo_id].append(val)
                if val not in omim_to_mondo:
                    omim_to_mondo[val] = mondo_id

    stats = {
        "total_mondo_nodes": mondo_count,
        "mondo_with_orpha": len(mondo_to_orpha),
        "mondo_with_omim": len(mondo_to_omim),
        "total_orpha_mappings": sum(len(v) for v in mondo_to_orpha.values()),
        "total_omim_mappings": sum(len(v) for v in mondo_to_omim.values()),
        "unique_orpha_ids": len(orpha_to_mondo),
        "unique_omim_ids": len(omim_to_mondo),
        "mondo_with_synonyms": len(mondo_synonyms),
        "mondo_with_names": len(mondo_names),
    }

    logger.info(f"\n  Crosswalk stats:")
    for k, v in stats.items():
        logger.info(f"    {k}: {v:,}")

    crosswalk = {
        "mondo_to_orpha": dict(mondo_to_orpha),
        "mondo_to_omim": dict(mondo_to_omim),
        "orpha_to_mondo": orpha_to_mondo,
        "omim_to_mondo": omim_to_mondo,
        "mondo_names": mondo_names,
        "mondo_synonyms": mondo_synonyms,
        "stats": stats,
    }

    return crosswalk


def reparse_orphadata_with_ids():
    """Re-parse Orphadata XML preserving OrphaCode for ID-based matching.

    Returns: dict with two lookup tables:
      by_orpha_id: {"Orphanet:XXXXX": {categories, min_age, max_age, name}}
      by_name: {"disease name lower": {categories, min_age, max_age, orpha_id}}
    """
    xml_file = VALIDATION_DIR / "orphadata_onset_ages.xml"
    if not xml_file.exists():
        logger.error(f"Orphadata XML not found: {xml_file}")
        return None

    logger.info(f"\nRe-parsing Orphadata XML with OrphaCode...")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    by_orpha_id = {}
    by_name = {}
    total = 0
    skipped_no_onset = 0

    # Navigate XML structure
    disorders = root.findall(".//Disorder")
    if not disorders:
        # Try alternative path
        disorders = root.findall(".//{http://www.orphadata.org}Disorder")
    if not disorders:
        # Flat search
        for elem in root.iter():
            if elem.tag == "Disorder":
                disorders.append(elem)
                if len(disorders) > 10000:
                    break

    logger.info(f"  Found {len(disorders)} Disorder elements")

    for disorder in disorders:
        total += 1

        # Extract OrphaCode
        orpha_code_elem = disorder.find("OrphaCode")
        if orpha_code_elem is None:
            continue
        orpha_code = orpha_code_elem.text
        orpha_id = f"Orphanet:{orpha_code}"

        # Extract name
        name_elem = disorder.find("Name")
        name = name_elem.text.lower().strip() if name_elem is not None and name_elem.text else None

        # Extract onset categories
        onset_list = disorder.find("AverageAgeOfOnsetList")
        if onset_list is None:
            skipped_no_onset += 1
            continue

        categories = []
        for onset in onset_list.findall("AverageAgeOfOnset"):
            onset_name = onset.find("Name")
            if onset_name is not None and onset_name.text:
                categories.append(onset_name.text)

        if not categories:
            skipped_no_onset += 1
            continue

        # Map categories to age range
        min_ages = []
        max_ages = []
        for cat in categories:
            age_range = ORPHA_ONSET_MAP.get(cat)
            if age_range is not None:
                min_ages.append(age_range[0])
                max_ages.append(age_range[1])

        if not min_ages:
            skipped_no_onset += 1
            continue

        entry = {
            "categories": categories,
            "min_age": min(min_ages),
            "max_age": max(max_ages),
            "name": name,
            "orpha_id": orpha_id,
        }

        by_orpha_id[orpha_id] = entry
        if name:
            by_name[name] = entry

    logger.info(f"  Total disorders: {total}")
    logger.info(f"  With onset data: {len(by_orpha_id)}")
    logger.info(f"  Skipped (no onset): {skipped_no_onset}")
    logger.info(f"  By name: {len(by_name)}")

    return {"by_orpha_id": by_orpha_id, "by_name": by_name}


def reparse_hpoa_with_ids():
    """Re-parse HPOA preserving OMIM IDs for ID-based matching.

    Returns: dict[omim_id] = {name, onset_min, onset_max, onset_terms}
    Disease-level aggregation (min/max across all phenotype annotations with onset).
    """
    HPO_ONSET_TO_AGE = {
        "HP:0030674": (0, 0),       # Antenatal
        "HP:0003577": (0, 0),       # Congenital
        "HP:0003623": (0, 0.08),    # Neonatal
        "HP:0003593": (0.08, 1),    # Infantile
        "HP:0011463": (1, 5),       # Childhood
        "HP:0003621": (5, 15),      # Juvenile
        "HP:0011462": (15, 40),     # Young adult
        "HP:0003584": (15, 120),    # Late onset
        "HP:0003581": (40, 120),    # Adult onset
        "HP:0025708": (60, 120),    # Middle age
    }

    hpoa_file = VALIDATION_DIR / "phenotype.hpoa"
    if not hpoa_file.exists():
        logger.error(f"HPOA not found: {hpoa_file}")
        return None

    logger.info(f"\nRe-parsing HPOA with disease IDs...")

    diseases = defaultdict(lambda: {
        "name": None,
        "onset_mins": [],
        "onset_maxs": [],
        "onset_terms": set(),
    })

    total_rows = 0
    rows_with_onset = 0

    with open(hpoa_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue
            if parts[0] == "database_id":
                continue  # header

            total_rows += 1
            db_id = parts[0]        # e.g., OMIM:619340
            disease_name = parts[1]
            onset_col = parts[6] if len(parts) > 6 else ""

            if not onset_col or onset_col not in HPO_ONSET_TO_AGE:
                continue

            rows_with_onset += 1
            age_min, age_max = HPO_ONSET_TO_AGE[onset_col]

            diseases[db_id]["name"] = disease_name.lower().strip()
            diseases[db_id]["onset_mins"].append(age_min)
            diseases[db_id]["onset_maxs"].append(age_max)
            diseases[db_id]["onset_terms"].add(onset_col)

    # Aggregate per disease
    result = {}
    for db_id, data in diseases.items():
        if not data["onset_mins"]:
            continue
        result[db_id] = {
            "name": data["name"],
            "min_age": min(data["onset_mins"]),
            "max_age": max(data["onset_maxs"]),
            "onset_terms": list(data["onset_terms"]),
        }

    logger.info(f"  Total HPOA rows: {total_rows:,}")
    logger.info(f"  Rows with onset: {rows_with_onset:,}")
    logger.info(f"  Diseases with onset: {len(result)}")
    logger.info(f"  OMIM IDs: {sum(1 for k in result if k.startswith('OMIM:'))}")
    logger.info(f"  ORPHA IDs: {sum(1 for k in result if k.startswith('ORPHA:'))}")

    return result


def main():
    logger.info("=" * 70)
    logger.info("Ontology Crosswalk Builder")
    logger.info("=" * 70)

    # Step 1: Build MONDO crosswalk
    crosswalk = build_mondo_crosswalk()
    if crosswalk is None:
        return

    # Step 2: Re-parse Orphadata with IDs
    orphadata = reparse_orphadata_with_ids()
    if orphadata is None:
        return

    # Step 3: Re-parse HPOA with IDs
    hpoa = reparse_hpoa_with_ids()
    if hpoa is None:
        return

    # Step 4: Compute match potential
    # How many of our 15,828 MONDO IDs can now map to Orphadata via ID?
    logger.info(f"\n{'=' * 70}")
    logger.info("MATCH POTENTIAL ANALYSIS")
    logger.info(f"{'=' * 70}")

    # Load our disease configs — normalize MONDO IDs to 7-digit zero-padded
    import yaml
    config_dir = PROJECT_ROOT / "config" / "diseases"
    our_mondo_ids = set()
    for yf in config_dir.glob("*.yaml"):
        try:
            with open(yf) as f:
                cfg = yaml.safe_load(f)
            if cfg and cfg.get("mondo_id"):
                raw_id = cfg["mondo_id"]
                # Normalize: MONDO:270 → MONDO:0000270
                prefix, num = raw_id.split(":", 1)
                padded = f"{prefix}:{num.zfill(7)}"
                our_mondo_ids.add(padded)
        except Exception:
            pass

    logger.info(f"\n  Our MONDO IDs (zero-padded): {len(our_mondo_ids)}")

    # MONDO → Orphanet → Orphadata
    mondo_to_orpha = crosswalk["mondo_to_orpha"]
    orpha_by_id = orphadata["by_orpha_id"]

    matched_via_id = 0
    matched_orpha_ids = set()
    for mondo_id in our_mondo_ids:
        orpha_ids = mondo_to_orpha.get(mondo_id, [])
        for oid in orpha_ids:
            if oid in orpha_by_id:
                matched_via_id += 1
                matched_orpha_ids.add(oid)
                break

    logger.info(f"  Our MONDO → Orphadata (via Orphanet ID): {matched_via_id}")
    logger.info(f"  Previous name-based match: 1,856 exact + 589 fuzzy = 2,445")
    logger.info(f"  Improvement: +{matched_via_id - 2445} diseases")

    # MONDO → OMIM → HPOA
    mondo_to_omim = crosswalk["mondo_to_omim"]
    matched_hpoa = 0
    for mondo_id in our_mondo_ids:
        omim_ids = mondo_to_omim.get(mondo_id, [])
        for oid in omim_ids:
            if oid in hpoa:
                matched_hpoa += 1
                break

    logger.info(f"  Our MONDO → HPOA (via OMIM ID): {matched_hpoa}")
    logger.info(f"  Previous name-based match: 405")
    logger.info(f"  Improvement: +{matched_hpoa - 405} diseases")

    # Save crosswalk
    crosswalk_file = VALIDATION_DIR / "mondo_crosswalk.json"
    with open(crosswalk_file, "w") as f:
        json.dump(crosswalk, f, indent=2)
    logger.info(f"\n  Saved crosswalk: {crosswalk_file}")

    # Save re-parsed Orphadata
    orpha_file = VALIDATION_DIR / "orphadata_with_ids.json"
    with open(orpha_file, "w") as f:
        json.dump(orphadata, f, indent=2)
    logger.info(f"  Saved Orphadata: {orpha_file}")

    # Save re-parsed HPOA
    hpoa_file = VALIDATION_DIR / "hpoa_with_ids.json"
    with open(hpoa_file, "w") as f:
        json.dump(hpoa, f, indent=2)
    logger.info(f"  Saved HPOA: {hpoa_file}")


if __name__ == "__main__":
    main()
