#!/usr/bin/env python3
"""
Triple-Level Post-Processing for ChronoMedKG
===============================================
Runs AFTER consolidate_entities.py to:
1. Reclassify type-mismatched relations based on entity types
2. Deduplicate remaining edges (reuses consolidate_entities logic)
3. Tag unresolvable mismatches without dropping them

Usage:
    python3 scripts/cleanup_triples.py                        # All diseases
    python3 scripts/cleanup_triples.py --disease MONDO_10000  # Single disease
    python3 scripts/cleanup_triples.py --dry-run              # Preview without writing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"

# Import consolidation helpers
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from consolidate_entities import normalize_entity_name, make_edge_key, merge_edges


# ──────────────────────────────────────────────────────────────────────
# Valid (source_type, relation, target_type) combinations from PrimeKG
# ──────────────────────────────────────────────────────────────────────

VALID_TYPE_COMBOS: dict[str, set[tuple[str, str]]] = {
    # relation -> set of valid (source_type, target_type)
    "disease_phenotype_positive": {("disease", "phenotype"), ("phenotype", "disease")},
    "disease_phenotype_negative": {("disease", "phenotype"), ("phenotype", "disease")},
    "disease_protein":           {("disease", "gene/protein"), ("gene/protein", "disease")},
    "indication":                {("drug", "disease"), ("disease", "drug")},
    "contraindication":          {("drug", "disease"), ("disease", "drug")},
    "off-label use":             {("drug", "disease"), ("disease", "drug")},
    "drug_effect":               {("drug", "phenotype"), ("drug", "biological_process"),
                                  ("phenotype", "drug"), ("biological_process", "drug")},
    "drug_protein":              {("drug", "gene/protein"), ("gene/protein", "drug")},
    "disease_disease":           {("disease", "disease")},
    "protein_protein":           {("gene/protein", "gene/protein")},
    "bioprocess_protein":        {("biological_process", "gene/protein"), ("gene/protein", "biological_process")},
    "molfunc_protein":           {("molecular_function", "gene/protein"), ("gene/protein", "molecular_function")},
    "cellcomp_protein":          {("cellular_component", "gene/protein"), ("gene/protein", "cellular_component")},
    "phenotype_protein":         {("phenotype", "gene/protein"), ("gene/protein", "phenotype")},
    "pathway_protein":           {("pathway", "gene/protein"), ("gene/protein", "pathway")},
    "anatomy_protein_expressed": {("anatomy", "gene/protein"), ("gene/protein", "anatomy")},
    "anatomy_protein_absent":    {("anatomy", "gene/protein"), ("gene/protein", "anatomy")},
    "exposure_disease":          {("exposure", "disease"), ("disease", "exposure")},
    "exposure_protein":          {("exposure", "gene/protein"), ("gene/protein", "exposure")},
    "exposure_bioprocess":       {("exposure", "biological_process"), ("biological_process", "exposure")},
    "exposure_molfunc":          {("exposure", "molecular_function"), ("molecular_function", "exposure")},
    "exposure_cellcomp":         {("exposure", "cellular_component"), ("cellular_component", "exposure")},
    # Carrier relations (from extraction, not yet mapped)
    "treats":                    {("drug", "disease"), ("disease", "drug")},
    "manifests_as":              {("disease", "phenotype"), ("phenotype", "disease")},
    "caused_by":                 {("disease", "gene/protein"), ("gene/protein", "disease")},
    "biomarker_for":             {("gene/protein", "disease"), ("disease", "gene/protein"),
                                  ("phenotype", "disease"), ("disease", "phenotype")},
    "progresses_to":             {("disease", "disease")},
    "differentiates":            {("disease", "disease"), ("phenotype", "disease")},
    "onset_at":                  {("disease", "phenotype"), ("phenotype", "disease")},
}


# ──────────────────────────────────────────────────────────────────────
# Reclassification rules: (source_type, current_relation, target_type) -> new_relation
# ──────────────────────────────────────────────────────────────────────

RECLASSIFY_RULES: dict[tuple[str, str, str], str] = {
    # disease→disease with phenotype relation → disease_disease
    ("disease", "disease_phenotype_positive", "disease"): "disease_disease",
    ("disease", "disease_phenotype_negative", "disease"): "disease_disease",
    ("disease", "manifests_as", "disease"):               "disease_disease",
    # disease→phenotype with disease_disease → disease_phenotype_positive
    ("disease", "disease_disease", "phenotype"):          "disease_phenotype_positive",
    ("phenotype", "disease_disease", "disease"):          "disease_phenotype_positive",
    # gene/protein→phenotype with disease_phenotype → phenotype_protein
    ("gene/protein", "disease_phenotype_positive", "phenotype"): "phenotype_protein",
    ("phenotype", "disease_phenotype_positive", "gene/protein"): "phenotype_protein",
    # drug→phenotype with indication → drug_effect
    ("drug", "indication", "phenotype"):                  "drug_effect",
    ("phenotype", "indication", "drug"):                  "drug_effect",
    # disease→drug with indication → swap source/target (handled separately)
    # gene/protein→disease with disease_phenotype → disease_protein
    ("gene/protein", "disease_phenotype_positive", "disease"): "disease_protein",
    ("disease", "disease_phenotype_positive", "gene/protein"): "disease_protein",
    # biological_process with disease_protein → bioprocess_protein
    ("disease", "disease_protein", "biological_process"):       "bioprocess_protein",
    ("biological_process", "disease_protein", "disease"):       "bioprocess_protein",
    ("biological_process", "disease_protein", "gene/protein"):  "bioprocess_protein",
    ("gene/protein", "disease_protein", "biological_process"):  "bioprocess_protein",
}

# Swap rules: (source_type, relation, target_type) -> swap source and target
SWAP_RULES: set[tuple[str, str, str]] = {
    ("disease", "indication", "drug"),  # should be drug→disease
}


def is_valid_combo(source_type: str, relation: str, target_type: str) -> bool:
    """Check if this (source_type, relation, target_type) is valid."""
    valid = VALID_TYPE_COMBOS.get(relation, set())
    return (source_type, target_type) in valid


def reclassify_edge(edge: dict) -> tuple[dict, str]:
    """
    Attempt to reclassify a type-mismatched edge.
    Returns (modified_edge, action) where action is one of:
      'valid', 'reclassified', 'swapped', 'tagged_unresolved'
    """
    source_type = (edge.get("source_type") or "").lower().strip()
    target_type = (edge.get("target_type") or "").lower().strip()
    relation = (edge.get("relation") or "").lower().strip()

    # Already valid
    if is_valid_combo(source_type, relation, target_type):
        return edge, "valid"

    # Try reclassification
    key = (source_type, relation, target_type)
    if key in RECLASSIFY_RULES:
        edge["relation"] = RECLASSIFY_RULES[key]
        return edge, "reclassified"

    # Try swap
    if key in SWAP_RULES:
        edge["source_id"], edge["target_id"] = edge["target_id"], edge["source_id"]
        edge["source_name"], edge["target_name"] = edge["target_name"], edge["source_name"]
        edge["source_type"], edge["target_type"] = edge["target_type"], edge["source_type"]
        return edge, "swapped"

    # Unresolvable — tag but keep
    edge["type_mismatch_unresolved"] = True
    return edge, "tagged_unresolved"


def cleanup_disease(disease_dir: Path, dry_run: bool = False) -> dict:
    """Clean up triples for a single disease."""
    vt_file = disease_dir / "validated_triples.jsonl"
    if not vt_file.exists():
        return {"status": "skipped", "reason": "no validated_triples.jsonl"}

    edges = []
    with open(vt_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    edges.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not edges:
        return {"status": "skipped", "reason": "empty file"}

    original_count = len(edges)

    # Step 1: Type mismatch reclassification
    actions = defaultdict(int)
    for i, edge in enumerate(edges):
        edges[i], action = reclassify_edge(edge)
        actions[action] += 1

    # Step 2: Normalize entity names
    name_changes = 0
    for edge in edges:
        for field in ("source_name", "target_name"):
            old_name = edge.get(field, "")
            new_name = normalize_entity_name(old_name)
            if new_name != old_name:
                edge[field] = new_name
                name_changes += 1

    # Step 3: Deduplicate
    edge_groups: dict[tuple, list[dict]] = defaultdict(list)
    for edge in edges:
        key = make_edge_key(edge)
        edge_groups[key].append(edge)

    consolidated = []
    duplicates_merged = 0
    for key, group in edge_groups.items():
        merged = merge_edges(group)
        consolidated.append(merged)
        if len(group) > 1:
            duplicates_merged += len(group) - 1

    stats = {
        "status": "success",
        "original_triples": original_count,
        "cleaned_triples": len(consolidated),
        "reclassified": actions.get("reclassified", 0),
        "swapped": actions.get("swapped", 0),
        "tagged_unresolved": actions.get("tagged_unresolved", 0),
        "already_valid": actions.get("valid", 0),
        "duplicates_merged": duplicates_merged,
        "name_changes": name_changes,
    }

    if not dry_run:
        # Backup original (only if no backup exists yet)
        backup = disease_dir / "validated_triples.jsonl.pre_cleanup"
        if not backup.exists():
            # Check if .bak exists (from consolidate_entities.py)
            bak = disease_dir / "validated_triples.jsonl.bak"
            if not bak.exists():
                import shutil
                shutil.copy2(vt_file, backup)

        with open(vt_file, "w") as f:
            for edge in consolidated:
                f.write(json.dumps(edge, default=str) + "\n")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Triple-level cleanup: type mismatches + dedup")
    parser.add_argument("--disease", type=str, help="Process single disease dir (e.g., MONDO_10000)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    if args.disease:
        disease_dir = EXTRACTED_DIR / args.disease
        if not disease_dir.exists():
            logger.error("Disease directory not found: %s", disease_dir)
            sys.exit(1)
        dirs = [disease_dir]
    else:
        dirs = sorted(d for d in EXTRACTED_DIR.iterdir()
                      if d.is_dir() and (d / "validated_triples.jsonl").exists())

    logger.info("Cleaning %d disease(s)%s...", len(dirs), " (dry run)" if args.dry_run else "")

    totals = defaultdict(int)
    disease_count = 0

    for disease_dir in dirs:
        stats = cleanup_disease(disease_dir, dry_run=args.dry_run)
        if stats["status"] == "success":
            disease_count += 1
            for k, v in stats.items():
                if isinstance(v, (int, float)) and k != "status":
                    totals[k] += v

    logger.info("\n" + "=" * 60)
    logger.info("TRIPLE CLEANUP SUMMARY")
    logger.info("=" * 60)
    logger.info("Diseases processed:    %d", disease_count)
    logger.info("Original triples:      %d", totals["original_triples"])
    logger.info("Cleaned triples:       %d", totals["cleaned_triples"])
    logger.info("  Already valid:       %d", totals["already_valid"])
    logger.info("  Reclassified:        %d", totals["reclassified"])
    logger.info("  Swapped:             %d", totals["swapped"])
    logger.info("  Tagged unresolved:   %d", totals["tagged_unresolved"])
    logger.info("  Duplicates merged:   %d", totals["duplicates_merged"])
    logger.info("  Name normalizations: %d", totals["name_changes"])
    if totals["original_triples"] > 0:
        reduction = (totals["original_triples"] - totals["cleaned_triples"]) / totals["original_triples"] * 100
        logger.info("  Overall reduction:   %.1f%%", reduction)

    # Save stats
    stats_file = PROJECT_ROOT / "data" / "benchmark" / "triple_cleanup_stats.json"
    stats_data = {
        "diseases_processed": disease_count,
        **{k: v for k, v in totals.items()},
    }
    if not args.dry_run:
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_file, "w") as f:
            json.dump(stats_data, f, indent=2)
        logger.info("\nStats saved to %s", stats_file)


if __name__ == "__main__":
    main()
