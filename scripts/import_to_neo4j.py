#!/usr/bin/env python3
"""
Import validated_triples.jsonl → Neo4j local instance
======================================================

Graph schema
------------
Nodes  : Disease | Drug | Phenotype | GeneProtein | BiologicalProcess |
         Exposure | Anatomy | Pathway | CellularComponent
         Properties: name, entity_id, entity_type

Edges  : DISEASE_PHENOTYPE_POSITIVE | DISEASE_PHENOTYPE_NEGATIVE |
         DISEASE_DISEASE | DISEASE_PROTEIN | INDICATION | CONTRAINDICATION |
         DRUG_EFFECT | DRUG_PROTEIN | BIOPROCESS_PROTEIN | PHENOTYPE_PROTEIN |
         PROTEIN_PROTEIN | PATHWAY_PROTEIN | OTHER
         Properties: edge_id, quality_grade, credibility_score, tier,
                     study_type, consensus_confidence, extraction_models,
                     pmids, evidence_text, is_retracted,
                     onset_age_min, onset_age_max, progression_stage,
                     temporal_qualifier, extraction_date, pipeline_version,
                     disease_profile_id

Prerequisites
-------------
  1. Neo4j running:  sudo systemctl start neo4j   (or neo4j start)
  2. Fill in .env:   NEO4J_PASSWORD=<your password>
  3. pip install neo4j (already in requirements.txt)

Usage
-----
  python -m scripts.import_to_neo4j
  python -m scripts.import_to_neo4j --input data/extracted/validated_triples.jsonl
  python -m scripts.import_to_neo4j --batch-size 2000 --wipe
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_env = PROJECT_ROOT / ".env"
if _env.exists():
    for _line in open(_env):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            if _v.strip():
                os.environ[_k.strip()] = _v.strip()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("neo4j_import")

# ---------------------------------------------------------------------------
# Schema mapping
# ---------------------------------------------------------------------------

_TYPE_TO_LABEL: dict[str, str] = {
    "disease":              "Disease",
    "drug":                 "Drug",
    "phenotype":            "Phenotype",
    "gene/protein":         "GeneProtein",
    "biological_process":   "BiologicalProcess",
    "exposure":             "Exposure",
    "anatomy":              "Anatomy",
    "pathway":              "Pathway",
    "cellular_component":   "CellularComponent",
}

_REL_TYPE: dict[str, str] = {
    "disease_phenotype_positive":  "DISEASE_PHENOTYPE_POSITIVE",
    "disease_phenotype_negative":  "DISEASE_PHENOTYPE_NEGATIVE",
    "disease_disease":             "DISEASE_DISEASE",
    "disease_protein":             "DISEASE_PROTEIN",
    "indication":                  "INDICATION",
    "contraindication":            "CONTRAINDICATION",
    "drug_effect":                 "DRUG_EFFECT",
    "drug_protein":                "DRUG_PROTEIN",
    "bioprocess_protein":          "BIOPROCESS_PROTEIN",
    "phenotype_protein":           "PHENOTYPE_PROTEIN",
    "protein_protein":             "PROTEIN_PROTEIN",
    "pathway_protein":             "PATHWAY_PROTEIN",
    "other":                       "OTHER",
}


def _entity_label(entity_type: str) -> str:
    return _TYPE_TO_LABEL.get(entity_type.lower(), "Entity")


def _rel_type(relation: str) -> str:
    return _REL_TYPE.get(relation.lower(), "OTHER")


# ---------------------------------------------------------------------------
# Triple parsing
# ---------------------------------------------------------------------------

def _parse_triple(t: dict) -> tuple[dict, dict, dict] | None:
    """Return (src_node, tgt_node, edge_props) or None if triple is malformed."""
    src_name = (t.get("source_name") or "").strip()
    tgt_name = (t.get("target_name") or "").strip()
    if not src_name or not tgt_name:
        return None

    src = {
        "name": src_name,
        "entity_id": (t.get("source_id") or src_name).strip(),
        "entity_type": (t.get("source_type") or "").strip(),
        "label": _entity_label(t.get("source_type") or ""),
    }
    tgt = {
        "name": tgt_name,
        "entity_id": (t.get("target_id") or tgt_name).strip(),
        "entity_type": (t.get("target_type") or "").strip(),
        "label": _entity_label(t.get("target_type") or ""),
    }

    ev = t.get("evidence") or {}
    temporal = t.get("temporal") or {}

    edge = {
        "edge_id":              t.get("edge_id", ""),
        "relation":             t.get("relation", ""),
        "quality_grade":        t.get("quality_grade", ""),
        "extraction_date":      t.get("extraction_date", ""),
        "pipeline_version":     t.get("pipeline_version", ""),
        "disease_profile_id":   t.get("disease_profile_id", ""),
        # evidence
        "credibility_score":    ev.get("credibility_score"),
        "tier":                 ev.get("tier"),
        "study_type":           ev.get("study_type", ""),
        "consensus_confidence": ev.get("consensus_confidence"),
        "extraction_models":    ",".join(ev.get("extraction_models") or []),
        "pmids":                ",".join(ev.get("source_ids") or []),
        "evidence_text":        (ev.get("evidence_text") or "")[:500],
        "is_retracted":         ev.get("is_retracted", False),
        # temporal
        "onset_age_min":        temporal.get("onset_age_min"),
        "onset_age_max":        temporal.get("onset_age_max"),
        "progression_stage":    temporal.get("progression_stage", ""),
        "temporal_qualifier":   temporal.get("temporal_qualifier", ""),
        "duration":             temporal.get("duration", ""),
    }
    # Strip None values to keep Neo4j properties clean
    edge = {k: v for k, v in edge.items() if v is not None and v != ""}

    return src, tgt, edge


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------

def _create_indexes(session) -> None:
    labels = list(_TYPE_TO_LABEL.values()) + ["Entity"]
    for label in labels:
        session.run(
            f"CREATE INDEX {label.lower()}_name IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.name)"
        )
    session.run(
        "CREATE INDEX edge_id IF NOT EXISTS "
        "FOR ()-[r:DISEASE_PHENOTYPE_POSITIVE]-() ON (r.edge_id)"
    )
    logger.info("Indexes created (or already exist)")


def _merge_nodes_batch(session, batch: list[dict]) -> None:
    """MERGE a batch of node dicts (each has label, name, entity_id, entity_type)."""
    # Group by label so we can use a single UNWIND per label
    by_label: dict[str, list[dict]] = {}
    for node in batch:
        by_label.setdefault(node["label"], []).append(node)

    for label, nodes in by_label.items():
        session.run(
            f"UNWIND $nodes AS n "
            f"MERGE (e:{label} {{name: n.name}}) "
            f"ON CREATE SET e.entity_id = n.entity_id, e.entity_type = n.entity_type",
            nodes=nodes,
        )


def _merge_edges_batch(session, batch: list[tuple[dict, dict, dict]]) -> None:
    """MERGE edges for a batch of (src, tgt, edge_props) tuples.

    Neo4j does not support dynamic relationship types in a single query,
    so we group by rel_type and issue one UNWIND per type.
    """
    by_rel: dict[str, list[dict]] = {}
    for src, tgt, edge in batch:
        rtype = _rel_type(edge.get("relation", ""))
        src_label = src["label"]
        tgt_label = tgt["label"]
        key = (rtype, src_label, tgt_label)
        by_rel.setdefault(key, []).append({
            "src_name": src["name"],
            "tgt_name": tgt["name"],
            **edge,
        })

    for (rtype, src_label, tgt_label), rows in by_rel.items():
        session.run(
            f"UNWIND $rows AS r "
            f"MATCH (src:{src_label} {{name: r.src_name}}) "
            f"MATCH (tgt:{tgt_label} {{name: r.tgt_name}}) "
            f"MERGE (src)-[e:{rtype} {{edge_id: r.edge_id}}]->(tgt) "
            f"ON CREATE SET e += r",
            rows=rows,
        )


# ---------------------------------------------------------------------------
# Main import
# ---------------------------------------------------------------------------

def load_triples(path: Path) -> list[tuple[dict, dict, dict]]:
    parsed = []
    skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            result = _parse_triple(t)
            if result is None:
                skipped += 1
            else:
                parsed.append(result)
    if skipped:
        logger.warning("Skipped %d malformed triples", skipped)
    return parsed


def run_import(args: argparse.Namespace) -> None:
    try:
        from neo4j import GraphDatabase
    except ImportError:
        logger.error("neo4j package not installed. Run: pip install neo4j")
        sys.exit(1)

    uri      = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user     = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    if not password:
        logger.error(
            "NEO4J_PASSWORD not set. Edit .env and set the actual password."
        )
        sys.exit(1)

    logger.info("Connecting to %s (db=%s)", uri, database)
    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        with driver.session(database=database) as session:
            session.run("RETURN 1")
        logger.info("Connection OK")
    except Exception as e:
        logger.error("Cannot connect to Neo4j: %s", e)
        logger.error("Start Neo4j first:  sudo systemctl start neo4j")
        driver.close()
        sys.exit(1)

    # Load
    input_path = Path(args.input)
    logger.info("Loading triples from %s", input_path)
    triples = load_triples(input_path)
    logger.info("Parsed %d valid triples", len(triples))

    with driver.session(database=database) as session:
        if args.wipe:
            logger.warning("Wiping database %s ...", database)
            session.run("MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS")
            logger.info("Database wiped")

        _create_indexes(session)

    # --- Import nodes ---
    logger.info("Importing nodes ...")
    all_nodes: list[dict] = []
    seen_nodes: set[tuple[str, str]] = set()
    for src, tgt, _ in triples:
        for node in (src, tgt):
            key = (node["name"], node["label"])
            if key not in seen_nodes:
                seen_nodes.add(key)
                all_nodes.append(node)

    logger.info("Unique nodes: %d", len(all_nodes))
    bs = args.batch_size
    t0 = time.time()
    for i in range(0, len(all_nodes), bs):
        batch = all_nodes[i : i + bs]
        with driver.session(database=database) as session:
            _merge_nodes_batch(session, batch)
        if (i // bs + 1) % 20 == 0:
            pct = 100 * (i + len(batch)) / len(all_nodes)
            logger.info("  Nodes: %d/%d  (%.0f%%)", i + len(batch), len(all_nodes), pct)

    logger.info("Nodes done in %.1fs", time.time() - t0)

    # --- Import edges ---
    logger.info("Importing edges ...")
    t0 = time.time()
    edge_bs = max(1, bs // 4)  # smaller batch for edges (more properties)
    for i in range(0, len(triples), edge_bs):
        batch = triples[i : i + edge_bs]
        with driver.session(database=database) as session:
            _merge_edges_batch(session, batch)
        if (i // edge_bs + 1) % 50 == 0:
            pct = 100 * (i + len(batch)) / len(triples)
            logger.info("  Edges: %d/%d  (%.0f%%)", i + len(batch), len(triples), pct)

    logger.info("Edges done in %.1fs", time.time() - t0)

    # --- Summary ---
    with driver.session(database=database) as session:
        n_nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        n_edges = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

    logger.info("Import complete: %d nodes, %d edges in %s", n_nodes, n_edges, database)
    driver.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import ChronoMedKG validated triples into Neo4j"
    )
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "extracted" / "validated_triples.jsonl"),
        help="Path to validated_triples.jsonl",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000,
        help="Node MERGE batch size (default: 1000)",
    )
    parser.add_argument(
        "--wipe", action="store_true",
        help="Delete all existing nodes/edges before importing",
    )
    args = parser.parse_args()
    run_import(args)


if __name__ == "__main__":
    main()
