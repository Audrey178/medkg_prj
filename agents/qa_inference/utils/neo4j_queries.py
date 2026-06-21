"""
Neo4j Cypher query templates for KG-RAG retrieval.

Node schema   : {name, entity_id, entity_type}  (labels: Disease, Drug, GeneProtein, ...)
Relationship  : {edge_id, relation, credibility_score, pmids, onset_age_min,
                 onset_age_max, temporal_qualifier, progression_stage,
                 evidence_text, quality_grade, study_type, is_retracted}
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING



# ---------------------------------------------------------------------------
# Driver factory
# ---------------------------------------------------------------------------

def get_driver():
    """Create a Neo4j driver from environment variables."""
    from neo4j import GraphDatabase  # local import — optional dependency

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    return GraphDatabase.driver(uri, auth=(user, password))


# ---------------------------------------------------------------------------
# Fetch all entities (for matching candidates + FAISS index)
# ---------------------------------------------------------------------------

_FETCH_ALL_ENTITIES = """
MATCH (n)
WHERE n.name IS NOT NULL AND n.entity_id IS NOT NULL
RETURN n.entity_id AS entity_id,
       n.name       AS name,
       n.entity_type AS entity_type
"""

def fetch_all_entities(driver) -> list[dict]:
    """
    Return all nodes as matching candidates.
    Result shape: [{cui, name, entity_type, aliases}]
    'aliases' is empty list — Neo4j schema has no alias field.
    """
    with driver.session() as session:
        records = session.run(_FETCH_ALL_ENTITIES)
        return [
            {
                "cui": r["entity_id"],
                "name": r["name"] or "",
                "entity_type": r["entity_type"] or "",
                "aliases": [],
            }
            for r in records
            if r["entity_id"] and r["name"]
        ]


# ---------------------------------------------------------------------------
# Fetch entity neighborhood (for context retrieval)
# Two separate queries with LIMIT — Cypher doesn't support dynamic list slicing
# ---------------------------------------------------------------------------

_FETCH_ENTITY_META = """
MATCH (n)
WHERE n.entity_id = $entity_id
RETURN n.entity_id AS entity_id, n.name AS name, n.entity_type AS entity_type
LIMIT 1
"""

_FETCH_OUTGOING = """
MATCH (n)-[r]->(m)
WHERE n.entity_id = $entity_id
RETURN 'outgoing'    AS direction,
       type(r)       AS relation,
       m.name        AS target_name,
       m.entity_id   AS target_id,
       m.entity_type AS target_type,
       r.credibility_score  AS credibility_score,
       r.pmids              AS pmids,
       r.onset_age_min      AS onset_age_min,
       r.onset_age_max      AS onset_age_max,
       r.temporal_qualifier AS temporal_qualifier,
       r.progression_stage  AS progression_stage,
       r.evidence_text      AS evidence_text,
       r.quality_grade      AS quality_grade,
       r.study_type         AS study_type,
       r.is_retracted       AS is_retracted
LIMIT $limit
"""

_FETCH_INCOMING = """
MATCH (k)-[r]->(n)
WHERE n.entity_id = $entity_id
RETURN 'incoming'    AS direction,
       type(r)       AS relation,
       k.name        AS source_name,
       k.entity_id   AS source_id,
       k.entity_type AS source_type,
       r.credibility_score  AS credibility_score,
       r.pmids              AS pmids,
       r.onset_age_min      AS onset_age_min,
       r.onset_age_max      AS onset_age_max,
       r.temporal_qualifier AS temporal_qualifier,
       r.progression_stage  AS progression_stage,
       r.evidence_text      AS evidence_text,
       r.quality_grade      AS quality_grade,
       r.study_type         AS study_type,
       r.is_retracted       AS is_retracted
LIMIT $limit
"""


def fetch_entity_neighborhood(
    driver,
    entity_id: str,
    max_rels: int = 80,
) -> tuple[dict, list[dict]]:
    """
    Return (entity_meta, relationships) for a single entity.

    entity_meta: {entity_id, name, entity_type}
    relationships: list of rel dicts (direction, relation, neighbour info, edge props)
    """
    half = max_rels // 2

    with driver.session() as session:
        meta_rec = session.run(_FETCH_ENTITY_META, entity_id=entity_id).single()
        if meta_rec is None:
            return {}, []

        entity_meta = {
            "entity_id": meta_rec["entity_id"],
            "name": meta_rec["name"],
            "entity_type": meta_rec["entity_type"],
        }

        rels: list[dict] = []
        for rec in session.run(_FETCH_OUTGOING, entity_id=entity_id, limit=half):
            rels.append(dict(rec))
        for rec in session.run(_FETCH_INCOMING, entity_id=entity_id, limit=half):
            rels.append(dict(rec))

        return entity_meta, rels


# ---------------------------------------------------------------------------
# Extract PMIDs from a batch of relationships
# ---------------------------------------------------------------------------

def extract_pmids(relationships: list[dict]) -> list[str]:
    """Collect unique PMIDs from a list of relationship dicts."""
    seen: set[str] = set()
    for r in relationships:
        raw = r.get("pmids") or ""
        for pmid in str(raw).split(","):
            pmid = pmid.strip()
            if pmid and pmid not in seen:
                seen.add(pmid)
    return sorted(seen)
