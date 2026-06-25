"""
Retrieval indexes for ChronoMedKG — Tier 1.3
==============================================
PMIDEdgeIndex: inverted index from PMID → edge IDs for citation-grounded retrieval.
AdjacencyIndex: entity-name → edge IDs for k-hop BFS in GraphRAG.

Both are built in a single pass over a list[TemporalEdge] and can be persisted
with pickle (same pattern as PrimeKGIndex).
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from core.models import TemporalEdge


@dataclass
class PMIDEdgeIndex:
    """Inverted index: PMID → list of edge_ids that cite it."""
    pmid_to_edge_ids: dict[str, list[str]] = field(default_factory=dict)
    edge_id_to_edge: dict[str, TemporalEdge] = field(default_factory=dict)

    @classmethod
    def build(cls, edges: list[TemporalEdge]) -> "PMIDEdgeIndex":
        pmid_to_edge_ids: dict[str, list[str]] = defaultdict(list)
        edge_id_to_edge: dict[str, TemporalEdge] = {}
        for e in edges:
            eid = e.edge_id
            edge_id_to_edge[eid] = e
            for pmid in e.evidence.source_ids:
                if pmid:
                    pmid_to_edge_ids[pmid].append(eid)
        return cls(dict(pmid_to_edge_ids), edge_id_to_edge)

    def edges_for_pmid(self, pmid: str) -> list[TemporalEdge]:
        return [self.edge_id_to_edge[eid]
                for eid in self.pmid_to_edge_ids.get(pmid, [])
                if eid in self.edge_id_to_edge]

    def to_pickle(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, path: Path) -> "PMIDEdgeIndex":
        with open(path, "rb") as f:
            return pickle.load(f)


@dataclass
class AdjacencyIndex:
    """Forward + reverse adjacency: entity_name_lower → list of edge_ids.

    Built alongside PMIDEdgeIndex in a single pass to avoid traversing edges twice.
    """
    entity_to_edge_ids: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def build(cls, edges: list[TemporalEdge]) -> "AdjacencyIndex":
        entity_to_edge_ids: dict[str, list[str]] = defaultdict(list)
        for e in edges:
            eid = e.edge_id
            entity_to_edge_ids[e.source_name.lower()].append(eid)
            entity_to_edge_ids[e.target_name.lower()].append(eid)
        return cls(dict(entity_to_edge_ids))

    def edges_for_entity(self, name: str) -> list[str]:
        return self.entity_to_edge_ids.get(name.lower(), [])

    def to_pickle(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, path: Path) -> "AdjacencyIndex":
        with open(path, "rb") as f:
            return pickle.load(f)


def build_indexes(
    edges: list[TemporalEdge],
) -> tuple[PMIDEdgeIndex, AdjacencyIndex]:
    """Build both indexes in a single pass. Use this instead of calling .build() twice."""
    pmid_to_edge_ids: dict[str, list[str]] = defaultdict(list)
    edge_id_to_edge: dict[str, TemporalEdge] = {}
    entity_to_edge_ids: dict[str, list[str]] = defaultdict(list)

    for e in edges:
        eid = e.edge_id
        edge_id_to_edge[eid] = e
        for pmid in e.evidence.source_ids:
            if pmid:
                pmid_to_edge_ids[pmid].append(eid)
        entity_to_edge_ids[e.source_name.lower()].append(eid)
        entity_to_edge_ids[e.target_name.lower()].append(eid)

    pmid_index = PMIDEdgeIndex(dict(pmid_to_edge_ids), edge_id_to_edge)
    adj_index = AdjacencyIndex(dict(entity_to_edge_ids))
    return pmid_index, adj_index
