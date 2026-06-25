"""
PrimeKG Schema Alignment
=========================
Loads PrimeKG edges into an indexed structure for:
1. Confirming extracted triples against known PrimeKG edges
2. Resolving entity names to PrimeKG node IDs
3. Mapping carrier relations (from LLM extraction) to PrimeKG relation types
4. Finding disease-specific subgraphs for context enrichment

This is the bridge between free-text LLM extraction and the structured PrimeKG schema.
"""

from __future__ import annotations

import csv
import logging
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.models import (
    CARRIER_TO_PRIMEKG,
    PrimeKGNodeType,
    PrimeKGRelationType,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PrimeKGNode:
    """A node in PrimeKG."""
    node_id: str
    node_type: str  # raw PrimeKG type string
    name: str
    source: str

    @property
    def name_lower(self) -> str:
        return self.name.lower().strip()


@dataclass
class PrimeKGEdge:
    """A single PrimeKG edge (no temporal/evidence metadata — that's what we add)."""
    relation: str
    display_relation: str
    x_id: str
    x_type: str
    x_name: str
    x_source: str
    y_id: str
    y_type: str
    y_name: str
    y_source: str


@dataclass
class ConfirmationResult:
    """Result of checking an extracted triple against PrimeKG."""
    is_confirmed: bool = False
    matching_edges: list[PrimeKGEdge] = field(default_factory=list)
    match_type: str = "none"  # "exact", "fuzzy_name", "same_relation", "related_entity"
    primekg_x_id: Optional[str] = None
    primekg_y_id: Optional[str] = None


class PrimeKGIndex:
    """
    Indexed PrimeKG for fast lookup during quality control.

    Indexes built:
    - name_to_nodes: entity name (lower) → list of PrimeKGNode
    - disease_edges: disease node_id → list of PrimeKGEdge (all edges involving that disease)
    - edge_lookup: (x_name_lower, relation, y_name_lower) → list of PrimeKGEdge
    - relation_pairs: (x_name_lower, y_name_lower) → set of relations
    """

    def __init__(self, kg_path: str | Path | None = None):
        self.kg_path = Path(kg_path) if kg_path else PROJECT_ROOT / "data" / "primekg" / "kg.csv"
        self.name_to_nodes: dict[str, list[PrimeKGNode]] = defaultdict(list)
        self.disease_edges: dict[str, list[PrimeKGEdge]] = defaultdict(list)
        self.edge_lookup: dict[tuple[str, str, str], list[PrimeKGEdge]] = defaultdict(list)
        self.relation_pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.disease_nodes: dict[str, PrimeKGNode] = {}  # id → node
        self._edge_lookup_lite: dict[tuple[str, str, str], tuple[str, str]] = {}  # lite: key → (x_id, y_id)
        self._lite_mode = False
        self._sqlite_mode = False
        self._db = None  # SQLite connection
        self._loaded = False

    def load(self, disease_ids: list[str] | None = None, lightweight: bool = True) -> None:
        """
        Load PrimeKG edges. Optionally filter to edges involving specific diseases.
        Uses a pickle cache for fast subsequent loads (~2s vs ~53s from CSV).

        Args:
            disease_ids: If provided, only load edges where x_id or y_id matches
                        a disease in this set. Dramatically reduces memory for
                        disease-focused analysis.
            lightweight: If True (default), use lightweight cache (~200MB) that stores
                        only (x_id, y_id) in edge_lookup instead of full PrimeKGEdge
                        objects, and skips disease_edges. Sufficient for extraction
                        pipeline (confirm_triple + entity normalization).
        """
        if self._loaded:
            return

        if not self.kg_path.exists():
            logger.warning("PrimeKG file not found: %s", self.kg_path)
            return

        # Try SQLite index first (~50MB RAM vs ~2GB+ for pickle)
        import sqlite3
        sqlite_path = self.kg_path.parent / "kg_index.db"
        if lightweight and sqlite_path.exists():
            try:
                t0 = time.monotonic()
                self._db = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
                self._db.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
                self._db.execute("PRAGMA cache_size=-50000")  # 50MB cache
                # Quick sanity check
                count = self._db.execute("SELECT COUNT(*) FROM edge_lookup").fetchone()[0]
                self._sqlite_mode = True
                self._lite_mode = True
                self._loaded = True
                logger.info("PrimeKG SQLite loaded in %.1fs (%d edge entries, ~50MB RAM)",
                            time.monotonic() - t0, count)
                return
            except Exception as e:
                logger.warning("SQLite load failed, falling back: %s", e)
                if self._db:
                    self._db.close()
                    self._db = None

        # Try full pickle cache (10-20x faster than CSV parsing)
        cache_path = self.kg_path.with_suffix(".pkl")
        if cache_path.exists() and cache_path.stat().st_mtime >= self.kg_path.stat().st_mtime:
            try:
                t0 = time.monotonic()
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                self.name_to_nodes = cached["name_to_nodes"]
                self.disease_edges = cached["disease_edges"]
                self.edge_lookup = cached["edge_lookup"]
                self.relation_pairs = cached["relation_pairs"]
                self.disease_nodes = cached["disease_nodes"]
                self._loaded = True
                logger.info("PrimeKG FULL loaded from cache in %.1fs (%d names, %d disease nodes)",
                            time.monotonic() - t0, len(self.name_to_nodes), len(self.disease_nodes))
                return
            except Exception as e:
                logger.warning("Cache load failed, rebuilding: %s", e)

        logger.info("Loading PrimeKG from %s...", self.kg_path)

        disease_id_set = set(disease_ids) if disease_ids else None
        edge_count = 0
        node_names_seen: set[str] = set()

        with open(self.kg_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                x_type = row["x_type"]
                y_type = row["y_type"]
                x_id = row["x_id"]
                y_id = row["y_id"]
                x_name = row["x_name"]
                y_name = row["y_name"]
                relation = row["relation"]

                # If filtering by disease, only keep edges touching our diseases
                is_disease_edge = (x_type == "disease" or y_type == "disease")
                if disease_id_set and is_disease_edge:
                    if x_id not in disease_id_set and y_id not in disease_id_set:
                        # Still index disease nodes by name for fuzzy matching
                        pass
                    # Always index disease node names regardless of filter

                edge = PrimeKGEdge(
                    relation=relation,
                    display_relation=row["display_relation"],
                    x_id=x_id,
                    x_type=x_type,
                    x_name=x_name,
                    x_source=row["x_source"],
                    y_id=y_id,
                    y_type=y_type,
                    y_name=y_name,
                    y_source=row["y_source"],
                )

                # Index nodes by name (for entity resolution)
                for nid, ntype, nname, nsource in [
                    (x_id, x_type, x_name, row["x_source"]),
                    (y_id, y_type, y_name, row["y_source"]),
                ]:
                    key = nname.lower().strip()
                    if key not in node_names_seen:
                        node_names_seen.add(key)
                        node = PrimeKGNode(node_id=nid, node_type=ntype, name=nname, source=nsource)
                        self.name_to_nodes[key].append(node)
                        if ntype == "disease":
                            self.disease_nodes[nid] = node

                # Index disease-specific edges
                if x_type == "disease":
                    self.disease_edges[x_id].append(edge)
                if y_type == "disease":
                    self.disease_edges[y_id].append(edge)

                # Index by (x_name, relation, y_name) for confirmation
                edge_key = (x_name.lower().strip(), relation, y_name.lower().strip())
                self.edge_lookup[edge_key].append(edge)

                # Index by entity pair → relations
                pair_key = (x_name.lower().strip(), y_name.lower().strip())
                self.relation_pairs[pair_key].add(relation)

                edge_count += 1

        self._loaded = True
        logger.info("PrimeKG loaded: %d edges, %d unique names, %d disease nodes",
                     edge_count, len(self.name_to_nodes), len(self.disease_nodes))

        # Save pickle cache for fast subsequent loads
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "name_to_nodes": dict(self.name_to_nodes),
                    "disease_edges": dict(self.disease_edges),
                    "edge_lookup": dict(self.edge_lookup),
                    "relation_pairs": dict(self.relation_pairs),
                    "disease_nodes": self.disease_nodes,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("PrimeKG cache saved to %s (%.1f MB)",
                         cache_path, cache_path.stat().st_size / 1e6)
        except Exception as e:
            logger.warning("Failed to save cache: %s", e)

    def resolve_name(self, name: str, entity_type: str | None = None) -> list[PrimeKGNode]:
        """
        Resolve an entity name to PrimeKG nodes.

        Args:
            name: Entity name (case-insensitive)
            entity_type: Optional type filter (e.g., "disease", "gene/protein")

        Returns:
            List of matching PrimeKGNode objects
        """
        key = name.lower().strip()

        if self._sqlite_mode:
            return self._sqlite_resolve_name(key, entity_type)

        candidates = self.name_to_nodes.get(key, [])

        if entity_type:
            type_lower = entity_type.lower().strip()
            # Map our types to PrimeKG types
            type_map = {
                "gene": "gene/protein",
                "protein": "gene/protein",
                "drug": "drug",
                "treatment": "drug",
                "phenotype": "effect/phenotype",
                "symptom": "effect/phenotype",
                "effect/phenotype": "effect/phenotype",
            }
            primekg_type = type_map.get(type_lower, type_lower)
            candidates = [n for n in candidates if n.node_type == primekg_type]

        return candidates

    def _sqlite_resolve_name(self, key: str, entity_type: str | None = None) -> list[PrimeKGNode]:
        """SQLite-backed name resolution."""
        type_map = {
            "gene": "gene/protein", "protein": "gene/protein",
            "drug": "drug", "treatment": "drug",
            "phenotype": "effect/phenotype", "symptom": "effect/phenotype",
            "effect/phenotype": "effect/phenotype",
        }
        if entity_type:
            primekg_type = type_map.get(entity_type.lower().strip(), entity_type.lower().strip())
            rows = self._db.execute(
                "SELECT node_id, node_type, name, source FROM name_to_nodes WHERE name_lower=? AND node_type=?",
                (key, primekg_type)
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT node_id, node_type, name, source FROM name_to_nodes WHERE name_lower=?",
                (key,)
            ).fetchall()
        return [PrimeKGNode(node_id=r[0], node_type=r[1], name=r[2], source=r[3]) for r in rows]

    def fuzzy_resolve_name(self, name: str, entity_type: str | None = None) -> list[PrimeKGNode]:
        """
        Fuzzy resolve entity name — checks substrings and common variants.
        More expensive than exact resolve, use as fallback.
        """
        key = name.lower().strip()

        # Exact match first
        exact = self.resolve_name(name, entity_type)
        if exact:
            return exact

        if self._sqlite_mode:
            return self._sqlite_fuzzy_resolve_nodes(key, entity_type)

        # Substring matching: check if query is contained in any PrimeKG name
        candidates = []
        for stored_name, nodes in self.name_to_nodes.items():
            if key in stored_name or stored_name in key:
                candidates.extend(nodes)

        if entity_type:
            type_lower = entity_type.lower().strip()
            type_map = {
                "gene": "gene/protein", "protein": "gene/protein",
                "drug": "drug", "phenotype": "effect/phenotype",
                "symptom": "effect/phenotype",
            }
            primekg_type = type_map.get(type_lower, type_lower)
            candidates = [n for n in candidates if n.node_type == primekg_type]

        return candidates[:10]  # Limit fuzzy results

    def confirm_triple(
        self,
        subject: str,
        relation: str,
        obj: str,
        subject_type: str | None = None,
        object_type: str | None = None,
    ) -> ConfirmationResult:
        """
        Check if an extracted triple is confirmed by PrimeKG.

        Checks (in order of strength):
        1. Exact match: (subject, relation, object) all match
        2. Same entities, different relation: entities match, relation differs
        3. Fuzzy name match: entity names partially match

        Returns ConfirmationResult with match details.
        """
        result = ConfirmationResult()
        subj_lower = subject.lower().strip()
        obj_lower = obj.lower().strip()
        rel_lower = relation.lower().strip()

        # Map carrier relation to PrimeKG relation
        primekg_rel = CARRIER_TO_PRIMEKG.get(rel_lower)
        primekg_rel_str = primekg_rel.value if primekg_rel else rel_lower

        # SQLite mode: all lookups go to disk
        if self._sqlite_mode:
            return self._confirm_triple_sqlite(subj_lower, primekg_rel_str, obj_lower,
                                                subject_type, object_type)

        # In-memory mode (full pkl or lite pkl)
        _lookup = self._edge_lookup_lite if self._lite_mode else self.edge_lookup

        # 1. Exact match (subject, primekg_relation, object)
        exact_key = (subj_lower, primekg_rel_str, obj_lower)
        if exact_key in _lookup:
            result.is_confirmed = True
            result.match_type = "exact"
            if self._lite_mode:
                result.primekg_x_id, result.primekg_y_id = _lookup[exact_key]
            else:
                result.matching_edges = _lookup[exact_key]
                if result.matching_edges:
                    result.primekg_x_id = result.matching_edges[0].x_id
                    result.primekg_y_id = result.matching_edges[0].y_id
            return result

        # Also check reverse direction
        reverse_key = (obj_lower, primekg_rel_str, subj_lower)
        if reverse_key in _lookup:
            result.is_confirmed = True
            result.match_type = "exact"
            if self._lite_mode:
                x_id, y_id = _lookup[reverse_key]
                result.primekg_x_id = y_id
                result.primekg_y_id = x_id
            else:
                result.matching_edges = _lookup[reverse_key]
                if result.matching_edges:
                    result.primekg_x_id = result.matching_edges[0].y_id
                    result.primekg_y_id = result.matching_edges[0].x_id
            return result

        # 2. Same entities, any relation
        pair_key = (subj_lower, obj_lower)
        if pair_key in self.relation_pairs:
            result.is_confirmed = True
            result.match_type = "same_relation"
            if self._lite_mode:
                for rel in self.relation_pairs[pair_key]:
                    ids = _lookup.get((subj_lower, rel, obj_lower))
                    if ids:
                        result.primekg_x_id, result.primekg_y_id = ids
                        break
            else:
                for rel in self.relation_pairs[pair_key]:
                    edges = _lookup.get((subj_lower, rel, obj_lower), [])
                    result.matching_edges.extend(edges)
                if result.matching_edges:
                    result.primekg_x_id = result.matching_edges[0].x_id
                    result.primekg_y_id = result.matching_edges[0].y_id
            return result

        # Check reverse pair
        reverse_pair = (obj_lower, subj_lower)
        if reverse_pair in self.relation_pairs:
            result.is_confirmed = True
            result.match_type = "same_relation"
            if self._lite_mode:
                for rel in self.relation_pairs[reverse_pair]:
                    ids = _lookup.get((obj_lower, rel, subj_lower))
                    if ids:
                        result.primekg_x_id = ids[1]
                        result.primekg_y_id = ids[0]
                        break
            else:
                for rel in self.relation_pairs[reverse_pair]:
                    edges = _lookup.get((obj_lower, rel, subj_lower), [])
                    result.matching_edges.extend(edges)
                if result.matching_edges:
                    result.primekg_x_id = result.matching_edges[0].y_id
                    result.primekg_y_id = result.matching_edges[0].x_id
            return result

        # 3. Fuzzy: resolve both entities and check if any pair exists
        subj_nodes = self.fuzzy_resolve_name(subject, subject_type)
        obj_nodes = self.fuzzy_resolve_name(obj, object_type)

        for sn in subj_nodes:
            for on in obj_nodes:
                pair = (sn.name_lower, on.name_lower)
                if pair in self.relation_pairs:
                    result.is_confirmed = True
                    result.match_type = "fuzzy_name"
                    result.primekg_x_id = sn.node_id
                    result.primekg_y_id = on.node_id
                    return result
                # Reverse
                rpair = (on.name_lower, sn.name_lower)
                if rpair in self.relation_pairs:
                    result.is_confirmed = True
                    result.match_type = "fuzzy_name"
                    result.primekg_x_id = on.node_id
                    result.primekg_y_id = sn.node_id
                    return result

        return result  # No match found

    def _confirm_triple_sqlite(
        self, subj_lower: str, primekg_rel_str: str, obj_lower: str,
        subject_type: str | None, object_type: str | None,
    ) -> ConfirmationResult:
        """SQLite-backed confirm_triple — near-zero RAM."""
        result = ConfirmationResult()
        db = self._db

        # 1. Exact match
        row = db.execute(
            "SELECT x_id, y_id FROM edge_lookup WHERE x_name=? AND relation=? AND y_name=? LIMIT 1",
            (subj_lower, primekg_rel_str, obj_lower)
        ).fetchone()
        if row:
            result.is_confirmed = True
            result.match_type = "exact"
            result.primekg_x_id, result.primekg_y_id = row
            return result

        # Reverse direction
        row = db.execute(
            "SELECT x_id, y_id FROM edge_lookup WHERE x_name=? AND relation=? AND y_name=? LIMIT 1",
            (obj_lower, primekg_rel_str, subj_lower)
        ).fetchone()
        if row:
            result.is_confirmed = True
            result.match_type = "exact"
            result.primekg_x_id = row[1]
            result.primekg_y_id = row[0]
            return result

        # 2. Same entities, any relation
        row = db.execute(
            "SELECT e.x_id, e.y_id FROM relation_pairs rp "
            "JOIN edge_lookup e ON e.x_name=rp.x_name AND e.relation=rp.relation AND e.y_name=rp.y_name "
            "WHERE rp.x_name=? AND rp.y_name=? LIMIT 1",
            (subj_lower, obj_lower)
        ).fetchone()
        if row:
            result.is_confirmed = True
            result.match_type = "same_relation"
            result.primekg_x_id, result.primekg_y_id = row
            return result

        # Reverse pair
        row = db.execute(
            "SELECT e.x_id, e.y_id FROM relation_pairs rp "
            "JOIN edge_lookup e ON e.x_name=rp.x_name AND e.relation=rp.relation AND e.y_name=rp.y_name "
            "WHERE rp.x_name=? AND rp.y_name=? LIMIT 1",
            (obj_lower, subj_lower)
        ).fetchone()
        if row:
            result.is_confirmed = True
            result.match_type = "same_relation"
            result.primekg_x_id = row[1]
            result.primekg_y_id = row[0]
            return result

        # 3. Fuzzy match via name_to_nodes
        subj_nodes = self._sqlite_fuzzy_resolve(subj_lower, subject_type)
        obj_nodes = self._sqlite_fuzzy_resolve(obj_lower, object_type)

        for sn_name, sn_id in subj_nodes:
            for on_name, on_id in obj_nodes:
                exists = db.execute(
                    "SELECT 1 FROM relation_pairs WHERE x_name=? AND y_name=? LIMIT 1",
                    (sn_name, on_name)
                ).fetchone()
                if exists:
                    result.is_confirmed = True
                    result.match_type = "fuzzy_name"
                    result.primekg_x_id = sn_id
                    result.primekg_y_id = on_id
                    return result
                # Reverse
                exists = db.execute(
                    "SELECT 1 FROM relation_pairs WHERE x_name=? AND y_name=? LIMIT 1",
                    (on_name, sn_name)
                ).fetchone()
                if exists:
                    result.is_confirmed = True
                    result.match_type = "fuzzy_name"
                    result.primekg_x_id = on_id
                    result.primekg_y_id = sn_id
                    return result

        return result

    def _sqlite_fuzzy_resolve(self, name_lower: str, entity_type: str | None = None) -> list[tuple[str, str]]:
        """SQLite-backed fuzzy name resolution. Returns [(name_lower, node_id), ...]."""
        db = self._db
        # Exact match first
        if entity_type:
            type_map = {
                "gene": "gene/protein", "protein": "gene/protein",
                "drug": "drug", "phenotype": "effect/phenotype",
                "symptom": "effect/phenotype", "effect/phenotype": "effect/phenotype",
            }
            primekg_type = type_map.get(entity_type.lower().strip(), entity_type.lower().strip())
            rows = db.execute(
                "SELECT name_lower, node_id FROM name_to_nodes WHERE name_lower=? AND node_type=? LIMIT 10",
                (name_lower, primekg_type)
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT name_lower, node_id FROM name_to_nodes WHERE name_lower=? LIMIT 10",
                (name_lower,)
            ).fetchall()
        if rows:
            return rows

        # Substring match (LIKE is slower but rare — only hits on fuzzy fallback)
        pattern = f"%{name_lower}%"
        if entity_type:
            rows = db.execute(
                "SELECT name_lower, node_id FROM name_to_nodes WHERE name_lower LIKE ? AND node_type=? LIMIT 10",
                (pattern, primekg_type)
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT name_lower, node_id FROM name_to_nodes WHERE name_lower LIKE ? LIMIT 10",
                (pattern,)
            ).fetchall()
        return rows

    def _sqlite_fuzzy_resolve_nodes(self, name_lower: str, entity_type: str | None = None) -> list[PrimeKGNode]:
        """SQLite-backed fuzzy name resolution returning PrimeKGNode objects."""
        type_map = {
            "gene": "gene/protein", "protein": "gene/protein",
            "drug": "drug", "phenotype": "effect/phenotype",
            "symptom": "effect/phenotype", "effect/phenotype": "effect/phenotype",
        }
        if entity_type:
            primekg_type = type_map.get(entity_type.lower().strip(), entity_type.lower().strip())
            rows = self._db.execute(
                "SELECT node_id, node_type, name, source FROM name_to_nodes WHERE name_lower LIKE ? AND node_type=? LIMIT 10",
                (f"%{name_lower}%", primekg_type)
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT node_id, node_type, name, source FROM name_to_nodes WHERE name_lower LIKE ? LIMIT 10",
                (f"%{name_lower}%",)
            ).fetchall()
        return [PrimeKGNode(node_id=r[0], node_type=r[1], name=r[2], source=r[3]) for r in rows]

    def get_disease_subgraph(self, disease_id: str) -> list[PrimeKGEdge]:
        """Get all PrimeKG edges for a given disease."""
        return self.disease_edges.get(disease_id, [])

    def get_disease_neighbors(self, disease_id: str) -> dict[str, list[str]]:
        """Get neighbor entities grouped by relation type for a disease."""
        neighbors: dict[str, list[str]] = defaultdict(list)
        for edge in self.disease_edges.get(disease_id, []):
            if edge.x_id == disease_id:
                neighbors[edge.relation].append(edge.y_name)
            else:
                neighbors[edge.relation].append(edge.x_name)
        return dict(neighbors)

    def find_disease_by_name(self, name: str) -> Optional[PrimeKGNode]:
        """Find a disease node by name (exact or fuzzy)."""
        candidates = self.resolve_name(name, entity_type="disease")
        if candidates:
            return candidates[0]
        # Fuzzy
        candidates = self.fuzzy_resolve_name(name, entity_type="disease")
        if candidates:
            return candidates[0]
        return None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @classmethod
    def build_sqlite_index(
        cls,
        kg_csv_path: str | Path | None = None,
        sqlite_path: str | Path | None = None,
        batch_size: int = 50_000,
    ) -> Path:
        """Build kg_index.db from kg.csv using streaming CSV → SQLite.

        Processes rows one at a time — never holds the full 8.1M edge dataset
        in RAM. Requires ~100MB peak RAM regardless of KG size.
        Only needs to run ONCE; subsequent loads use SQLite (~50MB RAM).

        Returns the path to the created SQLite file.
        """
        import sqlite3

        kg_path = Path(kg_csv_path) if kg_csv_path else PROJECT_ROOT / "data" / "primekg" / "kg.csv"
        out_path = Path(sqlite_path) if sqlite_path else kg_path.parent / "kg_index.db"

        if not kg_path.exists():
            raise FileNotFoundError(f"PrimeKG CSV not found: {kg_path}")

        logger.info("Building SQLite index from %s → %s", kg_path, out_path)
        t0 = time.monotonic()

        conn = sqlite3.connect(str(out_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-128000")  # 128MB page cache during build

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS edge_lookup (
                x_name TEXT NOT NULL,
                relation TEXT NOT NULL,
                y_name TEXT NOT NULL,
                x_id TEXT NOT NULL,
                y_id TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS relation_pairs (
                x_name TEXT NOT NULL,
                y_name TEXT NOT NULL,
                relation TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS name_to_nodes (
                name_lower TEXT NOT NULL,
                node_id TEXT NOT NULL,
                node_type TEXT NOT NULL,
                name TEXT NOT NULL,
                source TEXT NOT NULL
            );
        """)

        edge_count = 0
        node_names_seen: set[str] = set()
        relation_pairs_seen: set[tuple[str, str]] = set()
        batch_edges: list[tuple] = []
        batch_pairs: list[tuple] = []
        batch_nodes: list[tuple] = []

        def _flush():
            if batch_edges:
                conn.executemany(
                    "INSERT INTO edge_lookup(x_name,relation,y_name,x_id,y_id) VALUES(?,?,?,?,?)",
                    batch_edges,
                )
                batch_edges.clear()
            if batch_pairs:
                conn.executemany(
                    "INSERT INTO relation_pairs(x_name,y_name,relation) VALUES(?,?,?)",
                    batch_pairs,
                )
                batch_pairs.clear()
            if batch_nodes:
                conn.executemany(
                    "INSERT INTO name_to_nodes(name_lower,node_id,node_type,name,source) VALUES(?,?,?,?,?)",
                    batch_nodes,
                )
                batch_nodes.clear()
            conn.commit()

        with open(kg_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                x_name = row["x_name"].lower().strip()
                y_name = row["y_name"].lower().strip()
                relation = row["relation"]
                x_id = row["x_id"]
                y_id = row["y_id"]
                x_type = row["x_type"]
                y_type = row["y_type"]

                batch_edges.append((x_name, relation, y_name, x_id, y_id))

                pair_key = (x_name, y_name)
                if pair_key not in relation_pairs_seen:
                    relation_pairs_seen.add(pair_key)
                    batch_pairs.append((x_name, y_name, relation))

                for nid, ntype, nname_raw, nsource in [
                    (x_id, x_type, row["x_name"], row["x_source"]),
                    (y_id, y_type, row["y_name"], row["y_source"]),
                ]:
                    key = nname_raw.lower().strip()
                    if key not in node_names_seen:
                        node_names_seen.add(key)
                        batch_nodes.append((key, nid, ntype, nname_raw, nsource))

                edge_count += 1
                if edge_count % batch_size == 0:
                    _flush()
                    logger.info("  %d edges processed...", edge_count)

        _flush()
        logger.info("Rows inserted: %d edges, %d relation_pairs, %d unique nodes",
                    edge_count, len(relation_pairs_seen), len(node_names_seen))

        # Build indexes AFTER bulk insert (much faster)
        logger.info("Building indexes...")
        conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_edge_x ON edge_lookup(x_name);
            CREATE INDEX IF NOT EXISTS idx_edge_xyz ON edge_lookup(x_name, relation, y_name);
            CREATE INDEX IF NOT EXISTS idx_edge_y ON edge_lookup(y_name);
            CREATE INDEX IF NOT EXISTS idx_pairs_xy ON relation_pairs(x_name, y_name);
            CREATE INDEX IF NOT EXISTS idx_pairs_yx ON relation_pairs(y_name, x_name);
            CREATE INDEX IF NOT EXISTS idx_nodes_name ON name_to_nodes(name_lower);
            CREATE INDEX IF NOT EXISTS idx_nodes_type ON name_to_nodes(name_lower, node_type);
        """)
        conn.execute("ANALYZE")
        conn.close()

        elapsed = time.monotonic() - t0
        size_mb = out_path.stat().st_size / 1e6
        logger.info("SQLite index built in %.0fs — %.0f MB at %s", elapsed, size_mb, out_path)
        return out_path
