"""
Agent 1: Disease Profiler
=========================
Given a disease identifier, autonomously builds a complete DiseaseProfile
that parameterizes all downstream agents.

Replaces Paper 1's hand-curated DISEASE_CONFIGS with autonomous profiling.

Steps:
1. Resolve identity (OMIM, Orphanet, MONDO cross-references)
2. Check Tier 1 source availability (GeneReviews, OMIM, Orphanet)
3. Profile PrimeKG neighborhood
4. Estimate literature coverage (PubMed counts)
5. Generate extraction strategy
6. Identify differential diagnosis partners (via LLM)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from agents.base_agent import BaseAgent
from core.models import (
    AgentResult,
    BioASQProfile,
    CoverageFlag,
    DiseaseProfile,
    RawTriple,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DiseaseProfiler(BaseAgent):
    """Agent 1: Autonomous disease profiling."""

    def __init__(self, config: dict, primekg_data: dict | None = None):
        super().__init__(config, logger)
        self.primekg_data = primekg_data or {} # primekg_data lấy ở đâu
        self.ncbi_api_key = os.environ.get("NCBI_API_KEY", "")
        self.omim_api_key = os.environ.get("OMIM_API_KEY", "")
        self._config_dir = PROJECT_ROOT / "config" / "diseases"
        self._config_dir.mkdir(parents=True, exist_ok=True)

    async def run(self, input_data: dict) -> AgentResult:
        """
        Build a DiseaseProfile for the given disease, or process a BioASQ batch.

        input_data (disease-driven mode):
            disease_id: str (OMIM ID, Orphanet code, or MONDO ID)
            disease_name: str
        input_data (bioasq_data mode):
            mode: "bioasq_data"
            file_path: str  path to BioASQ JSON {"questions": [...]}
        """
        if input_data.get("mode") == "bioasq_data":
            return await self._run_bioasq_mode(input_data)

        disease_id = input_data["disease_id"]
        disease_name = input_data["disease_name"]

        self.logger.info("Profiling disease: %s (%s)", disease_name, disease_id)

        profile = DiseaseProfile(
            disease_id=disease_id,
            disease_name=disease_name,
        )

        # Step 1: Resolve identity and cross-references
        self._resolve_identity(profile)

        # Step 2: Check Tier 1 source availability
        self._check_tier1_sources(profile)

        # Step 3: Profile PrimeKG neighborhood
        self._profile_primekg(profile)

        # Step 4: Estimate literature coverage
        self._estimate_literature(profile)

        # Step 5: Generate extraction strategy
        self._generate_strategy(profile)

        # Step 6: Identify differential diagnosis partners
        await self._identify_differentials(profile)

        # Step 7: Convert differential_diseases to tier1_relations (ontology-derived, no LLM)
        self._build_tier1_relations(profile)

        # Save as YAML config
        config_path = self._save_config(profile)

        status = "success" if profile.has_sufficient_sources() else "partial"
        return AgentResult(
            agent_name="DiseaseProfiler",
            disease_id=disease_id,
            status=status,
            data={
                "profile": profile.to_dict(),
                "config_path": str(config_path),
            },
            metrics={
                "has_genereviews": profile.has_genereviews,
                "has_omim": profile.has_omim,
                "has_orphanet": profile.has_orphanet,
                "pubmed_count": profile.pubmed_article_count,
                "primekg_edges": profile.primekg_neighbor_count,
                "coverage_flag": profile.coverage_flag.value,
                "sufficient_sources": profile.has_sufficient_sources(),
            },
            timestamp=datetime.utcnow(),
        )

    async def _run_bioasq_mode(self, input_data: dict) -> AgentResult:
        """Process a BioASQ Task B JSON file; return one BioASQProfile per question."""
        import json as _json

        file_path = input_data["file_path"]
        self.logger.info("BioASQ mode: reading %s", file_path)

        with open(file_path, encoding="utf-8") as fh:
            data = _json.load(fh)

        items = data.get("questions", [])
        total_items = len(items)
        profiles: list[BioASQProfile] = []
        error_count = 0

        for item in items:
            if not all(k in item for k in ("id", "body", "documents")):
                self.logger.warning("BioASQ item missing required fields (id/body/documents), skipping: %s",
                                    str(item)[:120])
                error_count += 1
                continue

            try:
                pmids = [url.rsplit("/", 1)[-1] for url in item["documents"]]
                pmid_set = set(pmids)

                snippet_pmids = set()
                for snip in item.get("snippets", []):
                    doc_url = snip.get("document", "")
                    if doc_url:
                        snippet_pmids.add(doc_url.rsplit("/", 1)[-1])

                pmids_with_snippet = [p for p in pmids if p in snippet_pmids]
                pmids_missing_snippet = [p for p in pmids if p not in snippet_pmids]

                ideal = item.get("ideal_answer", [])
                if isinstance(ideal, str):
                    ideal = [ideal]

                profile = BioASQProfile(
                    bioasq_id=item["id"],
                    question_body=item["body"],
                    question_type=item.get("type", ""),
                    pmids=pmids,
                    document_urls=list(item["documents"]),
                    snippets=item.get("snippets", []),
                    concepts=item.get("concepts", []),
                    ideal_answer=ideal,
                    pmids_with_snippet=pmids_with_snippet,
                    pmids_missing_snippet=pmids_missing_snippet,
                )
                profiles.append(profile)
            except Exception as exc:
                self.logger.warning("Failed to build BioASQProfile for item %s: %s",
                                    item.get("id", "?"), exc)
                error_count += 1

        if profiles and error_count == 0:
            status = "success"
        elif profiles:
            status = "partial"
        else:
            status = "failed"

        self.logger.info("BioASQ mode: %d/%d valid profiles, %d errors",
                         len(profiles), total_items, error_count)

        return AgentResult(
            agent_name="DiseaseProfiler",
            disease_id="BIOASQ_BATCH",
            status=status,
            data={"profiles": [p.to_dict() for p in profiles]},
            metrics={
                "total_samples": total_items,
                "valid_profiles": len(profiles),
                "skipped_invalid_item": error_count,
                "skipped_no_snippet_match": sum(len(p.pmids_missing_snippet) for p in profiles),
            },
            timestamp=datetime.utcnow(),
        )

    def _resolve_identity(self, profile: DiseaseProfile) -> None:
        """Resolve OMIM, Orphanet, MONDO cross-references."""
        did = profile.disease_id

        # Parse ID type
        if did.startswith("OMIM:"):
            profile.omim_id = did
        elif did.startswith("ORPHA:") or did.startswith("Orphanet:"):
            profile.orphanet_id = did
        elif did.startswith("MONDO:"):
            profile.mondo_id = did

        # Try OMIM API for cross-references
        if profile.omim_id and self.omim_api_key:
            self._query_omim(profile)

        self.logger.info(
            "Identity resolved: OMIM=%s, Orphanet=%s, MONDO=%s",
            profile.omim_id, profile.orphanet_id, profile.mondo_id,
        )

    def _query_omim(self, profile: DiseaseProfile) -> None:
        """Query OMIM API for disease details."""
        if not self.omim_api_key:
            return

        try:
            import urllib.request
            import urllib.parse

            omim_num = profile.omim_id.replace("OMIM:", "")
            url = (
                f"https://api.omim.org/api/entry?"
                f"mimNumber={omim_num}&include=all"
                f"&format=json&apiKey={self.omim_api_key}"
            )
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())

            entries = data.get("omim", {}).get("entryList", [])
            if entries:
                entry = entries[0].get("entry", {})
                titles = entry.get("titles", {})

                # Synonyms from alternative titles
                alt_titles = titles.get("alternativeTitles", "") 
                if alt_titles:
                    profile.synonyms.extend(
                        [t.strip() for t in alt_titles.split(";;") if t.strip()]
                    )

                # Key genes from gene map
                gene_map = entry.get("geneMap", {})
                if gene_map:
                    genes = gene_map.get("geneSymbols", "")
                    if genes:
                        profile.key_genes.extend(
                            [g.strip() for g in genes.split(",") if g.strip()]
                        )

                # Inheritance pattern
                gene_map_list = entry.get("phenotypeMapList", [])
                for pm in gene_map_list:
                    inh = pm.get("phenotypeMap", {}).get("phenotypeInheritance", "")
                    if inh and not profile.inheritance_pattern:
                        profile.inheritance_pattern = inh

                profile.has_omim = True
                self.logger.info("OMIM data fetched: %d synonyms, %d genes",
                                 len(profile.synonyms), len(profile.key_genes))

        except Exception as e:
            self.logger.warning("OMIM API query failed: %s", e)

    def _check_tier1_sources(self, profile: DiseaseProfile) -> None:
        """Check availability of Tier 1 sources."""
        # GeneReviews check via NCBI E-utilities
        self._check_genereviews(profile)

        # OMIM already checked in _query_omim
        if not profile.has_omim and profile.omim_id:
            profile.has_omim = True  # we have an OMIM ID at minimum

        # Orphanet: check if we have an Orphanet ID
        if profile.orphanet_id:
            profile.has_orphanet = True

        # Build tier1_sources list
        if profile.has_genereviews:
            profile.tier1_sources.append("genereviews")
        if profile.has_omim:
            profile.tier1_sources.append("omim")
        if profile.has_orphanet:
            profile.tier1_sources.append("orphanet")

    def _check_genereviews(self, profile: DiseaseProfile) -> None:
        """Check if disease has a GeneReviews entry via NCBI."""
        try:
            import urllib.request
            import urllib.parse
            import xml.etree.ElementTree as ET

            query = urllib.parse.quote(f'"{profile.disease_name}"[Title] AND "GeneReviews"[Book]')
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=books&term={query}&retmax=1"
            if self.ncbi_api_key:
                url += f"&api_key={self.ncbi_api_key}"

            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                xml_data = resp.read()

            root = ET.fromstring(xml_data)
            count = int(root.findtext("Count", "0"))
            if count > 0:
                id_list = root.find("IdList")
                if id_list is not None:
                    first_id = id_list.findtext("Id")
                    profile.has_genereviews = True
                    profile.genereviews_id = first_id
                    self.logger.info("GeneReviews entry found: %s", first_id)

            time.sleep(0.15 if self.ncbi_api_key else 0.4)

        except Exception as e:
            self.logger.debug("GeneReviews check failed: %s", e)

    def _profile_primekg(self, profile: DiseaseProfile) -> None:
        """Profile the disease's neighborhood in PrimeKG."""
        if not self.primekg_data:
            return

        # Look up disease in PrimeKG disease profiles
        disease_profiles = self.primekg_data.get("disease_profiles", [])
        for dp in disease_profiles:
            if (dp.get("disease_name", "").lower() == profile.disease_name.lower() or
                    dp.get("disease_id") == profile.disease_id):
                profile.primekg_node_id = dp.get("disease_id")
                profile.primekg_neighbor_count = dp.get("edge_count", 0)
                profile.primekg_edge_types = dp.get("relation_types", [])
                self.logger.info("PrimeKG match: %d edges, types=%s",
                                 profile.primekg_neighbor_count,
                                 profile.primekg_edge_types[:5])
                break
            

    def _estimate_literature(self, profile: DiseaseProfile) -> None:
        """Estimate PubMed article count for this disease."""
        try:
            import urllib.request
            import urllib.parse
            import xml.etree.ElementTree as ET

            # Build search query
            terms = [f'"{profile.disease_name}"']
            for syn in profile.synonyms[:3]:
                terms.append(f'"{syn}"')
            query = " OR ".join(terms)
            encoded = urllib.parse.quote(query)

            url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                f"?db=pubmed&term={encoded}&rettype=count"
            )
            if self.ncbi_api_key:
                url += f"&api_key={self.ncbi_api_key}"

            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                xml_data = resp.read()

            root = ET.fromstring(xml_data)
            count = int(root.findtext("Count", "0"))
            profile.pubmed_article_count = count

            # PMC OA estimate (~10-15% of PubMed articles have PMC full text)
            profile.pmc_oa_count = int(count * 0.12)

            self.logger.info("PubMed articles: %d (est. PMC OA: %d)",
                             count, profile.pmc_oa_count)

            time.sleep(0.15 if self.ncbi_api_key else 0.4)

        except Exception as e:
            self.logger.warning("PubMed count estimation failed: %s", e)

    # [TODO]: Update 
    def _generate_strategy(self, profile: DiseaseProfile) -> None:
        """Determine extraction strategy based on available sources."""
        # Coverage flag
        if profile.pubmed_article_count >= 500 and len(profile.tier1_sources) >= 2:
            profile.coverage_flag = CoverageFlag.RICH
            profile.expected_yield = "high"
        elif profile.pubmed_article_count >= 50 or len(profile.tier1_sources) >= 1:
            profile.coverage_flag = CoverageFlag.MODERATE
            profile.expected_yield = "medium"
        else:
            profile.coverage_flag = CoverageFlag.SPARSE
            profile.expected_yield = "low"

        # Generate PubMed search queries
        base_name = profile.disease_name
        queries = [
            f'"{base_name}" AND (treatment OR therapy OR management)',
            f'"{base_name}" AND (pathogenesis OR mechanism OR genetics)',
            f'"{base_name}" AND (diagnosis OR biomarker OR prognosis)',
            f'"{base_name}" AND (natural history OR progression OR outcome)',
        ]
        # Add differential queries if we have partners
        for diff in profile.differential_diseases[:3]:
            queries.append(f'"{base_name}" AND "{diff}" AND (differential OR distinguish)')

        profile.recommended_pubmed_queries = queries

    async def _identify_differentials(self, profile: DiseaseProfile) -> None:
        """Identify differential diagnosis partners using LLM."""
        # For now, use PrimeKG disease-disease edges if available
        if self.primekg_data:
            dd_edges = self.primekg_data.get("disease_disease_edges", {})
            neighbors = dd_edges.get(profile.disease_name.lower(), [])
            if neighbors:
                profile.differential_diseases = neighbors[:10]
                self.logger.info("Differentials from PrimeKG: %s", profile.differential_diseases)
                return

        # Fallback: LLM-based identification (to be implemented with actual LLM call)
        # For Phase 0, leave empty — will be filled by LLM in Phase 1
        self.logger.info("No differential diseases found yet — will use LLM in extraction phase")

    def _build_tier1_relations(self, profile: DiseaseProfile) -> None:
        """Convert differential_diseases into tier1_relations (ontology-derived, not LLM).

        Kept separate from consensus_triples so downstream agents can distinguish
        'LLM extracted' from 'ontology asserted'.
        """
        source_id = profile.omim_id or profile.mondo_id or profile.orphanet_id or ""
        for dd_name in profile.differential_diseases:
            triple = RawTriple(
                subject=profile.disease_name,
                subject_type="disease",
                relation="differential_diagnosis",
                object=dd_name,
                object_type="disease",
                source_id=source_id,
                extraction_model="tier1_ontology",
                confidence=1.0,
                evidence_text="",  # Tier-1 structured field — no verbatim quote
            )
            profile.tier1_relations.append(triple)
        if profile.tier1_relations:
            self.logger.info("Built %d tier1_relations from differential_diseases",
                             len(profile.tier1_relations))

    def _save_config(self, profile: DiseaseProfile) -> Path:
        """Save DiseaseProfile as YAML config."""
        safe_id = profile.disease_id.replace(":", "_").replace("/", "_")
        config_path = self._config_dir / f"{safe_id}.yaml"

        config_data = {
            "disease_id": profile.disease_id,
            "disease_name": profile.disease_name,
            "synonyms": profile.synonyms,
            "omim_id": profile.omim_id,
            "orphanet_id": profile.orphanet_id,
            "mondo_id": profile.mondo_id,
            "disease_category": profile.disease_category,
            "inheritance_pattern": profile.inheritance_pattern,
            "disease_type": profile.disease_type,
            "differential_diseases": profile.differential_diseases,
            "primekg_node_id": profile.primekg_node_id,
            "primekg_neighbor_count": profile.primekg_neighbor_count,
            "has_genereviews": profile.has_genereviews,
            "genereviews_id": profile.genereviews_id,
            "has_omim": profile.has_omim,
            "has_orphanet": profile.has_orphanet,
            "tier1_sources": profile.tier1_sources,
            "key_genes": profile.key_genes,
            "key_phenotypes": profile.key_phenotypes,
            "pubmed_article_count": profile.pubmed_article_count,
            "pmc_oa_count": profile.pmc_oa_count,
            "recommended_pubmed_queries": profile.recommended_pubmed_queries,
            "expected_yield": profile.expected_yield,
            "coverage_flag": profile.coverage_flag.value,
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

        self.logger.info("Disease config saved to %s", config_path)
        return config_path
