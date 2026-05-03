"""
Agent 2: Evidence Harvester
============================
Given a DiseaseProfile, collect all relevant evidence from multiple tiers.
Applies credibility scoring to rank sources before passing to extraction.

Tier 1: GeneReviews, OMIM clinical synopsis, Orphanet
Tier 2: PubMed abstracts, PMC Open Access full-text

Reuses PubMed fetching patterns from Paper 1 (2_targeted_text_mining.py),
generalized to work with any DiseaseProfile instead of hardcoded configs.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import threading
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from agents.base_agent import BaseAgent
from core.credibility_scorer import CredibilityScorer
from core.models import (
    AgentResult,
    DiseaseProfile,
    EvidenceCollection,
    EvidenceTier,
    SourceDocument,
    StudyType,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Publication types to exclude (from Paper 1)
EXCLUDED_PUB_TYPES = {
    "editorial", "letter", "comment", "erratum",
    "retraction of publication", "retracted publication",
    "news", "published erratum", "biography",
}

MIN_ABSTRACT_WORDS = 100


def _load_evidence_json(cache_dir: Path) -> dict | None:
    """Load evidence collection JSON, preferring .json.gz with .json fallback.

    Returns the parsed dict, or None if neither file exists.
    """
    gz_file = cache_dir / "evidence_collection.json.gz"
    json_file = cache_dir / "evidence_collection.json"
    if gz_file.exists():
        with gzip.open(gz_file, "rt", encoding="utf-8") as f:
            return json.load(f)
    elif json_file.exists():
        with open(json_file) as f:
            return json.load(f)
    return None


class _NCBIRateLimiter:
    """
    Process-wide token bucket rate limiter for NCBI API calls.

    Shared across ALL PubMedClient instances so that parallel workers
    collectively stay under NCBI's rate limit (10 req/s with API key,
    3 req/s without). We target 8 req/s with API key as a safe margin.

    Thread-safe: uses a lock to coordinate token refill and consumption.
    """

    _instance: _NCBIRateLimiter | None = None
    _init_lock = threading.Lock()

    def __new__(cls) -> _NCBIRateLimiter:
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def configure(self, max_requests_per_second: float) -> None:
        """Set the rate limit. Safe to call multiple times; first call wins."""
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self._max_rps = max_requests_per_second
            self._min_interval = 1.0 / max_requests_per_second
            self._lock = threading.Lock()
            self._last_request_time = 0.0
            self._initialized = True
            logger.info(
                "NCBI shared rate limiter configured: %.1f req/s (%.3fs between requests)",
                max_requests_per_second,
                self._min_interval,
            )

    def acquire(self) -> None:
        """
        Block until it is safe to make the next NCBI request.

        Ensures a minimum interval between any two NCBI requests
        across all threads in the process.
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                wait = self._min_interval - elapsed
                time.sleep(wait)
            self._last_request_time = time.monotonic()


# Module-level singleton — created once, shared across all PubMedClient instances
_ncbi_rate_limiter = _NCBIRateLimiter()


class PubMedClient:
    """NCBI PubMed/PMC API client with rate limiting and retry."""

    def __init__(self, api_key: str | None = None):
        from Bio import Entrez
        self.Entrez = Entrez
        Entrez.email = "chronomedkg@sdu.dk"
        if api_key:
            Entrez.api_key = api_key
        # Configure the shared rate limiter (first call wins; subsequent calls are no-ops).
        # With API key: 8 req/s (safe margin under NCBI's 10 req/s limit).
        # Without API key: 2.5 req/s (safe margin under NCBI's 3 req/s limit).
        max_rps = 8.0 if api_key else 2.5
        _ncbi_rate_limiter.configure(max_rps)
        self._rate_limiter = _ncbi_rate_limiter
        self._max_retries = 2

    def _retry(self, fn, *args, **kwargs):
        """Execute with retry on transient NCBI errors."""
        for attempt in range(self._max_retries + 1):
            try:
                # Acquire slot from the shared process-wide rate limiter
                # BEFORE making the request, so all threads are coordinated.
                self._rate_limiter.acquire()
                result = fn(*args, **kwargs)
                return result
            except Exception as e:
                if attempt < self._max_retries:
                    wait = 2.0 * (attempt + 1)
                    logger.warning("NCBI error (attempt %d): %s. Retrying in %.0fs...",
                                   attempt + 1, e, wait)
                    time.sleep(wait)
                else:
                    logger.error("NCBI failed after %d attempts: %s", self._max_retries + 1, e)
                    return None

    def search(self, query: str, max_results: int = 200) -> list[str]:
        """Search PubMed, return PMIDs."""
        def _do():
            handle = self.Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
            record = self.Entrez.read(handle)
            handle.close()
            return record.get("IdList", [])

        result = self._retry(_do)
        return result or []

    def fetch_abstracts(self, pmids: list[str], batch_size: int = 500) -> list[dict]:
        """Fetch article metadata + abstracts using EPost + batched EFetch.

        Uses NCBI History Server: post all PMIDs once via EPost, then
        fetch in batches of 500 via EFetch with WebEnv — drastically
        reduces the number of API calls.
        """
        if not pmids:
            return []

        # Step 1: Post all PMIDs to History Server
        webenv, query_key = self._epost_pmids(pmids)

        if webenv and query_key:
            # Step 2: Fetch via WebEnv in batches of 500
            return self._efetch_via_webenv(webenv, query_key, len(pmids), batch_size)
        else:
            # Fallback: direct batched fetch (old method)
            logger.debug("EPost failed, falling back to direct fetch")
            return self._fetch_abstracts_direct(pmids, batch_size)

    def _epost_pmids(self, pmids: list[str]) -> tuple[str | None, str | None]:
        """Post PMIDs to NCBI History Server, return (WebEnv, QueryKey)."""
        def _do():
            handle = self.Entrez.epost(db="pubmed", id=",".join(pmids))
            record = self.Entrez.read(handle)
            handle.close()
            return record

        result = self._retry(_do)
        if result:
            return result.get("WebEnv"), result.get("QueryKey")
        return None, None

    def _efetch_via_webenv(self, webenv: str, query_key: str,
                           total: int, batch_size: int = 500) -> list[dict]:
        """Fetch articles from History Server in batches."""
        all_articles = []
        for start in range(0, total, batch_size):
            def _do(retstart=start):
                handle = self.Entrez.efetch(
                    db="pubmed", query_key=query_key, WebEnv=webenv,
                    retstart=retstart, retmax=batch_size,
                    rettype="xml", retmode="xml"
                )
                xml_data = handle.read()
                handle.close()
                return xml_data

            xml_data = self._retry(_do)
            if xml_data:
                articles = self._parse_pubmed_xml(xml_data)
                all_articles.extend(articles)
            logger.info("  Fetched %d/%d articles", len(all_articles), total)
        return all_articles

    def _fetch_abstracts_direct(self, pmids: list[str], batch_size: int = 500) -> list[dict]:
        """Direct batched fetch (fallback if EPost fails)."""
        all_articles = []
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            articles = self._fetch_batch(batch)
            if articles:
                all_articles.extend(articles)
            logger.info("  Fetched %d/%d articles", len(all_articles), len(pmids))
        return all_articles

    def _fetch_batch(self, pmids: list[str]) -> list[dict]:
        """Fetch a single batch of articles."""
        def _do():
            handle = self.Entrez.efetch(
                db="pubmed", id=",".join(pmids),
                rettype="xml", retmode="xml"
            )
            xml_data = handle.read()
            handle.close()
            return xml_data

        xml_data = self._retry(_do)
        if not xml_data:
            return []

        return self._parse_pubmed_xml(xml_data)

    def _parse_pubmed_xml(self, xml_data: bytes | str) -> list[dict]:
        """Parse PubMed XML to extract article metadata."""
        articles = []
        try:
            root = ET.fromstring(xml_data if isinstance(xml_data, bytes) else xml_data.encode())
        except ET.ParseError as e:
            logger.error("XML parse error: %s", e)
            return []

        for article in root.findall(".//PubmedArticle"):
            try:
                parsed = self._parse_article(article)
                if parsed:
                    articles.append(parsed)
            except Exception as e:
                logger.debug("Failed to parse article: %s", e)

        return articles

    def _parse_article(self, article_elem) -> dict | None:
        """Parse a single PubmedArticle XML element."""
        medline = article_elem.find(".//MedlineCitation")
        if medline is None:
            return None

        pmid = medline.findtext("PMID", "")
        art = medline.find("Article")
        if art is None:
            return None

        # Title
        title = art.findtext("ArticleTitle", "")

        # Abstract
        abstract_elem = art.find("Abstract")
        abstract_text = ""
        if abstract_elem is not None:
            parts = []
            for at in abstract_elem.findall("AbstractText"):
                label = at.get("Label", "")
                text = "".join(at.itertext()).strip()
                if label:
                    parts.append(f"{label}: {text}")
                else:
                    parts.append(text)
            abstract_text = "\n".join(parts)

        # Quality filter
        if len(abstract_text.split()) < MIN_ABSTRACT_WORDS:
            return None

        # Publication type exclusion
        pub_types = []
        for pt in art.findall(".//PublicationType"):
            pt_text = (pt.text or "").lower()
            pub_types.append(pt_text)
            if pt_text in EXCLUDED_PUB_TYPES:
                return None

        # Journal
        journal = art.findtext(".//Journal/Title", "")

        # Date
        pub_date = None
        date_elem = art.find(".//ArticleDate")
        if date_elem is None:
            date_elem = medline.find(".//DateCompleted")
        if date_elem is not None:
            try:
                year = int(date_elem.findtext("Year", "0"))
                month = int(date_elem.findtext("Month", "1"))
                day = int(date_elem.findtext("Day", "1"))
                if year > 0:
                    pub_date = date(year, min(month, 12), min(day, 28))
            except (ValueError, TypeError):
                pass

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract_text,
            "journal": journal,
            "publication_date": pub_date,
            "pub_types": pub_types,
        }

    def batch_pmid_to_pmcid(self, pmids: list[str]) -> dict[str, str]:
        """Convert a batch of PMIDs to PMCIDs in a single elink call.

        Returns dict mapping PMID -> PMCID for those that have PMC entries.
        Uses one API call instead of N calls.
        """
        if not pmids:
            return {}

        pmid_to_pmcid = {}
        # elink supports up to ~200 IDs per call reliably
        for i in range(0, len(pmids), 200):
            batch = pmids[i:i + 200]

            def _do(ids=batch):
                handle = self.Entrez.elink(dbfrom="pubmed", db="pmc", id=ids)
                record = self.Entrez.read(handle)
                handle.close()
                return record

            link_records = self._retry(_do)
            if not link_records:
                continue

            for rec in link_records:
                try:
                    src_pmid = str(rec.get("IdList", [""])[0])
                    link_sets = rec.get("LinkSetDb", [])
                    for ls in link_sets:
                        if ls.get("DbTo") == "pmc":
                            links = ls.get("Link", [])
                            if links:
                                pmid_to_pmcid[src_pmid] = links[0]["Id"]
                                break
                except (IndexError, KeyError):
                    continue

        return pmid_to_pmcid

    def batch_fetch_pmc_fulltext(self, pmid_pmcid_map: dict[str, str],
                                  batch_size: int = 20) -> list[dict]:
        """Fetch full-text for multiple PMC articles in batched efetch calls.

        Takes a {PMID: PMCID} dict, fetches PMC XML in batches,
        and returns parsed results.
        """
        results = []
        items = list(pmid_pmcid_map.items())

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            pmcids = [pmcid for _, pmcid in batch]

            def _do(ids=pmcids):
                handle = self.Entrez.efetch(db="pmc", id=",".join(ids), rettype="xml")
                xml_data = handle.read()
                handle.close()
                return xml_data

            xml_data = self._retry(_do)
            if not xml_data:
                continue

            # Parse the multi-article PMC XML
            try:
                root = ET.fromstring(xml_data if isinstance(xml_data, bytes) else xml_data.encode())
            except ET.ParseError:
                continue

            # PMC batch response wraps articles in <pmc-articleset>
            articles = root.findall(".//article")
            if not articles:
                # Single article response
                articles = [root] if root.tag == "article" else []

            for art_elem in articles:
                # Extract PMCID from article metadata
                art_pmcid = None
                for aid in art_elem.findall(".//article-id"):
                    if aid.get("pub-id-type") == "pmc":
                        art_pmcid = aid.text
                        break

                # Find matching PMID
                art_pmid = None
                for aid in art_elem.findall(".//article-id"):
                    if aid.get("pub-id-type") == "pmid":
                        art_pmid = aid.text
                        break

                if not art_pmcid:
                    # Try matching by position in batch
                    idx = len(results) - (i // batch_size) * batch_size
                    if idx < len(batch):
                        art_pmid, art_pmcid = batch[idx]

                if art_pmcid:
                    parsed = self._parse_pmc_article_elem(art_elem, art_pmid or "", art_pmcid)
                    if parsed:
                        results.append(parsed)

        return results

    def _parse_pmc_article_elem(self, article: ET.Element, pmid: str, pmcid: str) -> dict | None:
        """Parse a single PMC article XML element into structured sections."""
        sections = {}

        title_elem = article.find(".//article-title")
        title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""

        abstract_elem = article.find(".//abstract")
        if abstract_elem is not None:
            sections["abstract"] = "".join(abstract_elem.itertext()).strip()

        body = article.find(".//body")
        if body is not None:
            for sec in body.findall(".//sec"):
                sec_title_elem = sec.find("title")
                if sec_title_elem is not None:
                    sec_title = "".join(sec_title_elem.itertext()).strip().lower()
                    sec_text = "".join(sec.itertext()).strip()

                    if any(k in sec_title for k in ["introduction", "background"]):
                        sections["introduction"] = sec_text
                    elif any(k in sec_title for k in ["result", "finding"]):
                        sections["results"] = sec_text
                    elif any(k in sec_title for k in ["discussion", "interpretation"]):
                        sections["discussion"] = sec_text
                    elif any(k in sec_title for k in ["method", "material", "patient", "study design"]):
                        sections["methods"] = sec_text
                    elif any(k in sec_title for k in ["conclusion", "summary"]):
                        sections["conclusion"] = sec_text

        if not sections:
            return None

        focus_text = ""
        for key in ["results", "discussion", "conclusion"]:
            if key in sections:
                focus_text += f"\n## {key.title()}\n{sections[key]}\n"

        if not focus_text and "abstract" in sections:
            focus_text = sections["abstract"]

        return {
            "pmid": pmid,
            "pmcid": f"PMC{pmcid}",
            "title": title,
            "sections": sections,
            "focus_text": focus_text,
            "section_count": len(sections),
        }

    def fetch_pmc_fulltext(self, pmid: str) -> dict | None:
        """
        Fetch full-text from PMC Open Access for a given PMID.
        Returns structured sections (Introduction, Results, Discussion, etc.)
        NOTE: For bulk operations, use batch_pmid_to_pmcid + batch_fetch_pmc_fulltext instead.
        """
        mapping = self.batch_pmid_to_pmcid([pmid])
        if pmid not in mapping:
            return None

        results = self.batch_fetch_pmc_fulltext({pmid: mapping[pmid]})
        return results[0] if results else None

    # NOTE: _parse_pmc_xml removed — replaced by _parse_pmc_article_elem
    # which handles both single and batch PMC XML responses.

    def fetch_genereviews(self, genereviews_id: str) -> dict | None:
        """Fetch GeneReviews entry text via NCBI Books."""
        def _do():
            handle = self.Entrez.efetch(db="books", id=genereviews_id, rettype="xml")
            xml_data = handle.read()
            handle.close()
            return xml_data

        xml_data = self._retry(_do)
        if not xml_data:
            return None

        try:
            root = ET.fromstring(xml_data if isinstance(xml_data, bytes) else xml_data.encode())
            # Extract all text content
            text_parts = []
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    text_parts.append(elem.text.strip())
            full_text = "\n".join(text_parts)

            title_elem = root.find(".//BookTitle") or root.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else f"GeneReviews {genereviews_id}"

            return {
                "id": genereviews_id,
                "title": title,
                "text": full_text,
            }
        except Exception as e:
            logger.warning("Failed to parse GeneReviews %s: %s", genereviews_id, e)
            return None


class EvidenceHarvester(BaseAgent):
    """Agent 2: Multi-source evidence collection."""

    def __init__(self, config: dict):
        super().__init__(config, logger)
        api_key = os.environ.get("NCBI_API_KEY", "")
        self.pubmed = PubMedClient(api_key=api_key if api_key else None)
        self.scorer = CredibilityScorer()
        self._cache_dir = PROJECT_ROOT / "data" / "extracted"

    async def run(self, input_data: dict) -> AgentResult:
        """
        Collect all evidence for a disease.

        input_data:
            profile: DiseaseProfile (as dict)
        """
        profile = DiseaseProfile.from_dict(input_data["profile"])
        disease_id = profile.disease_id

        self.logger.info("Harvesting evidence for: %s (%s)", profile.disease_name, disease_id)

        collection = EvidenceCollection(disease_id=disease_id)
        start_time = time.monotonic()

        # Tier 1: Curated sources
        self._harvest_tier1(profile, collection)

        # Tier 2: Literature
        self._harvest_tier2(profile, collection)

        elapsed = time.monotonic() - start_time

        # Save to cache
        cache_dir = self._cache_dir / disease_id.replace(":", "_")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._save_collection(collection, cache_dir)

        collection.harvest_metrics = {
            "elapsed_seconds": round(elapsed, 1),
            "tier1_count": len(collection.tier1_documents),
            "tier2_count": len(collection.tier2_documents),
            "total_sources": collection.total_sources,
            "coverage": collection.coverage_quality.value,
        }

        status = "success" if collection.total_sources > 0 else "failed"
        return AgentResult(
            agent_name="EvidenceHarvester",
            disease_id=disease_id,
            status=status,
            data={"collection_summary": collection.harvest_metrics},
            metrics=collection.harvest_metrics,
            timestamp=datetime.utcnow(),
        )

    def _harvest_tier1(self, profile: DiseaseProfile, collection: EvidenceCollection) -> None:
        """Collect Tier 1 curated sources."""
        # GeneReviews
        if profile.has_genereviews and profile.genereviews_id:
            self.logger.info("  Fetching GeneReviews: %s", profile.genereviews_id)
            gr_data = self.pubmed.fetch_genereviews(profile.genereviews_id)
            if gr_data:
                score, _ = self.scorer.score_tier1_source("genereviews")
                doc = SourceDocument(
                    source_id=f"GeneReviews:{profile.genereviews_id}",
                    source_type="genereviews",
                    tier=EvidenceTier.TIER_1,
                    title=gr_data["title"],
                    text=gr_data["text"],
                    credibility_score=score,
                    study_type=StudyType.DATABASE,
                )
                collection.tier1_documents.append(doc)

        # OMIM
        if profile.has_omim and profile.omim_id:
            self.logger.info("  Fetching OMIM: %s", profile.omim_id)
            omim_doc = self._fetch_omim(profile)
            if omim_doc:
                collection.tier1_documents.append(omim_doc)

        # Orphanet
        if profile.has_orphanet and profile.orphanet_id:
            self.logger.info("  Noting Orphanet source: %s", profile.orphanet_id)
            score, _ = self.scorer.score_tier1_source("orphanet")
            doc = SourceDocument(
                source_id=f"Orphanet:{profile.orphanet_id}",
                source_type="orphanet",
                tier=EvidenceTier.TIER_1,
                title=f"{profile.disease_name} — Orphanet",
                text="",  # Orphanet data fetched separately via Orphadata XML
                credibility_score=score,
                study_type=StudyType.DATABASE,
            )
            collection.tier1_documents.append(doc)

        self.logger.info("  Tier 1: %d documents collected", len(collection.tier1_documents))

    def _fetch_omim(self, profile: DiseaseProfile) -> SourceDocument | None:
        """Fetch OMIM clinical synopsis."""
        omim_api_key = os.environ.get("OMIM_API_KEY", "")
        if not omim_api_key:
            self.logger.debug("No OMIM_API_KEY — skipping OMIM fetch")
            return None

        try:
            import urllib.request
            omim_num = profile.omim_id.replace("OMIM:", "")
            url = (
                f"https://api.omim.org/api/entry?"
                f"mimNumber={omim_num}&include=clinicalSynopsis&include=text"
                f"&format=json&apiKey={omim_api_key}"
            )
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())

            entries = data.get("omim", {}).get("entryList", [])
            if not entries:
                return None

            entry = entries[0].get("entry", {})

            # Collect text sections
            text_parts = []
            text_section = entry.get("textSectionList", [])
            for ts in text_section:
                section = ts.get("textSection", {})
                title = section.get("textSectionTitle", "")
                content = section.get("textSectionContent", "")
                if content:
                    text_parts.append(f"## {title}\n{content}")

            # Clinical synopsis
            clin = entry.get("clinicalSynopsisList", [])
            if clin:
                synopsis = clin[0].get("clinicalSynopsis", {})
                for key, val in synopsis.items():
                    if isinstance(val, str) and key not in ("mimNumber", "prefix"):
                        text_parts.append(f"## Clinical Synopsis — {key}\n{val}")

            full_text = "\n\n".join(text_parts)
            if not full_text.strip():
                return None

            score, _ = self.scorer.score_tier1_source("omim")
            return SourceDocument(
                source_id=profile.omim_id,
                source_type="omim",
                tier=EvidenceTier.TIER_1,
                title=f"{profile.disease_name} — OMIM {omim_num}",
                text=full_text,
                credibility_score=score,
                study_type=StudyType.DATABASE,
            )

        except Exception as e:
            self.logger.warning("OMIM fetch failed for %s: %s", profile.omim_id, e)
            return None

    def _harvest_tier2(self, profile: DiseaseProfile, collection: EvidenceCollection) -> None:
        """Collect Tier 2 literature from PubMed."""
        max_per_query = self.config.get("max_abstracts_per_query", 200)
        all_pmids = set()
        all_articles = []

        # Run each recommended query
        for i, query in enumerate(profile.recommended_pubmed_queries):
            self.logger.info("  PubMed query %d/%d: %s",
                             i + 1, len(profile.recommended_pubmed_queries), query[:80])
            pmids = self.pubmed.search(query, max_results=max_per_query)
            new_pmids = [p for p in pmids if p not in all_pmids]
            all_pmids.update(new_pmids)

            if new_pmids:
                articles = self.pubmed.fetch_abstracts(new_pmids)
                all_articles.extend(articles)

            self.logger.info("    Found %d PMIDs (%d new), fetched %d articles",
                             len(pmids), len(new_pmids), len(articles) if new_pmids else 0)

        # Score and convert to SourceDocuments
        for art in all_articles:
            study_type = self._classify_study_type(art.get("pub_types", []))
            score, _ = self.scorer.compute(
                journal_name=art.get("journal"),
                publication_date=art.get("publication_date"),
                citation_count=None,  # Not available from PubMed directly
                study_type=study_type,
                is_retracted=False,
            )

            doc = SourceDocument(
                source_id=f"PMID:{art['pmid']}",
                source_type="pubmed_abstract",
                tier=EvidenceTier.TIER_2,
                title=art.get("title", ""),
                text=art.get("abstract", ""),
                publication_date=art.get("publication_date"),
                journal=art.get("journal"),
                credibility_score=score,
                study_type=study_type,
            )
            collection.tier2_documents.append(doc)

        self.logger.info("  Tier 2 abstracts: %d documents from %d unique PMIDs",
                         len(collection.tier2_documents), len(all_pmids))

        # Attempt PMC full-text for top-ranked articles (Results + Discussion)
        # Uses batch elink + batch efetch for massive speedup
        max_fulltext = self.config.get("max_fulltext_per_disease", 50)
        ranked_docs = sorted(collection.tier2_documents, key=lambda d: d.credibility_score, reverse=True)

        # Step 1: Batch convert PMIDs to PMCIDs (1-2 API calls instead of 50)
        candidate_pmids = [doc.source_id.replace("PMID:", "") for doc in ranked_docs[:max_fulltext * 2]]
        pmid_to_pmcid = self.pubmed.batch_pmid_to_pmcid(candidate_pmids)

        # Step 2: Take top max_fulltext that have PMC entries
        pmid_doc_map = {doc.source_id.replace("PMID:", ""): doc for doc in ranked_docs}
        selected = {}
        for pmid in candidate_pmids:
            if pmid in pmid_to_pmcid and len(selected) < max_fulltext:
                selected[pmid] = pmid_to_pmcid[pmid]

        # Step 3: Batch fetch full-text (2-3 API calls instead of 50)
        fulltext_results = self.pubmed.batch_fetch_pmc_fulltext(selected) if selected else []
        fulltext_count = 0

        for pmc_data in fulltext_results:
            if pmc_data and pmc_data.get("focus_text"):
                pmid = pmc_data["pmid"]
                src_doc = pmid_doc_map.get(pmid)
                if not src_doc:
                    continue

                fulltext_doc = SourceDocument(
                    source_id=f"{src_doc.source_id}|{pmc_data['pmcid']}",
                    source_type="pmc_fulltext",
                    tier=EvidenceTier.TIER_2,
                    title=src_doc.title,
                    text=pmc_data["focus_text"],
                    sections=pmc_data.get("sections"),
                    publication_date=src_doc.publication_date,
                    journal=src_doc.journal,
                    credibility_score=src_doc.credibility_score * 1.1,
                    study_type=src_doc.study_type,
                )
                collection.tier2_documents.append(fulltext_doc)
                fulltext_count += 1

        self.logger.info("  Tier 2 full-text: %d PMC articles fetched (Results + Discussion)",
                         fulltext_count)

    def _classify_study_type(self, pub_types: list[str]) -> StudyType:
        """Classify study type from PubMed publication types."""
        pub_types_lower = {pt.lower() for pt in pub_types}

        if "meta-analysis" in pub_types_lower:
            return StudyType.META_ANALYSIS
        if "randomized controlled trial" in pub_types_lower:
            return StudyType.RCT
        if "practice guideline" in pub_types_lower or "guideline" in pub_types_lower:
            return StudyType.GUIDELINE
        if "systematic review" in pub_types_lower:
            return StudyType.META_ANALYSIS
        if "clinical trial" in pub_types_lower:
            return StudyType.COHORT
        if "case reports" in pub_types_lower:
            return StudyType.CASE_REPORT
        if "review" in pub_types_lower:
            return StudyType.REVIEW
        return StudyType.OTHER

    def _save_collection(self, collection: EvidenceCollection, cache_dir: Path) -> None:
        """Save evidence collection with FULL text for extraction pipeline."""

        def _doc_to_full_dict(doc: SourceDocument) -> dict:
            """Serialize with FULL text (not truncated) for extraction."""
            return {
                "source_id": doc.source_id,
                "source_type": doc.source_type,
                "tier": doc.tier.value,
                "title": doc.title,
                "text": doc.text,  # FULL text, not truncated
                "sections": doc.sections,  # Full dict, not just keys
                "publication_date": doc.publication_date.isoformat() if doc.publication_date else None,
                "journal": doc.journal,
                "credibility_score": doc.credibility_score,
                "study_type": doc.study_type.value if doc.study_type else None,
                "citation_count": doc.citation_count,
                "is_retracted": doc.is_retracted,
            }

        data = {
            "disease_id": collection.disease_id,
            "tier1_count": len(collection.tier1_documents),
            "tier2_count": len(collection.tier2_documents),
            "tier1_sources": [_doc_to_full_dict(d) for d in collection.tier1_documents],
            "tier2_sources": [_doc_to_full_dict(d) for d in collection.tier2_documents],
        }
        json_bytes = json.dumps(data, indent=2, default=str).encode("utf-8")
        with gzip.open(cache_dir / "evidence_collection.json.gz", "wb") as f:
            f.write(json_bytes)

    def get_collection(self, disease_id: str) -> EvidenceCollection | None:
        """Load a previously harvested collection from cache."""
        cache_dir = self._cache_dir / disease_id.replace(":", "_")
        data = _load_evidence_json(cache_dir)
        if data is None:
            return None

        collection = EvidenceCollection(disease_id=disease_id)
        # Reconstruct SourceDocuments from summary (text is truncated in summary)
        # For full reconstruction, would need separate full-text cache
        collection.harvest_metrics = {
            "tier1_count": data["tier1_count"],
            "tier2_count": data["tier2_count"],
            "loaded_from_cache": True,
        }
        return collection
