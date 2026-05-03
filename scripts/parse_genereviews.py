#!/usr/bin/env python3
"""
Parse GeneReviews content from NCBI Bookshelf for temporal biomedical data.

Fetches GeneReviews HTML via NCBI E-utilities, extracts structured temporal
information (onset ages, stages, phenotype timing, milestones) from the
Clinical Description, Natural History, and Management sections.

Usage:
    python3 scripts/parse_genereviews.py --limit 50       # first 50
    python3 scripts/parse_genereviews.py --limit 0        # all
    python3 scripts/parse_genereviews.py --resume          # resume from checkpoint
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import pickle
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "data" / "genereviews_parse.log"),
    ],
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Data classes                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class PhenotypeOnset:
    """A phenotype with its onset age range."""
    phenotype: str
    age_min_years: Optional[float] = None
    age_max_years: Optional[float] = None
    description: str = ""  # raw text snippet


@dataclass
class DiseaseStage:
    """A named disease stage with age bounds."""
    name: str
    age_min_years: Optional[float] = None
    age_max_years: Optional[float] = None
    description: str = ""


@dataclass
class Milestone:
    """A clinical milestone with typical age."""
    name: str
    typical_age_years: Optional[float] = None
    age_range: str = ""  # raw text like "by age 12"


@dataclass
class GeneReviewParsed:
    """Parsed temporal data from one GeneReviews entry."""
    nbk_id: str
    shortname: str
    title: str
    omim_ids: list[str] = field(default_factory=list)
    # Temporal extractions
    disease_onset_age_min: Optional[float] = None
    disease_onset_age_max: Optional[float] = None
    onset_description: str = ""
    phenotype_onsets: list[PhenotypeOnset] = field(default_factory=list)
    stages: list[DiseaseStage] = field(default_factory=list)
    milestones: list[Milestone] = field(default_factory=list)
    # Metadata
    fetch_success: bool = False
    parse_sections_found: list[str] = field(default_factory=list)
    raw_text_length: int = 0
    error: str = ""


# --------------------------------------------------------------------------- #
#  Age parsing utilities                                                        #
# --------------------------------------------------------------------------- #

# Map textual age descriptions to numeric years
AGE_TERM_MAP = {
    "neonatal": (0, 0.083),       # 0-1 month
    "newborn": (0, 0.083),
    "neonat": (0, 0.083),
    "infancy": (0, 2),
    "infant": (0, 2),
    "early childhood": (2, 6),
    "childhood": (2, 12),
    "child": (2, 12),
    "juvenile": (5, 16),
    "adolescence": (12, 18),
    "adolescent": (12, 18),
    "teenage": (12, 18),
    "young adult": (18, 35),
    "early adulthood": (18, 35),
    "adult": (18, 65),
    "adulthood": (18, 65),
    "middle age": (40, 65),
    "late onset": (40, 80),
    "elderly": (65, 90),
    "prenatal": (-0.75, 0),
    "congenital": (0, 0),
    "at birth": (0, 0),
    "perinatal": (-0.083, 0.083),
    "first decade": (0, 10),
    "second decade": (10, 20),
    "third decade": (20, 30),
    "fourth decade": (30, 40),
    "fifth decade": (40, 50),
    "sixth decade": (50, 60),
    "first year of life": (0, 1),
    "first year": (0, 1),
    "early infancy": (0, 0.5),
    "late infancy": (0.5, 2),
    "toddler": (1, 3),
    "preschool": (3, 5),
    "school age": (5, 12),
    "school-age": (5, 12),
    "puberty": (10, 16),
    "prepubertal": (5, 10),
    "postpubertal": (13, 18),
}


def _parse_age_years(text: str) -> tuple[Optional[float], Optional[float]]:
    """
    Extract age range (in years) from text.
    Returns (min_years, max_years). Either or both may be None.
    """
    text_lower = text.lower().strip()

    # Try explicit numeric patterns first
    # "X-Y years"
    m = re.search(r"(\d+(?:\.\d+)?)\s*[-–to]+\s*(\d+(?:\.\d+)?)\s*years?\b", text_lower)
    if m:
        return float(m.group(1)), float(m.group(2))

    # "X-Y months"
    m = re.search(r"(\d+(?:\.\d+)?)\s*[-–to]+\s*(\d+(?:\.\d+)?)\s*months?\b", text_lower)
    if m:
        return float(m.group(1)) / 12, float(m.group(2)) / 12

    # "age X years" or "by age X"
    m = re.search(r"(?:age|by age)\s+(\d+(?:\.\d+)?)\s*(?:years?|yr)", text_lower)
    if m:
        val = float(m.group(1))
        return val, val

    # "X years of age"
    m = re.search(r"(\d+(?:\.\d+)?)\s*years?\s+of\s+age", text_lower)
    if m:
        val = float(m.group(1))
        return val, val

    # "X months of age"
    m = re.search(r"(\d+(?:\.\d+)?)\s*months?\s+of\s+age", text_lower)
    if m:
        val = float(m.group(1)) / 12
        return val, val

    # "before age X"
    m = re.search(r"before\s+age\s+(\d+(?:\.\d+)?)", text_lower)
    if m:
        return None, float(m.group(1))

    # "after age X"
    m = re.search(r"after\s+age\s+(\d+(?:\.\d+)?)", text_lower)
    if m:
        return float(m.group(1)), None

    # "by age X" (upper bound)
    m = re.search(r"by\s+age\s+(\d+(?:\.\d+)?)", text_lower)
    if m:
        return None, float(m.group(1))

    # "within the first X years"
    m = re.search(r"within\s+the\s+first\s+(\d+)\s+years?", text_lower)
    if m:
        return 0, float(m.group(1))

    # "within the first X months"
    m = re.search(r"within\s+the\s+first\s+(\d+)\s+months?", text_lower)
    if m:
        return 0, float(m.group(1)) / 12

    # Try term-based mapping
    for term, (lo, hi) in sorted(AGE_TERM_MAP.items(), key=lambda x: -len(x[0])):
        if term in text_lower:
            return lo, hi

    return None, None


# --------------------------------------------------------------------------- #
#  NCBI fetcher                                                                 #
# --------------------------------------------------------------------------- #

NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
NCBI_BOOK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
REQUEST_INTERVAL = 0.15 if NCBI_API_KEY else 0.4  # 8 req/s with key, 2.5 without
_last_request_time = 0.0


def _rate_limit():
    """Enforce rate limiting between NCBI requests."""
    global _last_request_time
    now = time.monotonic()
    elapsed = now - _last_request_time
    if elapsed < REQUEST_INTERVAL:
        time.sleep(REQUEST_INTERVAL - elapsed)
    _last_request_time = time.monotonic()


def fetch_genereview_text(nbk_id: str) -> Optional[str]:
    """
    Fetch GeneReviews content as plain text from NCBI Bookshelf.

    Uses efetch with db=books and rettype=docsum first, then falls back
    to the printable HTML endpoint.
    """
    _rate_limit()

    # Try the printable HTML page (most reliable for GeneReviews)
    url = f"https://www.ncbi.nlm.nih.gov/books/{nbk_id}/?report=printable"
    params = {}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.text
        else:
            logger.warning("HTTP %d for %s", resp.status_code, nbk_id)
            return None
    except requests.RequestException as e:
        logger.warning("Request failed for %s: %s", nbk_id, e)
        return None


# --------------------------------------------------------------------------- #
#  HTML → section text extraction                                               #
# --------------------------------------------------------------------------- #

def _strip_html_tags(html: str) -> str:
    """Remove HTML tags, decode entities, collapse whitespace."""
    # Remove script/style blocks
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common entities
    for entity, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                         ("&nbsp;", " "), ("&#8211;", "–"), ("&#8212;", "—"),
                         ("&rsquo;", "'"), ("&lsquo;", "'"), ("&rdquo;", '"'),
                         ("&ldquo;", '"'), ("&#x2013;", "–"), ("&#x2014;", "—")]:
        text = text.replace(entity, char)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_sections(html: str) -> dict[str, str]:
    """
    Extract key sections from GeneReviews HTML.

    Returns dict of section_name -> plain text content.
    Target sections: Clinical Description, Natural History, Suggestive Findings,
    Diagnosis, Management, Genotype-Phenotype Correlations.
    """
    sections = {}

    # GeneReviews uses <h2> or <h3> tags with specific section titles
    # Also uses <div class="section"> or id attributes
    # We'll split by heading patterns

    # Find all heading positions
    heading_pattern = re.compile(
        r"<h[23][^>]*>(.*?)</h[23]>",
        re.IGNORECASE | re.DOTALL,
    )

    headings = list(heading_pattern.finditer(html))

    TARGET_SECTIONS = {
        "clinical description": "clinical_description",
        "natural history": "natural_history",
        "clinical characteristics": "clinical_description",
        "suggestive findings": "suggestive_findings",
        "clinical findings": "clinical_description",
        "diagnosis": "diagnosis",
        "management": "management",
        "genotype-phenotype correlations": "genotype_phenotype",
        "clinical features": "clinical_description",
        "phenotype": "clinical_description",
        "prognosis": "prognosis",
    }

    for i, match in enumerate(headings):
        heading_text = _strip_html_tags(match.group(1)).lower().strip()

        # Check if this heading matches any target section
        matched_key = None
        for pattern, key in TARGET_SECTIONS.items():
            if pattern in heading_text:
                matched_key = key
                break

        if matched_key:
            # Extract text from this heading to the next heading
            start = match.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(html)
            section_html = html[start:end]
            section_text = _strip_html_tags(section_html)

            # Append if we already have content for this section (some reviews
            # split clinical description across multiple subsections)
            if matched_key in sections:
                sections[matched_key] += " " + section_text
            else:
                sections[matched_key] = section_text

    return sections


# --------------------------------------------------------------------------- #
#  Temporal data extraction from section text                                   #
# --------------------------------------------------------------------------- #

def extract_onset(text: str) -> tuple[Optional[float], Optional[float], str]:
    """
    Extract disease onset age from clinical description text.
    Returns (min_years, max_years, description_snippet).
    """
    # Look for onset-related sentences
    onset_patterns = [
        r"(?:onset|presents?|begins?|manifests?|appears?)\s+(?:in|during|at|by)\s+([^.]{5,80})",
        r"(?:age\s+(?:of|at)\s+onset)\s+(?:is|ranges?|varies?)\s+([^.]{5,80})",
        r"(?:typically\s+(?:present|manifest|begin|appear)s?)\s+(?:in|during|at)\s+([^.]{5,80})",
        r"(?:first\s+symptoms?)\s+(?:appear|present|occur|develop)\s+([^.]{5,80})",
        r"(?:symptom\s+onset)\s+(?:is|occurs?|ranges?)\s+([^.]{5,80})",
    ]

    best_desc = ""
    best_min = None
    best_max = None

    for pat in onset_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            snippet = m.group(1).strip()
            lo, hi = _parse_age_years(snippet)
            if lo is not None or hi is not None:
                if best_min is None and best_max is None:
                    best_min, best_max, best_desc = lo, hi, snippet
                break  # take first match per pattern

    return best_min, best_max, best_desc


def extract_phenotype_onsets(text: str) -> list[PhenotypeOnset]:
    """
    Extract phenotype-specific onset timing from clinical text.

    Looks for patterns like:
    - "Hearing loss typically presents in the first decade"
    - "Cardiomyopathy develops by age 15"
    - "Seizures begin in infancy"
    """
    results = []
    seen_phenos = set()

    # Common phenotype terms to look for
    phenotype_terms = [
        "hearing loss", "seizures?", "cardiomyopathy", "intellectual disability",
        "developmental delay", "hypotonia", "spasticity", "ataxia", "dystonia",
        "neuropathy", "retinitis pigmentosa", "vision loss", "blindness",
        "renal (?:failure|disease|involvement)", "hepatomegaly", "splenomegaly",
        "scoliosis", "contractures?", "weakness", "myopathy", "respiratory (?:failure|insufficiency)",
        "epilepsy", "microcephaly", "macrocephaly", "short stature", "growth (?:failure|retardation)",
        "feeding difficulties", "dysphagia", "failure to thrive", "regression",
        "autism", "behavioral (?:problems|issues)", "cognitive (?:decline|impairment)",
        "dementia", "tremor", "chorea", "myoclonus", "optic atrophy",
        "cataracts?", "glaucoma", "liver (?:disease|failure)", "diabetes",
        "cardiac (?:defects?|anomalies)", "congenital heart (?:defects?|disease)",
        "skin (?:findings|lesions|abnormalities)", "skeletal (?:abnormalities|anomalies)",
        "joint (?:laxity|hypermobility)", "osteoporosis", "fractures?",
        "anemia", "thrombocytopenia", "immunodeficiency",
    ]

    for pheno_pat in phenotype_terms:
        # Look for sentences containing the phenotype + age info
        pattern = re.compile(
            rf"([^.]*?\b({pheno_pat})\b[^.]*?(?:onset|present|develop|begin|appear|occur|emerge|manifest|by age|in the|during|at age)[^.]*\.)",
            re.IGNORECASE,
        )
        for m in pattern.finditer(text):
            sentence = m.group(1).strip()
            pheno_name = m.group(2).strip().lower()

            # Normalize plural
            pheno_name = re.sub(r"s$", "", pheno_name) if pheno_name.endswith("s") and not pheno_name.endswith("ss") else pheno_name

            if pheno_name in seen_phenos:
                continue

            lo, hi = _parse_age_years(sentence)
            if lo is not None or hi is not None:
                seen_phenos.add(pheno_name)
                results.append(PhenotypeOnset(
                    phenotype=pheno_name,
                    age_min_years=lo,
                    age_max_years=hi,
                    description=sentence[:200],
                ))

    return results


def extract_stages(text: str) -> list[DiseaseStage]:
    """
    Extract disease stages from clinical text.

    Looks for patterns like:
    - "Stage 1 (early): ages 0-5..."
    - "The infantile form presents..."
    - "Classic (severe) form / Mild (attenuated) form"
    """
    stages = []
    seen = set()

    # Pattern: "Stage X: description"
    for m in re.finditer(
        r"(?:stage|phase)\s+(\w+)\s*[:(]\s*([^).\n]{5,150})",
        text, re.IGNORECASE,
    ):
        name = f"Stage {m.group(1)}"
        desc = m.group(2).strip()
        if name.lower() not in seen:
            lo, hi = _parse_age_years(desc)
            seen.add(name.lower())
            stages.append(DiseaseStage(name=name, age_min_years=lo, age_max_years=hi, description=desc[:200]))

    # Pattern: "X form" (infantile, juvenile, adult)
    form_pattern = re.compile(
        r"\b((?:infantile|juvenile|adult|late[- ]onset|early[- ]onset|neonatal|congenital|"
        r"classic|severe|mild|attenuated|intermediate|late[- ]infantile|childhood)\s+(?:form|type|variant|onset|phenotype))\b"
        r"\s*[^.]{0,100}",
        re.IGNORECASE,
    )
    for m in form_pattern.finditer(text):
        name = m.group(1).strip()
        context = m.group(0).strip()
        if name.lower() not in seen:
            lo, hi = _parse_age_years(context)
            seen.add(name.lower())
            stages.append(DiseaseStage(name=name, age_min_years=lo, age_max_years=hi, description=context[:200]))

    return stages


def extract_milestones(text: str) -> list[Milestone]:
    """
    Extract clinical milestones with typical ages.

    Looks for patterns like:
    - "Loss of ambulation by age 12"
    - "Death typically occurs in the second decade"
    - "Wheelchair dependence by age 10-12"
    """
    milestones = []
    seen = set()

    milestone_terms = [
        (r"loss of ambulation", "loss of ambulation"),
        (r"wheelchair\s+(?:dependence|bound|dependent|use)", "wheelchair dependence"),
        (r"(?:death|die|mortality|life\s*expectancy|survival)", "death/survival"),
        (r"loss of (?:independent\s+)?walking", "loss of walking"),
        (r"ventilat(?:or|ion|ory)\s+(?:support|dependence|assistance)", "ventilatory support"),
        (r"(?:non-?ambulatory|unable to walk)", "loss of ambulation"),
        (r"cardiac\s+(?:failure|death|transplant)", "cardiac event"),
        (r"(?:liver|hepatic)\s+(?:failure|transplant)", "liver failure"),
        (r"renal\s+(?:failure|replacement|dialysis|transplant)", "renal failure"),
        (r"developmental\s+regression", "developmental regression"),
        (r"loss of (?:speech|language)", "loss of speech"),
        (r"loss of (?:vision|sight)", "loss of vision"),
        (r"loss of hearing", "loss of hearing"),
        (r"(?:feeding\s+tube|gastrostomy|g-tube|peg)", "feeding tube"),
    ]

    for pattern, milestone_name in milestone_terms:
        pat = re.compile(
            rf"([^.]*?\b{pattern}\b[^.]*\.)",
            re.IGNORECASE,
        )
        for m in pat.finditer(text):
            if milestone_name in seen:
                break
            sentence = m.group(1).strip()
            lo, hi = _parse_age_years(sentence)
            typical = lo if lo is not None else hi
            if typical is not None or hi is not None:
                seen.add(milestone_name)
                milestones.append(Milestone(
                    name=milestone_name,
                    typical_age_years=typical,
                    age_range=sentence[:200],
                ))

    return milestones


# --------------------------------------------------------------------------- #
#  Main parse function                                                          #
# --------------------------------------------------------------------------- #

def parse_genereview(nbk_id: str, shortname: str, title: str,
                     omim_ids: list[str]) -> GeneReviewParsed:
    """
    Fetch and parse a single GeneReviews entry.
    """
    result = GeneReviewParsed(
        nbk_id=nbk_id,
        shortname=shortname,
        title=title,
        omim_ids=omim_ids,
    )

    html = fetch_genereview_text(nbk_id)
    if not html:
        result.error = "fetch_failed"
        return result

    result.fetch_success = True
    result.raw_text_length = len(html)

    # Extract sections
    sections = extract_sections(html)
    result.parse_sections_found = list(sections.keys())

    if not sections:
        result.error = "no_sections_found"
        return result

    # Combine relevant text for extraction
    clinical_text = " ".join([
        sections.get("clinical_description", ""),
        sections.get("natural_history", ""),
        sections.get("prognosis", ""),
    ]).strip()

    all_text = " ".join(sections.values()).strip()

    # Extract onset
    if clinical_text:
        result.disease_onset_age_min, result.disease_onset_age_max, result.onset_description = (
            extract_onset(clinical_text)
        )

    # Extract phenotype onsets
    if clinical_text:
        result.phenotype_onsets = extract_phenotype_onsets(clinical_text)

    # Extract stages
    if all_text:
        result.stages = extract_stages(all_text)

    # Extract milestones
    if clinical_text:
        result.milestones = extract_milestones(clinical_text)

    return result


# --------------------------------------------------------------------------- #
#  Load mapping files                                                           #
# --------------------------------------------------------------------------- #

def load_genereviews_catalog() -> list[dict]:
    """
    Load all GeneReviews entries from mapping files.
    Returns list of dicts with keys: nbk_id, shortname, title, omim_ids.
    """
    gr_dir = PROJECT_ROOT / "data" / "validation_sources" / "genereviews"

    # Load title + NBK mapping (most complete list)
    title_file = gr_dir / "GRtitle_shortname_NBKid.txt"
    entries = {}  # nbk_id -> dict

    with open(title_file, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                shortname = parts[0]
                title = parts[1]
                nbk_id = parts[2]
                if nbk_id not in entries:
                    entries[nbk_id] = {
                        "nbk_id": nbk_id,
                        "shortname": shortname,
                        "title": title,
                        "omim_ids": [],
                    }

    # Load OMIM mappings
    omim_file = gr_dir / "NBKid_shortname_OMIM.txt"
    with open(omim_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                nbk_id = parts[0]
                omim_id = parts[2]
                if nbk_id in entries and omim_id:
                    entries[nbk_id]["omim_ids"].append(omim_id)

    catalog = sorted(entries.values(), key=lambda x: x["nbk_id"])
    logger.info("Loaded %d unique GeneReviews entries", len(catalog))
    return catalog


# --------------------------------------------------------------------------- #
#  Checkpoint management                                                        #
# --------------------------------------------------------------------------- #

CHECKPOINT_DIR = PROJECT_ROOT / "data" / "validation_sources" / "genereviews_parsed"
CHECKPOINT_FILE = CHECKPOINT_DIR / "checkpoint.json.gz"
OUTPUT_FILE = PROJECT_ROOT / "data" / "validation_sources" / "genereviews_parsed.pkl"


def load_checkpoint() -> dict[str, dict]:
    """Load previously parsed results from checkpoint."""
    if CHECKPOINT_FILE.exists():
        with gzip.open(CHECKPOINT_FILE, "rt", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Loaded checkpoint with %d entries", len(data))
        return data
    return {}


def save_checkpoint(results: dict[str, dict]):
    """Save results to checkpoint file (compressed JSON)."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(CHECKPOINT_FILE, "wt", encoding="utf-8") as f:
        json.dump(results, f)


def _to_dict(obj) -> dict:
    """Convert dataclass to dict, handling nested dataclasses."""
    if hasattr(obj, "__dataclass_fields__"):
        d = {}
        for k, v in asdict(obj).items():
            d[k] = v
        return d
    return obj


# --------------------------------------------------------------------------- #
#  Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Parse GeneReviews for temporal data")
    parser.add_argument("--limit", type=int, default=50,
                        help="Max entries to process (0 = all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N entries")
    args = parser.parse_args()

    catalog = load_genereviews_catalog()

    # Load or initialize checkpoint
    if args.resume:
        checkpoint = load_checkpoint()
    else:
        checkpoint = {}

    # Determine which entries to process
    to_process = []
    for entry in catalog:
        if entry["nbk_id"] not in checkpoint:
            to_process.append(entry)

    if args.offset > 0:
        to_process = to_process[args.offset:]

    if args.limit > 0:
        to_process = to_process[:args.limit]

    logger.info(
        "Processing %d entries (checkpoint has %d, catalog has %d)",
        len(to_process), len(checkpoint), len(catalog),
    )

    # Process entries
    success = 0
    failed = 0
    with_temporal = 0

    for i, entry in enumerate(to_process):
        nbk_id = entry["nbk_id"]
        logger.info(
            "[%d/%d] Parsing %s: %s",
            i + 1, len(to_process), nbk_id, entry["title"],
        )

        try:
            result = parse_genereview(
                nbk_id=nbk_id,
                shortname=entry["shortname"],
                title=entry["title"],
                omim_ids=entry["omim_ids"],
            )

            result_dict = _to_dict(result)
            checkpoint[nbk_id] = result_dict

            if result.fetch_success:
                success += 1
                has_temporal = (
                    result.disease_onset_age_min is not None
                    or result.disease_onset_age_max is not None
                    or len(result.phenotype_onsets) > 0
                    or len(result.stages) > 0
                    or len(result.milestones) > 0
                )
                if has_temporal:
                    with_temporal += 1

                logger.info(
                    "  -> sections=%s, onset=(%s,%s), phenotypes=%d, stages=%d, milestones=%d",
                    result.parse_sections_found,
                    result.disease_onset_age_min,
                    result.disease_onset_age_max,
                    len(result.phenotype_onsets),
                    len(result.stages),
                    len(result.milestones),
                )
            else:
                failed += 1
                logger.warning("  -> FAILED: %s", result.error)

        except Exception as e:
            failed += 1
            logger.error("  -> Exception for %s: %s", nbk_id, e, exc_info=True)
            checkpoint[nbk_id] = {"nbk_id": nbk_id, "error": str(e), "fetch_success": False}

        # Save checkpoint every 10 entries
        if (i + 1) % 10 == 0:
            save_checkpoint(checkpoint)
            logger.info("  Checkpoint saved (%d total entries)", len(checkpoint))

    # Final save
    save_checkpoint(checkpoint)

    # Save as pickle for easy loading
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(checkpoint, f)

    # Summary
    total = success + failed
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("  Total processed: %d", total)
    logger.info("  Fetched OK:      %d (%.1f%%)", success, 100 * success / max(total, 1))
    logger.info("  With temporal:   %d (%.1f%% of fetched)", with_temporal, 100 * with_temporal / max(success, 1))
    logger.info("  Failed:          %d", failed)
    logger.info("  Checkpoint:      %d total entries", len(checkpoint))
    logger.info("  Output:          %s", OUTPUT_FILE)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
