#!/usr/bin/env python3
"""
ChronoMedKG RAG Experiment v3
================================
Tests frontier LLMs on ChronoMedKG-TQA v7.1 benchmark under 4 conditions:
  (a) No retrieval — LLM parametric knowledge only
  (b) PrimeKG RAG — static KG edges (no temporal data)
  (c) HPOA RAG — Human Phenotype Ontology Annotations (onset data only)
  (d) ChronoMedKG RAG — temporal triples with onset ages, stages, timelines

Key improvements over v2:
  1. HPOA RAG condition — tests whether curated onset data alone suffices
  2. Claude 3.5 Haiku added as 4th model (4 model families)
  3. Gemini Flash updated routing
  4. Smarter sampling: Tier 1 + static controls by default
  5. Cost logging at startup
  6. --dry-run flag for verification

Models: DeepSeek-V3, GPT-4o-mini, Gemini Flash, Claude 3.5 Haiku (4 families)

Cost estimate: ~$8-12 for 1000 questions × 4 models × 4 conditions

Usage:
    python3 scripts/run_rag_experiment_v3.py --sample 1000
    python3 scripts/run_rag_experiment_v3.py --sample 100 --models deepseek-v3  # quick test
    python3 scripts/run_rag_experiment_v3.py --full  # all 3341 questions (expensive)
    python3 scripts/run_rag_experiment_v3.py --dry-run  # load everything, 0 API calls
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import random
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load env
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    for line in open(env_file):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            key, val = k.strip(), v.strip()
            if val:
                os.environ[key] = val

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger("rag_v3")

EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
PRIMEKG_DIR = PROJECT_ROOT / "data" / "primekg"
RESULTS_DIR = BENCHMARK_DIR / "rag_v3_results"

# HPO onset term IDs to human-readable names
HPO_ONSET_NAMES = {
    "HP:0003577": "Congenital onset",
    "HP:0003623": "Neonatal onset",
    "HP:0003593": "Infantile onset",
    "HP:0011463": "Childhood onset",
    "HP:0003621": "Juvenile onset",
    "HP:0003581": "Adult onset",
    "HP:0011462": "Young adult onset",
    "HP:0003596": "Middle age onset",
    "HP:0003584": "Late onset",
    "HP:0030674": "Antenatal onset",
    "HP:0034199": "Embryonal onset",
    "HP:0011460": "Fetal onset",
    "HP:0410280": "Pediatric onset",
}


# ============================================================================
# DATA LOADING
# ============================================================================

def _normalize_disease_id(did):
    """Normalize disease IDs: strip leading zeros, uppercase prefix."""
    if ":" not in did:
        return did
    prefix, num = did.split(":", 1)
    try:
        return f"{prefix.upper()}:{int(num)}"
    except ValueError:
        return did


def load_chronomedkg_index():
    """Load validated triples indexed by disease_id, with robust name matching."""
    index = {}  # normalized disease_id -> list of triples
    name_to_id = {}  # lowercase disease name -> disease_id

    for d in EXTRACTED_DIR.iterdir():
        if not d.is_dir():
            continue
        vt = d / "validated_triples.jsonl"
        if not vt.exists():
            continue
        did = d.name.replace("_", ":")
        did_norm = _normalize_disease_id(did)
        triples = []
        names_seen = set()
        with open(vt) as f:
            for line in f:
                if line.strip():
                    try:
                        t = json.loads(line)
                        triples.append(t)
                        # Collect ALL disease name variants
                        for field in ("source_name", "target_name"):
                            name = t.get(field, "").lower().strip()
                            if name and t.get("source_type") == "disease" and field == "source_name":
                                names_seen.add(name)
                            elif name and t.get("target_type") == "disease" and field == "target_name":
                                names_seen.add(name)
                    except Exception:
                        pass
        if triples:
            index[did_norm] = triples
            # Also index under original (non-normalized) key
            if did != did_norm:
                index[did] = triples
            for name in names_seen:
                name_to_id[name] = did_norm
            # First triple source_name as primary
            src_name = triples[0].get("source_name", "").lower().strip()
            if src_name:
                name_to_id[src_name] = did_norm

    logger.info("ChronoMedKG: %d diseases, %d name mappings", len(index), len(name_to_id))
    return index, name_to_id


def load_primekg_index():
    """Load PrimeKG edges indexed by disease name (lowercase), with type sub-index."""
    index = defaultdict(list)  # disease_name_lower -> edges
    kg_file = PRIMEKG_DIR / "kg.csv"

    if kg_file.exists():
        import csv
        with open(kg_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel = row.get("display_relation", row.get("relation", ""))
                x_name = row.get("x_name", "")
                y_name = row.get("y_name", "")
                x_type = row.get("x_type", "")
                y_type = row.get("y_type", "")

                if y_type == "disease":
                    index[y_name.lower().strip()].append({
                        "source": x_name,
                        "source_type": x_type,
                        "relation": rel,
                        "target": y_name,
                        "target_type": y_type,
                    })
                if x_type == "disease":
                    index[x_name.lower().strip()].append({
                        "source": x_name,
                        "source_type": x_type,
                        "relation": rel,
                        "target": y_name,
                        "target_type": y_type,
                    })

    logger.info("PrimeKG: %d diseases loaded", len(index))
    return dict(index)


def load_hpoa_index():
    """Load HPOA onset data indexed by OMIM ID and disease name (lowercase).

    Returns:
        hpoa_index: dict mapping OMIM ID or lowercase disease name -> onset record
        mondo_crosswalk: dict with mondo_to_omim mapping
    """
    hpoa_path = PROJECT_ROOT / "data" / "validation_sources" / "hpoa_with_ids.json"
    crosswalk_path = PROJECT_ROOT / "data" / "validation_sources" / "mondo_crosswalk.json"

    hpoa_index = {}  # keyed by OMIM ID and lowercase disease name
    mondo_crosswalk = {}

    if hpoa_path.exists():
        with open(hpoa_path) as f:
            hpoa_raw = json.load(f)
        for omim_id, record in hpoa_raw.items():
            # Index by OMIM ID (both raw and normalized)
            hpoa_index[omim_id] = record
            omim_norm = _normalize_disease_id(omim_id)
            if omim_norm != omim_id:
                hpoa_index[omim_norm] = record
            # Index by disease name (lowercase)
            name = record.get("name", "").lower().strip()
            if name:
                hpoa_index[name] = record
        logger.info("HPOA: %d entries loaded (%d with name index)",
                     len(hpoa_raw), len(hpoa_index))
    else:
        logger.warning("HPOA file not found: %s", hpoa_path)

    if crosswalk_path.exists():
        with open(crosswalk_path) as f:
            mondo_crosswalk = json.load(f)
        logger.info("MONDO crosswalk: %d MONDO->OMIM mappings",
                     len(mondo_crosswalk.get("mondo_to_omim", {})))
    else:
        logger.warning("MONDO crosswalk not found: %s", crosswalk_path)

    return hpoa_index, mondo_crosswalk


# ============================================================================
# QUESTION-AWARE RETRIEVAL
# ============================================================================

def extract_question_keywords(question):
    """Extract key phenotype/entity terms from question text for relevance matching."""
    q_text = question["question"].lower()
    # Remove common filler
    for filler in ["which of the following", "based on typical", "is most consistent",
                   "what phenotypes are", "rank the following", "is age",
                   "within the typical onset window", "does not typically present"]:
        q_text = q_text.replace(filler, "")
    # Extract meaningful words (>3 chars, not stopwords)
    # NOTE: Do NOT strip temporal terms (onset, age, stage, period) — they are
    # the most important retrieval signals for temporal questions!
    stopwords = {"the", "and", "for", "with", "from", "this", "that", "have", "has",
                 "are", "was", "were", "been", "being", "their", "which", "what",
                 "does", "during", "typically", "following", "clinical",
                 "most", "consistent", "presentation", "typical"}
    words = set(w.strip(".,?!()") for w in q_text.split() if len(w) > 3 and w not in stopwords)
    return words


def retrieve_chronomedkg(question, ta_index, ta_name_to_id, max_triples=15):
    """
    Question-aware retrieval from ChronoMedKG.
    Retrieves triples relevant to the specific question, not just random top-10.
    """
    q_type = question["type"]
    disease_ids = []

    # Step 1: Identify relevant disease(s)
    if q_type == "temporal_differential_dx":
        # Retrieve for ALL 4 diseases in the MCQ
        options = question.get("options", {})
        for opt_name in options.values():
            # Try to find disease ID by name
            opt_lower = opt_name.lower().strip()
            did = ta_name_to_id.get(opt_lower)
            if did:
                disease_ids.append(did)
    elif q_type == "cross_disease_comparison":
        # Retrieve for BOTH diseases
        gs = question["gold_standard"]
        for key in ["earlier_disease", "later_disease"]:
            name = gs.get(key, "").lower().strip()
            did = ta_name_to_id.get(name)
            if did:
                disease_ids.append(did)
    else:
        # Single disease — try normalized ID, then name
        did = question.get("disease_id", "")
        did_norm = _normalize_disease_id(did) if did else ""
        if did_norm and did_norm in ta_index:
            disease_ids.append(did_norm)
        elif did and did in ta_index:
            disease_ids.append(did)
        elif question.get("disease_name"):
            dn = question["disease_name"].lower().strip()
            did = ta_name_to_id.get(dn)
            if did:
                disease_ids.append(did)

    if not disease_ids:
        return ""  # No data — let model use parametric knowledge instead of confusing it

    # Step 2: Get triples and rank by relevance to question
    keywords = extract_question_keywords(question)
    all_scored = []

    for did in disease_ids:
        triples = ta_index.get(did, [])
        for t in triples:
            # Score: how many question keywords appear in triple content
            triple_text = (
                f"{t.get('source_name', '')} {t.get('target_name', '')} "
                f"{t.get('relation', '')} "
                f"{json.dumps(t.get('temporal', {}))}"
            ).lower()
            score = sum(1 for kw in keywords if kw in triple_text)

            # Bonus for triples with temporal data (what we're testing)
            temp = t.get("temporal", {}) or {}
            if temp.get("onset_age_min") is not None:
                score += 2
            if temp.get("progression_stage"):
                score += 1
            if temp.get("temporal_qualifier"):
                score += 1

            all_scored.append((score, t, did))

    # Sort by relevance score, take top N
    all_scored.sort(key=lambda x: -x[0])

    # Noise threshold: if best triple scores < 2, retrieval is too noisy — skip
    if all_scored and all_scored[0][0] < 2:
        return ""  # Low confidence retrieval — let model use parametric knowledge

    selected = all_scored[:max_triples]

    # Format as COMPACT structured table — only include columns that have data
    # This avoids sparse tables with 70% empty cells that confuse LLMs
    raw_rows = []
    for score, t, did in selected:
        temporal = t.get("temporal", {}) or {}
        row = {
            "Disease": t.get("source_name", "?"),
            "Relation": t.get("relation", "?"),
            "Entity": t.get("target_name", "?"),
        }
        # Only add temporal columns if they have data
        onset_min = temporal.get("onset_age_min")
        if onset_min is not None:
            row["Onset Age"] = f"{onset_min}-{temporal.get('onset_age_max', '?')}y"
        stage = temporal.get("progression_stage", "")
        if stage and stage.lower() not in ("unknown", "null", "none"):
            row["Stage"] = stage
        milestone = temporal.get("milestone", "")
        if milestone and milestone.lower() not in ("unknown", "null", "none"):
            row["Milestone"] = milestone
        timing = temporal.get("temporal_qualifier", "")
        if timing and timing.lower() not in ("unknown", "null", "none"):
            row["Timing"] = timing
        raw_rows.append(row)

    if not raw_rows:
        return ""

    # Compute aggregate onset range across all triples (for temporal_window context)
    onset_ages = []
    for score, t, did in selected:
        temp = t.get("temporal", {}) or {}
        if temp.get("onset_age_min") is not None:
            try:
                onset_ages.append(float(temp["onset_age_min"]))
            except (ValueError, TypeError):
                pass
        if temp.get("onset_age_max") is not None:
            try:
                onset_ages.append(float(temp["onset_age_max"]))
            except (ValueError, TypeError):
                pass

    # Build summary line with aggregate onset range
    summary = ""
    if onset_ages:
        min_onset = min(onset_ages)
        max_onset = max(onset_ages)
        disease_name = selected[0][1].get("source_name", "this disease") if selected else "this disease"
        summary = f"Overall onset range for {disease_name}: {min_onset:.1f}-{max_onset:.1f} years\n\n"

    # Determine which columns have data in ANY row (keep table dense)
    all_cols = ["Disease", "Relation", "Entity", "Onset Age", "Stage", "Milestone", "Timing"]
    active_cols = [c for c in all_cols if any(c in r for r in raw_rows)]

    header = "| " + " | ".join(active_cols) + " |"
    sep = "|" + "|".join("---" for _ in active_cols) + "|"
    rows = []
    for r in raw_rows:
        cells = [r.get(c, "") for c in active_cols]
        rows.append("| " + " | ".join(cells) + " |")

    return f"Biomedical knowledge graph data:\n{summary}{header}\n{sep}\n" + "\n".join(rows)


def retrieve_primekg(question, primekg_index, max_edges=15):
    """
    Question-aware retrieval from PrimeKG.
    Matches on disease name(s) from the question.
    """
    q_type = question["type"]
    disease_names = []

    if q_type == "temporal_differential_dx":
        options = question.get("options", {})
        disease_names = [n.lower().strip() for n in options.values()]
    elif q_type == "cross_disease_comparison":
        gs = question["gold_standard"]
        for key in ["earlier_disease", "later_disease"]:
            disease_names.append(gs.get(key, "").lower().strip())
    else:
        dn = question.get("disease_name", "").lower().strip()
        if dn:
            disease_names.append(dn)

    all_edges = []
    for dn in disease_names:
        edges = primekg_index.get(dn, [])
        all_edges.extend(edges)

    if not all_edges:
        return ""  # No data — let model use parametric knowledge

    # Rank by keyword relevance
    keywords = extract_question_keywords(question)
    scored = []
    for e in all_edges:
        edge_text = f"{e['source']} {e['relation']} {e['target']} {e['target_type']}".lower()
        score = sum(1 for kw in keywords if kw in edge_text)
        scored.append((score, e))

    scored.sort(key=lambda x: -x[0])
    selected = scored[:max_edges]

    # Format as structured table
    header = "| Source | Relation | Target | Type |"
    sep = "|--------|----------|--------|------|"
    rows = []
    for _, e in selected:
        rows.append(f"| {e['source']} | {e['relation']} | {e['target']} | {e['target_type']} |")

    if not rows:
        return ""

    return f"Biomedical knowledge graph data:\n{header}\n{sep}\n" + "\n".join(rows)


def _resolve_disease_to_omim_ids(disease_id, disease_name, mondo_crosswalk):
    """Resolve a MONDO disease ID (or disease name) to OMIM IDs via crosswalk.

    Returns a list of OMIM IDs (may be empty).
    """
    omim_ids = []
    if disease_id:
        did_norm = _normalize_disease_id(disease_id)
        # Direct OMIM ID
        if did_norm.startswith("OMIM:"):
            omim_ids.append(did_norm)
        # MONDO -> OMIM via crosswalk
        elif did_norm.startswith("MONDO:"):
            mondo_to_omim = mondo_crosswalk.get("mondo_to_omim", {})
            mapped = mondo_to_omim.get(did_norm, [])
            omim_ids.extend(mapped)
    return omim_ids


def retrieve_hpoa(question, hpoa_index, mondo_crosswalk):
    """Retrieve onset data from HPOA for the disease(s) in a question.

    Returns formatted context string, or empty string if no data found.
    """
    q_type = question["type"]
    disease_lookups = []  # list of (disease_id, disease_name) tuples

    if q_type == "temporal_differential_dx":
        # Multi-disease: look up all option diseases
        options = question.get("options", {})
        for opt_name in options.values():
            disease_lookups.append(("", opt_name))
    elif q_type == "cross_disease_comparison":
        gs = question["gold_standard"]
        for key in ["earlier_disease", "later_disease"]:
            name = gs.get(key, "")
            disease_lookups.append(("", name))
    else:
        # Single disease
        disease_lookups.append((
            question.get("disease_id", ""),
            question.get("disease_name", ""),
        ))

    results = []
    for did, dname in disease_lookups:
        record = None

        # Strategy 1: Resolve via MONDO -> OMIM crosswalk
        omim_ids = _resolve_disease_to_omim_ids(did, dname, mondo_crosswalk)
        for omim_id in omim_ids:
            if omim_id in hpoa_index:
                record = hpoa_index[omim_id]
                break

        # Strategy 2: Direct OMIM ID lookup (if disease_id is already OMIM)
        if record is None and did:
            did_norm = _normalize_disease_id(did)
            if did_norm in hpoa_index:
                record = hpoa_index[did_norm]

        # Strategy 3: Disease name lookup (lowercase)
        if record is None and dname:
            record = hpoa_index.get(dname.lower().strip())

        if record is None:
            continue

        # Format the onset data
        name = record.get("name", dname or "Unknown")
        min_age = record.get("min_age")
        max_age = record.get("max_age")
        onset_terms = record.get("onset_terms", [])

        # Resolve HPO onset term IDs to names
        onset_names = []
        for term in onset_terms:
            readable = HPO_ONSET_NAMES.get(term, term)
            onset_names.append(readable)

        onset_str = ", ".join(onset_names) if onset_names else "Unknown"
        age_str = f"{min_age}-{max_age} years" if min_age is not None and max_age is not None else "Unknown"

        results.append(f"Disease: {name} | Onset: {onset_str} | Age range: {age_str}")

    if not results:
        return ""

    return "Disease onset data from Human Phenotype Ontology Annotations (HPOA):\n" + "\n".join(results)


# ============================================================================
# NEUTRAL PROMPTS (no bias toward either KG)
# ============================================================================

def build_prompt(question, condition, context=""):
    """Build LLM prompt — neutral framing for all conditions."""
    q_text = question["question"]

    if condition == "no_retrieval" or not context.strip():
        # No retrieval, or retrieval returned empty — use parametric knowledge
        return (
            f"Answer the following biomedical question. "
            f"Be specific about ages, dates, and time periods where relevant.\n\n"
            f"Question: {q_text}\n\n"
            f"Answer:"
        )
    else:
        # RAG condition — provide context but don't override parametric knowledge
        return (
            f"Answer the following biomedical question. "
            f"You may use the knowledge graph data below if it is relevant. "
            f"You may also use your own medical knowledge. "
            f"Be specific about ages, dates, and time periods where relevant.\n\n"
            f"{context}\n\n"
            f"Question: {q_text}\n\n"
            f"Answer:"
        )


# ============================================================================
# TYPE-SPECIFIC SCORING
# ============================================================================

def score_mcq(llm_answer, correct_letter, options):
    """Score MCQ questions (differential dx, negative temporal, static controls)."""
    llm_lower = llm_answer.strip().lower()

    # Direct letter match
    if llm_lower.startswith(correct_letter.lower()):
        return True, "direct_letter_match"

    # Check if the correct option text appears in the answer
    correct_text = options.get(correct_letter, "").lower()
    if correct_text and correct_text in llm_lower:
        return True, "option_text_match"

    # Check if a WRONG option text appears (to distinguish from "don't know")
    for letter, text in options.items():
        if letter != correct_letter and text.lower() in llm_lower:
            return False, f"chose_wrong_option_{letter}"

    return False, "no_clear_answer"


def score_temporal_window(llm_answer, correct_answer):
    """Score Yes/No temporal window questions — flexible parsing."""
    llm_lower = llm_answer.strip().lower()
    correct_lower = correct_answer.lower()

    # Direct yes/no detection (look throughout the answer, not just first 50 chars)
    yes_signals = ["yes", "is within", "falls within", "consistent with", "within the typical",
                   "within the onset", "within this range", "is typical"]
    no_signals = ["no", "is not within", "outside", "not within", "not typical",
                  "does not fall", "is outside", "unlikely", "not consistent",
                  "no specific information", "no data"]

    if correct_lower == "yes":
        if any(s in llm_lower for s in yes_signals):
            # Make sure it's not negated
            if not any(neg in llm_lower[:100] for neg in ["no,", "no.", "no ", "not within", "outside"]):
                return True, "correct_yes"
    elif correct_lower == "no":
        if any(s in llm_lower for s in no_signals):
            if not any(pos in llm_lower[:100] for pos in ["yes,", "yes.", "yes "]):
                return True, "correct_no"

    return False, "incorrect_or_unclear"


def score_cross_disease(llm_answer, correct_disease):
    """Score cross-disease comparison (which has earlier onset)."""
    llm_lower = llm_answer.lower()
    if correct_disease.lower() in llm_lower:
        return True, "correct_disease_mentioned"
    return False, "correct_disease_not_found"


def score_onset_age(llm_answer, gold_min, gold_max):
    """Score onset age questions (phenopackets, temporal window with age)."""
    llm_lower = llm_answer.lower()

    # Try to extract age ranges
    patterns = [
        r'(\d+\.?\d*)\s*(?:to|-|–|and)\s*(\d+\.?\d*)\s*(?:years|yr)',
        r'age\s*(\d+\.?\d*)\s*(?:to|-|–|and)\s*(\d+\.?\d*)',
        r'between\s*(\d+\.?\d*)\s*and\s*(\d+\.?\d*)',
    ]
    for pat in patterns:
        match = re.search(pat, llm_lower)
        if match:
            try:
                llm_min = float(match.group(1))
                llm_max = float(match.group(2))
                if llm_min <= gold_max + 2 and llm_max >= gold_min - 2:
                    return True, f"age_overlap:{llm_min}-{llm_max}"
            except ValueError:
                pass

    # Try single age
    single = re.findall(r'(\d+\.?\d*)\s*years?\b', llm_lower)
    if single:
        try:
            age = float(single[0])
            if gold_min - 2 <= age <= gold_max + 2:
                return True, f"single_age_match:{age}"
        except ValueError:
            pass

    # Category keyword fallback
    category_kw = {
        "birth": (0, 0.1), "congenital": (0, 0.1), "prenatal": (0, 0),
        "neonatal": (0, 0.08), "infantile": (0.08, 2), "infancy": (0.08, 2),
        "childhood": (1, 11), "juvenile": (5, 15),
        "adolescen": (10, 18), "young adult": (15, 40),
        "adult": (18, 65), "elderly": (60, 120),
    }
    for kw, (kw_min, kw_max) in category_kw.items():
        if kw in llm_lower:
            if kw_min <= gold_max + 5 and kw_max >= gold_min - 5:
                return True, f"category_match:{kw}"

    return False, "no_match"


def score_ordering(llm_answer, correct_order, answer_detail):
    """Score phenotype ordering questions — flexible parsing from prose.

    The model typically outputs numbered paragraphs with milestone names.
    We extract the ORDER in which milestones are mentioned and compare to gold.
    """
    llm_lower = llm_answer.lower()
    # Strip markdown bold markers
    llm_clean = llm_lower.replace("**", "").replace("*", "")
    n_items = len(correct_order.split(" → "))
    milestones_gold_order = [d["milestone"].lower().strip() for d in answer_detail]

    # Method 1: Find each gold-order milestone's FIRST position in the LLM answer
    # This handles prose, numbered lists, bold formatting, etc.
    milestone_positions = []
    for m in milestones_gold_order:
        pos = llm_clean.find(m)
        if pos < 0:
            # Partial match — try key words (>4 chars, not generic)
            generic = {"onset", "presentation", "clinical", "early", "late", "stage",
                       "disease", "symptom", "birth", "adult", "initial"}
            key_words = [w for w in m.split() if len(w) > 4 and w not in generic]
            for w in key_words:
                pos = llm_clean.find(w)
                if pos >= 0:
                    break
        milestone_positions.append(pos)

    valid = [(i, p) for i, p in enumerate(milestone_positions) if p >= 0]
    if len(valid) >= n_items:
        # All found — check order
        positions_only = [p for _, p in valid]
        if positions_only == sorted(positions_only):
            return True, f"milestone_order_match_{len(valid)}/{n_items}"
    elif len(valid) >= 2:
        positions_only = [p for _, p in valid]
        if positions_only == sorted(positions_only):
            return True, f"partial_order_match_{len(valid)}/{n_items}"

    # Method 2: The model writes numbered lists — extract what's after each number
    # e.g., "1. **Fetal Diagnosis**" ... "2. **Diagnosis**" ... "3. **Primary Amenorrhea**"
    numbered_items = re.findall(r'(?:^|\n)\s*(\d+)\.\s*\**([^*\n:]+)', llm_clean)
    if len(numbered_items) >= n_items:
        # Map each numbered item to the closest gold milestone
        llm_order_names = [name.strip().lower() for _, name in numbered_items[:n_items]]
        # Check if the LLM's numbered order matches the gold order
        matched_gold_indices = []
        for llm_name in llm_order_names:
            best_match = -1
            best_overlap = 0
            for gi, gm in enumerate(milestones_gold_order):
                # Word overlap
                llm_words = set(llm_name.split())
                gold_words = set(gm.split())
                overlap = len(llm_words & gold_words)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = gi
            matched_gold_indices.append(best_match)

        # Check if matched indices are in ascending order (= correct)
        if (len(set(matched_gold_indices)) == len(matched_gold_indices) and
                matched_gold_indices == sorted(matched_gold_indices)):
            return True, f"numbered_list_match_{len(numbered_items)}/{n_items}"

    # Method 3: Extract ages and check if they MATCH gold standard ages (within 5y tolerance)
    # Then check if they're in the correct relative order
    ages_in_answer = re.findall(r'(\d+\.?\d*)\s*(?:years?|yr)', llm_clean)
    if len(ages_in_answer) >= 2 and answer_detail:
        try:
            llm_ages = [float(a) for a in ages_in_answer if 0 < float(a) <= 120]
            gold_ages = [d["age"] for d in answer_detail]  # Already in correct order

            # For each gold age, find the closest LLM age within 5-year tolerance
            matched_pairs = []
            used_llm_indices = set()
            for g_idx, g_age in enumerate(gold_ages):
                best_match = None
                best_gap = float('inf')
                for l_idx, l_age in enumerate(llm_ages):
                    if l_idx in used_llm_indices:
                        continue
                    gap = abs(l_age - g_age)
                    if gap < best_gap and gap <= 5.0:
                        best_gap = gap
                        best_match = l_idx
                if best_match is not None:
                    matched_pairs.append((g_idx, best_match, llm_ages[best_match]))
                    used_llm_indices.add(best_match)

            # Need at least n_items-1 matched pairs, in correct relative order
            if len(matched_pairs) >= n_items - 1:
                llm_positions = [mp[1] for mp in matched_pairs]
                if llm_positions == sorted(llm_positions):
                    return True, f"age_order_match_{len(matched_pairs)}/{n_items}"
        except (ValueError, IndexError):
            pass

    return False, "order_not_matched"


def score_stage_conditional(llm_answer, gold_phenotypes):
    """Score stage-conditional questions (list of phenotypes).

    Accepts if model mentions >=30% of gold phenotypes (many stages have 10+ phenotypes,
    and the LLM won't list all of them in a short answer).
    Also matches partial phenotype names (e.g., "hypotonia" matches "neonatal hypotonia").
    """
    llm_lower = llm_answer.lower()
    if not gold_phenotypes:
        return False, "no_gold_phenotypes"

    # Count matches — use both exact and partial matching
    matched = 0
    for p in gold_phenotypes:
        p_lower = p.lower()
        if p_lower in llm_lower:
            matched += 1
        else:
            # Try key words from the phenotype (e.g., "hypotonia" from "neonatal hypotonia")
            key_words = [w for w in p_lower.split() if len(w) > 4
                        and w not in ("early", "late", "severe", "mild", "chronic", "acute",
                                     "progressive", "congenital", "neonatal", "infantile")]
            if any(w in llm_lower for w in key_words):
                matched += 1

    ratio = matched / len(gold_phenotypes)

    # Threshold: 30% for large gold sets, 50% for small
    threshold = 0.3 if len(gold_phenotypes) >= 5 else 0.5
    if ratio >= threshold:
        return True, f"matched_{matched}/{len(gold_phenotypes)}"
    elif matched > 0:
        return False, f"partial_{matched}/{len(gold_phenotypes)}"
    return False, "no_phenotypes_found"


def score_question(llm_answer, question):
    """Route to type-specific scorer."""
    q_type = question["type"]
    gold = question["gold_standard"]

    if q_type in ("temporal_differential_dx", "negative_temporal_mcq",
                  "static_control_drug", "static_control_gene"):
        return score_mcq(llm_answer, question["answer"], question.get("options", {}))

    elif q_type == "temporal_window":
        return score_temporal_window(llm_answer, question["answer"])

    elif q_type == "cross_disease_comparison":
        return score_cross_disease(llm_answer, question["answer"])

    elif q_type == "phenopackets_onset":
        return score_onset_age(llm_answer, gold.get("onset_min", 0), gold.get("onset_max", 120))

    elif q_type == "phenotype_ordering":
        return score_ordering(llm_answer, question["answer"], question.get("answer_detail", []))

    elif q_type == "stage_conditional":
        return score_stage_conditional(llm_answer, question.get("answer", []))

    return False, "unknown_type"


# ============================================================================
# LLM CLIENTS
# ============================================================================

def call_openai(model, prompt, api_key):
    """Call OpenAI-compatible API."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=400,
    )
    return response.choices[0].message.content


def call_deepseek(prompt, api_key):
    """Call DeepSeek API (OpenAI-compatible)."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=400,
    )
    return response.choices[0].message.content


def call_gemini(prompt, api_key):
    """Call Google Gemini API."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0.1, max_output_tokens=400),
    )
    return response.text


def call_anthropic(prompt, api_key):
    """Call Anthropic Claude 3.5 Haiku API."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.content[0].text


def call_model(model_name, prompt):
    """Route to appropriate API."""
    if model_name.startswith("gpt-"):
        return call_openai(model_name, prompt, os.environ["OPENAI_API_KEY"])
    elif model_name == "deepseek-v3":
        return call_deepseek(prompt, os.environ["DEEPSEEK_API_KEY"])
    elif model_name.startswith("gemini") or model_name == "gemini-flash":
        return call_gemini(prompt, os.environ["GOOGLE_API_KEY"])
    elif model_name == "claude-haiku":
        return call_anthropic(prompt, os.environ["ANTHROPIC_API_KEY"])
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

def bootstrap_ci(scores, n_boot=1000, ci=0.95):
    """Compute bootstrap confidence interval for accuracy."""
    if not scores:
        return 0.0, 0.0, 0.0
    scores = np.array(scores, dtype=float)
    mean = scores.mean()
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        boot_means.append(sample.mean())
    boot_means = sorted(boot_means)
    lo = boot_means[int((1 - ci) / 2 * n_boot)]
    hi = boot_means[int((1 + ci) / 2 * n_boot)]
    return mean, lo, hi


# ============================================================================
# COST ESTIMATION
# ============================================================================

def estimate_cost(n_questions, n_models, n_conditions, models):
    """Estimate API cost based on question count, models, and conditions.

    Rough per-call cost estimates (400 output tokens, ~500 input tokens avg):
    - deepseek-v3: ~$0.0003/call
    - gpt-4o-mini: ~$0.0002/call
    - gemini-flash: ~$0.0001/call
    - claude-haiku: ~$0.0004/call
    """
    cost_per_call = {
        "deepseek-v3": 0.0003,
        "gpt-4o-mini": 0.0002,
        "gemini-flash": 0.0001,
        "claude-haiku": 0.0004,
        "gpt-4o": 0.005,
        "gpt-4.1-nano": 0.0002,
    }
    total = 0.0
    for m in models:
        c = cost_per_call.get(m, 0.001)  # default $0.001 for unknown models
        total += n_questions * n_conditions * c
    return total


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(questions, models, conditions,
                   ta_index, ta_name_to_id, primekg_index,
                   hpoa_index, mondo_crosswalk, dry_run=False):
    """Run full RAG experiment with per-question results and checkpointing."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Checkpoint: load existing results if resuming
    checkpoint_file = RESULTS_DIR / "checkpoint.jsonl"
    completed_qids = set()
    per_question_results = []

    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    per_question_results.append(r)
                    completed_qids.add((r["question_id"], r["model"], r["condition"]))
        logger.info("Resumed from checkpoint: %d results (%d unique questions)",
                     len(per_question_results), len(set(r["question_id"] for r in per_question_results)))

    total_calls = len(questions) * len(models) * len(conditions)
    completed = len(per_question_results)
    start = time.monotonic()
    errors = 0

    if dry_run:
        logger.info("DRY RUN: would make %d API calls. Exiting.", total_calls - completed)
        return per_question_results

    # Open checkpoint for appending
    ckpt_f = open(checkpoint_file, "a")

    for q_idx, question in enumerate(questions):
        # Pre-compute retrieval contexts for all conditions
        ta_context = retrieve_chronomedkg(question, ta_index, ta_name_to_id)
        pk_context = retrieve_primekg(question, primekg_index)
        hpoa_context = retrieve_hpoa(question, hpoa_index, mondo_crosswalk)

        for model in models:
            for condition in conditions:
                # Skip if already completed (checkpoint resume)
                if (question["id"], model, condition) in completed_qids:
                    continue

                if condition == "no_retrieval":
                    context = ""
                elif condition == "primekg_rag":
                    context = pk_context
                elif condition == "hpoa_rag":
                    context = hpoa_context
                elif condition == "chronomedkg_rag":
                    context = ta_context
                else:
                    context = ""

                prompt = build_prompt(question, condition, context)

                try:
                    llm_answer = call_model(model, prompt)
                except Exception as e:
                    llm_answer = f"ERROR: {e}"
                    errors += 1

                correct, reason = score_question(llm_answer, question)

                result = {
                    "question_id": question["id"],
                    "question_type": question["type"],
                    "disease": question.get("disease_name", ""),
                    "model": model,
                    "condition": condition,
                    "correct": correct,
                    "reason": reason,
                    "llm_answer": llm_answer[:500],
                    "context_length": len(context),
                }
                per_question_results.append(result)

                # Incremental checkpoint save
                ckpt_f.write(json.dumps(result) + "\n")
                ckpt_f.flush()

                completed += 1

                # Rate limiting between API calls
                time.sleep(0.2)

            # Small extra delay between models per question
            time.sleep(0.1)

        if (q_idx + 1) % 25 == 0:
            elapsed = time.monotonic() - start
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = (total_calls - completed) / rate if rate > 0 else 0
            logger.info(
                "Progress: %d/%d Qs (%d/%d calls, %.0f min left, %d errors)",
                q_idx + 1, len(questions), completed, total_calls, remaining / 60, errors,
            )

    ckpt_f.close()
    return per_question_results


def main():
    parser = argparse.ArgumentParser(description="ChronoMedKG RAG Experiment v3")
    parser.add_argument("--sample", type=int, default=1000, help="Questions to sample from Tier 1 (0=all)")
    parser.add_argument("--full", action="store_true", help="Run on all questions (no sampling)")
    parser.add_argument("--models", nargs="+",
                        default=["deepseek-v3", "gpt-4o-mini", "gemini-flash", "claude-haiku"],
                        help="Models to test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-controls", action="store_true", default=True,
                        help="Include all static control questions (default: True)")
    parser.add_argument("--no-controls", action="store_true",
                        help="Exclude static control questions")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load everything but make 0 API calls (for verification)")
    args = parser.parse_args()

    if args.no_controls:
        args.include_controls = False

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Type classifications
    TIER1_TYPES = {"temporal_window", "temporal_differential_dx", "cross_disease_comparison",
                   "phenopackets_onset"}
    TIER2_TYPES = {"phenotype_ordering", "stage_conditional"}
    CONTROL_TYPES = {"static_control_drug", "static_control_gene"}

    CONDITIONS = ["no_retrieval", "primekg_rag", "hpoa_rag", "chronomedkg_rag"]

    # Load benchmark
    logger.info("Loading benchmark...")
    with open(BENCHMARK_DIR / "chronomedkg_tqa_v6.json") as f:
        tqa = json.load(f)
    questions = tqa["questions"]
    logger.info("Total: %d questions, version: %s", len(questions), tqa.get("version", "?"))

    # Sampling strategy
    if args.full:
        sampled = questions
    else:
        # Tier 1 questions: stratified sample of --sample from these types
        tier1_qs = [q for q in questions if q["type"] in TIER1_TYPES]
        control_qs = [q for q in questions if q["type"] in CONTROL_TYPES]

        # Stratified sample from Tier 1
        by_type = defaultdict(list)
        for q in tier1_qs:
            by_type[q["type"]].append(q)

        sampled_tier1 = []
        total_tier1 = len(tier1_qs)
        for qtype, qs in by_type.items():
            random.shuffle(qs)
            # Proportional allocation
            n = max(5, int(len(qs) / total_tier1 * args.sample))
            sampled_tier1.extend(qs[:n])

        random.shuffle(sampled_tier1)
        sampled_tier1 = sampled_tier1[:args.sample]

        # Add static controls (all 500)
        if args.include_controls:
            sampled = sampled_tier1 + control_qs
        else:
            sampled = sampled_tier1

        random.shuffle(sampled)

    logger.info("Sampled %d questions", len(sampled))
    type_counts = Counter(q["type"] for q in sampled)
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        logger.info("  %s: %d", t, c)

    # Cost estimate
    est_cost = estimate_cost(len(sampled), len(args.models), len(CONDITIONS), args.models)
    logger.info("Estimated cost: $%.2f (%d Qs x %d models x %d conditions = %d API calls)",
                est_cost, len(sampled), len(args.models), len(CONDITIONS),
                len(sampled) * len(args.models) * len(CONDITIONS))

    # Load all data sources
    logger.info("Loading ChronoMedKG index...")
    ta_index, ta_name_to_id = load_chronomedkg_index()

    logger.info("Loading PrimeKG index...")
    primekg_index = load_primekg_index()

    logger.info("Loading HPOA index...")
    hpoa_index, mondo_crosswalk = load_hpoa_index()

    # Run
    logger.info("Running: %d Qs x %d models x %d conditions = %d API calls",
                len(sampled), len(args.models), len(CONDITIONS),
                len(sampled) * len(args.models) * len(CONDITIONS))
    logger.info("Models: %s", args.models)
    logger.info("Conditions: %s", CONDITIONS)

    results = run_experiment(
        sampled, args.models, CONDITIONS,
        ta_index, ta_name_to_id, primekg_index,
        hpoa_index, mondo_crosswalk,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        logger.info("Dry run complete. No results to aggregate.")
        return

    # Aggregate results
    agg = defaultdict(lambda: defaultdict(list))  # model -> condition -> [0/1]
    by_type_agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # type -> model -> condition -> [0/1]

    for r in results:
        agg[r["model"]][r["condition"]].append(1 if r["correct"] else 0)
        by_type_agg[r["question_type"]][r["model"]][r["condition"]].append(1 if r["correct"] else 0)

    # Print results with CIs
    print(f"\n{'='*100}")
    print(f"ChronoMedKG RAG Experiment v3 — Results")
    print(f"{'='*100}")
    print(f"Benchmark: TQA v{tqa.get('version', '7.1')} ({len(sampled)} questions)")
    print(f"Models: {', '.join(args.models)}")
    print(f"Conditions: {', '.join(CONDITIONS)}")
    print()

    print(f"{'Model':<18s} {'No Retrieval':>16s} {'+ PrimeKG':>16s} {'+ HPOA':>16s} {'+ ChronoMedKG':>18s} {'TA-NR':>8s} {'TA-PK':>8s} {'TA-HPOA':>8s}")
    print("-" * 110)

    for model in args.models:
        row = {}
        for cond in CONDITIONS:
            scores = agg[model][cond]
            mean, lo, hi = bootstrap_ci(scores)
            row[cond] = (mean, lo, hi)

        nr = row["no_retrieval"]
        pk = row["primekg_rag"]
        hp = row["hpoa_rag"]
        ta = row["chronomedkg_rag"]
        gain_nr = ta[0] - nr[0]
        gain_pk = ta[0] - pk[0]
        gain_hp = ta[0] - hp[0]

        print(f"{model:<18s} "
              f"{nr[0]*100:5.1f}% [{nr[1]*100:.0f}-{nr[2]*100:.0f}] "
              f"{pk[0]*100:5.1f}% [{pk[1]*100:.0f}-{pk[2]*100:.0f}] "
              f"{hp[0]*100:5.1f}% [{hp[1]*100:.0f}-{hp[2]*100:.0f}] "
              f"{ta[0]*100:5.1f}% [{ta[1]*100:.0f}-{ta[2]*100:.0f}] "
              f"{gain_nr*100:+6.1f}% "
              f"{gain_pk*100:+6.1f}% "
              f"{gain_hp*100:+6.1f}%")

    # Per-type breakdown
    print(f"\n{'='*100}")
    print("Per-Type Breakdown (ChronoMedKG RAG accuracy)")
    print(f"{'='*100}")
    print(f"{'Type':<30s}", end="")
    for model in args.models:
        print(f" {model:>12s}", end="")
    print()
    print("-" * (30 + 13 * len(args.models)))

    for qtype in sorted(by_type_agg.keys()):
        print(f"{qtype:<30s}", end="")
        for model in args.models:
            scores = by_type_agg[qtype][model]["chronomedkg_rag"]
            if scores:
                acc = sum(scores) / len(scores) * 100
                print(f" {acc:11.1f}%", end="")
            else:
                print(f" {'N/A':>12s}", end="")
        print()

    # Temporal vs Static comparison (the key finding)
    print(f"\n{'='*100}")
    print("HEADLINE: Temporal vs Static Question Performance")
    print(f"{'='*100}")

    temporal_types = {"temporal_window", "temporal_differential_dx", "cross_disease_comparison",
                      "phenotype_ordering", "stage_conditional", "negative_temporal_mcq",
                      "phenopackets_onset"}
    static_types = {"static_control_drug", "static_control_gene"}

    for model in args.models:
        temp_scores = {cond: [] for cond in CONDITIONS}
        stat_scores = {cond: [] for cond in CONDITIONS}

        for r in results:
            if r["model"] != model:
                continue
            if r["question_type"] in temporal_types:
                temp_scores[r["condition"]].append(1 if r["correct"] else 0)
            elif r["question_type"] in static_types:
                stat_scores[r["condition"]].append(1 if r["correct"] else 0)

        print(f"\n{model}:")
        for label, scores in [("Temporal Qs", temp_scores), ("Static Qs", stat_scores)]:
            nr = np.mean(scores["no_retrieval"]) * 100 if scores["no_retrieval"] else 0
            pk = np.mean(scores["primekg_rag"]) * 100 if scores["primekg_rag"] else 0
            hp = np.mean(scores["hpoa_rag"]) * 100 if scores["hpoa_rag"] else 0
            ta = np.mean(scores["chronomedkg_rag"]) * 100 if scores["chronomedkg_rag"] else 0
            print(f"  {label:15s}: NR {nr:5.1f}% | +PK {pk:5.1f}% | +HPOA {hp:5.1f}% | +TA {ta:5.1f}% (TA-NR: {ta-nr:+.1f}%)")

    # Save everything
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Per-question results
    pq_file = RESULTS_DIR / "per_question_results.json"
    with open(pq_file, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    summary = {
        "experiment": "ChronoMedKG RAG Experiment v3",
        "benchmark": f"TQA v{tqa.get('version', '7.1')}",
        "questions_sampled": len(sampled),
        "models": args.models,
        "conditions": CONDITIONS,
        "aggregate": {
            model: {
                cond: {
                    "accuracy": float(np.mean(agg[model][cond])),
                    "ci_lower": float(bootstrap_ci(agg[model][cond])[1]),
                    "ci_upper": float(bootstrap_ci(agg[model][cond])[2]),
                    "n": len(agg[model][cond]),
                }
                for cond in CONDITIONS
            }
            for model in args.models
        },
        "by_type": {
            qtype: {
                model: {
                    cond: {
                        "accuracy": float(np.mean(by_type_agg[qtype][model][cond]))
                        if by_type_agg[qtype][model][cond] else None,
                        "n": len(by_type_agg[qtype][model][cond]),
                    }
                    for cond in CONDITIONS
                }
                for model in args.models
            }
            for qtype in sorted(by_type_agg.keys())
        },
    }

    sum_file = RESULTS_DIR / "experiment_summary.json"
    with open(sum_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved per-question results: {pq_file}")
    print(f"Saved summary: {sum_file}")


if __name__ == "__main__":
    main()
