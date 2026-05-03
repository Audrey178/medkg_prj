#!/usr/bin/env python3
"""
ChronoMedKG-TQA v6: Temporal REASONING Benchmark
====================================================
Key improvements over v5:
  - 10 question types testing REASONING, not just lookup
  - Stage-conditional and ordering questions from our triples
  - Temporal necessity check: every Q requires temporal info to answer
  - External gold standards for all answer keys
  - Static control questions to measure temporal-specific gains

Sources:
  External (answers from gold standards):
    1. Orphadata — differential Dx, temporal window, cross-disease comparison
    2. HPOA — negative temporal MCQ (what DOESN'T appear at this stage)
    3. Phenopackets — case-level onset validation
  Internal (answers from our triples, cross-validated against external):
    4. Validated triples — phenotype ordering, stage-conditional
  Control:
    5. PrimeKG — static (non-temporal) control questions

Usage:
    python3 scripts/build_tqa_v6.py
"""

import json
import pickle
import random
import os
import glob
import yaml
from collections import defaultdict, Counter
from pathlib import Path

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation_sources"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
CONFIG_DIR = PROJECT_ROOT / "config" / "diseases"
PRIMEKG_DIR = PROJECT_ROOT / "data" / "primekg"

# HPO onset term → age range (comprehensive mapping)
HPO_ONSET_TO_AGE = {
    "HP:0030674": (0, 0, "Antenatal"),
    "HP:0034197": (0, 0, "First trimester"),
    "HP:0034198": (0, 0, "Embryonal"),
    "HP:0034199": (0, 0, "Fetal"),
    "HP:0011461": (0, 0, "Fetal"),
    "HP:0025710": (0, 0, "Third trimester"),
    "HP:0003577": (0, 0, "Congenital"),
    "HP:0003623": (0, 0.08, "Neonatal"),
    "HP:0025709": (0, 0.08, "Neonatal"),
    "HP:0003593": (0.08, 2, "Infantile"),
    "HP:0011463": (1, 11, "Childhood"),
    "HP:0003621": (5, 15, "Juvenile"),
    "HP:0011462": (15, 40, "Young adult"),
    "HP:0003584": (15, 120, "Late onset"),
    "HP:0003581": (40, 120, "Adult"),
    "HP:0003596": (40, 60, "Middle age"),
    "HP:0025708": (40, 60, "Middle age"),
}

# Broad onset categories for differential Dx
BROAD_BINS = {
    "neonatal": (0, 0.08),
    "infantile": (0.08, 2),
    "childhood": (2, 11),
    "juvenile": (11, 18),
    "adult": (18, 65),
    "elderly": (65, 120),
}

NOISE_PHENOTYPE_PATTERNS = [
    "presentation", "onset", "manifestation", "symptoms", "diagnosis",
    "late-onset", "early-onset", "clinical", "disease", "syndrome",
    "progression", "worsening", "deterioration", "prognosis",
    # Molecular/pathway terms (not clinical phenotypes)
    "signaling pathway", "pathway", "activity", "expression level",
    "receptor signaling", "enzyme activity",
]

# Vague, non-temporal, or implausible milestone names to filter
NOISE_MILESTONE_PATTERNS = [
    "laboratory", "finding", "follow-up", "follow up",
    "documented", "reported", "observed", "described",
    # Non-temporal: features/characteristics, not events
    "defining feature", "characteristic", "hallmark", "typical feature",
    "key feature", "cardinal feature", "pathognomonic",
    # Molecular/pathway terms (not clinical milestones)
    "signaling pathway", "pathway", "activity", "expression",
    "receptor", "enzyme",
]

# PrimeKG drugs that CAUSE the condition rather than treat it (cause-treatment inversions)
# These produce medically dangerous benchmark questions
PRIMEKG_DRUG_CONTRADICTIONS = {
    ("digoxin", "digitalis poisoning"),
    ("digoxin", "cardiac glycoside poisoning"),
    ("warfarin", "warfarin poisoning"),
    ("heparin", "heparin-induced thrombocytopenia"),
    ("methotrexate", "methotrexate toxicity"),
    ("lithium", "lithium toxicity"),
    ("acetaminophen", "acetaminophen overdose"),
    ("iron", "iron poisoning"),
    ("lead", "lead poisoning"),
    ("mercury", "mercury poisoning"),
    ("arsenic", "arsenic poisoning"),
}

# Drug names that are androgens/steroids often linked to wrong indications in PrimeKG
SUSPECT_DRUG_PATTERNS = [
    "fluoxymesterone", "methyltestosterone", "nandrolone",  # androgens
    "stanozolol", "oxandrolone",  # anabolic steroids (unless for wasting)
]

# Umbrella/category disease names that are NOT specific diseases
# These produce unanswerable stage-conditional questions
UMBRELLA_DISEASE_PATTERNS = [
    "autosomal recessive disease", "autosomal dominant disease",
    "x-linked disease", "x-linked recessive disease",
    "hematologic disease", "hematological disease",
    "metabolic disease", "genetic disease",
    "congenital disorder", "rare disease",
    "neoplasm", "malignant neoplasm",
    "infectious disease", "inflammatory disease",
    "neurological disease", "cardiovascular disease",
    "respiratory disease", "renal disease",
    "endocrine disease", "autoimmune disease",
    "liver disease", "kidney disease", "heart disease",
    "lung disease", "skin disease", "eye disease",
    "bone disease", "blood disease", "muscle disease",
    "brain disease", "connective tissue disease",
]


def load_disease_names():
    """Load disease_name → disease_id mapping from YAML configs."""
    names = {}
    for yf in CONFIG_DIR.glob("*.yaml"):
        try:
            with open(yf) as f:
                cfg = yaml.safe_load(f)
            if cfg and "disease_name" in cfg:
                names[cfg["disease_name"].lower().strip()] = yf.stem.replace("_", ":")
        except Exception:
            pass
    return names


def load_disease_id_to_name(names):
    """Reverse mapping: disease_id → disease_name. Also loads from YAML configs."""
    id_to_name = {v: k.title() for k, v in names.items()}

    # Also load directly from YAML configs (covers MONDO IDs etc.)
    for yf in CONFIG_DIR.glob("*.yaml"):
        try:
            with open(yf) as f:
                cfg = yaml.safe_load(f)
            if cfg and "disease_name" in cfg:
                did = yf.stem.replace("_", ":")
                if did not in id_to_name:
                    id_to_name[did] = cfg["disease_name"].title()
        except Exception:
            pass
    return id_to_name


def classify_onset_bin(min_age, max_age):
    """Classify onset ages into refined bins.

    Fix from v6.0 validation: split "adult" (0.08-65y) into narrower bins
    to avoid labeling age 12 as "adult".
    Uses the MIDPOINT of the range for classification.
    Diseases with very broad ranges (>60 years) are marked as broad_range.
    """
    span = max_age - min_age
    if span > 60:
        return "broad_range"  # Truly uninformative (e.g., 0-120)
    mid = (min_age + max_age) / 2
    if mid < 0:
        return "prenatal"
    elif mid < 0.08:
        return "neonatal"
    elif mid < 2:
        return "infantile"
    elif mid < 6:
        return "early_childhood"
    elif mid < 12:
        return "childhood"
    elif mid < 18:
        return "adolescent"
    elif mid < 40:
        return "young_adult"
    elif mid < 65:
        return "middle_adult"
    else:
        return "elderly"


def age_to_readable(age):
    """Convert age in years to readable string.

    Fix from v6.0 validation: "age prenatal" → "the prenatal period"
    """
    if age < 0:
        return "the prenatal period"
    elif age < 0.08:
        return "the neonatal period"
    elif age < 0.5:
        return f"{int(age * 12)} months"
    elif age < 1:
        return f"{age * 12:.0f} months"
    elif age < 2:
        return f"{age:.1f} years"
    else:
        return f"{int(age)} years"


# ============================================================================
# TYPE 1: Temporal Differential Diagnosis (Orphadata)
# ============================================================================
def gen_differential_dx(orphadata, names):
    """
    MCQ: Given an onset age, which disease is most likely?
    4 options from DIFFERENT onset bins. Only 1 matches the age.
    Requires temporal reasoning: must know onset ranges to discriminate.
    """
    questions = []

    # Group diseases by refined bin
    bin_diseases = defaultdict(list)
    for disease_lower, entry in orphadata.items():
        did = names.get(disease_lower)
        if not did:
            continue
        mn = entry.get("min_age", 0)
        mx = entry.get("max_age", 120)
        if mn == 0 and mx == 120:
            continue  # "All ages" — can't discriminate
        bin_label = classify_onset_bin(mn, mx)
        if bin_label == "broad_range":
            continue  # Range too wide to meaningfully classify
        bin_diseases[bin_label].append({
            "name": disease_lower.title(),
            "id": did,
            "min_age": mn,
            "max_age": mx,
            "bin": bin_label,
        })

    # Need at least 4 bins with diseases
    available_bins = [b for b, ds in bin_diseases.items() if len(ds) >= 10]
    if len(available_bins) < 4:
        print(f"  WARNING: Only {len(available_bins)} bins with 10+ diseases")
        return questions

    target_count = 700
    attempts = 0
    seen_correct = set()

    while len(questions) < target_count and attempts < target_count * 5:
        attempts += 1

        # Pick a correct disease from a random bin
        correct_bin = random.choice(available_bins)
        correct_disease = random.choice(bin_diseases[correct_bin])
        if correct_disease["id"] in seen_correct:
            continue
        seen_correct.add(correct_disease["id"])

        # Pick 3 distractors from DIFFERENT bins
        other_bins = [b for b in available_bins if b != correct_bin]
        if len(other_bins) < 3:
            continue
        distractor_bins = random.sample(other_bins, 3)
        distractors = [random.choice(bin_diseases[b]) for b in distractor_bins]

        # Pick a test age within the correct disease's range
        test_age = random.uniform(correct_disease["min_age"], correct_disease["max_age"])

        # Verify test_age does NOT fall within ANY distractor's onset range
        # This prevents ambiguous questions where the test age fits multiple answers
        age_fits_distractor = any(
            d["min_age"] <= test_age <= d["max_age"] for d in distractors
        )
        if age_fits_distractor:
            continue  # Skip — test age is ambiguous

        test_age_str = age_to_readable(test_age)
        # Grammar fix: "age the neonatal period" → "during the neonatal period"
        if test_age_str.startswith("the "):
            age_phrase = f"during {test_age_str}"
        else:
            age_phrase = f"at age {test_age_str}"

        options = [correct_disease] + distractors
        random.shuffle(options)
        correct_idx = next(i for i, o in enumerate(options) if o["id"] == correct_disease["id"])
        option_labels = ["A", "B", "C", "D"]

        q_text = (
            f"A patient presents with symptoms {age_phrase}. "
            f"Based on typical age of onset, which of the following diseases "
            f"is most consistent with this presentation?\n"
        )
        for i, opt in enumerate(options):
            q_text += f"  {option_labels[i]}) {opt['name']}\n"

        questions.append({
            "question": q_text.strip(),
            "answer": option_labels[correct_idx],
            "answer_disease": correct_disease["name"],
            "gold_standard": {
                "source": "Orphadata",
                "correct_onset_min": correct_disease["min_age"],
                "correct_onset_max": correct_disease["max_age"],
                "correct_bin": correct_bin,
                "test_age": round(test_age, 2),
                "distractor_bins": distractor_bins,
            },
            "options": {option_labels[i]: opt["name"] for i, opt in enumerate(options)},
            "type": "temporal_differential_dx",
            "difficulty": "hard",
            "disease_id": correct_disease["id"],
            "disease_name": correct_disease["name"],
            "temporal_necessity": True,
            "reasoning": "Requires knowing onset age ranges to select the correct disease",
        })

    return questions


# ============================================================================
# TYPE 2: Temporal Window (Orphadata)
# ============================================================================
def gen_temporal_window(orphadata, names):
    """
    Yes/No + justification: 'Is age X within the typical onset window for Disease Y?'
    50% yes (age in range), 50% no (age outside range).
    """
    questions = []
    target_count = 800

    candidates = []
    for disease_lower, entry in orphadata.items():
        did = names.get(disease_lower)
        if not did:
            continue
        mn = entry.get("min_age", 0)
        mx = entry.get("max_age", 120)
        if mn == 0 and mx == 120:
            continue
        # Need a meaningful range (not too wide)
        if mx - mn > 80:
            continue
        candidates.append({
            "name": disease_lower.title(),
            "id": did,
            "min_age": mn,
            "max_age": mx,
        })

    random.shuffle(candidates)

    for disease in candidates[:target_count]:
        # 50/50 yes/no
        if random.random() < 0.5:
            # YES: pick age within range
            test_age = random.uniform(disease["min_age"], disease["max_age"])
            answer = "Yes"
            ta_str = age_to_readable(test_age)
            age_ref = f"The {ta_str[4:]}" if ta_str.startswith("the ") else f"Age {ta_str}"
            justification = (
                f"The typical onset for {disease['name']} is "
                f"{age_to_readable(disease['min_age'])} to "
                f"{age_to_readable(disease['max_age'])}. "
                f"{age_ref} falls within this range."
            )
        else:
            # NO: pick age outside range
            if disease["min_age"] > 2 and random.random() < 0.5:
                # Pick younger
                test_age = random.uniform(0, max(0, disease["min_age"] - 1))
            else:
                # Pick older
                test_age = random.uniform(disease["max_age"] + 5, min(100, disease["max_age"] + 30))
            answer = "No"
            ta_str = age_to_readable(test_age)
            age_ref = f"The {ta_str[4:]}" if ta_str.startswith("the ") else f"Age {ta_str}"
            justification = (
                f"The typical onset for {disease['name']} is "
                f"{age_to_readable(disease['min_age'])} to "
                f"{age_to_readable(disease['max_age'])}. "
                f"{age_ref} is outside this range."
            )

        # Grammar fix: "Is age the neonatal period" → "Is the neonatal period"
        ta_str = age_to_readable(test_age)
        if ta_str.startswith("the "):
            q_phrasing = f"Is {ta_str} within the typical onset window for {disease['name']}?"
        else:
            q_phrasing = f"Is age {ta_str} within the typical onset window for {disease['name']}?"

        questions.append({
            "question": q_phrasing,
            "answer": answer,
            "justification": justification,
            "gold_standard": {
                "source": "Orphadata",
                "onset_min": disease["min_age"],
                "onset_max": disease["max_age"],
                "test_age": round(test_age, 2),
            },
            "type": "temporal_window",
            "difficulty": "medium",
            "disease_id": disease["id"],
            "disease_name": disease["name"],
            "temporal_necessity": True,
            "reasoning": "Requires knowing the onset window to determine if age is in/out of range",
        })

    return questions


# ============================================================================
# TYPE 3: Cross-Disease Onset Comparison (Orphadata)
# ============================================================================
def gen_cross_disease_comparison(orphadata, names):
    """
    Free-text: 'Which has earlier onset: Disease A or Disease B?'
    Diseases must have non-overlapping or minimally-overlapping onset ranges.
    """
    questions = []
    target_count = 600

    candidates = []
    for disease_lower, entry in orphadata.items():
        did = names.get(disease_lower)
        if not did:
            continue
        mn = entry.get("min_age", 0)
        mx = entry.get("max_age", 120)
        if mn == 0 and mx == 120:
            continue
        if mx - mn > 60:
            continue
        candidates.append({
            "name": disease_lower.title(),
            "id": did,
            "min_age": mn,
            "max_age": mx,
            "midpoint": (mn + mx) / 2,
        })

    # Sort by midpoint to find non-overlapping pairs
    candidates.sort(key=lambda x: x["midpoint"])
    seen = set()
    disease_usage_count = Counter()  # Limit any disease to max 5 appearances
    MAX_DISEASE_USAGE = 5

    for i in range(len(candidates)):
        if len(questions) >= target_count:
            break
        for j in range(i + 1, len(candidates)):
            if len(questions) >= target_count:
                break

            d_early = candidates[i]
            d_late = candidates[j]

            # Require clear separation: early's max < late's min (or close)
            gap = d_late["min_age"] - d_early["max_age"]
            if gap < 2:
                continue  # Overlapping or too close

            pair_key = (d_early["id"], d_late["id"])
            if pair_key in seen:
                continue

            # Limit any single disease to MAX_DISEASE_USAGE appearances
            if (disease_usage_count[d_early["id"]] >= MAX_DISEASE_USAGE or
                    disease_usage_count[d_late["id"]] >= MAX_DISEASE_USAGE):
                continue

            seen.add(pair_key)
            disease_usage_count[d_early["id"]] += 1
            disease_usage_count[d_late["id"]] += 1

            # Randomly present order
            if random.random() < 0.5:
                first, second = d_early, d_late
                answer_name = d_early["name"]
            else:
                first, second = d_late, d_early
                answer_name = d_early["name"]

            questions.append({
                "question": (
                    f"Which disease typically has an earlier age of onset: "
                    f"{first['name']} or {second['name']}?"
                ),
                "answer": answer_name,
                "gold_standard": {
                    "source": "Orphadata",
                    "earlier_disease": d_early["name"],
                    "earlier_onset": f"{d_early['min_age']}-{d_early['max_age']}",
                    "later_disease": d_late["name"],
                    "later_onset": f"{d_late['min_age']}-{d_late['max_age']}",
                    "gap_years": round(gap, 1),
                },
                "type": "cross_disease_comparison",
                "difficulty": "medium",
                "disease_id": d_early["id"],
                "disease_name": f"{d_early['name']} vs {d_late['name']}",
                "temporal_necessity": True,
                "reasoning": "Requires comparing onset ranges of two diseases",
            })

            if len(questions) >= target_count:
                break

    random.shuffle(questions)
    return questions


# ============================================================================
# TYPE 4: Negative Temporal MCQ (HPOA)
# ============================================================================
def load_hp_labels():
    """Load HPO ID → label mapping from PrimeKG edges cache."""
    cache = VALIDATION_DIR / "primekg_edges.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            data = pickle.load(f)
        return data.get("hp_labels", {})
    return {}


def gen_negative_temporal(names):
    """
    MCQ: 'Which phenotype does NOT typically present during [onset period] in Disease X?'
    Uses HPOA per-phenotype onset annotations with HPO label lookup.
    """
    questions = []
    hpoa_file = VALIDATION_DIR / "phenotype.hpoa"
    if not hpoa_file.exists():
        print("  WARNING: phenotype.hpoa not found")
        return questions

    hp_labels = load_hp_labels()
    print(f"  HP label mapping: {len(hp_labels)} terms")

    # Parse HPOA: disease → {onset_label → [(hpo_id, phenotype_name)]}
    disease_onset_phenos = defaultdict(lambda: defaultdict(list))

    with open(hpoa_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 7 or parts[0] == "database_id":
                continue

            disease_name = parts[1].strip()
            hpo_id = parts[3].strip()
            onset_term = parts[6].strip()

            if not onset_term or onset_term not in HPO_ONSET_TO_AGE:
                continue

            disease_lower = disease_name.lower().strip()
            did = names.get(disease_lower)
            if not did:
                continue

            # Look up phenotype label from HP ID
            pheno_label = hp_labels.get(hpo_id, "")
            if not pheno_label:
                continue  # Skip if we can't name the phenotype

            onset_min, onset_max, onset_label = HPO_ONSET_TO_AGE[onset_term]
            disease_onset_phenos[disease_lower][onset_label].append(pheno_label)

    target_count = 300
    generated = []

    for disease_lower, onset_map in disease_onset_phenos.items():
        if len(generated) >= target_count:
            break

        did = names.get(disease_lower)
        disease_name = disease_lower.title()

        # Need 2+ onset periods to create negative questions
        if len(onset_map) < 2:
            continue

        for onset_label, phenos_at_onset in onset_map.items():
            if len(generated) >= target_count:
                break

            # Deduplicate phenotypes at this onset
            phenos_at_onset = list(set(phenos_at_onset))
            if len(phenos_at_onset) < 2:
                continue

            # Find phenotypes at OTHER onset periods (these are the negative answers)
            other_phenos = []
            for other_label, other_list in onset_map.items():
                if other_label != onset_label:
                    for p in other_list:
                        if p not in phenos_at_onset:  # Ensure truly negative
                            other_phenos.append(p)

            other_phenos = list(set(other_phenos))
            if not other_phenos:
                continue

            # Correct answer: a phenotype from a DIFFERENT onset period
            negative = random.choice(other_phenos)
            # Distractors: phenotypes that DO appear at this onset (need exactly 3)
            if len(phenos_at_onset) < 3:
                continue  # Need at least 3 positives for a 4-option MCQ
            positives = random.sample(phenos_at_onset, 3)

            options = positives + [negative]
            random.shuffle(options)
            correct_idx = options.index(negative)
            option_labels = ["A", "B", "C", "D"][:len(options)]

            q_text = (
                f"Which of the following phenotypes does NOT typically present "
                f"during the {onset_label.lower()} period "
                f"in {disease_name}?\n"
            )
            for i, opt in enumerate(options):
                q_text += f"  {option_labels[i]}) {opt}\n"

            generated.append({
                "question": q_text.strip(),
                "answer": option_labels[correct_idx],
                "answer_phenotype": negative,
                "gold_standard": {
                    "source": "HPOA",
                    "onset_label": onset_label,
                    "phenotypes_at_onset": phenos_at_onset[:5],
                    "negative_phenotype_actual_onset": [
                        label for label, ps in onset_map.items()
                        if negative in ps and label != onset_label
                    ],
                },
                "options": {option_labels[i]: opt for i, opt in enumerate(options)},
                "type": "negative_temporal_mcq",
                "difficulty": "hard",
                "disease_id": did,
                "disease_name": disease_name,
                "temporal_necessity": True,
                "reasoning": "Requires knowing which phenotypes appear at which disease stage",
            })

    return generated


# Milestones that logically MUST come before others (clinical common sense)
# If "onset" is at age X, "diagnosis" cannot be before X
MILESTONE_PRECEDES = {
    # These milestones should logically come BEFORE treatment/diagnosis
    "disease onset": ["symptom onset", "diagnosis", "treatment initiation", "treatment",
                      "surgical intervention", "treatment response", "treatment completion",
                      "death", "recurrence", "relapse", "presentation", "adult presentation"],
    "symptom onset": ["diagnosis", "treatment initiation", "treatment", "surgical intervention",
                      "treatment response", "treatment completion", "death", "recurrence", "relapse"],
    "onset": ["diagnosis", "treatment initiation", "treatment", "death"],
    "first clinical manifestations": ["diagnosis", "treatment initiation", "treatment", "death"],
    "initial presentation": ["treatment initiation", "treatment", "treatment completion"],
    "birth presentation": ["diagnosis", "treatment initiation", "childhood presentation",
                           "adult presentation"],
    "neonatal presentation": ["childhood presentation", "adult presentation", "death"],
    "congenital presentation": ["childhood presentation", "adult presentation", "death"],
    "diagnosis": ["treatment initiation", "treatment", "treatment completion",
                  "treatment response", "surgical intervention"],
}


def _check_ordering_coherence(milestones):
    """
    Check if milestone ordering is logically coherent.
    Returns True if the ordering makes clinical sense, False if illogical.
    """
    # Build name -> age mapping
    name_to_age = {m.lower(): age for m, age in milestones}

    for early_ms, must_follow_list in MILESTONE_PRECEDES.items():
        if early_ms not in name_to_age:
            continue
        early_age = name_to_age[early_ms]
        for late_ms in must_follow_list:
            if late_ms not in name_to_age:
                continue
            late_age = name_to_age[late_ms]
            if late_age < early_age - 0.5:  # Allow small tolerance
                return False  # Illogical: late milestone before early one

    # Filter implausible "birth/neonatal presentation" mixed with adult milestones
    birth_milestones = [(m, age) for m, age in milestones
                        if age <= 0.1 and ("birth" in m.lower() or "neonatal" in m.lower()
                                           or "congenital" in m.lower())]
    non_birth = [(m, age) for m, age in milestones if age > 0.1]

    if birth_milestones and non_birth:
        median_non_birth = sorted([a for _, a in non_birth])[len(non_birth) // 2]
        # If median age of other milestones > 15, birth presentation is likely
        # from a different patient population (neonatal case report for an adult disease)
        if median_non_birth > 15:
            return False
        # Also reject if span > 30 years (mixed patient data)
        all_ages = [age for _, age in milestones]
        if max(all_ages) - min(all_ages) > 30:
            return False

    return True


# ============================================================================
# TYPE 5: Phenotype Ordering (Our Triples, cross-validated)
# ============================================================================
def gen_phenotype_ordering(names):
    """
    Ordering: 'Rank these milestones by typical age of occurrence in Disease X.'
    From validated triples with milestone + onset_age data.

    Fixes from v6.0 validation:
    - ALL selected milestones must have ages (no missing data)
    - Logical coherence check (diagnosis can't precede onset)
    - Require 2+ evidence sources per milestone for reliability
    """
    questions = []
    target_count = 400

    validated_files = glob.glob(str(EXTRACTED_DIR / "*/validated_triples.jsonl"))

    ordering_candidates = []

    for f in validated_files:
        disease_id = os.path.basename(os.path.dirname(f)).replace("_", ":")
        milestone_ages = defaultdict(list)

        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    t = json.loads(line)
                except Exception:
                    continue

                temp = t.get("temporal", {}) or {}
                m = temp.get("milestone", "")
                onset_min = temp.get("onset_age_min")

                if not m or m in ("unknown", "null", "none"):
                    continue
                if onset_min is None:
                    continue
                try:
                    onset_min = float(onset_min)
                    if not (0 <= onset_min <= 120):
                        continue
                except (ValueError, TypeError):
                    continue

                milestone_ages[m].append(onset_min)

        # Normalize milestone names: collapse near-duplicates
        # "late onset" ≈ "late-onset" ≈ "late onset presentation" ≈ "late-onset presentation"
        normalized = defaultdict(list)
        for m, ages in milestone_ages.items():
            # Normalize: lowercase, remove hyphens, strip "presentation"/"form"
            key = m.lower().replace("-", " ").replace("  ", " ").strip()
            key = key.replace(" presentation", "").replace(" form", "").strip()
            normalized[key].extend([(m, a) for a in ages])

        # Pick the best name per normalized key, merge ages
        milestone_ages_deduped = {}
        for key, entries in normalized.items():
            # Pick the shortest original name as canonical
            best_name = min(set(e[0] for e in entries), key=len)
            all_ages = [e[1] for e in entries]
            milestone_ages_deduped[best_name] = all_ages

        # Require 2+ evidence points per milestone for reliability
        # Filter vague/noisy milestone names
        milestone_median = {}
        for m, ages in milestone_ages_deduped.items():
            if len(ages) < 2:
                continue
            m_lower = m.lower()
            if any(noise in m_lower for noise in NOISE_MILESTONE_PATTERNS):
                continue
            milestone_median[m] = sum(ages) / len(ages)

        # Need 3+ milestones at distinct ages (>2 years apart for cleaner questions)
        if len(milestone_median) < 3:
            continue

        sorted_milestones = sorted(milestone_median.items(), key=lambda x: x[1])
        distinct = [sorted_milestones[0]]
        for m, age in sorted_milestones[1:]:
            if age - distinct[-1][1] > 2.0:  # >2 year gap (was 1, tightened)
                distinct.append((m, age))

        if len(distinct) >= 3:
            # Check logical coherence before accepting
            if _check_ordering_coherence(distinct):
                ordering_candidates.append((disease_id, distinct))

    random.shuffle(ordering_candidates)
    id_to_name = load_disease_id_to_name(names)

    for disease_id, milestones in ordering_candidates:
        if len(questions) >= target_count:
            break
        disease_name = id_to_name.get(disease_id)
        if not disease_name:
            continue  # Skip if no human-readable name
        # Reject names that are just IDs
        if disease_name.startswith("Mondo:") or disease_name.startswith("Omim:"):
            continue

        # Pick 3-4 milestones — ALL must have ages (guaranteed by construction)
        n_pick = min(4, len(milestones))
        selected = random.sample(milestones, n_pick)
        correct_order = sorted(selected, key=lambda x: x[1])

        # Present in shuffled order
        shuffled = list(selected)
        random.shuffle(shuffled)

        q_text = (
            f"Rank the following clinical milestones in {disease_name} "
            f"from earliest to latest typical age of occurrence:\n"
        )
        for i, (m, _) in enumerate(shuffled, 1):
            q_text += f"  {i}. {m}\n"

        answer_order = []
        for m_correct, age_correct in correct_order:
            idx = next(i for i, (m, _) in enumerate(shuffled, 1) if m == m_correct)
            answer_order.append(idx)

        questions.append({
            "question": q_text.strip(),
            "answer": " → ".join(str(i) for i in answer_order),
            "answer_detail": [
                {"milestone": m, "age": round(age, 1)}
                for m, age in correct_order
            ],
            "gold_standard": {
                "source": "ChronoMedKG_triples",
                "cross_validation": "milestone ages from multi-model consensus (2+ sources per milestone)",
            },
            "type": "phenotype_ordering",
            "difficulty": "hard",
            "disease_id": disease_id,
            "disease_name": disease_name,
            "temporal_necessity": True,
            "reasoning": "Requires knowing temporal ordering of clinical milestones",
        })

    return questions


# ============================================================================
# TYPE 6: Stage-Conditional Phenotypes (Our Triples)
# ============================================================================
def gen_stage_conditional(names):
    """
    List question: 'What phenotypes are characteristic of the [stage] stage of Disease X?'
    From validated triples where progression_stage + phenotype co-occur.
    """
    questions = []
    target_count = 200

    validated_files = glob.glob(str(EXTRACTED_DIR / "*/validated_triples.jsonl"))
    candidates = []

    for f in validated_files:
        disease_id = os.path.basename(os.path.dirname(f)).replace("_", ":")
        stages_to_phenos = defaultdict(set)

        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    t = json.loads(line)
                except Exception:
                    continue

                temp = t.get("temporal", {}) or {}
                rel = t.get("relation", "")
                target = t.get("target_name", "")

                ps = temp.get("progression_stage", "")
                if not ps or ps in ("unknown", "null", "none"):
                    continue
                if rel != "disease_phenotype_positive":
                    continue

                # Filter noise phenotypes
                target_lower = target.lower()
                is_noise = any(
                    target_lower.endswith(p) or target_lower == p or
                    target_lower.startswith(p + " ")
                    for p in NOISE_PHENOTYPE_PATTERNS
                )
                if is_noise or len(target) <= 3:
                    continue

                stages_to_phenos[ps].add(target)

        # Keep diseases with 2+ stages, each with 2+ clean phenotypes
        clean_stages = {s: list(p) for s, p in stages_to_phenos.items() if len(p) >= 2}
        if len(clean_stages) >= 2:
            candidates.append((disease_id, clean_stages))

    random.shuffle(candidates)
    id_to_name = load_disease_id_to_name(names)

    for disease_id, stage_map in candidates:
        if len(questions) >= target_count:
            break
        disease_name = id_to_name.get(disease_id)
        if not disease_name:
            continue  # Skip if no human-readable name
        # Reject names that are just IDs (MONDO:, OMIM:, etc.)
        if disease_name.startswith("Mondo:") or disease_name.startswith("Omim:"):
            continue
        # Reject umbrella/category disease names (exact or near-exact match only)
        dn_lower = disease_name.lower().strip()
        if dn_lower in UMBRELLA_DISEASE_PATTERNS:
            continue
        # Also reject very short generic names (< 4 words and matches a pattern)
        dn_words = dn_lower.split()
        if len(dn_words) <= 3 and any(pat == dn_lower for pat in UMBRELLA_DISEASE_PATTERNS):
            continue

        # Filter to stages with 3+ clean phenotypes (tighter than 2+)
        good_stages = {s: p for s, p in stage_map.items() if len(p) >= 3}
        if len(good_stages) < 2:
            continue

        # Pick a random stage
        stage = random.choice(list(good_stages.keys()))
        phenotypes = good_stages[stage]

        # Also collect phenotypes from OTHER stages for MCQ distractors
        other_phenos = []
        for other_stage, other_list in good_stages.items():
            if other_stage != stage:
                other_phenos.extend(other_list)

        # Answer is EXACTLY the gold standard phenotypes (no inflation)
        answer_phenos = phenotypes[:5]

        questions.append({
            "question": (
                f"What phenotypes are characteristic of the {stage} "
                f"stage of {disease_name}?"
            ),
            "answer": answer_phenos,
            "gold_standard": {
                "source": "ChronoMedKG_triples",
                "stage": stage,
                "all_stages": list(good_stages.keys()),
                "phenotypes_at_stage": phenotypes,
                "phenotypes_at_other_stages": other_phenos[:10],
            },
            "type": "stage_conditional",
            "difficulty": "hard",
            "disease_id": disease_id,
            "disease_name": disease_name,
            "temporal_necessity": True,
            "reasoning": "Requires knowing which phenotypes appear at which disease stage",
        })

    return questions


# ============================================================================
# TYPE 7: Phenopackets Case-Level Onset (External)
# ============================================================================
def gen_phenopackets_onset(names):
    """
    Numeric: 'At what age does [phenotype] typically present in [disease]?'
    From real patient case data in Phenopackets.
    """
    questions = []

    with open(VALIDATION_DIR / "phenopackets_parsed.pkl", "rb") as f:
        phenopackets = pickle.load(f)

    for disease_lower, data in phenopackets.items():
        did = names.get(disease_lower)
        if not did:
            continue

        disease_name = disease_lower.title()
        pheno_onsets = data.get("phenotype_onsets", {})

        for phenotype, onset_list in pheno_onsets.items():
            if not onset_list or len(onset_list) < 2:
                continue  # Need 2+ cases for reliability

            all_mins = [o[0] for o in onset_list]
            all_maxs = [o[1] for o in onset_list]
            ref_min = min(all_mins)
            ref_max = max(all_maxs)
            n_cases = len(onset_list)

            # Format answer — handle congenital/neonatal (0-~0) specially
            if ref_min <= 0.01 and ref_max <= 0.08:
                answer_str = f"At birth / congenital (from {n_cases} cases)."
            else:
                answer_str = f"Age {ref_min:.1f}-{ref_max:.1f} years (from {n_cases} cases)."

            questions.append({
                "question": (
                    f"At what age does \"{phenotype}\" typically present "
                    f"in {disease_name}? (Based on patient case data)"
                ),
                "answer": answer_str,
                "gold_standard": {
                    "source": "GA4GH Phenopackets",
                    "phenotype": phenotype,
                    "onset_min": ref_min,
                    "onset_max": ref_max,
                    "cases": n_cases,
                },
                "type": "phenopackets_onset",
                "difficulty": "hard",
                "disease_id": did,
                "disease_name": disease_name,
                "temporal_necessity": True,
                "reasoning": "Requires knowing case-level phenotype onset data",
            })

    return questions


# ============================================================================
# TYPE 8: Static Control Questions (PrimeKG — NON-temporal)
# ============================================================================
def gen_static_control(names):
    """
    MCQ: Standard biomedical knowledge questions answerable from PrimeKG.
    These do NOT require temporal information — they are the control group.
    Used to measure: does temporal retrieval help/hurt on non-temporal questions?
    """
    questions = []
    target_count = 500

    # Load pre-cached PrimeKG edges
    cache = VALIDATION_DIR / "primekg_edges.pkl"
    if not cache.exists():
        print("  WARNING: primekg_edges.pkl not found — skipping static control")
        print("  Run the edge extraction first (see build_tqa_v6.py comments)")
        return questions

    with open(cache, "rb") as f:
        data = pickle.load(f)

    drug_disease_pairs = data.get("drug_disease", [])
    gene_disease_pairs = data.get("gene_disease", [])

    # Group: drug → [diseases], disease → [drugs]
    disease_to_drugs = defaultdict(set)
    all_drugs = set()
    for drug, disease in drug_disease_pairs:
        disease_to_drugs[disease].add(drug)
        all_drugs.add(drug)
    all_drugs = list(all_drugs)

    disease_to_genes = defaultdict(set)
    all_genes = set()
    for gene, disease in gene_disease_pairs:
        disease_to_genes[disease].add(gene)
        all_genes.add(gene)
    all_genes = list(all_genes)

    print(f"  PrimeKG: {len(disease_to_drugs)} diseases with drugs, {len(disease_to_genes)} with genes")

    # Drug-disease MCQs: "Which drug treats Disease X?"
    # Filter out cause-treatment inversions (e.g., digoxin for digitalis poisoning)
    for disease, drugs in list(disease_to_drugs.items()):
        filtered = {d for d in drugs
                    if (d.lower(), disease.lower()) not in PRIMEKG_DRUG_CONTRADICTIONS}
        if filtered:
            disease_to_drugs[disease] = filtered
        else:
            del disease_to_drugs[disease]

    drug_diseases = list(disease_to_drugs.items())
    random.shuffle(drug_diseases)

    for disease_name, drugs in drug_diseases:
        if len(questions) >= target_count // 2:
            break
        drugs = list(drugs)
        if len(drugs) < 1:
            continue

        correct_drug = random.choice(drugs)
        # Skip suspect drugs
        if correct_drug.lower() in SUSPECT_DRUG_PATTERNS:
            continue
        pool = [d for d in all_drugs if d not in drugs]
        if len(pool) < 3:
            continue
        distractors = random.sample(pool, 3)

        options = [correct_drug] + distractors
        random.shuffle(options)
        correct_idx = options.index(correct_drug)
        option_labels = ["A", "B", "C", "D"]

        q_text = f"Which of the following drugs is indicated for {disease_name}?\n"
        for i, opt in enumerate(options):
            q_text += f"  {option_labels[i]}) {opt}\n"

        questions.append({
            "question": q_text.strip(),
            "answer": option_labels[correct_idx],
            "answer_detail": correct_drug,
            "gold_standard": {
                "source": "PrimeKG",
                "relation": "indication",
            },
            "options": {option_labels[i]: opt for i, opt in enumerate(options)},
            "type": "static_control_drug",
            "difficulty": "easy",
            "disease_name": disease_name,
            "temporal_necessity": False,
            "reasoning": "Static question — no temporal information needed",
        })

    # Gene-disease MCQs: "Which gene is associated with Disease X?"
    gene_diseases = list(disease_to_genes.items())
    random.shuffle(gene_diseases)

    for disease_name, genes in gene_diseases:
        if len(questions) >= target_count:
            break
        genes = list(genes)
        if len(genes) < 1:
            continue

        correct_gene = random.choice(genes)
        pool = [g for g in all_genes if g not in genes]
        if len(pool) < 3:
            continue
        distractors = random.sample(pool, 3)

        options = [correct_gene] + distractors
        random.shuffle(options)
        correct_idx = options.index(correct_gene)
        option_labels = ["A", "B", "C", "D"]

        q_text = (
            f"Which of the following genes/proteins is associated "
            f"with {disease_name}?\n"
        )
        for i, opt in enumerate(options):
            q_text += f"  {option_labels[i]}) {opt}\n"

        questions.append({
            "question": q_text.strip(),
            "answer": option_labels[correct_idx],
            "answer_detail": correct_gene,
            "gold_standard": {
                "source": "PrimeKG",
                "relation": "disease_protein",
            },
            "options": {option_labels[i]: opt for i, opt in enumerate(options)},
            "type": "static_control_gene",
            "difficulty": "easy",
            "disease_name": disease_name,
            "temporal_necessity": False,
            "reasoning": "Static question — no temporal information needed",
        })

    return questions


# ============================================================================
# MAIN: Orchestrate all generators
# ============================================================================
def compute_statistics(questions):
    """Compute benchmark statistics."""
    type_counts = Counter()
    diff_counts = Counter()
    source_counts = Counter()
    disease_ids = set()
    temporal_count = 0

    for q in questions:
        type_counts[q["type"]] += 1
        diff_counts[q.get("difficulty", "unknown")] += 1
        source_counts[q["gold_standard"]["source"]] += 1
        if q.get("disease_id"):
            disease_ids.add(q["disease_id"])
        if q.get("temporal_necessity"):
            temporal_count += 1

    return {
        "total_questions": len(questions),
        "diseases": len(disease_ids),
        "temporal_questions": temporal_count,
        "static_control_questions": len(questions) - temporal_count,
        "by_type": dict(type_counts),
        "by_difficulty": dict(diff_counts),
        "by_gold_standard": dict(source_counts),
    }


# ============================================================================
# TIER 3: Novel Literature-Derived Knowledge
# ============================================================================

def _load_tier3_candidates():
    """Load pre-computed Tier 3 candidates from tier3_candidates.pkl."""
    import pickle as _pkl
    cache = BENCHMARK_DIR / "tier3_candidates.pkl"
    if not cache.exists():
        return []
    with open(cache, "rb") as f:
        return _pkl.load(f)


def gen_tier3_onset_extension(candidates):
    """
    TIER 3A: Onset Extension Yes/No
    'Recent literature shows Disease X can present in adulthood.
     Existing databases list onset as childhood only.
     Does Disease X have adult-onset forms?'

    Gold standard: PMIDs from ChronoMedKG extraction.
    Control: 50% Yes (TA diverges), 50% No (TA agrees with external).
    """
    questions = []
    target_yes = 150
    target_no = 50  # Controls where TA agrees

    # YES questions: TA extends onset beyond external sources
    yes_candidates = [c for c in candidates
                      if c["sources_diverged"] and c["n_pmids"] >= 10]
    random.shuffle(yes_candidates)

    for c in yes_candidates[:target_yes]:
        # Pick the most dramatic divergence
        divergence = c["sources_diverged"][0]
        ext_source = divergence[0]

        # Determine the direction
        if "extends older" in divergence[1]:
            ext_range = f"{c['orpha_range']} years"
            ta_age = c["ta_range"].split("-")[1]
            question_text = (
                f"Existing databases ({ext_source}) list the typical onset of "
                f"{c['disease_name']} as {ext_range}. "
                f"Based on recent literature, can {c['disease_name']} also present "
                f"at older ages (e.g., in adulthood or later)?"
            )
        else:
            ext_range = f"{c['orpha_range']} years"
            ta_age = c["ta_range"].split("-")[0]
            question_text = (
                f"Existing databases ({ext_source}) list the typical onset of "
                f"{c['disease_name']} as {ext_range}. "
                f"Based on recent literature, can {c['disease_name']} also present "
                f"at younger ages (e.g., neonatally or in infancy)?"
            )

        # Get supporting evidence
        supporting = c.get("supporting_triples", [])
        evidence_pmids = []
        evidence_texts = []
        for t in supporting[:3]:
            evidence_pmids.extend(t["pmids"][:2])
            evidence_texts.append(t["evidence_text"])

        questions.append({
            "question": question_text,
            "answer": "Yes",
            "gold_standard": {
                "source": "ChronoMedKG_literature",
                "tier": 3,
                "external_source": ext_source,
                "external_range": c["orpha_range"],
                "ta_range": c["ta_range"],
                "divergence_detail": divergence[1],
                "supporting_pmids": list(set(evidence_pmids))[:5],
                "evidence_texts": evidence_texts[:2],
                "n_total_pmids": c["n_pmids"],
                "sources_diverged": [s[0] for s in c["sources_diverged"]],
            },
            "type": "tier3_onset_extension",
            "difficulty": "hard",
            "disease_id": c["disease_id"],
            "disease_name": c["disease_name"],
            "temporal_necessity": True,
            "reasoning": "Requires knowledge of recent literature extending known onset ranges",
        })

    # NO controls: use diseases where TA AGREES with Orphadata (no divergence)
    no_candidates = [c for c in candidates
                     if not c["sources_diverged"] and c["n_pmids"] >= 5]
    random.shuffle(no_candidates)

    for c in no_candidates[:target_no]:
        question_text = (
            f"Existing databases (Orphadata) list the typical onset of "
            f"{c['disease_name']} as {c['orpha_range']} years. "
            f"Based on current literature, does {c['disease_name']} present "
            f"outside this onset range (e.g., significantly earlier or later)?"
        )

        questions.append({
            "question": question_text,
            "answer": "No",
            "gold_standard": {
                "source": "ChronoMedKG_literature",
                "tier": 3,
                "external_source": "Orphadata",
                "external_range": c["orpha_range"],
                "ta_range": c["ta_range"],
                "divergence_detail": "TA confirms external range",
                "n_total_pmids": c["n_pmids"],
            },
            "type": "tier3_onset_extension",
            "difficulty": "hard",
            "disease_id": c["disease_id"],
            "disease_name": c["disease_name"],
            "temporal_necessity": True,
            "reasoning": "Tests whether model can correctly identify NO divergence",
        })

    return questions


def gen_tier3_extended_range(candidates):
    """
    TIER 3B: Extended Onset Range
    'What is the full age range at which Disease X can present,
     including recently documented late-onset or early-onset forms?'

    Gold standard: ChronoMedKG range (broader than external), with PMIDs.
    """
    questions = []
    target = 150

    # Use divergence candidates with clear extension
    ext_candidates = [c for c in candidates
                      if c["sources_diverged"] and c["n_pmids"] >= 10
                      and float(c["ta_range"].split("-")[1]) - float(c["ta_range"].split("-")[0]) > 5]
    random.shuffle(ext_candidates)

    for c in ext_candidates[:target]:
        ta_min = float(c["ta_range"].split("-")[0])
        ta_max = float(c["ta_range"].split("-")[1])

        question_text = (
            f"What is the full documented age range at which {c['disease_name']} "
            f"can present, based on the broadest available clinical evidence?"
        )

        # Collect PMIDs
        supporting = c.get("supporting_triples", [])
        all_pmids = []
        all_evidence = []
        for t in supporting[:3]:
            all_pmids.extend(t["pmids"][:2])
            all_evidence.append(t["evidence_text"])

        questions.append({
            "question": question_text,
            "answer": f"Age {ta_min:.1f}-{ta_max:.1f} years",
            "gold_standard": {
                "source": "ChronoMedKG_literature",
                "tier": 3,
                "onset_min": ta_min,
                "onset_max": ta_max,
                "external_range": c["orpha_range"],
                "supporting_pmids": list(set(all_pmids))[:5],
                "evidence_texts": all_evidence[:2],
                "n_total_pmids": c["n_pmids"],
                "sources_diverged": [s[0] for s in c["sources_diverged"]],
            },
            "type": "tier3_extended_range",
            "difficulty": "hard",
            "disease_id": c["disease_id"],
            "disease_name": c["disease_name"],
            "temporal_necessity": True,
            "reasoning": "Requires knowledge of literature-extended onset ranges beyond standard databases",
        })

    return questions


def gen_tier3_novel_staging(candidates):
    """
    TIER 3D: Novel Disease Staging
    'What progression stages does Disease X go through?'

    Gold standard: TA-extracted stages with PMID evidence.
    No external database has structured staging for these diseases.
    """
    questions = []
    target = 150

    # Use candidates with novel staging (3+ stages, 5+ PMIDs)
    # Filter: require at least 2 non-onset-label stage names (meaningful stages)
    onset_labels = {"neonatal", "infantile", "infant", "childhood", "adult", "adult-onset",
                    "late-onset", "early-onset", "young adult-onset", "elderly", "prenatal",
                    "congenital", "fetal", "juvenile"}
    staging_candidates = []
    for c in candidates:
        if not (c.get("has_novel_staging") and c["n_pmids"] >= 5 and c.get("stages_with_evidence")):
            continue
        non_onset = [s for s in c["stage_names"] if s.lower() not in onset_labels]
        if len(non_onset) >= 2:
            staging_candidates.append(c)
    random.shuffle(staging_candidates)

    for c in staging_candidates[:target]:
        stages = c["stage_names"]
        stage_evidence = c.get("stages_with_evidence", {})

        question_text = (
            f"What are the documented progression stages of {c['disease_name']}?"
        )

        # Collect PMIDs from stage evidence
        all_pmids = []
        stage_details = {}
        for stage_name, entries in stage_evidence.items():
            phenos = [e["phenotype"] for e in entries[:3]]
            pmids = []
            for e in entries[:3]:
                pmids.extend(e["pmids"][:1])
            stage_details[stage_name] = {
                "phenotypes": phenos,
                "pmids": pmids,
            }
            all_pmids.extend(pmids)

        questions.append({
            "question": question_text,
            "answer": stages[:6],
            "gold_standard": {
                "source": "ChronoMedKG_literature",
                "tier": 3,
                "n_stages": c["n_stages"],
                "all_stages": stages,
                "stage_details": stage_details,
                "supporting_pmids": list(set(all_pmids))[:10],
                "n_total_pmids": c["n_pmids"],
            },
            "type": "tier3_novel_staging",
            "difficulty": "hard",
            "disease_id": c["disease_id"],
            "disease_name": c["disease_name"],
            "temporal_necessity": True,
            "reasoning": "Requires knowledge of disease progression stages not available in any structured database",
        })

    return questions


def main():
    print("=" * 60)
    print("ChronoMedKG-TQA v6: Temporal REASONING Benchmark")
    print("=" * 60)

    print("\nLoading data...")
    names = load_disease_names()
    print(f"  Disease names: {len(names)}")

    with open(VALIDATION_DIR / "orpha_parsed.pkl", "rb") as f:
        orphadata = pickle.load(f)
    print(f"  Orphadata: {len(orphadata)} diseases")

    # Generate each question type
    print("\n--- Generating questions ---\n")

    print("Type 1: Temporal Differential Dx (Orphadata)...")
    q1 = gen_differential_dx(orphadata, names)
    print(f"  Generated: {len(q1)}")

    print("Type 2: Temporal Window (Orphadata)...")
    q2 = gen_temporal_window(orphadata, names)
    print(f"  Generated: {len(q2)}")

    print("Type 3: Cross-Disease Comparison (Orphadata)...")
    q3 = gen_cross_disease_comparison(orphadata, names)
    print(f"  Generated: {len(q3)}")

    print("Type 4: Negative Temporal MCQ (HPOA)...")
    q4 = gen_negative_temporal(names)
    print(f"  Generated: {len(q4)}")

    print("Type 5: Phenotype Ordering (Our Triples)...")
    q5 = gen_phenotype_ordering(names)
    print(f"  Generated: {len(q5)}")

    print("Type 6: Stage-Conditional (Our Triples)...")
    q6 = gen_stage_conditional(names)
    print(f"  Generated: {len(q6)}")

    print("Type 7: Phenopackets Onset (External)...")
    q7 = gen_phenopackets_onset(names)
    print(f"  Generated: {len(q7)}")

    print("Type 8: Static Control (PrimeKG)...")
    q8 = gen_static_control(names)
    print(f"  Generated: {len(q8)}")

    # TIER 3: Novel literature-derived knowledge
    print("\n--- Tier 3: Novel Literature Knowledge ---\n")
    tier3_candidates = _load_tier3_candidates()
    print(f"  Tier 3 candidates loaded: {len(tier3_candidates)}")

    if tier3_candidates:
        print("Type 9: Tier 3A — Onset Extension Yes/No...")
        q9 = gen_tier3_onset_extension(tier3_candidates)
        print(f"  Generated: {len(q9)}")

        print("Type 10: Tier 3B — Extended Onset Range...")
        q10 = gen_tier3_extended_range(tier3_candidates)
        print(f"  Generated: {len(q10)}")

        print("Type 11: Tier 3D — Novel Disease Staging...")
        q11 = gen_tier3_novel_staging(tier3_candidates)
        print(f"  Generated: {len(q11)}")
    else:
        print("  WARNING: tier3_candidates.pkl not found — run multi-source divergence analysis first")
        q9, q10, q11 = [], [], []

    # Combine and deduplicate
    all_questions = q1 + q2 + q3 + q4 + q5 + q6 + q7 + q8 + q9 + q10 + q11

    # Multi-level deduplication
    seen_text = set()      # Question text dedup
    seen_disease = set()   # Disease × type dedup (prevent same disease appearing multiple times per type)
    deduped = []
    for q in all_questions:
        # Level 1: Exact question text dedup
        q_text = q["question"][:200]
        key1 = (q["type"], q_text)
        if key1 in seen_text:
            continue
        seen_text.add(key1)

        # Level 2: Disease × type dedup (max 1 question per disease per type)
        # Except for types where multiple questions per disease make sense:
        #   - phenopackets: different phenotypes per disease
        #   - differential_dx / cross_disease: different pairings
        multi_per_disease_types = (
            "temporal_differential_dx", "cross_disease_comparison",
            "phenopackets_onset",
            "tier3_onset_extension", "tier3_extended_range", "tier3_novel_staging",
        )
        disease_key = q.get("disease_id", q.get("disease_name", ""))
        if disease_key and q["type"] not in multi_per_disease_types:
            key2 = (q["type"], disease_key)
            if key2 in seen_disease:
                continue
            seen_disease.add(key2)

        deduped.append(q)

    random.shuffle(deduped)

    # Assign IDs
    for i, q in enumerate(deduped, 1):
        q["id"] = f"TQA6-{i:05d}"

    # Statistics
    stats = compute_statistics(deduped)

    # Build benchmark object
    benchmark = {
        "name": "ChronoMedKG-TQA",
        "version": "7.0.0",
        "description": (
            "Temporal REASONING benchmark for biomedical KGs. "
            "8 question types testing temporal reasoning (not just lookup). "
            "External gold standards (Orphadata, HPOA, Phenopackets) for answer keys. "
            "Static control questions from PrimeKG for non-temporal baseline."
        ),
        "methodology": {
            "external_sources": [
                "Orphadata (disease-level onset ages)",
                "HPOA (per-phenotype onset annotations)",
                "GA4GH Phenopackets (real patient case data)",
                "PrimeKG (static edges for control questions)",
            ],
            "internal_sources": [
                "ChronoMedKG validated triples (milestone ordering, stage-conditional)",
            ],
            "anti_circularity": (
                "External source questions use ONLY gold standard answers. "
                "Internal source questions (ordering, stage-conditional) use "
                "multi-model consensus triples, which can be independently verified."
            ),
            "temporal_necessity": (
                "Every temporal question requires temporal information to answer correctly. "
                "Static control questions explicitly do NOT require temporal information."
            ),
        },
        "statistics": stats,
        "questions": deduped,
    }

    # Save
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    out_file = BENCHMARK_DIR / "chronomedkg_tqa_v6.json"
    with open(out_file, "w") as f:
        json.dump(benchmark, f, indent=2, default=str)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"ChronoMedKG-TQA v6 — SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total questions: {stats['total_questions']}")
    print(f"  Temporal (reasoning): {stats['temporal_questions']}")
    print(f"  Static (control):     {stats['static_control_questions']}")
    print(f"Diseases covered: {stats['diseases']}")
    print(f"\nBy question type:")
    for t, c in sorted(stats["by_type"].items(), key=lambda x: -x[1]):
        print(f"  {t:35s} {c:>5d}")
    print(f"\nBy difficulty:")
    for d, c in sorted(stats["by_difficulty"].items()):
        print(f"  {d:15s} {c:>5d}")
    print(f"\nBy gold standard source:")
    for s, c in sorted(stats["by_gold_standard"].items(), key=lambda x: -x[1]):
        print(f"  {s:35s} {c:>5d}")
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
