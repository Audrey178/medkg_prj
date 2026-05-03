# Clinician Evaluation — ChronoMedKG Temporal Triples

**Time commitment:** ~15-20 minutes for 20 triples

**What you'll see:** 20 statements about disease onset timing extracted from medical literature by our knowledge graph pipeline. Each includes the source PMID so you can verify if needed.

**What we're asking:** Quick clinical judgment on accuracy, usefulness, and novelty.

---

## Background (2 paragraphs, for your context)

ChronoMedKG is a biomedical knowledge graph project at King's College London that extracts **temporal information** (when phenotypes appear, disease progression stages, onset age ranges) from the medical literature for 13,431 rare and common diseases. Unlike existing knowledge graphs like PrimeKG or Hetionet that provide static disease-phenotype associations (e.g., "DMD is associated with cardiomyopathy"), ChronoMedKG captures **WHEN** these associations manifest (e.g., "cardiomyopathy in DMD typically presents between ages 10-18").

We're preparing for submission to NeurIPS 2026 Datasets & Benchmarks track. Expert clinician validation on a small sample would significantly strengthen the paper. Your ratings will be cited (anonymously or named, your choice) in the validation section.

---

## How to rate each triple

For each of the 20 disease–phenotype–timing statements, please rate:

### 1. Clinical Accuracy (1-5)
- **1** — Clearly wrong; I would correct this
- **2** — Mostly wrong; some truth but misleading
- **3** — Partially correct; needs caveat
- **4** — Mostly correct; minor imprecision acceptable
- **5** — Fully accurate; matches clinical experience

### 2. Clinical Usefulness (1-5)
- **1** — Not useful in practice
- **2** — Rarely useful
- **3** — Sometimes useful
- **4** — Often useful
- **5** — Critical for clinical decision-making

### 3. Novelty (select one)
- **A** — Widely known in standard clinical references (UpToDate, textbooks)
- **B** — Known to specialists but not easily found in standard references
- **C** — Novel/under-documented; would be useful to systematize

### 4. Optional comment (free text)
- Anything that would help us improve or calibrate the extraction

---

## Example triples (format you'll see)

**Triple 1**
- Disease: Duchenne muscular dystrophy
- Phenotype: Loss of independent ambulation
- Typical onset: 8–12 years
- Source: PMID:34567890 (2022)

Accuracy (1-5): ___ | Usefulness (1-5): ___ | Novelty (A/B/C): ___
Comment: _______________________________________

---

## Specific triples we'd like rated

[20 triples will be inserted here, sampled from:
 - Common rare diseases the clinician's specialty covers
 - Novel temporal claims not in Orphadata/HPOA
 - Cross-validation triples (where we know the answer) for calibration]

---

## Questions for the clinician (optional, 5 minutes)

1. **In your clinical practice, how often do you need to estimate when a specific phenotype will appear or has appeared in a patient's disease course?**
   - Rarely / Sometimes / Often / Daily

2. **What clinical resources do you currently use for phenotype-onset timing?**
   (UpToDate, GeneReviews, Orphanet, specialty texts, experience, etc.)

3. **What temporal clinical questions are hardest to answer with current resources?**
   (e.g., "Should we screen for X now or wait?", "Is this too early/late for Y?")

4. **Would a queryable database of temporally-grounded phenotypes be useful for:**
   - [ ] Differential diagnosis
   - [ ] Screening timing decisions
   - [ ] Patient counselling (what to expect next)
   - [ ] Teaching / residency training
   - [ ] Research
   - [ ] Other: _____

5. **What would make ChronoMedKG more useful to you as a clinician?** (free text)

---

## Notes for the PI (Shamim)

### Who to recruit — priority order:
1. **Rare disease specialists** (neuromuscular, metabolic, genetic) — our core demographic
2. **General paediatricians** — bread-and-butter diagnostic validators
3. **Clinical geneticists** — know rare conditions well
4. **Internal medicine with rare disease interest**

### Sample selection strategy (for the 20 triples):

For each clinician, include:
- **5 triples from their specialty** (high accuracy expected — tests extraction quality)
- **10 novel triples** (no gold standard — tests if TA adds new clinical knowledge)
- **3 calibration triples** (known answers from Orphadata/GeneReviews — tests clinician consistency)
- **2 "trap" triples** (deliberately wrong triples — tests if clinicians catch errors)

### For citation:
- Ask if they want to be named in Acknowledgments or Validation section
- Some may prefer "anonymous expert clinician (rare disease specialist)"
- If they contribute substantially (40+ triples), consider co-authorship

### Realistic expectations:
- Getting 1-2 clinicians by May 4 is already a strong addition
- Even 10 rated triples from 1 clinician is citable as "expert spot-check"
- If >2 clinicians, we can report inter-rater agreement

### What to send them:
1. This document (2 min read)
2. 20 triples in a Google Sheet (pre-filled with accuracy/usefulness/novelty dropdowns)
3. Deadline: whatever they can do — don't push unrealistic timelines
