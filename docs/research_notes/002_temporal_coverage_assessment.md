# Research Note 002: Is Our Temporal Coverage Rich Enough?

> Date: 2026-03-24
> Context: Evaluating whether temporal metadata satisfies clinician and pharma expectations
> Verdict: **INSUFFICIENT for landmark — needs 3 specific improvements**

---

## What We Currently Capture

### Temporal signal breakdown (best case: DMD, 200 docs)

| Signal | Coverage | What it means |
|--------|:--------:|---------------|
| Any temporal | 65% | At least one temporal field populated |
| discovery_date | 13% | Publication year of source paper |
| validity_start | 13% | Same as discovery_date (auto-inferred) |
| validity_end | **0%** | When knowledge became obsolete — NEVER set |
| onset_age | 21% | Age range when condition/treatment applies |
| progression_stage | 42% | Clinical stage (early, advanced, etc.) |
| duration | 29% | How long a treatment/outcome lasts |

### Resolution quality
- **100% year-level only** — "2019", never "March 2019" or "Q1 2019"
- No week-level precision for treatment responses
- No day-level precision for acute interventions

## What Clinicians Actually Need

### A neurologist evaluating a DMD patient wants:

| Clinical question | Required temporal precision | Do we have it? |
|------------------|:-------------------------:|:--------------:|
| When to start corticosteroids? | Age range (4-6 years) | Partially (onset_age) |
| How long until deflazacort shows effect? | Weeks (6-12 weeks) | **NO** |
| When does ambulation loss typically occur? | Age range (9-13 years) | Partially (progression) |
| When to start cardiac monitoring? | Age (10 years) | **NO** (not extracted as milestone) |
| When was ataluren approved? | Year (2014 EU) | **NO** (no regulatory dates) |
| Is prednisone still first-line? | Current validity | **NO** (validity_end = 0%) |

### Verdict for clinicians: **~30% of what they need**
We capture onset ages and progression stages at a coarse level. But treatment timelines,
monitoring schedules, and guideline currency are missing.

## What Pharma Expects

### A pharma company building a disease model wants:

| Pharma use case | Required temporal data | Do we have it? |
|----------------|:---------------------:|:--------------:|
| Natural history model | Age-at-event milestones | Partially |
| Treatment response curves | Weeks/months to response | **NO** |
| Adverse event onset windows | Days/weeks after drug start | **NO** |
| Drug lifecycle (approval → withdrawal) | Regulatory dates | **NO** |
| Market evolution (standard of care changes) | Supersession chains | **NO** (0 detected) |
| Clinical trial endpoint timing | Months to endpoint | **NO** |
| Real-world evidence timelines | Continuous time series | **NO** |

### Verdict for pharma: **~15% of what they need**
We provide disease progression staging and some onset ages. But the high-value temporal
data (treatment response curves, adverse event windows, regulatory timelines) is not
being extracted.

## Why Our Coverage Is Low (Root Causes)

### 1. Extraction prompt doesn't ask for specific temporal types
Current prompt asks for `temporal_context` as a free-text field. LLMs fill it with
whatever they notice — usually just the publication year. Need structured temporal fields:
- `treatment_response_time`
- `monitoring_schedule`
- `regulatory_dates`
- `natural_history_milestones`

### 2. Abstracts lack temporal detail
Abstracts mention findings, not timelines. Full-text Methods and Results sections contain
the actual temporal data ("patients showed improvement at week 12", "ambulation loss
occurred at median age 10.2 years"). We fetch PMC full-text but only for top 46 articles.

### 3. No structured temporal extraction template
We extract free-form triples. A structured approach would define expected temporal
milestones per disease and specifically query for them.

### 4. Year-only resolution from discovery_date
We set `validity_start` = publication year. This is a proxy, not a real temporal signal.
True validity dates come from guideline publications, FDA approvals, clinical trials.

## Three Improvements to Reach Landmark Quality

### Improvement 1: Structured Temporal Extraction Prompt (HIGH IMPACT)
Instead of: "Extract temporal context"
Use: "For each relationship, extract ALL of these temporal signals if present:
- discovery/first-reported year
- treatment initiation age/stage
- time-to-response (weeks/months)
- duration of effect
- monitoring schedule
- age-at-milestone for disease progression
- regulatory approval/withdrawal dates
- guideline change dates"

**Expected impact:** temporal coverage 65% → 80%+, with richer signal types

### Improvement 2: Disease-Specific Milestone Templates (MEDIUM IMPACT)
For each disease, define expected temporal milestones from GeneReviews/OMIM:
- DMD: onset, ambulation loss, scoliosis, cardiomyopathy, ventilation, death
- MG: onset, crisis episodes, thymectomy timing, remission
- CIDP: onset, treatment response, relapse cycles

Then specifically query for these milestones across all documents.

**Expected impact:** progression_stage coverage 42% → 70%+

### Improvement 3: Guideline + Regulatory Date Layer (MEDIUM IMPACT)
Add a dedicated harvester for:
- FDA/EMA approval dates (via OpenFDA API or DailyMed)
- Clinical practice guideline publication dates (via NICE, AAN, etc.)
- Cochrane review dates

These give ground-truth validity_start and validity_end dates.

**Expected impact:** validity_start from proxy → real dates, validity_end from 0% → 20%+

## Projected Coverage After Improvements

| Signal | Current (DMD) | After improvements | Target for landmark |
|--------|:------------:|:-----------------:|:-------------------:|
| Any temporal | 65% | **85%** | 80%+ |
| Treatment timelines | ~0% | **30%** | 25%+ |
| Natural history milestones | ~20% | **50%** | 40%+ |
| Regulatory dates | 0% | **15%** | 10%+ |
| Validity_end (supersession) | 0% | **10%** | 5%+ |
| Sub-year resolution | 0% | **20%** | 15%+ |

## Action Items

1. **Redesign extraction prompt** with structured temporal fields (1 day)
2. **Build milestone template system** per disease category (2 days)
3. **Add regulatory date harvester** from OpenFDA + DailyMed (1 day)
4. **Re-run DMD as proof-of-concept** with improved extraction (2 hrs)
5. **Clinician spot-check**: Have 1 neurologist review 50 temporal triples

## Bottom Line

Our temporal coverage is the paper's core differentiator — it's what makes PrimeKG-T
different from PrimeKG, KARMA, and iKraph. But **65% coverage with year-only resolution
and no treatment timelines won't impress Nature Methods reviewers.**

The three improvements above would push us to **85% coverage with clinically meaningful
temporal signals**. That's genuinely landmark territory — a temporally-grounded KG that
clinicians and pharma can actually use for decision-making.

**Estimated effort: 4 days of development + 1 re-extraction run.**
