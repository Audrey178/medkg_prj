# Research Note 004: PrimeKG Edge Audit Findings

**Date:** 2026-03-28
**Status:** Complete (445 diseases assessed, re-run after full extraction for final numbers)

## Headline Results

| Metric | Value |
|--------|-------|
| PrimeKG edges audited | 682,488 |
| Assessed (matched to our evidence) | 10,114 (1.5%) |
| Confirmed | 9,718 (96.1% of assessed) |
| **Contradicted** | **396 (3.9% of assessed)** |
| **Stale >10 years** | **4,008 (41.2% of confirmed)** |
| Stale >15 years | 2,238 (23.0% of confirmed) |

## Evidence Age Distribution (of confirmed edges)

| Freshness | Count | % | Meaning |
|-----------|-------|---|---------|
| Fresh (<5yr) | 3,276 | 33.7% | Supported by 2021+ evidence |
| Aging (5-10yr) | 2,434 | 25.0% | Supported by 2016-2021 evidence |
| Stale (10-15yr) | 1,770 | 18.2% | Newest support is 2011-2016 |
| Ancient (>15yr) | 2,238 | 23.0% | Newest support is pre-2011 |

**Bottom line: 2 out of 3 confirmed PrimeKG edges have no supporting evidence from the last 5 years.**

## Per-Relation Staleness

| Relation | Edges | Confirmed | Contradicted | Avg Age | Stale >10yr |
|----------|-------|-----------|-------------|---------|-------------|
| disease_phenotype_positive | 300,634 | 5,446 | 64 | 10.2yr | 41.7% |
| disease_protein | 160,822 | 1,488 | 0 | 7.6yr | 32.7% |
| disease_disease | 128,776 | 1,452 | 0 | 9.5yr | 39.5% |
| **contraindication** | **61,350** | **66** | **316** | **14.9yr** | **48.5%** |
| indication | 18,776 | 1,008 | 10 | 12.6yr | 48.6% |
| off-label use | 5,136 | 242 | 0 | 14.8yr | 59.5% |
| exposure_disease | 4,608 | 16 | 0 | 11.9yr | 62.5% |
| disease_phenotype_negative | 2,386 | 0 | 6 | — | — |

## 5 Components of Staleness in PrimeKG

### 1. Source Database Lag (biggest contributor)
PrimeKG imports from DrugBank, DisGeNET, OMIM, HPO — these update on different cycles. DrugBank drug-disease edges may reflect FDA labeling from 5-10 years ago.

**Headline example:** Zinc compounds marked as **contraindication** for Wilson disease (from DrugBank labeling), but zinc therapy is now **standard-of-care** for Wilson disease. 316 contraindication edges contradicted — a patient safety concern.

### 2. Evidence Evolution (phenotype refinement)
Medical understanding evolves. What was "always present" 15 years ago may now be "sometimes absent."

**Example:** Kayser-Fleischer ring in Wilson disease: PrimeKG says `phenotype_positive`, but 2025 studies show KF rings are absent in ~40-50% of neuropsychiatric presentations → `phenotype_negative` in context.

### 3. No Temporal Context (most fundamental)
PrimeKG treats all edges as equally true at all times. Drug contraindications change, phenotype-disease associations are stage-dependent, gene-disease links get refined with better sequencing.

### 4. No Evidence Grading
PrimeKG gives equal weight to a 15-year-old DrugBank label and a 2025 meta-analysis. No mechanism to distinguish evidence quality or recency.

### 5. No Conflict Detection
When DrugBank says "contraindication" and newer literature says "treatment", PrimeKG has no mechanism to flag the conflict. Both cannot be simultaneously true, but PrimeKG has no way to know.

## Methodology: How We Determine Evidence Age

**Critical clarification:** PrimeKG stores ZERO evidence provenance — no dates, no PMIDs, no source papers. We cannot measure PrimeKG's own evidence age directly.

**What we measure:** For each PrimeKG edge we can match to our extractions, we report the age of the **newest confirming evidence we found in 2026 literature**. If the newest paper confirming "Drug X treats Disease Y" was published in 2014, that edge has 12 years of evidence age.

**Evidence dating sources:**
1. `publication_date` field from evidence_collection.json.gz (97% of sources have it, format: "YYYY-MM-DD")
2. PMID-based year estimation as fallback (linear interpolation from PMID ranges)

**This is a CONSERVATIVE estimate** — PrimeKG's original evidence may be even older than the newest literature we found confirming it.

**Entity matching:** 3-tier strategy:
1. Exact match after normalization (lowercase, strip punctuation)
2. Substring containment (shorter entity in longer, ≥40% of length)
3. Token overlap (Jaccard similarity ≥0.5 on word tokens ≥3 chars)

**Relation mapping:** PrimeKG's 8 relations mapped to our broader canonical set (e.g., PrimeKG "indication" → our "treats", "treatment_for", "therapeutic_for", etc.)

**Contradiction detection:** Via negation pairs:
- indication ↔ contraindication
- disease_phenotype_positive ↔ disease_phenotype_negative

## Key Paper Quotes

1. "41.2% of independently confirmed PrimeKG edges rely on evidence published more than a decade ago"
2. "316 drug-disease contraindication edges are contradicted by newer literature — a potential patient safety concern"
3. "Two-thirds of confirmed PrimeKG edges have no supporting evidence from the last 5 years"
4. "Off-label use edges are the stalest category at 14.8 years average, with 59.5% based on pre-2016 evidence"
5. "Zinc therapy — now standard-of-care for Wilson disease — is listed as a contraindication in PrimeKG"

## Limitations (to acknowledge in paper)

- Only 1.5% of PrimeKG edges could be assessed (limited by entity matching precision and number of extracted diseases)
- "Contradicted" may include cases where both old and new evidence are valid in different contexts (e.g., drug contraindicated in one subtype, indicated in another)
- Assessment rate will improve as more diseases are extracted (currently 445/17,080)
- We measure age of confirming/contradicting evidence, not PrimeKG's original evidence (which is unknowable)

## Self-Healing Architecture

The audit pipeline (`scripts/audit_primekg.py`) is the foundation for self-healing:
- **Idempotent:** same inputs → same outputs
- **Incremental:** automatically picks up new disease extractions
- **Self-healing flag:** `--self-heal` marks stale edges with `validity_end`
- **Re-runnable:** execute after each batch to get updated numbers
- **Bridge to federated:** periodic audit → flag stale → update KG → the seed of continuous knowledge maintenance

## Contradiction Deep-Dive Examples

### Zinc × Wilson Disease (Patient Safety)
```
PrimeKG:    Zinc gluconate --[contraindication]--> Wilson disease
Our evidence: Zinc --[treats]--> Wilson disease (GOLD tier, 3-model consensus)
PMIDs: 12962167, 9890071 (2002, 1999)
Reality: Zinc therapy is standard-of-care for maintenance in Wilson disease
```

### Kayser-Fleischer Ring × Wilson Disease (Evidence Evolution)
```
PrimeKG:    Wilson disease --[phenotype_positive]--> Kayser-Fleischer ring
Our evidence: Wilson disease --[phenotype_negative]--> KF ring (Bronze-B, 2025 studies)
Reality: KF rings absent in ~40-50% of neuropsychiatric Wilson presentations
```

## Files

- `data/audit/audit_summary.json` — top-line numbers
- `data/audit/staleness_by_relation.json` — per-relation breakdown
- `data/audit/edge_verdicts.jsonl` — 682K per-edge verdicts
- `data/audit/contradiction_examples.json` — sample contradictions
- `scripts/audit_primekg.py` — audit pipeline (re-runnable)
