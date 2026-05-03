# Research Note 009: TQA Benchmark Quality Audit & 4-Layer Validation Plan

**Date:** 2026-03-29 (audit), 2026-03-31 (Layer 1 cleanup EXECUTED)
**Status:** Layer 1 DONE — v2 benchmark produced. Layers 2-4 pending.

---

## Problem Statement

ChronoMedKG-TQA v1 has 29,514 questions across 5,816 diseases, but:
1. ~50% have quality issues (garbage answers, duplicates, tautologies)
2. Questions are generated FROM our own triples — circular benchmarking
3. No external validation of answer correctness
4. No human ceiling score
5. No difficulty calibration beyond easy/medium

A NeurIPS reviewer would reject this in its current form.

---

## Quality Audit Results

| Issue | Count | % | Severity |
|-------|------:|--:|----------|
| Same min/max age ("Between age 5 and 5") | 5,764 | 19.5% | 🔴 |
| "Age 0" answers | 4,527 | 15.3% | 🟡 |
| Treatment Qs with non-temporal answers | 5,333 | 69.7% of treatment | 🔴 |
| Duplicate questions | 1,202 extra | 4.1% | 🟡 |
| Tautological stage Qs | 2,008 | 35.8% of stage | 🔴 |

**Estimated clean after filtering:** ~8,000-10,000 questions

---

## 4-Layer Validation Strategy

### Layer 1: Cleanup ($0, 2 hours) — ✅ DONE 2026-03-31
- ✅ Dropped "Between age X and X" (5,764 questions)
- ✅ Dropped treatment Qs without temporal expressions (3,691)
- ✅ Dropped tautological stage Qs (5,555)
- ✅ Added hard difficulty tier (1,707 questions)
- **Result: 29,514 → 14,504** (50.9% removed)
- Scripts: `scripts/cleanup_tqa_benchmark.py`, `scripts/cleanup_triples.py`
- Output: `data/benchmark/temporalatlas_tqa_v2.json`

### Layer 2: External Gold Standard ($0, 1 day)
- 500 stratified Qs cross-referenced against GeneReviews/OMIM/Orphanet
- Verify answer exists in source PMID paper (provenance check)
- Target: >93% externally confirmed

### Layer 3: Clinician Validation (~$200-500, 1 week)
- 2-3 clinicians rate 200 Qs
- Metrics: factual accuracy (>90%), clinical relevance (>80%), human ceiling
- Inter-annotator: Cohen's κ > 0.7

### Layer 4: LLM Parametric Baseline ($5-10, 2 hours)
- GPT-4o + Claude on 500 Qs without retrieval
- Sweet spot: 30-60% (proves KG necessity)

---

## Why NOT MedQA/USMLE/PubMedQA

- These test LLM clinical reasoning, not KG completeness
- No KG paper (KARMA, iKraph, PrimeKG) uses them
- All KGs would perform identically (KGs don't reason, they store)
- Would confound KG quality with LLM ability

---

## Execution Dependencies

| Step | Depends on | Cost |
|------|-----------|------|
| Layer 1 (cleanup) | Nothing — can run now | $0 |
| Layer 2 (gold standard) | Nothing — can run now | $0 |
| Layer 3 (clinicians) | Budget approval | $200-500 |
| Layer 4 (LLM baseline) | API keys restored | $5-10 |

**Layers 1 and 2 can be executed immediately at $0 cost.**
