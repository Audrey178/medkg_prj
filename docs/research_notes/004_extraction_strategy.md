# 004: Multi-Pass Extraction Strategy

> **Created:** 2026-03-26
> **Status:** Active — Pass 1 in progress
> **Relevance:** Paper Methods §3, Supplementary Table S2

---

## Overview

ChronoMedKG uses a literature-adaptive, multi-pass extraction strategy to maximize temporal coverage while minimizing cost. Rather than a single expensive pass, we decompose extraction into three passes of increasing specificity.

---

## Pass 1: Abstracts-Only Broad Coverage (Current)

**Scope:** All 17,080 diseases
**Input:** PubMed abstracts only (skip PMC full-text)
**Method:** 3-LLM consensus (GPT-4o-mini + Claude 3 Haiku + Gemini Flash) via batch APIs (50% cost)
**Fallback chain:**
- GPT-4o-mini → DeepSeek V3 if 0 triples
- Gemini Flash → DeepSeek V3 if 0 triples
- Claude Haiku: no fallback needed (different LLM family provides diversity)

**Output:** ~500K+ consensus triples across all diseases
**Duration:** ~3-4 days
**Cost:** ~$200-400 (batch API pricing)

**Rationale:**
- Abstracts are fast to harvest (~15s/disease vs ~2.5 min with full-text)
- For ~12K rare diseases with <100 papers, abstracts capture all available information — full-text adds nothing
- Provides a quality baseline for every disease before investing in targeted enrichment
- Phase 1 validation showed 71% temporal coverage from abstracts alone

### Batch API Architecture

| Provider | Mode | Chunk Size | Discount |
|----------|------|------------|----------|
| OpenAI (GPT-4o-mini) | Batch API (JSONL upload, 24h window) | 10K requests/batch | 50% |
| Anthropic (Claude 3 Haiku) | Message Batches API (inline, 24h window) | 10K requests/batch | 50% |
| Gemini 2.5 Flash | Real-time (10-concurrent async) | 50 docs/wave | Standard |
| DeepSeek V3 | Real-time fallback | On-demand | Standard |

**Pipeline per chunk (100 diseases):**
1. Profile + Harvest (4 workers, cached evidence reused)
2. Submit to OpenAI + Anthropic batch APIs
3. Run Gemini 2.5 Flash real-time (10-concurrent async, adaptive backoff, DeepSeek V3 fallback) while batches process
4. Poll batch APIs until completion (max 4h)
5. Retrieve results, compute per-document consensus, run QC

### Custom ID Format (≤64 chars, Anthropic spec)
```
{disease_component}__{doc_component}__{model_suffix}
```
Built by `_make_custom_id()` which dynamically allocates character budget between disease and doc components to guarantee total ≤64 chars.

---

## Pass 2: Targeted Full-Text Enrichment (After Pass 1)

**Scope:** ~2,800 diseases scored for enrichment need
**Input:** PMC full-text (Results + Discussion sections)
**Method:** `TEMPORAL_REEXTRACTION_PROMPT` — second-pass prompt requiring ≥2 temporal fields per triple

### Disease Selection Scoring
```
enrichment_priority = (1 - temporal_coverage) × log(pubmed_count) × genereviews_boost
```

| Tier | Diseases | Selection Criteria | Full-text per Disease |
|------|----------|-------------------|----------------------|
| A | ~800 | GeneReviews-linked + >500 PubMed hits | Top 50 PMC articles |
| B | ~2,000 | Low temporal coverage from Pass 1 (<30%) | Top 20 PMC articles |
| C | ~12,000 | Adequate from abstracts | Skip — not needed |

### Efficiency: Local PMCID Lookup
- Download `PMC-ids.csv.gz` from NCBI FTP (~100MB, one-time)
- Cross-reference cached PMIDs from Pass 1 against local PMCID map
- Zero API calls to determine which articles have full-text
- Batch efetch 20 articles/call (no elink needed)

### Merge Strategy
- Full-text triples: `evidence_tier: "tier2_fulltext_enrichment"`, credibility +0.1 boost
- Marked as single-LLM extraction (supplementary, not replacement)
- Existing 3-LLM consensus triples from Pass 1 retained as primary
- Deduplication against existing triples via `{existing_triples_summary}` in prompt

---

## Pass 3: GeneReviews Structured Parsing (Optional, ~800 diseases)

**Scope:** Diseases with GeneReviews entries
**Input:** GeneReviews structured text
**Method:** `GENREVIEWS_TEMPORAL_PROMPT` — specialized prompt for:
- **Stage definitions:** named stages with age ranges and diagnostic criteria
- **Stage-specific phenotypes:** what features are present/absent/emerging per stage
- **Genotype-specific timelines:** how different mutations alter disease course
- **Survival data:** life expectancy by genotype/intervention
- **Treatment timing windows:** when interventions are most effective

**Output:** Temporal Phenotype Profile (TPP) data — CORE contribution for the paper
**Evidence tier:** Tier 1 (curated source) — highest credibility

---

## Cost Comparison

| Strategy | Duration | Cost | Temporal Coverage |
|----------|----------|------|-------------------|
| Single pass (abstracts + full-text, all diseases) | ~15-20 days | ~$800-1200 | ~75% |
| **Multi-pass (our approach)** | **~6 days total** | **~$500-700** | **~75-80%** |

Multi-pass is both faster and cheaper because it avoids wasting full-text effort on diseases that don't benefit.

---

## Fallback Chain Design

Every document gets at least 2-3 independent LLM extractions:

```
Document → GPT-4o-mini (batch)  ─── 0 triples? ──→ DeepSeek V3 (real-time)
         → Claude 3 Haiku (batch)
         → Gemini Flash (real-time) ─ 0 triples? ──→ DeepSeek V3 (real-time)
```

**Consensus rule:** A triple is accepted if ≥2 out of 3 LLM families agree on:
- Same source entity (fuzzy match)
- Same relation type (exact match)
- Same target entity (fuzzy match)

This 3-LLM consensus from different model families (OpenAI, Anthropic, Google) reduces hallucination and extraction noise. DeepSeek V3 serves as a safety net ensuring no document produces zero triples from all providers.

---

## Paper Framing

### For Methods Section
> "We employ a literature-adaptive multi-pass extraction strategy. Pass 1 processes all 17,080 PrimeKG diseases using abstracts with 3-LLM consensus (GPT-4o-mini, Claude 3 Haiku, Gemini Flash) via batch APIs at 50% cost. Pass 2 selectively enriches ~2,800 diseases with PMC full-text using a temporal re-extraction prompt, prioritized by temporal coverage gaps from Pass 1. Pass 3 parses GeneReviews entries for ~800 diseases to populate Temporal Phenotype Profiles."

### For Efficiency/Cost Argument (vs competitors)
> "By scoring diseases for enrichment need after Pass 1, ChronoMedKG avoids the common approach of uniformly processing all literature at maximum depth. This reduces total extraction cost by ~40% and processing time by ~60% compared to a single-pass strategy, while achieving equivalent temporal coverage on the diseases that matter most."

### Key Numbers for Paper
- 17,080 diseases processed (all PrimeKG diseases)
- 3-LLM consensus with DeepSeek V3 fallback chain
- Batch API pricing: 50% cost reduction on 2/3 providers
- Literature-adaptive: 3 tiers of extraction depth based on coverage analysis
- ~$500-700 total extraction cost (reproducibility-friendly)

---

## Parallel Execution Architecture (3 Processes)

```
Process 1: complete_chunk.py (chunk N — reuse existing batch IDs)
    └── 10-concurrent Gemini 2.5 Flash + poll existing OpenAI/Anthropic batches
    └── Zero wasted money: batches already submitted and processing

Process 2: run_batch_17k.py --resume-from N+1 (future chunks)
    └── Full pipeline: harvest → batch submit → concurrent Gemini → poll → consensus

Process 3: pre_harvest.py (cache ahead, abstracts only)
    └── Profile + harvest diseases for future chunks before they're needed
```

### Gemini Concurrency Design
```python
sem = asyncio.Semaphore(10)       # 10 concurrent (well within 300 RPM Tier 1)
# Wave processing: 50 docs/wave for progress logging
# On 429 rate limit: backoff 5s + retry once
# On persistent 429 (>30% of wave): halve semaphore to 5
# On total failure: DeepSeek V3 per document (no data loss)
```

**Impact:** Gemini phase dropped from ~15 hours/chunk (sequential) to ~1.5 hours/chunk (10-concurrent). 10x speedup.

---

## Bugs Fixed During Implementation

| Bug | Impact | Fix |
|-----|--------|-----|
| `custom_id > 64 chars` | All Anthropic batches rejected | `_make_custom_id()` with dynamic budget allocation |
| OpenAI JSONL > 200MB | All OpenAI batches rejected | Reduced chunk size 50K → 10K requests |
| OpenAI 2M token enqueue limit | Second batch rejected | 30s stagger between batch submissions |
| Phase 5 key mismatch | Batch results silently dropped for long-ID diseases | Unified key generation via `_make_custom_id()` in both submit and lookup |

---

## Related Notes
- `001_triple_count_sufficiency.md` — triple count targets
- `002_temporal_coverage_assessment.md` — temporal field coverage analysis
- `003_temporal_data_loss_bug.md` — QC mapping fix
