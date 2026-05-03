# Research Note 020: Tier 3 Feasibility Analysis — Novel Knowledge Claims

**Date:** 2026-04-05
**Status:** CRITICAL FINDING — most Tier 3 claims are not defensible

---

## Summary

Tier 3 was designed to demonstrate ChronoMedKG extracts **novel temporal knowledge** not present in existing databases (Orphadata, HPOA, Phenopackets). The intended claims:
- 681 diseases with onset range extensions
- 764 diseases with novel disease staging
- 44 diseases contradicting 2+ external sources

**After evidence audit, only ~16% of onset divergences are defensible.** The rest suffer from two upstream extraction errors.

---

## Upstream Errors

### Error 1: Age-of-Patient vs Age-of-Onset Conflation
The extraction pipeline treats patient ages at time of study/complication/death as onset ages.

Examples:
- "A 58-year-old with 35-year history" → pipeline records onset as 58 (actual onset: ~23)
- "18 months prior" → pipeline extracts 18 as age (actually duration in months)
- "average age of 48, range 11-88" → pipeline picks 11 as onset_min (it's minimum from a case series)

### Error 2: Age-Category Labels as Disease Stages
The pipeline extracts "neonatal", "infantile", "adult-onset" as disease progression stages. These are patient subgroup classifiers from case series, not temporal progression stages.

Example: Kienböck disease (avascular necrosis of lunate) gets stages "neonatal, infantile, juvenile, geriatric" — but neonates don't get Kienböck disease. The actual Lichtman staging (I→II→IIIA→IIIB→IV) is based on radiographic findings.

---

## Evidence Audit Results (200 divergences checked)

| Category | Count | % |
|----------|------:|--:|
| Defensible (evidence text confirms onset age) | 32 | 16% |
| Mixed (patient age language, ambiguous) | 62 | 31% |
| Likely garbage (evidence contradicts claim) | 6 | 3% |
| Unverifiable (no numeric ages in evidence text) | 100 | 50% |

Extrapolated to full 681 divergences:
- Defensible: ~108
- Questionable: ~231
- Unverifiable: ~340

---

## LLM Audit of 15 Tier 3 Questions

Conducted by external LLM reviewer. Results:
- **2/15 plausible (13% pass rate)**
- 5/5 novel_staging questions invalid (age categories, not real stages)
- 5/5 extended_range questions had evidence issues
- 3/5 onset_extension questions had unsupported claims

---

## What IS Defensible

~108 onset divergences where:
1. Evidence text contains explicit numeric age
2. That age exceeds the Orphadata range
3. The language is onset-specific (not patient-age-at-study)

These could form a small, ultra-high-confidence Tier 3 (~50-100 questions).

---

## Recommendation

### For current submission:
- Do NOT include Tier 3 in benchmark
- Do NOT claim "681 divergences" in paper
- CAN mention "~100 evidence-supported onset extensions" as resource characterization
- Frame as: "preliminary evidence of novel knowledge extraction, with systematic verification planned"

### For revision/future work:
- Fix extraction prompts to distinguish onset age vs patient age
- Fix stage extraction to distinguish progression stages vs age categories
- Re-extract with improved prompts
- Build verified Tier 3 from clean extractions

### For the paper narrative:
- Tier 1 (validation) + Tier 2 (utility) carry the submission
- Gap analysis is mentioned as resource characterization, not as a benchmark claim
- The extraction limitation is acknowledged honestly — reviewers respect this

---

## Key Lesson

**LLM extraction pipelines conflate "age mentioned in text" with "age of onset."** This is a fundamental challenge for temporal KG construction that affects all LLM-based approaches (KARMA likely has the same issue but never checked). Documenting this openly could itself be a contribution.
