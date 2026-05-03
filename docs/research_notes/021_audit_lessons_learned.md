# Research Note 021: Audit Lessons Learned — False Claims and Root Causes

**Date:** 2026-04-07
**Status:** CRITICAL — methodology notes for all future work

---

## Summary

During the ChronoMedKG paper preparation, three significant false claims were made and subsequently caught through rigorous auditing. This note documents each error, its root cause, how it was caught, and the corrective principle.

---

## Error 1: RAG Gain Inflated from +6.1pp to +33.9pp

**False claim:** "GPT-4o scores 47.6% → 81.5% with ChronoMedKG (+33.9pp)"

**Actual result:** +6.1pp overall (+24pp on stage-conditional, no gain on ordering)

**Root cause:** The `age_order_match` scorer in `score_ordering()` checked whether ANY ages mentioned in the LLM's answer appeared in ascending order — not whether those ages matched the gold standard milestone ages. Since LLMs naturally describe events chronologically, almost any answer with ascending age mentions was marked "correct."

**How caught:** User requested audit. Manual inspection of 50 "age_order_match" cases showed 96% were false positives — extracted ages didn't match gold standard ages (e.g., gold: [0.5, 8.9, 11.0, 22.7], found: [12.0, 1.8]).

**Corrective principle:** **Always manually verify 10+ scored examples before reporting aggregate numbers.** Automated scorers must be validated against human judgment on a sample before trusting at scale.

**Fix applied:** Rewrote scorer to require extracted ages to be within 5 years of gold standard ages AND in correct relative order. Also computed Kendall's tau as a continuous metric.

---

## Error 2: Multi-Model Consensus Claimed as Non-Existent

**False claim:** "99.4% of validated triples come from a single model — multi-LLM consensus is not real"

**Actual reality:** Multi-model consensus IS genuinely implemented. 2 LLMs run per document in parallel. The `extraction_models` metadata field only stores the representative model name, not all agreeing models. Consensus agreement is encoded in the confidence field (1.0 = full agreement, 0.67 = 2-of-3).

**Root cause:** Checked the DATA (validated triple metadata) without reading the CODE (knowledge_extractor.py). The metadata was misleading — it stores one model name per triple even though 2+ models contributed to the consensus decision.

**How caught:** User asked to "check the code properly." Full code audit of `knowledge_extractor.py` revealed the cascading consensus strategy (lines 481-551) and the `_compute_consensus` method (lines 783-871) with genuine cross-model agreement via Union-Find clustering.

**Corrective principle:** **Always trace claims back to source code, not just data files.** Metadata can be lossy. The code is the ground truth for how data was produced.

**Verified facts:**
- 2 models (DeepSeek V3.2 + GPT-4.1-nano) run per document in parallel
- Claude 3 Haiku invoked as tiebreaker for ~17% of documents
- Consensus requires same canonical relation + entity fuzzy match ≥80% (rapidfuzz token_sort_ratio)
- 82.9% of consensus triples have full agreement, 17.1% have 2-of-3 agreement
- 3.5% consensus rate (460K from 13M raw) reflects genuine cross-model stringency

---

## Error 3: 681 Onset Divergences Claimed Without Verification

**False claim:** "ChronoMedKG extends onset ranges for 681 diseases beyond Orphadata"

**Actual verified count:** ~21 with strict evidence verification (explicit onset-age language in source PMID + disease name proximity check)

**Root cause:** Compared TA `onset_age_min/max` fields against Orphadata ranges without checking whether the source evidence actually supported the claimed onset age. The extraction pipeline conflates patient age at time of study with true disease onset age in many cases.

**How caught:** External LLM audit of 15 Tier 3 questions showed 13% pass rate. Subsequent programmatic verification against full paper text confirmed only ~3.5% of claimed divergences have explicit onset-specific language in the source literature.

**Corrective principle:** **Never trust extracted data at face value. Always verify a sample against source evidence.** LLM-extracted fields (especially numeric ones like ages) should be treated as hypotheses until confirmed against the source text.

**Additional finding:** The age-of-patient vs age-of-onset conflation is a systematic extraction pipeline limitation affecting all LLM-based temporal extraction. This itself is a contribution worth documenting.

---

## Common Thread

All three errors share the same anti-pattern: **trusting intermediate outputs without first-principles verification.**

| What I trusted | What I should have done |
|---|---|
| Scorer verdict (correct/wrong) | Manually checked 10+ examples against human judgment |
| Metadata field (extraction_models) | Read the code that produces the metadata |
| Pipeline output (onset_age_min) | Verified against source evidence text |

---

## Corrective Checklist for Future Work

Before reporting ANY number in a paper:

1. **Scorer validation:** Manually verify 10+ scored examples per metric. If <90% agree with human judgment, the scorer is broken.

2. **Code tracing:** For any claim about methodology (e.g., "multi-model consensus"), read the actual code path end-to-end. Don't infer from data alone.

3. **Source verification:** For any claim about extracted knowledge (e.g., "X diseases have Y property"), verify a random sample against the original source documents.

4. **Sensitivity analysis:** Report results under multiple scoring variants (strict, partial, continuous) to show robustness. If results change dramatically across variants, the primary metric is fragile.

5. **Adversarial audit:** Before finalizing, ask: "What would a hostile reviewer check first?" Then check that yourself.

---

## Impact on Paper

| Claim | Before audit | After audit | Impact |
|---|---|---|---|
| RAG gain | +33.9pp | +24pp (staging only) | Changed headline but still strong |
| Multi-model consensus | "Not real" | Verified as genuine | Strengthened the paper |
| Novel discoveries | 681 | ~21 verified | Moved to supplementary table |
| Overall paper quality | Would have been rejected | Defensible | Saved the submission |

The auditing process, while time-consuming, transformed the paper from one containing three independently rejectable claims into one with honest, verified, defensible results.
