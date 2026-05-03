# Research Note 001: Are Our Triple Counts Sufficient for a Landmark Paper?

> Date: 2026-03-24
> Context: After 200-doc extraction for DMD and MG, assessing if output is landmark-worthy
> Verdict: **NOT YET — but the path is clear**

---

## Current Numbers (Hard Data)

| Disease | Docs processed | Consensus triples | Per-doc yield |
|---------|:--------------:|:-----------------:|:-------------:|
| DMD | 200 | 240 | 1.2 |
| MG | 200 | 382 | 1.9 |
| CIDP | 100 | 112 | 1.1 |
| GBS | 100 | 122 | 1.2 |
| LEMS | 100 | 87 | 0.87 |
| **Total** | | **943** | |

## What Competitors Produce

| System | Total triples | Diseases | Per-disease avg | Temporal? |
|--------|:------------:|:--------:|:---------------:|:---------:|
| **PrimeKG (original)** | 8.1M edges | 17K diseases | ~485 | No |
| **KARMA (NeurIPS 2025)** | ~15K | Generic KG | N/A | No |
| **iKraph (Nature MI 2025)** | 689K triples | All PubMed | N/A | No |
| **AutoBioKG** | ~50K | Generic | N/A | No |
| **Us (current)** | 943 | 5 diseases | ~189 | **Yes** |
| **Us (projected, 200 docs/disease)** | ~50K-75K | 200 diseases | ~250-375 | **Yes** |

## Honest Assessment

### What's GOOD:
1. **Quality over quantity** — every triple is 2/3 LLM consensus, schema-validated, credibility-scored
2. **Temporal metadata on 34-65% of triples** — no competitor has ANY
3. **Novel triples: 81-99% per disease** — genuinely new knowledge not in PrimeKG
4. **Reproducible pipeline** — re-runnable, costs ~$1.50/disease
5. **Super-linear scaling** — 2x docs gave 2.0-2.7x triples, more docs = more overlap = higher consensus

### What's NOT ENOUGH for a landmark paper:
1. **5 diseases is a demo, not a resource paper** — need 50-200 minimum
2. **~200 triples/disease is thin** — clinicians expect comprehensive disease profiles
3. **No downstream evaluation** — no benchmark, no clinical validation
4. **Temporal coverage gaps** — no validity_end, no treatment timelines, year-resolution only

## What Would Make It Landmark?

### Tier 1 (Must-have for top venue):
- [ ] Scale to **50+ diseases** (neuromuscular + metabolic + neurological)
- [ ] **PrimeKG-TQA benchmark**: 500+ temporal QA pairs with gold answers
- [ ] **Clinical validation**: 2-3 Gottingen neurologists evaluate 100 triples each
- [ ] **Downstream task**: temporal differential diagnosis or drug repurposing

### Tier 2 (Differentiator):
- [ ] Scale to **200+ diseases** → resource paper territory
- [ ] **Treatment timeline extraction** — "response within X weeks"
- [ ] **Supersession chains** — "Drug A replaced by Drug B in 2018"
- [ ] **Semantic Scholar citation velocity** for credibility scoring
- [ ] **Neo4j deployment** with interactive explorer

### Tier 3 (Nice to have):
- [ ] 800+ diseases (full PrimeKG disease coverage)
- [ ] Integration with MIMIC-IV for real-world temporal validation
- [ ] Federated learning component

## Bottom Line

**Current state: strong proof-of-concept, not yet landmark.**

The 200-doc runs showing 240-382 triples per disease with 60-65% temporal coverage
prove the pipeline works. But 5 diseases × ~200 triples = ~1,000 total triples won't
make reviewers say "this changes the field."

**Path to landmark:** Run 200 diseases × 200 docs each. At $1.50/disease = $300 total.
Expected yield: ~50K-75K temporally-grounded, evidence-graded triples across 200
diseases. THAT would be a genuine resource that doesn't exist anywhere else.

**Timeline estimate:** 200 diseases × ~2 hrs each = ~400 hrs sequential. With
parallelization (5 concurrent) = ~80 hrs = 3-4 days continuous. Feasible.
