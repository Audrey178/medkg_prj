# Research Note 005: Federated Medicine & Agent Institutions Framing

**Date:** 2026-03-28
**For:** Paper 2 Discussion & Future Directions
**Source:** Evans, Bratton & Agüera y Arcas (2026). "Agentic AI and the next intelligence explosion." arXiv:2603.20639v1

---

## Core Argument

ChronoMedKG is not "PrimeKG + temporal dimension." It is a **foundation layer for context-aware, continuously updatable medical knowledge** — where temporal phenotype progression, evidence grading, and (eventually) geographic/demographic metadata create a living diagnostic substrate.

## Key Concepts

### 1. Federated Medical Knowledge (Future Direction)
- Current: diagnostic knowledge fragmented across geographies; guidelines update on 3-5 year cycles
- Vision: federated KG construction with evidence grading — local clinical observations feed shared knowledge, weighted by evidence quality, geographic context, demographic relevance
- Evidence tier system (Tier 1 curated, Tier 2 LLM-extracted, Gold/Silver/Bronze consensus) = **trust architecture for heterogeneous evidence**
- Analogy: federated learning solves privacy but not semantics; federated KG construction with evidence tiers solves both

### 2. Standardized Reasoning, Parameterized Priors
- Don't want uniform diagnosis — want **standardized reasoning frameworks with contextual adaptation**
- Ontology = global (ICD-11, HPO, OMIM), inference rules = standardized, priors = localized
- Temporal KG with demographic/geographic edge metadata enables: same architecture, different evidence weighting
- Examples: DMD progression differs with corticosteroid access; CYP2D6 polymorphisms across populations

### 3. Agentic Pipeline as "Agent Institution" (Discussion)
- 4-agent pipeline (Profiler → Harvester → Extractor → QC) maps to Evans et al.'s "agent institutions"
- Specialized roles with defined protocols, not monolithic
- Parallel: courtrooms function because "judge/attorney/jury" are well-defined slots; our pipeline works because each agent has constrained role with clear I/O contracts
- Design advantage, not implementation detail — principled choice grounded in social intelligence theory

### 4. Self-Healing KG (Bridge to Federated)
- Audit pipeline (`scripts/audit_primekg.py`) IS the self-healing mechanism
- Re-run after each extraction batch → stale edges flagged → contradictions surfaced
- Current: periodic batch audit → flag stale → publish updated KG
- Future: continuous clinical observation stream → evidence grading → parameterized updates

## Paper Placement

| Section | Content |
|---------|---------|
| Intro motivation | "Static KGs cannot support continuously evolving, context-dependent medical knowledge" |
| Results §4.2 | Evidence Decay Analysis: 41% stale >10yr, 316 contradicted contraindications |
| Discussion §5.1 | Agent institution architecture (cite Evans et al.) |
| Discussion §5.2 | Self-healing audit pipeline as practical contribution |
| Future Directions §6 | Federated medical KG vision; demographic/geographic parameterization |

## What NOT to Overclaim
- No clinical feedback data beyond Göttingen collaborators
- Federated infrastructure is post-PhD scope — plant flag, don't promise
- Demographic/geographic parameterization = genuine future work

## Citation
```bibtex
@article{evans2026agentic,
  title={Agentic AI and the next intelligence explosion},
  author={Evans, James and Bratton, Benjamin and Ag{\"u}era y Arcas, Blaise},
  journal={arXiv preprint arXiv:2603.20639},
  year={2026}
}
```
