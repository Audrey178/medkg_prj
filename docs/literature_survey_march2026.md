# PrimeKG-T Literature Survey & Competitive Landscape
**Date:** March 2026
**Purpose:** Deep research informing Paper 2 architecture decisions
**Target venue:** NeurIPS 2026 Datasets & Benchmarks

---

## Executive Summary

The landscape has shifted dramatically since we scoped Paper 2. Three major systems occupy adjacent territory (KARMA at NeurIPS 2025, iKraph in Nature Machine Intelligence, AutoBioKG on bioRxiv). However, **none** addresses temporal grounding or evidence hierarchy on biomedical KGs. The gap is real and widening.

Our unique intersection — **temporal + evidence hierarchy + agentic autonomy** — is genuinely novel. Every individual component has precedent; the combination does not exist anywhere.

---

## 1. Competitor Deep Dive

### 1.1 KARMA (NeurIPS 2025 Poster)

**Paper:** "Leveraging Multi-Agent LLMs for Automated Knowledge Graph Enrichment" (Lu & Wang, 2025)
**URL:** https://arxiv.org/abs/2502.06472 | https://openreview.net/forum?id=k0wyi4cOGy
**GitHub:** https://github.com/YuxingLu613/KARMA

**Architecture — 9 Agents:**
1. Ingestion Agents (IA): Document retrieval and normalization
2. Reader Agents (RA): Filter and parse documents into segments
3. Summarizer Agents (SA): Create concise summaries
4. Entity Extraction Agents (EEA): Identify and normalize entities
5. Relationship Extraction Agents (REA): Infer relationships
6. Schema Alignment Agents (SAA): Ensure conformity to existing KG schema
7. Conflict Resolution Agents (CRA): LLM debate mechanism for contradictions
8. Evaluator Agents (EA): Aggregate outputs, confidence/clarity/relevance scoring
9. Central Controller: Orchestrates all agents

**Results:** 38,230 new entities from 1,200 PubMed articles, 83.1% LLM-verified correctness, 18.6% conflict reduction.

**Exploitable Weaknesses:**
- **LLM-only evaluation** — no human expert validation. We have Gottingen clinicians.
- **Single LLM backbone** (DeepSeek-v3) — we use 3-family consensus (GPT-4o-mini, Claude, Gemini)
- **Performance drops -12% on sparse domains** (metabolomics) — rare diseases ARE sparse
- **No temporal dimension** — all edges are static
- **No evidence hierarchy** — case report = meta-analysis
- **Only 1,200 articles** — tiny scale
- **Manual domain specification** required — not disease-autonomous
- Was a **Poster** (not Spotlight as initially reported)

**Citation strategy:** "KARMA demonstrated multi-agent LLM architectures can effectively enrich knowledge graphs. However, KARMA treats all extracted facts as temporally static and evidence-equivalent. PrimeKG-T extends this paradigm by adding temporal grounding and evidence hierarchy as first-class dimensions."

---

### 1.2 iKraph (Nature Machine Intelligence, March 2025)

**Paper:** "A comprehensive large-scale biomedical knowledge graph for AI-powered data-driven biomedical research" (Zhang et al., 2025)
**URL:** https://www.nature.com/articles/s42256-025-01014-w

**Architecture:** Fine-tuned NLP pipeline (LitCoin Challenge winner, human-level accuracy):
- NER: RoBERTa-large + PubMedBERT with ensemble modeling + label smoothing
- RE: Multi-sentence document-level relation prediction
- Integration: 40 public database sources merged

**Scale:** ALL PubMed abstracts (34M+), 10,686,927 unique entities, 30,758,640 unique relations.

**Results:** Human-level extraction accuracy. Drug repurposing demos (COVID-19, cystic fibrosis). Probabilistic semantic reasoning.

**Exploitable Weaknesses:**
- **Completely static** — no temporal dimension whatsoever
- **No evidence grading** — 2005 pilot study = 2024 Phase III RCT
- **Traditional pipeline** — no agentic architecture, no conflict resolution
- **No clinical utility demo** beyond drug repurposing
- **Cannot answer temporal queries** — "What was standard of care in 2015 vs 2024?"

**Citation strategy:** "iKraph achieves unprecedented scale by processing all PubMed abstracts with human-level accuracy. However, like all existing large-scale biomedical KGs, iKraph treats every extracted fact as equally valid and temporally unbounded."

---

### 1.3 AutoBioKG (bioRxiv, January 2026)

**Paper:** "Automating Biomedical Knowledge Graph Construction For Context-Aware Scientific Inference"
**URL:** https://www.biorxiv.org/content/10.64898/2026.01.14.699420v1

**Key Innovations:**
- **Composite triples:** Enriched with contextual conditions and entity attributes (e.g., "Protein X phosphorylates Protein Y under hypoxia conditions"). Fine-grained attributes and contextual dependencies enhance semantic density.
- **Self-evolution learning:** Model iteratively improves by learning from its own high-confidence predictions. Synergistic confidence-entropy filtering.

**Results:** Outperforms GPT-4o zero-shot by 18.5-20.7% F1 on DDI, ChemProt, BioRED. Outperforms on BioASQ complex queries.

**Exploitable Weaknesses:**
- **No temporal dimension**
- **Not multi-agent** — single model self-evolution
- **No PrimeKG integration** — standalone schema
- **bioRxiv only** — not peer-reviewed
- **No clinical utility demo**

**What to adopt:** Composite triple concept → our `ConditionalContext` on edges (age_group, disease_stage, genetic_subtype). We already have this in our schema.

---

### 1.4 EvoReasoner + EvoKG (MIT CSAIL + IBM, September 2025)

**Paper:** "Temporal Reasoning with Large Language Models Augmented by Evolving Knowledge Graphs"
**URL:** https://arxiv.org/abs/2509.15464

**Architecture:**
- **EvoKG:** Incrementally updates KG from unstructured documents. Confidence-based contradiction resolution + temporal trend tracking. Maintains multiple candidates with confidence scores.
- **EvoReasoner:** Temporal-aware multi-hop reasoning with global-local entity grounding, multi-route decomposition, temporally grounded scoring.

**Result:** 8B model + EvoKG matches 671B model performance.

**Critical finding for us:** This is the closest system to our temporal approach, but in general domain, not biomedical. Our differentiation:
- EvoKG treats all sources equally → we have Tier 1/2 evidence hierarchy
- EvoKG is general domain → we're biomedical with clinical validation
- EvoKG has no benchmark → we have PrimeKG-TQA
- EvoKG has no clinical utility demo → we have temporal differential diagnosis

**Citation strategy:** Cite as evidence that temporal KG evolution matters, then differentiate on evidence hierarchy and biomedical specialization.

---

### 1.5 Other Competitors

**PKG 2.0** (Scientific Data, June 2025): 36M papers + patents + clinical trials, 482M entity linkages. Has document timestamps but NOT fact-level temporal grounding.

**KG4Diagnosis** (AAAI Bridge 2025): Multi-agent + 362 diseases. Diagnosis-focused, no KG construction.

**Knez & Zitnik** (CL4Health, LREC-COLING 2024): Augments PrimeKG with temporal info for **patient-level** temporal relation extraction from clinical notes. NOT literature-level evidence evolution. Workshop paper, preliminary.

**BioKGrapher** (CSBJ, October 2024): Automated KG from PubMed, KL-divergence weighting, 6 conditions. Not temporal, not agentic.

---

## 2. Entity Normalization SOTA (2025-2026)

### 2.1 SapBERT Is No Longer SOTA

**BioLORD-2023** (Remy et al., JAMIA 2024):
- Fuses LLM-generated definitions with SNOMED-CT ontology via contrastive learning
- SOTA on MedSTS (clinical sentences) and EHR-Rel-B (biomedical concepts)
- Better than SapBERT for clinical concept similarity
- URL: https://huggingface.co/FremyCompany/BioLORD-2023

**CODER-BERT / GEBERT:** Knowledge-infused and graph-based extensions of SapBERT.

**KrissBERT** (Zhang et al., NAACL 2024): Knowledge-Rich Self-Supervised learning. Strong on rare entity normalization.

### 2.2 LLM-Based Entity Normalization (Breakthrough)

**LLM as Entity Disambiguator** (Ye & Mitchell, ACL 2025):
- LLM as second-stage disambiguator after bi-encoder candidate generation
- **+16 accuracy points** over previous SOTA on multiple biomedical datasets
- Zero fine-tuning required
- URL: https://aclanthology.org/2025.acl-short.25/

**Generative Relevance Feedback** (Bioinformatics 2026):
- RAG-based re-ranking with GPT-4o for entity linking
- LLM contextual understanding enables nuanced alignment
- URL: https://academic.oup.com/bioinformatics/article/42/2/btag011/8426181

### 2.3 Our Innovation: 3-Stage Pipeline

No existing biomedical KG construction system uses LLM-in-the-loop entity disambiguation. Our pipeline:

| Stage | Method | When Used |
|-------|--------|-----------|
| 1. Dictionary | PrimeKG index (128K nodes, type-aware) | Always — exact/fuzzy match |
| 2. Embedding | BioLORD-2023 dense retrieval + cosine similarity | When dictionary fails |
| 3. LLM | Claude Haiku / GPT-4o-mini disambiguation | Only for ambiguous cases (confidence-gated) |

This is a publishable contribution — the confidence-gated routing between fast embeddings and expensive LLM is novel for KG construction.

---

## 3. Temporal KG Landscape

### 3.1 Existing Temporal KG Methods Are Event-Centric

| Method | Year | Approach | Domain |
|--------|------|----------|--------|
| TTransE | 2018 | Timestamp as translation vector | General events |
| HyTE | 2018 | Time-specific hyperplanes | General events |
| DE-SimplE | 2020 | Diachronic entity embeddings | ICEWS/GDELT |
| TNTComplEx | 2020 | Tensor decomposition + time | General events |
| BoxTE | 2022 | Box embeddings + temporal | General events |
| TiRGN | 2022 | Temporal interaction-relational GNN | General events |

**Critical distinction:** All of these model `(entity, relation, entity, point_timestamp)` for events like "(Obama, visited, China, 2013-03-22)". Our model is fundamentally different: `(entity, relation, entity, [validity_start, validity_end, superseded_by])` — interval-based evidence validity with supersession. Must frame this distinction clearly in Related Work.

### 3.2 Biomedical KGs Are ALL Static

| KG | Temporal? | Evidence Grading? |
|----|-----------|------------------|
| PrimeKG (2023) | No | No |
| Hetionet (2017) | No | No |
| DRKG (2020) | No | No |
| iKraph (2025) | No | No |
| PrimeKG++ (2025) | No | No |
| **PrimeKG-T (Ours)** | **Yes — fact-level** | **Yes — Tier 1/2 + credibility** |

### 3.3 Evidence Evolution Is an Empty Field

- **Retraction Watch:** 47K+ retracted papers tracked, but no system propagates retractions to KGs
- **Schneider et al. (2020):** Retracted findings persist in KGs for years after retraction
- **Cochrane Living Reviews:** Track changing conclusions but NOT in graph form
- **Clinical guideline updates:** MAGIC/OpenCPG formalize guidelines but don't track temporal evolution in KGs

**Nobody has built a system that detects when biomedical KG facts become outdated.** This is our central contribution.

---

## 4. Temporal QA Benchmarks

### 4.1 No Biomedical Temporal QA Benchmark Exists

| Benchmark | Year | Domain | Temporal? | Biomedical? |
|-----------|------|--------|-----------|-------------|
| BioASQ | 2013+ | Biomedical | No | Yes |
| PubMedQA | 2019 | PubMed | No | Yes |
| PrimeKGQA | 2024 | PrimeKG | No | Yes |
| CronQuestions | 2021 | Wikidata | Yes | No |
| TimeQA | 2021 | Wikipedia | Yes | No |
| ComplexTempQA | 2023 | Open | Yes | No |
| **PrimeKG-TQA (Ours)** | **2026** | **PrimeKG-T** | **Yes** | **Yes** |

PrimeKG-TQA is the **first** benchmark at the intersection of temporal QA and biomedicine.

### 4.2 PrimeKGQA (Zitnik Lab, ECAI 2024)

- 83,999 QA pairs from PrimeKG
- 2-4 node subgraph questions
- Tests static KG reasoning ("What drugs treat X?", "What genes cause Y?")
- **ZERO temporal dimension** — all answerable from static snapshot
- URL: https://zenodo.org/records/13829395

---

## 5. NeurIPS D&B 2025/2026 Standards

### 5.1 Review Criteria
- Papers held to **same rigor as main track**
- Must host on Harvard Dataverse / Kaggle / HuggingFace / OpenML
- **Must include Croissant machine-readable format** (new 2025 requirement)
- Impact = enabling future research, broadening applicability, **challenging dominant evaluation paradigms**

### 5.2 What Makes Landmark
- **Define the evaluation paradigm** — PrimeKGQA became standard because they defined the task. PrimeKG-TQA can do the same for temporal biomedical reasoning.
- **Show something is wrong with the status quo** — demonstrate that static KGs propagate outdated/retracted findings
- **Resource + Benchmark + Utility Demo** = triple contribution (extremely strong for D&B track)

---

## 6. Novelty Assessment (Final)

| Claim | Status | Risk |
|-------|--------|------|
| Fact-level temporal scope on biomedical KG | **GENUINELY NOVEL** | Low — confirmed empty field |
| Evidence hierarchy on KG edges | **NOVEL in KGs** | Low — EBM pyramids exist conceptually, never on edges |
| Evidence supersession tracking | **NOVEL** | Low — retracted findings persist everywhere |
| First biomedical temporal QA benchmark | **CONFIRMED FIRST** | Medium — someone could publish before us |
| LLM-in-the-loop entity normalization for KG construction | **NOVEL combination** | Low — nobody uses this for KG building |
| Disease-autonomous agentic construction | Incrementally novel | Medium — KARMA could add this |

---

## 7. Recommended Related Work Structure

1. **Biomedical Knowledge Graphs** (PrimeKG, Hetionet, iKraph, DRKG — all static)
2. **Agentic KG Construction** (KARMA, AutoBioKG, BioKGrapher)
3. **Temporal KG Methods** (TTransE, DE-SimplE, TNTComplEx — note: event-centric, NOT evidence-validity-centric)
4. **Temporal QA Benchmarks** (CronQuestions, TimeQA — note: none biomedical)
5. **Evidence-Based Medicine Informatics** (living reviews, retraction tracking — none produce temporal KGs)

---

## 8. Landmark Elevation Strategies

1. **Find a concrete "PrimeKG is wrong, PrimeKG-T is right" example.** A withdrawn drug or retracted finding that persists in static PrimeKG but is correctly flagged in PrimeKG-T.
2. **PrimeKG-TQA must show a large accuracy delta** (target: 40%+) between temporal-aware and static systems.
3. **Bench-to-bedside temporal chain:** Gene discovery → drug development → FDA approval as traversable KG path.
4. **Killer intro sentence:** "Existing biomedical KGs are frozen snapshots that treat a 2005 case report and a 2024 meta-analysis as equivalent evidence."
