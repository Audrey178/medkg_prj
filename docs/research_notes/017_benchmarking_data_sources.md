# Research Note 017: Benchmarking Data Sources for ChronoMedKG

**Date:** 2026-04-02
**Status:** CATALOGED — ready for integration

---

## Currently Using

| Source | Coverage | What we validate |
|--------|----------|-----------------|
| Orphadata (en_product9_ages.xml) | 6,729 diseases | Disease-level onset age bins |
| HPO (phenotype.hpoa) | 1,512 diseases | Per-phenotype onset annotations |
| PrimeKG | 17,080 diseases | Static edge confirmation |
| iKraph | 33,914 diseases | Cross-KG coverage comparison |
| Hetionet | 136 diseases | Cross-KG coverage comparison |

---

## Priority 1: Temporal Ground Truth (NEW)

### 1. GA4GH Phenopackets Corpus (phenopacket-store)
- **URL:** https://zenodo.org/records/14907503 / https://github.com/monarch-initiative/phenopacket-store
- **Content:** 6,668 phenopackets from 959 publications — 475 Mendelian diseases, 423 genes, 3,834 pathogenic alleles. Each phenopacket includes **age of onset per phenotypic feature**, sex, age at last examination
- **Format:** GA4GH Phenopacket Schema (JSON/protobuf), ISO-approved standard
- **Benchmark use:** Case-level temporal validation — compare ChronoMedKG stage-specific phenotypes against individual patient phenopackets with per-feature onset ages. **Arguably the highest-quality temporal ground truth available**
- **Availability:** Fully open, CC-BY
- **Priority:** CRITICAL — download and integrate immediately

### 2. Open Targets Platform (timestamped evidence)
- **URL:** https://platform.opentargets.org/downloads
- **Content:** Target-disease evidence from 22 sources. **NEW in 2025: comprehensive timestamping across millions of evidence points + target novelty metric**
- **Format:** Parquet files via bulk download, API
- **Benchmark use:** The only major KG-like resource besides ChronoMedKG that has added temporal evidence tracking. Direct head-to-head comparison of temporal coverage approaches
- **Availability:** Free, open access (CC0)
- **Priority:** CRITICAL — only direct temporal competitor

### 3. FDA Drugs@FDA + NME Compilation
- **URL:** https://catalog.data.gov/dataset/drugsfda-database
- **Content:** Drug approval dates since 1939 (most complete from 1985+). NME CSV contains curated new molecular entity approvals. 11 tables in CSV/SAS/Stata
- **Format:** CSV, structured tables
- **Benchmark use:** Gold-standard timestamps for treatment timeline edges (when drugs became available for diseases)
- **Availability:** Free, public domain
- **Priority:** HIGH — easy to integrate, validates treatment_timeline edges

### 4. ClinVar (Variant Classification Timeline)
- **URL:** https://www.ncbi.nlm.nih.gov/clinvar/
- **Content:** "First in ClinVar" date, "Last evaluated" date, full reclassification history for every variant. Variants reclassified at ~0.3% per variant-month
- **Format:** XML/TSV bulk download from NCBI FTP
- **Benchmark use:** Evidence trajectory validation — timestamped variant classifications provide ground truth for how gene-disease evidence evolves over time
- **Availability:** Free, public domain
- **Priority:** HIGH — validates evidence_trajectory TQA questions

### 5. Age-Phenome Knowledge-base (APK)
- **URL:** https://link.springer.com/article/10.1186/2193-1801-1-4
- **Content:** 35,683 entries linking ages to phenotypes (onset, diagnosis, observation) extracted from 1.5M PubMed abstracts using NLP. 5 relationship types including "Age of onset" and "Age of diagnosis"
- **Format:** Structured database with 4 tables
- **Benchmark use:** Cross-validate ChronoMedKG onset ages against literature-derived age-phenotype associations at scale
- **Availability:** Free
- **Priority:** HIGH — large-scale onset age validation

### 6. ClinicalTrials.gov API v2.0
- **URL:** https://clinicaltrials.gov/data-api
- **Content:** Trial start/end dates, eligibility criteria (including age), interventions, conditions, phases. REST API with OpenAPI 3.0 spec
- **Coverage:** 400K+ trials worldwide
- **Benchmark use:** Treatment timeline validation; age-based eligibility criteria can validate disease onset ranges
- **Availability:** Free, public
- **Priority:** MEDIUM — useful but requires matching logic

### 7. FAERS / OnSIDES
- **URL:** https://open.fda.gov/data/faers/ / https://onsidesdb.org/
- **Content:** FAERS: post-marketing adverse event reports with time-to-onset. OnSIDES (2025): 7.1M drug-ADE pairs for 4,097 drug ingredients from 51,460 labels (USA, EU, UK, Japan)
- **Benchmark use:** Validate drug side-effect onset timing
- **Availability:** Free, public domain / open-source
- **Priority:** MEDIUM

### 8. DECIPHER
- **URL:** https://www.deciphergenomics.org/
- **Content:** 40,000+ patient records with 51,000+ variants and 172,000+ HPO phenotype terms. Records support age of onset, pace of progression, severity
- **Coverage:** Rare disease patients from 250+ projects, ~40 countries
- **Benchmark use:** Patient-level phenotype onset data; progression pace annotations
- **Availability:** Open-access aggregated data; individual-level requires membership
- **Priority:** MEDIUM — rich but access-limited

---

## Priority 2: Cross-KG Comparison (NEW)

### 9. SPOKE (Scalable Precision Medicine Open Knowledge Engine)
- **URL:** https://spoke.ucsf.edu/
- **Content:** 27M nodes (21 types), 53M edges (55 types) from 41 databases
- **Benchmark use:** Largest open biomedical KG by node count. No temporal metadata — ChronoMedKG advantage
- **Availability:** Free, open access
- **Priority:** HIGH — impressive scale comparison

### 10. Monarch Initiative Knowledge Graph
- **URL:** https://monarchinitiative.org/
- **Content:** Cross-species gene-phenotype-disease associations. 6,499 Mondo disease terms, 15,538 phenotype terms
- **Format:** KGX TSV, Neo4j, API, R package
- **Benchmark use:** Rich phenotype-disease associations; lacks temporal grounding
- **Availability:** Free, CC-BY
- **Priority:** HIGH — strong phenotype coverage

### 11. DisGeNET v25.2
- **URL:** https://disgenet.com/
- **Content:** Comprehensive gene-disease associations with variant-disease, mode of inheritance annotations. NLP-extracted + curated
- **Format:** TSV, Cytoscape, R package, SPARQL
- **Benchmark use:** Gene-disease association comparison; has evidence scores but no temporal
- **Availability:** Free for academic use
- **Priority:** MEDIUM

### 12. CTD (Comparative Toxicogenomics Database)
- **URL:** https://ctdbase.org/
- **Content:** 17,100 chemicals, 54,300 genes, 6,100 phenotypes, 7,270 diseases. Manually curated
- **Benchmark use:** Chemical-disease-gene triadic relationships; no temporal
- **Availability:** Free, open access
- **Priority:** MEDIUM

### 13. PharmKG
- **URL:** https://github.com/biomed-AI/PharmKG
- **Content:** 500,000+ interconnections between genes, drugs, diseases. 29 relation types
- **Benchmark use:** Drug-centric comparison; no temporal
- **Availability:** Free, open source
- **Priority:** LOW

### 14. PKG 2.0 (PubMed Knowledge Graph 2.0)
- **URL:** https://pubmedkg.github.io/
- **Content:** 36M papers, 1.3M patents, 0.48M clinical trials. 482M biomedical entity linkages
- **Benchmark use:** Document-level timestamps vs ChronoMedKG fact-level temporal grounding
- **Availability:** Free, Scientific Data 2025
- **Priority:** LOW — document-level only

### 15. Samyama Biomedical KGs (March 2026)
- **URL:** https://arxiv.org/abs/2603.15080
- **Content:** Three open KGs: Pathways KG (835K edges), Clinical Trials KG (27M edges), Drug Interactions KG (192K edges)
- **Availability:** Open-source, Apache 2.0
- **Priority:** LOW — very recent, worth mentioning

---

## Priority 3: Evaluation Frameworks & QA Benchmarks

### 16. BioKGBench
- **URL:** https://github.com/westlake-autolab/BioKGBench
- **Content:** 2,000+ data points for KGQA and Scientific Claim Verification + 225 annotated agent tasks. Discovered 90+ factual errors in popular KGs
- **Benchmark use:** Directly evaluates AI agents on biomedical KG fact checking
- **Priority:** HIGH — directly relevant methodology

### 17. TGB 2.0 (Temporal Graph Benchmark)
- **URL:** https://arxiv.org/abs/2406.09639
- **Content:** NeurIPS 2024 D&B. 8 temporal graph datasets, up to 53M edges
- **Benchmark use:** Evaluation methodology and metrics for temporal graph learning; reviewer expectations
- **Priority:** HIGH — methodology reference for NeurIPS submission

### 18. TemporalFC (ISWC 2023)
- **URL:** https://link.springer.com/chapter/10.1007/978-3-031-47240-4_25
- **Content:** Temporal fact-checking benchmark
- **Benchmark use:** Framework for evaluating temporal fact validity
- **Priority:** MEDIUM

### 19. Standard Medical QA (for LLM baseline)
- MedQA: https://www.vals.ai/benchmarks/medqa
- PubMedQA: https://pubmedqa.github.io/
- BioASQ: http://bioasq.org/
- MedMCQA: 193K+ MCQs
- **Benchmark use:** LLM parametric baseline (Layer 4 of validation)
- **Priority:** LOW for KG benchmarking, needed for LLM baseline experiment

---

## Priority 4: Clinical Validation

### 20. GeneReviews (structured parsing)
- **URL:** https://www.ncbi.nlm.nih.gov/books/NBK1116/
- **Content:** Expert-authored reviews for 800+ genetic conditions with staging, onset, management timelines
- **Priority:** HIGH — but requires NLP parsing (no structured bulk download)

### 21. OMIM (genemap2 + Clinical Synopsis)
- **URL:** https://omim.org/downloads/
- **Priority:** HIGH — need API key (currently missing)

### 22. UniProt Disease Annotations
- **URL:** https://www.uniprot.org/
- **Content:** 81,000+ curated variants in 13,000 human proteins; 30,000+ Mendelian disease variants
- **Priority:** MEDIUM

### 23. MONDO Disease Ontology
- **URL:** https://mondo.monarchinitiative.org/
- **Priority:** LOW (already using MONDO IDs)

---

## Competitor Systems to Position Against

| System | Venue | Key difference from ChronoMedKG |
|--------|-------|----------------------------------|
| KARMA | NeurIPS 2025 Spotlight | No temporal, 1,200 articles only |
| AutoBioKG | bioRxiv Jan 2026 | No temporal, not agentic |
| KG-Orchestra | bioRxiv Feb 2026 | Evidence-based but no temporal |
| Open Targets | Platform | Timestamp on evidence, not on facts |
| iKraph | Nat Mach Intel 2025 | 34M papers but zero structured temporal |

---

## Recommended Integration Order

### Immediate ($0, 1-2 days each)
1. **Phenopackets** — download from Zenodo, parse JSON, cross-validate per-phenotype onset ages
2. **FDA Drugs@FDA** — download CSV, match drug-disease pairs, validate treatment timeline edges
3. **ClinVar timeline** — download TSV, extract first/last dates per variant, validate evidence trajectory

### Next sprint ($0, needs matching logic)
4. **Open Targets** — download Parquet, compare temporal evidence coverage head-to-head
5. **SPOKE** — download Neo4j dump, compare edge coverage at scale
6. **Monarch** — download KGX TSV, compare phenotype-disease coverage

### Paper experiments
7. **BioKGBench** — run ChronoMedKG through their fact-checking framework
8. **Age-Phenome KB** — cross-validate onset ages at scale
9. **ClinicalTrials.gov** — validate treatment timelines via trial dates

---

## Key Insight for Paper

**Phenopackets is the single most valuable addition.** 6,668 real patient cases with per-phenotype onset ages — this is exactly the granularity ChronoMedKG claims to provide. If we show 85%+ agreement with Phenopackets at the per-phenotype level, that's a much stronger validation than the current Orphadata comparison (which is disease-level bins).

**Open Targets is the positioning threat.** They added timestamping in 2025 — we need to clearly differentiate: they timestamp evidence documents, we timestamp facts. Their temporal is "paper X was published in 2023", ours is "onset age of phenotype Y in disease Z is 3-5 years."
