# PrimeKG-T Implementation Flow

## High-Level Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR                                  │
│  Input: Disease list (or "all GeneReviews diseases")                 │
│  Output: PrimeKG-T resource files + Neo4j database                   │
│  For each disease, runs Agent 1 → 2 → 3 → 4 sequentially            │
└──────────┬───────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  AGENT 1: Disease Profiler                                            │
│                                                                       │
│  Input:  Disease ID (e.g., OMIM:310200) + name                       │
│  Steps:  1. Resolve identity (OMIM/Orphanet/MONDO cross-refs)        │
│          2. Check Tier 1 sources (GeneReviews, OMIM, Orphanet)       │
│          3. Profile PrimeKG neighborhood (edges, types)               │
│          4. Count PubMed articles + PMC OA availability               │
│          5. Generate extraction strategy (queries, yield estimate)    │
│          6. Identify differential diagnosis partners                  │
│  Output: config/diseases/{disease_id}.yaml (DiseaseProfile)          │
│                                                                       │
│  Decision gate: has_sufficient_sources() ?                            │
│    → YES: proceed to Agent 2                                          │
│    → NO:  mark as "sparse", include PrimeKG-only edges               │
└──────────┬───────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  AGENT 2: Evidence Harvester                                          │
│                                                                       │
│  Input:  DiseaseProfile from Agent 1                                  │
│  Steps:  Tier 1 collection:                                           │
│            1. Fetch GeneReviews full text (NCBI Books API)            │
│            2. Fetch OMIM clinical synopsis + text (OMIM API)          │
│            3. Fetch Orphanet data (Orphadata)                         │
│            4. Score each with credibility_scorer (Tier 1 = ~0.93)    │
│          Tier 2 collection:                                           │
│            5. Run PubMed queries from DiseaseProfile                  │
│            6. Fetch abstracts (batch, rate-limited)                   │
│            7. Classify study type (meta-analysis/RCT/cohort/etc.)    │
│            8. Score credibility (6-signal system)                     │
│            9. Rank and filter by credibility                          │
│  Output: data/extracted/{disease_id}/evidence_collection.json         │
│          EvidenceCollection (Tier 1 + Tier 2 SourceDocuments)        │
└──────────┬───────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  AGENT 3: Knowledge Extractor                                         │
│                                                                       │
│  Input:  EvidenceCollection + DiseaseProfile                          │
│  Steps:  For each document:                                           │
│            1. Build universal prompt (parameterized by profile)       │
│            2. Send to 3 LLMs: GPT-4o-mini, Claude Sonnet, Gemini    │
│            3. Parse structured JSON output from each                  │
│            4. Extract: subject, relation, object, temporal_context,   │
│               conditions, evidence_text, confidence                   │
│          Then:                                                        │
│            5. Normalize entities (3-stage: Dictionary → BioLORD →    │
│               LLM disambiguator → PrimeKG ID resolver)               │
│            6. Multi-LLM consensus: group by (S,R,O), require ≥2/3   │
│            7. Consensus confidence = fraction of models agreeing      │
│  Output: data/extracted/{disease_id}/raw_triples.jsonl                │
│          data/extracted/{disease_id}/consensus_triples.jsonl           │
│          ExtractionResult (raw + consensus RawTriples)                │
└──────────┬───────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  AGENT 4: Quality Controller                                          │
│                                                                       │
│  Input:  ExtractionResult + DiseaseProfile + PrimeKG edges            │
│  Steps:  For each consensus triple:                                   │
│            1. Validate fields (non-empty, sufficient confidence)      │
│            2. Check temporal plausibility (age range, date order)     │
│            3. Check PrimeKG confirmation (boost if existing edge)     │
│            4. Convert RawTriple → TemporalEdge (PrimeKG-aligned)     │
│               - Map relation to PrimeKG's 30 edge types              │
│               - Map entity types to PrimeKG's 10 node types          │
│               - Attach temporal metadata (discovery_date, validity)   │
│               - Attach evidence metadata (tier, credibility, PMIDs)  │
│               - Attach conditional context (age_group, stage, etc.)  │
│          Then:                                                        │
│            5. Detect conflicts (indication vs contraindication, etc.) │
│            6. Compute quality metrics (temporal coverage, grade)      │
│            7. Assign quality grade: A (>90%), B (70-90%), C (<70%)   │
│  Output: data/extracted/{disease_id}/validated_triples.jsonl          │
│          data/extracted/{disease_id}/quality_report.json              │
│          List[TemporalEdge] ready for Neo4j ingestion                │
└──────────┬───────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  POST-PIPELINE (after all diseases processed)                         │
│                                                                       │
│  Step 8: Neo4j KG Construction                                        │
│    - Ingest all TemporalEdges into Neo4j                              │
│    - Create temporal indexes (discovery_date, validity_start/end)     │
│    - Create evidence indexes (tier, credibility_score)                │
│    - PrimeKG-compatible schema with extended properties               │
│                                                                       │
│  Step 9: Export PrimeKG-T Resource                                    │
│    - CSV (PrimeKG-compatible + temporal columns)                      │
│    - JSON-LD (semantic web)                                           │
│    - Neo4j dump                                                       │
│    - Croissant format (NeurIPS D&B 2026 requirement)                 │
│    - Upload to Harvard Dataverse / Zenodo                             │
│                                                                       │
│  Step 10: PrimeKG-TQA Benchmark Generation                           │
│    - Generate temporal QA pairs from validated edges                   │
│    - 5 task types: fact retrieval, evidence evolution,                │
│      temporal diff dx, contradiction detection, multi-hop             │
│    - Upload to HuggingFace                                            │
│                                                                       │
│  Step 11: Downstream Evaluation                                       │
│    - Temporal differential diagnosis (PrimeKG-T vs static PrimeKG)   │
│    - Evidence-recency drug recommendation                             │
│    - Clinician evaluation pack for Gottingen                          │
└──────────────────────────────────────────────────────────────────────┘
```

## Data Flow Per Disease

```
OMIM:310200 (DMD)
    │
    ├─→ config/diseases/OMIM_310200.yaml          ← Agent 1 output
    │
    ├─→ data/extracted/OMIM_310200/
    │       ├── evidence_collection.json            ← Agent 2 output
    │       ├── raw_triples.jsonl                   ← Agent 3 output (all models)
    │       ├── consensus_triples.jsonl             ← Agent 3 output (≥2/3 agree)
    │       ├── validated_triples.jsonl             ← Agent 4 output (TemporalEdges)
    │       ├── rejected_triples.json               ← Agent 4 rejects
    │       ├── quality_report.json                 ← Agent 4 quality metrics
    │       └── extraction_stats.json               ← Agent 3 model agreement
    │
    └─→ Neo4j (batch, after all diseases)           ← Step 8
```

## Phased Execution

```
PHASE 0 (Done): Foundation
  ✅ PrimeKG profiled (17,080 diseases, 7.3M edges)
  ✅ Disease Profiler Agent (tested on DMD, MG, CIDP)
  ✅ Credibility Scorer (six-signal system)
  ✅ Universal schema (TemporalEdge, DiseaseProfile, etc.)
  ✅ Entity Normalizer (3-stage: BioLORD + LLM + PrimeKG, 92.9% resolution)

PHASE 1 (Current): Proof of Concept
  🔄 Run full pipeline on DMD (Agent 1→2→3→4)
  ⬜ Run on MG/LEMS, CIDP/GBS
  ⬜ Compare agent output vs Paper 1 hand-curated triples (P/R/F1)
  ⬜ Tune prompts and thresholds
  ⬜ Quality baseline document

PHASE 2: Neuromuscular Expansion
  ⬜ Disease Profiler on all neuromuscular diseases (~50-80)
  ⬜ Batch pipeline execution
  ⬜ PMC full-text extraction
  ⬜ PrimeKG-TQA prototype (200-500 QA pairs)
  ⬜ PrimeKG schema alignment verification

PHASE 3: Rare Disease Scale-Up
  ⬜ All GeneReviews diseases (~800+)
  ⬜ Self-evolution (prompt refinement between batches)
  ⬜ PrimeKG-TQA at scale (5K-10K QA pairs)
  ⬜ Downstream clinical demos
  ⬜ Quality audit (50 diseases, manual review)

PHASE 4: Paper + Submission
  ⬜ Neo4j KG construction (Step 8)
  ⬜ Resource export (Step 9)
  ⬜ Benchmark release (Step 10)
  ⬜ Paper writing (NeurIPS D&B format)
  ⬜ Clinician evaluation (Gottingen)
```

## File Map

```
primekg-t/
├── agents/
│   ├── base_agent.py              # Abstract base with retry + metrics
│   ├── disease_profiler.py        # Agent 1: Autonomous disease profiling
│   ├── evidence_harvester.py      # Agent 2: PubMed + Tier 1 collection
│   ├── knowledge_extractor.py     # Agent 3: Multi-LLM extraction + consensus
│   ├── quality_controller.py      # Agent 4: Validation + conflict detection
│   └── orchestrator.py            # Pipeline controller (1→2→3→4)
├── core/
│   ├── models.py                  # TemporalEdge, DiseaseProfile, etc.
│   ├── credibility_scorer.py      # Six-signal paper credibility scoring
│   └── entity_normalizer.py       # 3-stage: BioLORD + LLM + PrimeKG
├── config/
│   ├── default.yaml               # Global pipeline config
│   └── diseases/                  # Auto-generated per-disease YAML
├── data/
│   ├── primekg/                   # PrimeKG profiling outputs (+ kg.csv)
│   ├── extracted/                 # Per-disease extraction outputs
│   ├── primekg_t/                 # Final PrimeKG-T resource (Phase 4)
│   └── benchmark/                 # PrimeKG-TQA files (Phase 3)
├── evaluation/                    # Downstream tasks (Phase 3-4)
├── scripts/
│   ├── profile_primekg.py         # PrimeKG analysis
│   └── test_profiler.py           # Disease Profiler validation
├── docs/
│   ├── literature_survey_march2026.md  # This survey
│   └── implementation_flow.md          # This document
├── tests/
├── .env                           # API keys
├── .env.example
├── requirements.txt
└── .claude/skills/primekg-t-paper2/   # Skill + 5 reference docs
```
