"""
Agent 3: Knowledge Extractor
==============================
Multi-LLM triple extraction with consensus.
Takes EvidenceCollection + DiseaseProfile, outputs consensus triples.

Key differences from Paper 1 (3_llm_extraction.py):
- Universal prompt template parameterized by DiseaseProfile (no per-disease code)
- PrimeKG-aligned entity/relation types
- Temporal + conditional context extraction
- Multi-LLM consensus across 3 model families
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from agents.base_agent import BaseAgent
from core.models import (
    AgentResult,
    DiseaseProfile,
    EvidenceTier,
    ExtractionResult,
    RawTriple,
    SourceDocument,
    StudyType,
)
from core.entity_normalizer import EntityNormalizer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Temporal-first extraction prompt — the core novelty of ChronoMedKG
# Every triple MUST carry temporal metadata. Static facts without temporal
# grounding are less valuable than temporally-anchored facts.
EXTRACTION_PROMPT = """You are a temporal biomedical knowledge extraction system. Your PRIMARY task is to extract relationships WITH TEMPORAL GROUNDING from the text about {disease_name}.

CRITICAL: Every relationship you extract MUST include temporal information when available. We are building a TEMPORAL knowledge graph where WHEN something is true matters as much as WHAT is true.

## What to extract (priority order)
1. TEMPORAL FACTS (highest priority): onset ages, disease milestones, progression timelines, treatment timing, when findings were discovered/published
2. EVIDENCE-DATED FACTS: relationships where the publication year or study date anchors when the evidence was established
3. CONDITIONAL FACTS: relationships that depend on age group, disease stage, genetic subtype
4. STATIC FACTS (lowest priority): general relationships without temporal context

## Output format
Return a JSON array. Each object MUST have:
- subject: entity name
- subject_type: one of [disease, gene/protein, drug, phenotype, symptom, anatomy, pathway, biological_process, exposure]
  (Use "symptom" for acute/presenting findings, e.g. "patient presents with"; use "phenotype" for chronic disease characteristics or HPO terms)
- relation: one of [disease_protein, indication, contraindication, disease_phenotype_positive, disease_phenotype_negative, drug_protein, drug_effect, protein_protein, disease_disease, bioprocess_protein, pathway_protein, treats, manifests_as, caused_by, biomarker_for, progresses_to, differentiates, onset_at, differential_diagnosis, first_line_treatment, second_line_treatment, risk_factor_for, complication_of]
- object: entity name
- object_type: same types as subject_type
- confidence: high, medium, or low
- evidence_text: exact supporting quote from the source (max 200 chars)
- temporal_context: REQUIRED object with fields:
    - onset_age_min: number in years (e.g., 3.0) or null — ALWAYS convert qualitative ages to numbers: "infancy"→0.5, "early childhood"→2, "childhood"→5, "adolescence"→13, "young adult"→20, "middle age"→45, "elderly"→65, "neonatal"→0, "first decade"→1, "second decade"→10, "third decade"→20, "fourth decade"→30, "teens"→13
    - onset_age_max: number in years (e.g., 12.0) or null — convert "childhood"→12, "adolescence"→18, "young adult"→35, "middle age"→60, "elderly"→85, "first decade"→10, "second decade"→20, "third decade"→30, "fourth decade"→40, "teens"→19
    - progression_stage: string (e.g., "ambulatory", "non-ambulatory", "early", "late") or null
    - duration: string (e.g., "2-3 years", "lifelong", "acute") or null
    - discovery_year: integer year when this fact was established or null
    - milestone: string describing the clinical milestone (e.g., "loss of ambulation", "ventilation required") or null
    - treatment_start_age: number in years or null — ALWAYS convert to numeric years
    - temporal_qualifier: string (e.g., "before age 10", "by late teens", "within 6 months") or null

IMPORTANT: ALWAYS provide numeric onset_age_min/max values when ANY age-related information is mentioned in the text. Convert qualitative descriptors to numeric ranges. If the text says "childhood onset", set onset_age_min: 2, onset_age_max: 12. If "elderly", set onset_age_min: 65, onset_age_max: 85. Do NOT leave these null when age information is present in any form.
- conditions: object with optional fields: age_group (pediatric/adult/elderly), genetic_subtype, disease_stage, treatment_line, sex

## Examples of temporal extraction for {disease_name}
{temporal_examples}

## Disease context
{disease_context}

## Known genes: {known_genes}
## Known phenotypes: {known_phenotypes}

## Source text ({source_section})
{text}

Extract ALL temporally-grounded relationships. Return a JSON object with key "triples" containing the array: {{"triples": [...]}}"""


# Second-pass prompt: temporal-only re-extraction for missed temporal facts
TEMPORAL_REEXTRACTION_PROMPT = """You are a specialized temporal biomedical extraction system performing a SECOND PASS focused EXCLUSIVELY on temporal facts about {disease_name}. The first extraction pass already captured general relationships. Your job is to find TEMPORAL INFORMATION THAT WAS MISSED.

## YOUR SOLE FOCUS: Temporal facts
Extract ONLY relationships that carry temporal grounding. Skip any fact that lacks age, timing, duration, stage, or progression information.

## What to extract (all require temporal grounding)
1. ONSET AGES: When symptoms, signs, or complications first appear
2. DISEASE STAGES: Named stages with age ranges and defining features
3. STAGE TRANSITIONS: When patients move from one stage to another
4. PROGRESSION TIMELINES: How fast disease worsens, stabilizes, or remits
5. TREATMENT WINDOWS: When interventions should start, how long they remain effective
6. MILESTONE AGES: Loss of function, need for devices, ventilation, mortality
7. STAGE-SPECIFIC PHENOTYPES: Which features are present, absent, or emerging at each stage

## MANDATORY: Convert ALL qualitative ages to numeric ranges (years)

| Term | onset_age_min | onset_age_max |
|------|--------------|--------------|
| prenatal / in utero | -0.5 | 0 |
| neonatal / at birth / congenital | 0 | 0.08 |
| first weeks of life | 0.02 | 0.12 |
| first months of life | 0.08 | 0.5 |
| infancy / infant | 0 | 1 |
| early childhood | 2 | 5 |
| childhood | 2 | 12 |
| late childhood | 8 | 12 |
| school age | 5 | 12 |
| teens / teenage | 13 | 19 |
| adolescence | 13 | 18 |
| juvenile | 5 | 16 |
| young adult | 18 | 35 |
| middle age | 40 | 60 |
| elderly / geriatric | 65 | 85 |
| first decade | 0 | 10 |
| second decade | 10 | 20 |
| third decade | 20 | 30 |
| fourth decade | 30 | 40 |

For "by age X" → onset_age_max = X. For "after age X" → onset_age_min = X.
For "around age X" → onset_age_min = X-2, onset_age_max = X+2.

## STAGE TRANSITION extraction
relation: "progresses_to", temporal_context MUST include onset_age_min/max, progression_stage, prior_stage, milestone.

## Output format
Return a JSON object with key "triples". Each object MUST have:
- subject, subject_type, relation, object, object_type, confidence, evidence_text
- temporal_context: REQUIRED — at least TWO fields must be non-null:
    onset_age_min, onset_age_max, progression_stage, prior_stage, duration,
    discovery_year, milestone, treatment_start_age, temporal_qualifier
- conditions: optional object with age_group, genetic_subtype, disease_stage, sex

RULE: If you cannot fill at least TWO temporal_context fields, DO NOT extract that triple.

## Disease context
{disease_context}
## Known genes: {known_genes}
## Known phenotypes: {known_phenotypes}

## Source text ({source_section})
{text}

## Previously extracted triples (DO NOT re-extract these)
{existing_triples_summary}

Extract ALL temporal facts not already captured. Return: {{"triples": [...]}}"""


# GeneReviews-specific prompt for structured temporal data
GENREVIEWS_TEMPORAL_PROMPT = """You are a specialized parser for GeneReviews content about {disease_name}. GeneReviews articles contain highly structured temporal and clinical staging information. Extract EVERY temporal fact.

## GeneReviews sections to parse
1. **Clinical Characteristics**: stage-by-stage phenotype descriptions
2. **Natural History**: disease progression timeline, milestone ages, survival data
3. **Genotype-Phenotype Correlations**: mutation-specific onset ages and severity
4. **Management / Surveillance**: recommended treatment start ages, screening intervals

## Extraction priorities

### A. DISEASE STAGE DEFINITIONS
Extract each stage: relation: "onset_at", temporal_context with onset_age_min/max and progression_stage.

### B. STAGE-SPECIFIC PHENOTYPES
For each stage, extract EVERY phenotype with its stage assignment:
- relation: "disease_phenotype_positive" or "disease_phenotype_negative"
- temporal_context.progression_stage and conditions.disease_stage MUST be filled

### C. STAGE TRANSITIONS
relation: "progresses_to" with prior_stage, progression_stage, onset_age_min/max, milestone.

### D. SURVIVAL AND MORTALITY
relation: "onset_at" with object: cause of death, temporal_context with age range.

### E. GENOTYPE-SPECIFIC TIMELINES
Separate triples per mutation/genotype with conditions.genetic_subtype.

### F. TREATMENT TIMING
treatment_start_age, treatment_end_age, temporal_qualifier for timing recommendations.

## MANDATORY: Numeric age conversion table
| Term | onset_age_min | onset_age_max |
|------|--------------|--------------|
| prenatal | -0.5 | 0 |
| neonatal / at birth / congenital | 0 | 0.08 |
| infancy | 0 | 1 |
| early childhood | 2 | 5 |
| childhood | 2 | 12 |
| adolescence | 13 | 18 |
| young adult | 18 | 35 |
| middle age | 40 | 60 |
| elderly | 65 | 85 |
| first decade | 0 | 10 |
| second decade | 10 | 20 |
| third decade | 20 | 30 |

For "by age X" → onset_age_max = X. For "after age X" → onset_age_min = X.
For "mean age X (range Y-Z)" → onset_age_min = Y, onset_age_max = Z.

## Output format
Return a JSON object with key "triples". Each object:
- subject, subject_type, relation, object, object_type
- confidence: high (GeneReviews is curated — most facts should be high)
- evidence_text: exact quote (max 200 chars)
- temporal_context: REQUIRED with onset_age_min, onset_age_max, progression_stage, etc.
- conditions: optional with age_group, genetic_subtype, disease_stage, sex

## Special GeneReviews rules
1. Parse tables: each stage-feature cell = separate triple
2. Frequency terms: "common"→high, "occasional"→medium, "rare"→low confidence
3. "may develop"→medium confidence, "invariably"→high confidence
4. Surveillance = implicit treatment_start_age
5. Classic vs non-classic forms = separate triple sets with conditions.genetic_subtype

## Disease context
{disease_context}
## Known genes: {known_genes}
## Known phenotypes: {known_phenotypes}

## GeneReviews text for {disease_name}
{text}

Extract EVERY temporal fact exhaustively. Return: {{"triples": [...]}}"""


class LLMClient:
    """Unified interface for multi-LLM extraction."""

    def __init__(self):
        self._clients = {}
        self._init_clients()

    def _init_clients(self):
        """Initialize available LLM clients.

        Model lineup (cascading consensus):
          Primary pair: DeepSeek V3.2 + GPT-4.1-nano (run in parallel)
          Tiebreaker:   Claude 3 Haiku (only when primary pair disagrees)
          Fallback:     GPT-4.1-nano replaces any failed primary call
        """
        import openai as openai_mod

        # OpenAI — GPT-4.1-nano (primary, cheapest quality model)
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if openai_key:
            try:
                self._clients["gpt-4.1-nano"] = openai_mod.OpenAI(api_key=openai_key, timeout=120)
                # Keep gpt-4o-mini available as legacy/fallback
                self._clients["gpt-4o-mini"] = self._clients["gpt-4.1-nano"]
            except Exception:
                logger.warning("Failed to init OpenAI client")

        # Anthropic (Claude 3 Haiku — tiebreaker only, not called on every doc)
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if anthropic_key:
            try:
                import anthropic
                self._clients["claude-haiku"] = anthropic.Anthropic(api_key=anthropic_key, timeout=120)
            except ImportError:
                logger.warning("anthropic package not installed")

        # Gemini REMOVED — 42% of cost, only 14% of triples

        # DeepSeek V3.2 — multi-key round-robin for higher throughput
        deepseek_keys = []
        for env_var in ["DEEPSEEK_API_KEY"] + [f"DEEPSEEK_API_KEY{i}" for i in range(2, 10)]:
            k = os.environ.get(env_var, "").strip()
            if k:
                deepseek_keys.append(k)
        if deepseek_keys:
            try:
                self._deepseek_clients = [
                    openai_mod.OpenAI(
                        api_key=k,
                        base_url="https://api.deepseek.com",
                        timeout=120,
                    )
                    for k in deepseek_keys
                ]
                self._deepseek_counter = 0
                self._clients["deepseek-v4"] = self._deepseek_clients[0]
                logger.info("DeepSeek: %d API key(s) loaded (round-robin enabled)",
                            len(deepseek_keys))
            except Exception:
                logger.warning("Failed to init DeepSeek clients")

        logger.info("LLM clients initialized: %s", list(self._clients.keys()))

    @property
    def available_models(self) -> list[str]:
        return list(self._clients.keys())

    def extract(self, model_name: str, prompt: str) -> list[dict]:
        """Run extraction with a specific model. Returns parsed triples."""
        if model_name not in self._clients:
            logger.warning("Model %s not available", model_name)
            return []

        try:
            if model_name == "gpt-4.1-nano":
                return self._extract_openai(prompt, model_id="gpt-4.1-nano")
            elif model_name == "gpt-4o-mini":
                return self._extract_openai(prompt, model_id="gpt-4o-mini")
            elif model_name == "claude-haiku":
                return self._extract_anthropic(prompt)
            elif model_name == "deepseek-v4":
                return self._extract_deepseek(prompt)
        except Exception as e:
            logger.error("Extraction failed with %s: %s", model_name, e)
            return []

    async def extract_async(self, model_name: str, prompt: str) -> list[dict]:
        """Async wrapper — runs blocking LLM call in thread pool for concurrency."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.extract, model_name, prompt)

    def _extract_openai(self, prompt: str, model_id: str = "gpt-4.1-nano") -> list[dict]:
        client_key = "gpt-4.1-nano" if model_id in ("gpt-4.1-nano", "gpt-4o-mini") else model_id
        client = self._clients.get(client_key) or self._clients.get("gpt-4.1-nano")
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content
        return self._parse_json_response(text)

    def _extract_anthropic(self, prompt: str) -> list[dict]:
        client = self._clients["claude-haiku"]
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        text = response.content[0].text
        return self._parse_json_response(text)

    # Gemini REMOVED — 42% of cost, only 14% of triples

    def _extract_deepseek(self, prompt: str) -> list[dict]:
        # Round-robin across multiple DeepSeek keys for higher throughput
        clients = getattr(self, '_deepseek_clients', [self._clients["deepseek-v3"]])
        idx = getattr(self, '_deepseek_counter', 0) % len(clients)
        self._deepseek_counter = idx + 1
        client = clients[idx]
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content
        return self._parse_json_response(text)

    def _parse_json_response(self, text: str) -> list[dict]:
        """Parse JSON response, handling various formats."""
        text = text.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                # Models wrap in {"triples": [...]} or other keys
                for key in ("triples", "results", "relationships", "data",
                            "extracted_relationships", "extracted_triples",
                            "extractions", "output", "entities"):
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]
                # If dict has exactly one key with a list value, use it
                list_vals = [(k, v) for k, v in parsed.items() if isinstance(v, list)]
                if len(list_vals) == 1:
                    return list_vals[0][1]
                # Single triple as dict — wrap it
                if "subject" in parsed and "object" in parsed:
                    return [parsed]
                return []
            return []
        except json.JSONDecodeError:
            # Try to find JSON array in the text
            import re
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            logger.warning("Could not parse LLM response as JSON: %s", text[:200] if text else "(empty)")
            return []


class KnowledgeExtractor(BaseAgent):
    """Agent 3: Multi-LLM knowledge extraction with consensus."""

    def __init__(self, config: dict, primekg_index=None):
        super().__init__(config, logger)
        self.llm = LLMClient()
        self.consensus_threshold = config.get("consensus_threshold", 2)
        self._cache_dir = PROJECT_ROOT / "data" / "extracted"

        # Entity normalizer (3-stage: Dictionary → BioLORD → LLM)
        # Pass shared PrimeKG index to avoid re-parsing kg.csv per worker
        use_embeddings = config.get("use_embeddings", False)  # Heavy — off by default for speed
        use_llm_norm = config.get("use_llm_normalization", False)
        self.normalizer = EntityNormalizer(
            use_embeddings=use_embeddings, use_llm=use_llm_norm,
            shared_primekg_index=primekg_index,
        )
        self._normalizer_initialized = False

    async def run(self, input_data: dict) -> AgentResult:
        """
        Extract triples from evidence collection.

        input_data:
            profile: DiseaseProfile dict
            documents: list of SourceDocument dicts
        """
        profile = DiseaseProfile.from_dict(input_data["profile"])
        documents = input_data.get("documents", [])

        self.logger.info("Extracting knowledge for: %s (%d documents)",
                         profile.disease_name, len(documents))

        result = ExtractionResult(disease_id=profile.disease_id)
        start_time = time.monotonic()

        # Extract from each document with each available model
        for i, doc_data in enumerate(documents):
            if isinstance(doc_data, dict):
                # Filter to only valid SourceDocument fields (cached JSON may have extras)
                valid_fields = {f.name for f in SourceDocument.__dataclass_fields__.values()}
                filtered = {k: v for k, v in doc_data.items() if k in valid_fields}
                # Reconstruct enums
                if "tier" in filtered and isinstance(filtered["tier"], int):
                    filtered["tier"] = EvidenceTier(filtered["tier"])
                if "study_type" in filtered and isinstance(filtered["study_type"], str):
                    try:
                        filtered["study_type"] = StudyType(filtered["study_type"])
                    except ValueError:
                        filtered["study_type"] = None
                # sections: if it's a list (from to_dict()), set to None since we lost the text
                if "sections" in filtered and isinstance(filtered["sections"], list):
                    filtered["sections"] = None
                if "publication_date" in filtered and isinstance(filtered["publication_date"], str):
                    from datetime import date as date_cls
                    try:
                        filtered["publication_date"] = date_cls.fromisoformat(filtered["publication_date"])
                    except (ValueError, TypeError):
                        filtered["publication_date"] = None
                doc = SourceDocument(**filtered)
            else:
                doc = doc_data
            self.logger.info("  Document %d/%d: %s (%s)",
                             i + 1, len(documents), doc.source_id, doc.source_type)

            prompt = self._build_prompt(profile, doc)

            # ── CASCADING CONSENSUS ──
            # Phase 1: Run DeepSeek V3.2 + GPT-4.1-nano in parallel (cheapest pair)
            # Phase 2: If they agree on >=50% of triples → accept (skip Claude)
            #          If they disagree → call Claude 3 Haiku as tiebreaker
            per_model_triples = {}

            async def _extract_one(model_name):
                raw = await self.llm.extract_async(model_name, prompt)
                parsed = [self._parse_triple(t, doc, model_name) for t in raw]
                return model_name, [t for t in parsed if t is not None]

            # Phase 1: Primary pair (parallel)
            primary_pair = []
            if "deepseek-v4" in self.llm.available_models:
                primary_pair.append("deepseek-v4")
            if "gpt-4.1-nano" in self.llm.available_models:
                primary_pair.append("gpt-4.1-nano")
            elif "gpt-4o-mini" in self.llm.available_models:
                primary_pair.append("gpt-4o-mini")

            tasks = [_extract_one(m) for m in primary_pair]
            model_results = await asyncio.gather(*tasks, return_exceptions=True)

            for mr in model_results:
                if isinstance(mr, Exception):
                    self.logger.error("    LLM extraction error: %s", mr)
                    continue
                model_name, parsed = mr
                per_model_triples[model_name] = parsed
                result.raw_triples.extend(parsed)
                self.logger.info("    %s: %d triples extracted", model_name, len(parsed))

            # Phase 2: Check agreement — do we need the tiebreaker?
            need_tiebreaker = True
            if len(per_model_triples) == 2:
                models = list(per_model_triples.keys())
                count_a = len(per_model_triples[models[0]])
                count_b = len(per_model_triples[models[1]])
                # If both models found triples, they have SOME signal — accept 2-model consensus
                # Only call tiebreaker if one model got 0 triples (total disagreement)
                # or if the counts are wildly different (>5x ratio suggesting extraction failure)
                if count_a > 0 and count_b > 0:
                    ratio = max(count_a, count_b) / max(1, min(count_a, count_b)) # Chỉ gọi khi một model có 0 triples hoặc tỷ lệ >5x
                    if ratio <= 5:
                        need_tiebreaker = False
                        self.logger.info("    2-model extraction: %s=%d, %s=%d (ratio %.1f) — skipping tiebreaker",
                                         models[0], count_a, models[1], count_b, ratio)

            # Phase 3: Call Claude as tiebreaker if needed
            if need_tiebreaker and "claude-haiku" in self.llm.available_models:
                self.logger.info("    Calling Claude 3 Haiku tiebreaker...")
                try:
                    claude_name, claude_parsed = await _extract_one("claude-haiku")
                    per_model_triples[claude_name] = claude_parsed
                    result.raw_triples.extend(claude_parsed)
                    self.logger.info("    %s (tiebreaker): %d triples extracted", claude_name, len(claude_parsed))
                except Exception as e:
                    self.logger.error("    Claude tiebreaker failed: %s", e)

            # Fallback: if any primary returned 0 triples, try the other OpenAI model
            for model_name in list(per_model_triples.keys()):
                if len(per_model_triples[model_name]) == 0 and model_name != "claude-haiku":
                    fallback = "gpt-4o-mini" if model_name != "gpt-4o-mini" else "gpt-4.1-nano"
                    if fallback in self.llm.available_models:
                        try:
                            fb_name, fb_parsed = await _extract_one(fallback)
                            per_model_triples[model_name] = fb_parsed
                            result.raw_triples.extend(fb_parsed)
                            self.logger.info("    %s (fallback for %s): %d triples", fallback, model_name, len(fb_parsed))
                        except Exception:
                            pass

            # Per-document consensus
            consensus = self._compute_consensus(per_model_triples)
            result.consensus_triples.extend(consensus)

        # Normalize entities in consensus triples → PrimeKG IDs
        self._normalize_triples(result.consensus_triples)

        elapsed = time.monotonic() - start_time

        # Agreement stats
        result.model_agreement_stats = {
            "total_raw": len(result.raw_triples),
            "total_consensus": len(result.consensus_triples),
            "consensus_rate": (
                len(result.consensus_triples) / max(1, len(result.raw_triples))
            ),
            "models_used": self.llm.available_models,
        }
        result.extraction_metrics = {
            "elapsed_seconds": round(elapsed, 1),
            "documents_processed": len(documents),
        }

        # Save to cache
        cache_dir = self._cache_dir / profile.disease_id.replace(":", "_")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._save_results(result, cache_dir)

        return AgentResult(
            agent_name="KnowledgeExtractor",
            disease_id=profile.disease_id,
            status="success" if result.consensus_triples else "partial",
            data={
                "raw_count": len(result.raw_triples),
                "consensus_count": len(result.consensus_triples),
            },
            metrics={**result.extraction_metrics, **result.model_agreement_stats},
            timestamp=datetime.utcnow(),
        )

    def _build_prompt(self, profile: DiseaseProfile, doc: SourceDocument) -> str:
        """Build extraction prompt parameterized by disease profile."""
        # Truncate text to fit context window
        # Full-text sources (PMC, GeneReviews, guidelines) get 30K chars;
        # abstracts get 6K chars to avoid wasting tokens on padding.
        if doc.source_type in ("pmc_fulltext", "genereviews", "guideline"):
            max_text_chars = 30000
        else:
            max_text_chars = 6000
        text = doc.text[:max_text_chars]
        if len(doc.text) > max_text_chars:
            text += "\n[... truncated ...]"

        disease_context = (
            f"Disease: {profile.disease_name}\n"
            f"Category: {profile.disease_category or 'unknown'}\n"
            f"Inheritance: {profile.inheritance_pattern or 'unknown'}\n"
            f"Differential diagnoses: {', '.join(profile.differential_diseases) or 'unknown'}"
        )

        # Determine source section label
        source_section = doc.source_type
        if doc.sections:
            source_section = f"{doc.source_type} — sections: {', '.join(doc.sections.keys())}"

        # Generate disease-specific temporal examples
        temporal_examples = self._generate_temporal_examples(profile.disease_name)

        return EXTRACTION_PROMPT.format(
            disease_name=profile.disease_name,
            disease_context=disease_context,
            known_genes=", ".join(profile.key_genes) if profile.key_genes else "not specified",
            known_phenotypes=", ".join(profile.key_phenotypes) if profile.key_phenotypes else "not specified",
            source_section=source_section,
            temporal_examples=temporal_examples,
            text=text,
        )

    def _generate_temporal_examples(self, disease_name: str) -> str:
        """
        Generate disease-specific temporal extraction examples.
        This is critical — DMD had 87% temporal coverage with DMD-specific examples,
        while MG/LEMS/CIDP/GBS had only 11-22% with those same DMD examples.
        """
        name_lower = disease_name.lower()

        # Disease-specific temporal example banks
        TEMPORAL_EXAMPLES = {
            # Neuromuscular - dystrophinopathies
            "duchenne": [
                ('Text: "DMD patients typically lose ambulation between ages 10 and 13 without treatment"',
                 'subject: "Duchenne muscular dystrophy", relation: "onset_at", object: "loss of ambulation",\n'
                 '  temporal_context: {{onset_age_min: 10, onset_age_max: 13, milestone: "loss of ambulation", temporal_qualifier: "without treatment"}}'),
                ('Text: "Deflazacort was approved for DMD in 2017"',
                 'subject: "deflazacort", relation: "treats", object: "Duchenne muscular dystrophy",\n'
                 '  temporal_context: {{discovery_year: 2017, temporal_qualifier: "FDA approved"}}'),
                ('Text: "Cardiomyopathy develops in virtually all DMD patients by late teens"',
                 'subject: "Duchenne muscular dystrophy", relation: "disease_phenotype_positive", object: "cardiomyopathy",\n'
                 '  temporal_context: {{onset_age_min: 15, onset_age_max: 19, progression_stage: "late", milestone: "cardiac involvement"}}'),
            ],
            "becker": [
                ('Text: "BMD patients typically retain ambulation beyond age 16"',
                 'subject: "Becker muscular dystrophy", relation: "disease_phenotype_positive", object: "preserved ambulation",\n'
                 '  temporal_context: {{onset_age_min: 16, milestone: "ambulation preserved", temporal_qualifier: "beyond age 16"}}'),
                ('Text: "Cardiac involvement in BMD often manifests in the third to fourth decade"',
                 'subject: "Becker muscular dystrophy", relation: "disease_phenotype_positive", object: "cardiomyopathy",\n'
                 '  temporal_context: {{onset_age_min: 20, onset_age_max: 40, milestone: "cardiac involvement", progression_stage: "adult"}}'),
            ],
            # Neuromuscular junction disorders
            "myasthenia gravis": [
                ('Text: "MG onset peaks bimodally: women in their 20s-30s and men in their 60s-70s"',
                 'subject: "Myasthenia gravis", relation: "onset_at", object: "symptom onset",\n'
                 '  temporal_context: {{onset_age_min: 20, onset_age_max: 30, temporal_qualifier: "bimodal: young women 20-30, older men 60-70"}}'),
                ('Text: "Efgartigimod was FDA-approved for generalized MG in 2021"',
                 'subject: "efgartigimod", relation: "treats", object: "Myasthenia gravis",\n'
                 '  temporal_context: {{discovery_year: 2021, temporal_qualifier: "FDA approved for generalized MG"}}'),
                ('Text: "Myasthenic crisis occurs in 15-20% of patients, typically within the first 2 years"',
                 'subject: "Myasthenia gravis", relation: "progresses_to", object: "myasthenic crisis",\n'
                 '  temporal_context: {{duration: "within first 2 years", milestone: "myasthenic crisis", temporal_qualifier: "15-20% of patients"}}'),
                ('Text: "Thymectomy is recommended early in the disease course for anti-AChR positive patients"',
                 'subject: "thymectomy", relation: "treats", object: "Myasthenia gravis",\n'
                 '  temporal_context: {{progression_stage: "early", temporal_qualifier: "early disease, anti-AChR positive", conditions: {{"disease_stage": "early"}}}}'),
            ],
            "lambert-eaton": [
                ('Text: "LEMS symptoms typically develop over weeks to months"',
                 'subject: "Lambert-Eaton myasthenic syndrome", relation: "onset_at", object: "proximal weakness",\n'
                 '  temporal_context: {{duration: "weeks to months", milestone: "symptom development", temporal_qualifier: "subacute onset"}}'),
                ('Text: "50-60% of LEMS patients have an underlying malignancy, usually diagnosed within 2 years"',
                 'subject: "Lambert-Eaton myasthenic syndrome", relation: "disease_disease", object: "small cell lung cancer",\n'
                 '  temporal_context: {{duration: "within 2 years of LEMS diagnosis", temporal_qualifier: "50-60% paraneoplastic"}}'),
                ('Text: "3,4-DAP has been the first-line treatment for LEMS since the 1990s"',
                 'subject: "3,4-diaminopyridine", relation: "treats", object: "Lambert-Eaton myasthenic syndrome",\n'
                 '  temporal_context: {{discovery_year: 1990, temporal_qualifier: "first-line since 1990s", conditions: {{"treatment_line": "first-line"}}}}'),
            ],
            # Peripheral neuropathies
            "chronic inflammatory demyelinating": [
                ('Text: "CIDP follows a chronic progressive or relapsing course over at least 8 weeks"',
                 'subject: "CIDP", relation: "onset_at", object: "progressive weakness",\n'
                 '  temporal_context: {{duration: "at least 8 weeks", milestone: "symptom progression", temporal_qualifier: "chronic progressive or relapsing"}}'),
                ('Text: "IVIg was established as first-line treatment for CIDP following the ICE trial in 2008"',
                 'subject: "IVIg", relation: "treats", object: "CIDP",\n'
                 '  temporal_context: {{discovery_year: 2008, temporal_qualifier: "ICE trial, established as first-line", conditions: {{"treatment_line": "first-line"}}}}'),
                ('Text: "Anti-NF155 antibody-positive CIDP shows younger onset, typically before age 40"',
                 'subject: "CIDP", relation: "disease_phenotype_positive", object: "anti-NF155 positive subtype",\n'
                 '  temporal_context: {{onset_age_max: 40, temporal_qualifier: "younger onset, before age 40", conditions: {{"genetic_subtype": "anti-NF155 positive"}}}}'),
                ('Text: "Up to 50% of CIDP patients become treatment-dependent within 5 years"',
                 'subject: "CIDP", relation: "progresses_to", object: "treatment dependence",\n'
                 '  temporal_context: {{duration: "within 5 years", temporal_qualifier: "50% become treatment-dependent"}}'),
            ],
            "guillain-barre": [
                ('Text: "GBS typically reaches nadir within 2-4 weeks of symptom onset"',
                 'subject: "Guillain-Barre syndrome", relation: "onset_at", object: "nadir",\n'
                 '  temporal_context: {{duration: "2-4 weeks", milestone: "nadir of weakness", temporal_qualifier: "from symptom onset"}}'),
                ('Text: "Most GBS patients recover within 6-12 months but 20% remain disabled"',
                 'subject: "Guillain-Barre syndrome", relation: "progresses_to", object: "recovery",\n'
                 '  temporal_context: {{duration: "6-12 months", milestone: "functional recovery", temporal_qualifier: "80% recover, 20% remain disabled"}}'),
                ('Text: "IVIg and plasma exchange are equally effective when started within 2 weeks of onset"',
                 'subject: "IVIg", relation: "treats", object: "Guillain-Barre syndrome",\n'
                 '  temporal_context: {{duration: "within 2 weeks of onset", temporal_qualifier: "equally effective as PLEX when started early"}}'),
                ('Text: "Preceding infection occurs 1-4 weeks before GBS onset in 70% of cases"',
                 'subject: "infection", relation: "caused_by", object: "Guillain-Barre syndrome",\n'
                 '  temporal_context: {{duration: "1-4 weeks before onset", temporal_qualifier: "preceding infection in 70%"}}'),
            ],
        }

        # Find matching examples
        examples = []
        for key, exs in TEMPORAL_EXAMPLES.items():
            if key in name_lower:
                examples = exs
                break

        # Fallback: generate generic temporal examples using disease name
        # These examples MUST show concrete numeric ages to guide LLMs
        if not examples:
            examples = [
                (f'Text: "{disease_name} onset typically occurs in childhood"',
                 f'subject: "{disease_name}", relation: "onset_at", object: "symptom onset",\n'
                 '  temporal_context: {{onset_age_min: 2, onset_age_max: 12, milestone: "symptom onset", temporal_qualifier: "childhood onset"}}'),
                (f'Text: "Late-onset {disease_name} presents in the fourth to sixth decade"',
                 f'subject: "{disease_name}", relation: "onset_at", object: "late-onset presentation",\n'
                 '  temporal_context: {{onset_age_min: 30, onset_age_max: 60, progression_stage: "late-onset", temporal_qualifier: "fourth to sixth decade"}}'),
                (f'Text: "ACE inhibitor is first-line therapy for {disease_name} per current guidelines"',
                 f'subject: "ACE inhibitor", relation: "first_line_treatment", object: "{disease_name}",\n'
                 '  temporal_context: {{discovery_year: 2018, temporal_qualifier: "current guidelines, first-line"}}'),
                (f'Text: "Drug X was approved for {disease_name} in 2015"',
                 f'subject: "Drug X", relation: "treats", object: "{disease_name}",\n'
                 '  temporal_context: {{discovery_year: 2015, temporal_qualifier: "approved"}}'),
                (f'Text: "{disease_name} must be distinguished from Disease Y in the differential"',
                 f'subject: "{disease_name}", relation: "differential_diagnosis", object: "Disease Y",\n'
                 '  temporal_context: {{temporal_qualifier: "differential diagnosis"}}'),
                (f'Text: "Hypertension is a known risk factor for {disease_name}"',
                 f'subject: "hypertension", relation: "risk_factor_for", object: "{disease_name}",\n'
                 '  temporal_context: {{temporal_qualifier: "established risk factor"}}'),
                (f'Text: "{disease_name} progresses to organ failure over 5-10 years"',
                 f'subject: "{disease_name}", relation: "complication_of", object: "organ failure",\n'
                 '  temporal_context: {{duration: "5-10 years", milestone: "organ failure"}}'),
                (f'Text: "Neonatal {disease_name} is the most severe form, presenting at birth"',
                 f'subject: "{disease_name}", relation: "onset_at", object: "neonatal presentation",\n'
                 '  temporal_context: {{onset_age_min: 0, onset_age_max: 0.08, progression_stage: "neonatal", milestone: "birth presentation"}}'),
            ]

        # Format as text
        lines = []
        for text_ex, triple_ex in examples:
            lines.append(f'{text_ex}\n→ {triple_ex}\n')
        return "\n".join(lines)

    def _parse_triple(self, raw: dict, doc: SourceDocument, model_name: str) -> RawTriple | None:
        """Parse a raw LLM output dict into a RawTriple."""
        try:
            subject = raw.get("subject", "").strip()
            obj = raw.get("object", "").strip()
            relation = raw.get("relation", "").strip()

            if not subject or not obj or not relation:
                return None

            confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.4}
            conf_str = raw.get("confidence", "medium")
            confidence = confidence_map.get(conf_str, 0.5) if isinstance(conf_str, str) else float(conf_str)

            return RawTriple(
                subject=subject,
                subject_type=raw.get("subject_type", "disease"),
                relation=relation,
                object=obj,
                object_type=raw.get("object_type", "phenotype"),
                temporal_context=raw.get("temporal_context"),
                conditions=raw.get("conditions"),
                evidence_text=raw.get("evidence_text", "")[:300],
                source_id=doc.source_id,
                extraction_model=model_name,
                confidence=confidence,
            )
        except Exception as e:
            logger.debug("Failed to parse triple: %s", e)
            return None

    def _compute_consensus(self, per_model_triples: dict[str, list[RawTriple]]) -> list[RawTriple]:
        """
        Compute multi-LLM consensus using semantic matching.

        Two triples are considered to agree if:
        1. Their relations map to the same canonical form
        2. Their subjects are semantically similar (fuzzy match >= 80)
        3. Their objects are semantically similar (fuzzy match >= 80)

        This catches cases where models phrase the same fact differently:
        - "corticosteroid" vs "corticosteroid treatment"
        - "prednisone/prednisolone" vs "prednisone"
        - "deflazacort 0.45 mg/kg/d" vs "deflazacort"
        """
        all_triples = []
        for model_name, triples in per_model_triples.items():
            all_triples.extend(triples)

        if not all_triples:
            return []

        # Stage 1: Normalize entity names and relations
        normalized = []
        for t in all_triples:
            normalized.append({
                "triple": t,
                "subj_norm": self._normalize_entity_name(t.subject),
                "rel_norm": self._canonicalize_relation(t.relation),
                "obj_norm": self._normalize_entity_name(t.object),
            })

        # Stage 2: Group by canonical relation, then cluster by semantic similarity
        from rapidfuzz import fuzz

        rel_groups: dict[str, list[dict]] = defaultdict(list)
        for n in normalized:
            rel_groups[n["rel_norm"]].append(n)

        # Union-Find for clustering similar triples
        parent = list(range(len(normalized)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Build index for O(n) lookup within relation groups
        idx_map = {id(n["triple"]): i for i, n in enumerate(normalized)}

        for rel, group in rel_groups.items():
            if len(group) < 2:
                continue

            # Compare all pairs within same relation (n^2 but n is small per-document)
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    # Skip same model (consensus requires different models)
                    if a["triple"].extraction_model == b["triple"].extraction_model:
                        continue

                    # FIX 2+3: Relaxed matching with fuzzy (70) + substring containment
                    subj_match = self._entity_match(a["subj_norm"], b["subj_norm"], fuzz)
                    obj_match = self._entity_match(a["obj_norm"], b["obj_norm"], fuzz)

                    if subj_match and obj_match:
                        union(idx_map[id(a["triple"])], idx_map[id(b["triple"])])

        # Stage 3: Collect clusters and check consensus threshold
        clusters: dict[int, list[RawTriple]] = defaultdict(list)
        for i, n in enumerate(normalized):
            clusters[find(i)].append(n["triple"])

        consensus = []
        for cluster_id, group in clusters.items():
            models_agreeing = {t.extraction_model for t in group}
            if len(models_agreeing) >= self.consensus_threshold:
                # Pick representative: highest confidence, prefer most specific entity name
                best = max(group, key=lambda t: (t.confidence, len(t.subject) + len(t.object)))
                best.confidence = len(models_agreeing) / len(per_model_triples)
                consensus.append(best)

        return consensus

    @staticmethod
    def _normalize_entity_name(name: str) -> str:
        """
        Normalize entity name for semantic matching. Only lowercase, strip whitespace, and remove dosage/parenthetical noise.

        Handles:
        - Case and whitespace
        - Dosage removal: "deflazacort 0.45 mg/kg/d" → "deflazacort"
        - Suffix stripping: "corticosteroid treatment" → "corticosteroid"
        - Slash splitting: "prednisone/prednisolone" → "prednisone prednisolone"
        - Parenthetical removal: "adeno-associated virus (AAV)" → "adeno-associated virus"
        """
        import re

        if not name:
            return ""
        if not isinstance(name, str):
            name = str(name)
        name = name.lower().strip()

        # Remove dosage patterns: "0.45 mg/kg/d", "2 mg/kg per day", "100 mg"
        name = re.sub(r'\b\d+\.?\d*\s*(mg|g|kg|ml|µg|iu|u)[\w/]*\b', '', name)
        name = re.sub(r'\bper\s+(day|week|month|dose)\b', '', name)

        # Remove parenthetical abbreviations: "(AAV)", "(DMD)", "(IVIg)"
        name = re.sub(r'\([^)]{1,10}\)', '', name)

        # Replace slashes with spaces: "prednisone/prednisolone" → "prednisone prednisolone"
        name = name.replace('/', ' ')

        # Remove common suffixes that don't change meaning
        noise_suffixes = [
            ' treatment', ' therapy', ' supplementation', ' administration',
            ' expression', ' signaling', ' pathway', ' level', ' levels',
            '-based', '-mediated',
        ]
        for suffix in noise_suffixes:
            if name.endswith(suffix) and len(name) > len(suffix) + 3:
                name = name[:-len(suffix)]

        # Collapse whitespace
        name = re.sub(r'\s+', ' ', name).strip()

        return name

    @staticmethod
    def _canonicalize_relation(relation: str) -> str:
        """
        Canonicalize relation names to handle model-specific phrasing. Only lowercase and strip whitespace; no semantic mapping here. Use _map_relation_synonyms for that.

        Maps synonymous relations to a single canonical form.
        """
        rel = relation.lower().strip()

        # Relation synonym groups — broad mapping to maximize cross-model agreement
        # FIX 1: Extended mappings for common LLM phrasings
        synonyms = {
            "treats": ["treats", "treatment_for", "used_to_treat", "therapeutic_for",
                        "indication", "managed_by", "managed_with", "therapy_for",
                        "first_line_treatment", "second_line_treatment", "standard_of_care"],
            "disease_phenotype_positive": ["disease_phenotype_positive", "manifests_as",
                        "presents_with", "associated_with_phenotype", "symptom_of",
                        "clinical_feature", "characterized_by", "sign_of",
                        "has_symptom", "phenotype_of", "clinical_manifestation"],
            "disease_phenotype_negative": ["disease_phenotype_negative", "absent_in",
                        "not_associated_with", "absent_phenotype"],
            "caused_by": ["caused_by", "etiology", "pathogenesis", "results_from",
                        "due_to", "attributed_to", "mechanism"],
            "biomarker_for": ["biomarker_for", "diagnostic_marker", "indicator_of",
                        "diagnostic_test", "detected_by", "measured_by"],
            "progresses_to": ["progresses_to", "leads_to", "evolves_into",
                        "develops_into", "complications", "sequela", "worsens_to"],
            "onset_at": ["onset_at", "onset_age", "age_of_onset", "typical_onset",
                        "age_at_diagnosis", "presents_at_age"],
            "disease_protein": ["disease_protein", "gene_disease", "associated_gene",
                        "caused_by_mutation_in", "gene_variant", "genetic_basis",
                        "mutation_in", "pathogenic_variant"],
            "drug_effect": ["drug_effect", "side_effect", "adverse_effect",
                        "adverse_reaction", "toxicity", "drug_interaction"],
            "disease_disease": ["disease_disease", "comorbidity", "differential_diagnosis",
                        "differentiates", "associated_disease", "co_occurs_with",
                        "misdiagnosed_as", "overlaps_with"],
            "contraindication": ["contraindication", "contraindicated_for",
                        "avoid_in", "not_recommended"],
            "exposure_disease": ["exposure_disease", "risk_factor", "risk_factor_for",
                        "environmental_cause", "trigger", "precipitant"],
        }

        for canonical, variants in synonyms.items():
            if rel in variants:
                return canonical

        return rel

    @staticmethod
    def _entity_match(name_a: str, name_b: str, fuzz) -> bool:
        """
        Check if two normalized entity names refer to the same entity. Only 

        Conservative matching to avoid union-find mega-clusters:
        1. Exact match after normalization
        2. Substring containment ONLY if shorter is >= 60% of longer (avoids "acid" matching everything)
        3. Fuzzy token_sort_ratio >= 80 (proven threshold from Paper 1)
        """
        if not name_a or not name_b:
            return False

        # Exact match
        if name_a == name_b:
            return True

        # Substring containment — conservative: shorter must be substantial part of longer
        shorter, longer = (name_a, name_b) if len(name_a) <= len(name_b) else (name_b, name_a)
        if len(shorter) >= 6 and shorter in longer and len(shorter) >= 0.6 * len(longer):
            return True

        # Fuzzy match — keep at 80 to prevent transitive mega-cluster merging
        score = fuzz.token_sort_ratio(name_a, name_b)
        return score >= 80

    def _normalize_triples(self, triples: list[RawTriple]) -> None:
        """Normalize entity names to PrimeKG IDs using 3-stage normalizer."""
        if not self._normalizer_initialized:
            self.normalizer.initialize()
            self._normalizer_initialized = True

        for triple in triples:
            # Normalize subject
            subj_result = self.normalizer.normalize(
                triple.subject, entity_type=triple.subject_type
            ) # Not context aware yet; could add context from disease profile if needed
            if subj_result.primekg_id:
                triple.subject_id = subj_result.primekg_id
                if subj_result.confidence >= 0.80:
                    triple.subject = subj_result.normalized_name

            # Normalize object
            obj_result = self.normalizer.normalize(
                triple.object, entity_type=triple.object_type
            ) # Not context aware yet; could add context from disease profile if needed
            if obj_result.primekg_id:
                triple.object_id = obj_result.primekg_id
                if obj_result.confidence >= 0.80:
                    triple.object = obj_result.normalized_name

        stats = self.normalizer.get_stats()
        self.logger.info("Entity normalization: %d entities, %.0f%% resolved (%s)",
                         stats["total"], stats["resolution_rate"] * 100,
                         f"dict={stats['dictionary']}, embed={stats['embedding']}, llm={stats['llm']}")

    def _save_results(self, result: ExtractionResult, cache_dir: Path) -> None:
        """Save extraction results to cache."""
        with open(cache_dir / "raw_triples.jsonl", "w") as f:
            for t in result.raw_triples:
                f.write(json.dumps(t.to_dict(), default=str) + "\n")

        with open(cache_dir / "consensus_triples.jsonl", "w") as f:
            for t in result.consensus_triples:
                f.write(json.dumps(t.to_dict(), default=str) + "\n")

        with open(cache_dir / "extraction_stats.json", "w") as f:
            json.dump({
                "model_agreement_stats": result.model_agreement_stats,
                "extraction_metrics": result.extraction_metrics,
            }, f, indent=2, default=str)
