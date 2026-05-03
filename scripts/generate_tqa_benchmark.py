#!/usr/bin/env python3
"""
ChronoMedKG-TQA Benchmark Generator

Generates temporal QA pairs from validated ChronoMedKG triples.
Five question types:
  1. Temporal Fact Retrieval (easy)
  2. Evidence Evolution (medium)
  3. Temporal Differential Diagnosis (hard)
  4. Contradiction Detection (hard)
  5. Multi-Hop Temporal Reasoning (very hard)

Usage:
    python scripts/generate_tqa_benchmark.py [--output data/benchmark/primekg_tqa.json]
"""
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TQAQuestion:
    """A single temporal QA question."""
    question_id: str
    question: str
    answer: str
    task_type: str  # temporal_fact, evidence_evolution, temporal_diff_dx, contradiction, multi_hop
    difficulty: str  # easy, medium, hard, very_hard
    disease_id: str
    disease_name: str
    evidence_edges: list[dict] = field(default_factory=list)  # supporting triples
    temporal_constraint: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class TQABenchmarkGenerator:
    """Generate temporal QA pairs from ChronoMedKG validated triples."""

    def __init__(self):
        self.data_dir = PROJECT_ROOT / "data" / "extracted"
        self.diseases = {}  # disease_id -> list of validated triples
        self.questions = []
        self._load_all_diseases()

    def _load_all_diseases(self):
        """Load validated triples for all diseases."""
        # Disease name lookup from config yamls
        config_dir = PROJECT_ROOT / "config" / "diseases"
        name_map = {}
        if config_dir.exists():
            import yaml
            for cfg in config_dir.glob("*.yaml"):
                try:
                    with open(cfg) as f:
                        data = yaml.safe_load(f)
                    if data and "disease_id" in data and "disease_name" in data:
                        name_map[data["disease_id"]] = data["disease_name"]
                except Exception:
                    pass

        for d in sorted(self.data_dir.iterdir()):
            if not d.is_dir():
                continue
            validated = d / "validated_triples.jsonl"
            if not validated.exists():
                continue

            disease_id = d.name.replace("_", ":")
            triples = []
            for line in open(validated):
                if line.strip():
                    triples.append(json.loads(line))

            if triples:
                disease_name = name_map.get(disease_id, d.name)
                self.diseases[disease_id] = {
                    "name": disease_name,
                    "triples": triples,
                }

        print(f"Loaded {len(self.diseases)} diseases with "
              f"{sum(len(d['triples']) for d in self.diseases.values())} total triples")

    def generate_all(self) -> list[TQAQuestion]:
        """Generate all question types."""
        self.questions = []

        for disease_id, disease_data in self.diseases.items():
            triples = disease_data["triples"]
            name = disease_data["name"]

            # Type 1: Temporal Fact Retrieval
            self._generate_temporal_facts(disease_id, name, triples)

            # Type 2: Evidence Evolution
            self._generate_evidence_evolution(disease_id, name, triples)

            # Type 3: Age-based queries
            self._generate_age_queries(disease_id, name, triples)

        # Type 3b: Differential Diagnosis (cross-disease)
        self._generate_differential_diagnosis()

        # Type 5: Multi-hop
        self._generate_multi_hop()

        # Assign IDs
        random.shuffle(self.questions)
        for i, q in enumerate(self.questions):
            q.question_id = f"TQA-{i+1:04d}"

        print(f"\nGenerated {len(self.questions)} questions:")
        type_counts = defaultdict(int)
        diff_counts = defaultdict(int)
        for q in self.questions:
            type_counts[q.task_type] += 1
            diff_counts[q.difficulty] += 1
        for t, c in sorted(type_counts.items()):
            print(f"  {t}: {c}")
        for d, c in sorted(diff_counts.items()):
            print(f"  [{d}]: {c}")

        return self.questions

    def _generate_temporal_facts(self, disease_id: str, name: str, triples: list):
        """Type 1: Questions about when facts were established."""

        # Treatment discovery questions
        treatment_triples = [t for t in triples
                           if t.get("relation") in ("treats", "indication")]
        for t in treatment_triples:
            temporal = t.get("temporal", {})
            disc_date = temporal.get("discovery_date")
            qualifier = temporal.get("temporal_qualifier", "")
            drug = t.get("source_name", t.get("source", ""))

            if disc_date:
                year = disc_date[:4]
                self.questions.append(TQAQuestion(
                    question_id="",
                    question=f"When was {drug} first reported as a treatment for {name}?",
                    answer=f"{drug} was first reported for {name} in {year}.",
                    task_type="temporal_fact",
                    difficulty="easy",
                    disease_id=disease_id,
                    disease_name=name,
                    evidence_edges=[t],
                    temporal_constraint={"year": year},
                ))

            if qualifier and "FDA" in qualifier:
                self.questions.append(TQAQuestion(
                    question_id="",
                    question=f"What is the regulatory status of {drug} for {name}?",
                    answer=f"{drug}: {qualifier}",
                    task_type="temporal_fact",
                    difficulty="easy",
                    disease_id=disease_id,
                    disease_name=name,
                    evidence_edges=[t],
                    temporal_constraint={"regulatory": True},
                ))

        # Milestone questions
        milestone_triples = [t for t in triples if t.get("temporal", {}).get("milestone")]
        for t in milestone_triples:
            temporal = t.get("temporal", {})
            milestone = temporal["milestone"]
            onset_min = temporal.get("onset_age_min")
            onset_max = temporal.get("onset_age_max")
            subject = t.get("source_name", t.get("source", ""))
            obj = t.get("target_name", t.get("target", ""))

            if onset_min is not None and onset_max is not None:
                self.questions.append(TQAQuestion(
                    question_id="",
                    question=f"At what age does '{milestone}' typically occur in {name}?",
                    answer=f"'{milestone}' typically occurs between age {onset_min:.0f} and {onset_max:.0f} in {name}.",
                    task_type="temporal_fact",
                    difficulty="easy",
                    disease_id=disease_id,
                    disease_name=name,
                    evidence_edges=[t],
                    temporal_constraint={"onset_age_min": onset_min, "onset_age_max": onset_max},
                ))
            elif onset_min is not None:
                self.questions.append(TQAQuestion(
                    question_id="",
                    question=f"What is the typical age for '{milestone}' in {name}?",
                    answer=f"'{milestone}' typically occurs around age {onset_min:.0f} in {name}.",
                    task_type="temporal_fact",
                    difficulty="easy",
                    disease_id=disease_id,
                    disease_name=name,
                    evidence_edges=[t],
                    temporal_constraint={"onset_age_min": onset_min},
                ))

        # Duration questions
        duration_triples = [t for t in triples if t.get("temporal", {}).get("duration")]
        for t in duration_triples[:5]:  # Limit per disease
            temporal = t.get("temporal", {})
            duration = temporal["duration"]
            subject = t.get("source_name", t.get("source", ""))
            relation = t.get("relation", "")
            obj = t.get("target_name", t.get("target", ""))

            self.questions.append(TQAQuestion(
                question_id="",
                question=f"How long does the relationship between {subject} and {obj} typically last in {name}?",
                answer=f"Duration: {duration}.",
                task_type="temporal_fact",
                difficulty="easy",
                disease_id=disease_id,
                disease_name=name,
                evidence_edges=[t],
                temporal_constraint={"duration": duration},
            ))

    def _generate_evidence_evolution(self, disease_id: str, name: str, triples: list):
        """Type 2: Questions about how evidence changed over time."""

        # Group triples by (subject, relation, object) to find evolving evidence
        grouped = defaultdict(list)
        for t in triples:
            key = (
                t.get("source_name", t.get("source", "")).lower(),
                t.get("relation", ""),
                t.get("target_name", t.get("target", "")).lower(),
            )
            grouped[key].append(t)

        # Find groups with multiple triples (evidence from different sources/times)
        for key, group in grouped.items():
            if len(group) < 2:
                continue

            # Sort by discovery date
            dated = [(t, t.get("temporal", {}).get("discovery_date", "")) for t in group]
            dated = [(t, d) for t, d in dated if d]
            if len(dated) < 2:
                continue

            dated.sort(key=lambda x: x[1])
            earliest = dated[0]
            latest = dated[-1]

            subject, relation, obj = key
            self.questions.append(TQAQuestion(
                question_id="",
                question=f"How has the evidence for the relationship between {subject} and {obj} ({relation}) in {name} evolved over time?",
                answer=f"Earliest evidence from {earliest[1][:4]}, most recent from {latest[1][:4]}. {len(dated)} evidence sources spanning {int(latest[1][:4]) - int(earliest[1][:4])} years.",
                task_type="evidence_evolution",
                difficulty="medium",
                disease_id=disease_id,
                disease_name=name,
                evidence_edges=group,
                temporal_constraint={"earliest": earliest[1], "latest": latest[1]},
            ))

        # Treatment landscape evolution
        treatments = [t for t in triples if t.get("relation") in ("treats", "indication")]
        if len(treatments) >= 3:
            dated_tx = [(t, t.get("temporal", {}).get("discovery_date", "")) for t in treatments]
            dated_tx = [(t, d) for t, d in dated_tx if d]
            if len(dated_tx) >= 3:
                dated_tx.sort(key=lambda x: x[1])
                tx_names = [t.get("source_name", t.get("source", "")) for t, d in dated_tx[:5]]
                years = [d[:4] for _, d in dated_tx[:5]]

                self.questions.append(TQAQuestion(
                    question_id="",
                    question=f"How has the treatment landscape for {name} evolved? List treatments in chronological order of evidence.",
                    answer=f"Treatment evolution: " + ", ".join(f"{n} ({y})" for n, y in zip(tx_names, years)),
                    task_type="evidence_evolution",
                    difficulty="medium",
                    disease_id=disease_id,
                    disease_name=name,
                    evidence_edges=[t for t, _ in dated_tx[:5]],
                    temporal_constraint={"chronological": True},
                ))

    def _generate_age_queries(self, disease_id: str, name: str, triples: list):
        """Type 3: Age-conditional queries."""

        # Find triples with onset age or progression stage
        age_triples = [t for t in triples
                      if t.get("temporal", {}).get("onset_age_min") is not None
                      or t.get("temporal", {}).get("progression_stage")]

        if len(age_triples) < 2:
            return

        # Group by progression stage
        by_stage = defaultdict(list)
        for t in age_triples:
            stage = t.get("temporal", {}).get("progression_stage", "unknown")
            if isinstance(stage, list):
                stage = stage[0] if stage else "unknown"
            by_stage[stage].append(t)

        for stage, group in by_stage.items():
            if stage == "unknown" or len(group) < 2:
                continue

            # Collect unique features (excluding the disease name itself)
            raw_features = [t.get("target_name", t.get("target", "")) for t in group]
            features = list(dict.fromkeys(f for f in raw_features if f.lower() != name.lower()))[:5]
            if not features:
                # Try source_name if target was the disease
                features = list(dict.fromkeys(
                    t.get("source_name", t.get("source", "")) for t in group
                    if t.get("source_name", t.get("source", "")).lower() != name.lower()
                ))[:5]
            if not features:
                continue
            self.questions.append(TQAQuestion(
                question_id="",
                question=f"What features are associated with the '{stage}' stage of {name}?",
                answer=f"During the '{stage}' stage: " + ", ".join(features),
                task_type="temporal_fact",
                difficulty="medium",
                disease_id=disease_id,
                disease_name=name,
                evidence_edges=group[:3],
                temporal_constraint={"progression_stage": stage},
            ))

    def _generate_differential_diagnosis(self):
        """Type 3b: Cross-disease temporal differential diagnosis."""

        # Define disease pairs for comparison
        pairs = [
            ("OMIM:310200", "OMIM:300376"),  # DMD vs BMD
            ("OMIM:254200", "ORPHA:43393"),   # MG vs LEMS
            ("ORPHA:93930", "OMIM:139393"),   # CIDP vs GBS
        ]

        for id_a, id_b in pairs:
            if id_a not in self.diseases or id_b not in self.diseases:
                continue

            name_a = self.diseases[id_a]["name"]
            name_b = self.diseases[id_b]["name"]
            triples_a = self.diseases[id_a]["triples"]
            triples_b = self.diseases[id_b]["triples"]

            # Compare onset ages
            ages_a = [(t.get("temporal", {}).get("onset_age_min"),
                       t.get("temporal", {}).get("onset_age_max"),
                       t.get("target_name", t.get("target", "")))
                      for t in triples_a
                      if t.get("temporal", {}).get("onset_age_min") is not None]

            ages_b = [(t.get("temporal", {}).get("onset_age_min"),
                       t.get("temporal", {}).get("onset_age_max"),
                       t.get("target_name", t.get("target", "")))
                      for t in triples_b
                      if t.get("temporal", {}).get("onset_age_min") is not None]

            if ages_a and ages_b:
                avg_a = sum(a[0] for a in ages_a) / len(ages_a)
                avg_b = sum(a[0] for a in ages_b) / len(ages_b)

                self.questions.append(TQAQuestion(
                    question_id="",
                    question=f"How do the temporal onset patterns of {name_a} differ from {name_b}?",
                    answer=f"{name_a} has average onset age ~{avg_a:.0f} years, "
                           f"while {name_b} has average onset age ~{avg_b:.0f} years.",
                    task_type="temporal_diff_dx",
                    difficulty="hard",
                    disease_id=f"{id_a}|{id_b}",
                    disease_name=f"{name_a} vs {name_b}",
                    temporal_constraint={"comparison": True},
                ))

            # Compare progression stages
            stages_a = set(t.get("temporal", {}).get("progression_stage", "")
                          for t in triples_a if t.get("temporal", {}).get("progression_stage"))
            stages_b = set(t.get("temporal", {}).get("progression_stage", "")
                          for t in triples_b if t.get("temporal", {}).get("progression_stage"))

            if stages_a and stages_b:
                self.questions.append(TQAQuestion(
                    question_id="",
                    question=f"What progression stages are documented for {name_a} versus {name_b}?",
                    answer=f"{name_a} stages: {', '.join(sorted(stages_a))}. "
                           f"{name_b} stages: {', '.join(sorted(stages_b))}.",
                    task_type="temporal_diff_dx",
                    difficulty="hard",
                    disease_id=f"{id_a}|{id_b}",
                    disease_name=f"{name_a} vs {name_b}",
                    temporal_constraint={"stages": True},
                ))

            # Compare treatment timelines
            tx_a = [t for t in triples_a if t.get("relation") in ("treats", "indication")]
            tx_b = [t for t in triples_b if t.get("relation") in ("treats", "indication")]

            if tx_a and tx_b:
                drugs_a = list(set(t.get("source_name", t.get("source", "")) for t in tx_a))[:3]
                drugs_b = list(set(t.get("source_name", t.get("source", "")) for t in tx_b))[:3]

                self.questions.append(TQAQuestion(
                    question_id="",
                    question=f"Compare the treatment options for {name_a} and {name_b}. How do they differ?",
                    answer=f"{name_a} treatments: {', '.join(drugs_a)}. "
                           f"{name_b} treatments: {', '.join(drugs_b)}.",
                    task_type="temporal_diff_dx",
                    difficulty="hard",
                    disease_id=f"{id_a}|{id_b}",
                    disease_name=f"{name_a} vs {name_b}",
                    temporal_constraint={"treatment_comparison": True},
                ))

    def _generate_multi_hop(self):
        """Type 5: Multi-hop temporal reasoning."""

        for disease_id, disease_data in self.diseases.items():
            triples = disease_data["triples"]
            name = disease_data["name"]

            # Gene → disease → treatment chain with temporal data
            gene_triples = [t for t in triples
                           if t.get("relation") in ("disease_protein", "caused_by")]
            treatment_triples = [t for t in triples
                               if t.get("relation") in ("treats", "indication")]

            if gene_triples and treatment_triples:
                gene = gene_triples[0].get("source_name", gene_triples[0].get("source", ""))
                gene_date = gene_triples[0].get("temporal", {}).get("discovery_date", "")
                drug = treatment_triples[0].get("source_name", treatment_triples[0].get("source", ""))
                drug_date = treatment_triples[0].get("temporal", {}).get("discovery_date", "")

                if gene_date and drug_date:
                    gap = int(drug_date[:4]) - int(gene_date[:4])
                    if gap > 0:
                        self.questions.append(TQAQuestion(
                            question_id="",
                            question=f"How many years passed between the identification of {gene}'s role in {name} and the development of {drug} as a treatment?",
                            answer=f"{gene} was linked to {name} in {gene_date[:4]}. {drug} was reported as treatment in {drug_date[:4]}. Bench-to-bedside time: {gap} years.",
                            task_type="multi_hop",
                            difficulty="very_hard",
                            disease_id=disease_id,
                            disease_name=name,
                            evidence_edges=[gene_triples[0], treatment_triples[0]],
                            temporal_constraint={"gene_year": gene_date[:4], "drug_year": drug_date[:4]},
                        ))

    def save(self, output_path: Path):
        """Save benchmark to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        benchmark = {
            "name": "ChronoMedKG-TQA",
            "version": "0.1.0",
            "description": "Temporal QA Benchmark for Biomedical Knowledge Graphs",
            "statistics": {
                "total_questions": len(self.questions),
                "diseases": len(self.diseases),
                "by_type": dict(defaultdict(int,
                    **{q.task_type: sum(1 for qq in self.questions if qq.task_type == q.task_type)
                       for q in self.questions})),
                "by_difficulty": dict(defaultdict(int,
                    **{q.difficulty: sum(1 for qq in self.questions if qq.difficulty == q.difficulty)
                       for q in self.questions})),
            },
            "questions": [asdict(q) for q in self.questions],
        }

        with open(output_path, "w") as f:
            json.dump(benchmark, f, indent=2, default=str)

        print(f"\nSaved benchmark to {output_path}")
        print(f"  Questions: {len(self.questions)}")
        print(f"  Diseases: {len(self.diseases)}")


def main():
    parser = argparse.ArgumentParser(description="Generate ChronoMedKG-TQA Benchmark")
    parser.add_argument("--output", type=str,
                        default="data/benchmark/temporal_atlas_tqa.json",
                        help="Output path for benchmark JSON")
    args = parser.parse_args()

    generator = TQABenchmarkGenerator()
    generator.generate_all()
    generator.save(PROJECT_ROOT / args.output)


if __name__ == "__main__":
    main()
