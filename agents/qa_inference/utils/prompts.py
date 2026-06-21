"""
Prompt templates for the KG-RAG QA pipeline.

Sections:
  - TRANSLATE: language detection + translation
  - NER: biomedical entity extraction
  - ANSWER: 6 answer templates (added in TIP-009)
"""

# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

TRANSLATE_SYSTEM = (
    "You are a biomedical translation assistant. "
    "Detect the language of the user query and translate it to English if needed. "
    "Return valid JSON only."
)

TRANSLATE_USER = """\
Query: {query}

Return JSON with exactly these keys:
{{
  "language": "<ISO 639-1 code, e.g. en, vi, fr>",
  "query_en": "<the query in English — unchanged if already English>"
}}"""


# ---------------------------------------------------------------------------
# Biomedical NER
# ---------------------------------------------------------------------------

NER_SYSTEM = (
    "You are a biomedical named entity recognition system. "
    "Extract the PRIMARY SUBJECT entities that the question is asking about. "
    "Focus on: specific named diseases, specific drugs/treatments, specific genes/proteins, "
    "specific biological processes or pathways. "
    "SKIP: generic clinical terms (fever, fatigue, pain, weakness, headache, rash), "
    "patient demographics, routine lab tests (hemoglobin, glucose), "
    "non-specific procedures (MRI, biopsy) unless the question is specifically about them. "
    "Return valid JSON only."
)

NER_USER = """\
Query: {query}

Return JSON with exactly this key:
{{
  "entities": ["<entity 1>", "<entity 2>", ...]
}}

Rules:
- Extract entities this question is specifically ASKING ABOUT, not all entities mentioned
- Include full names (e.g. "Duchenne muscular dystrophy", not just "muscular dystrophy")
- Include abbreviations as separate entries if they appear in the query
- MAX 6 entities — prioritize specific named concepts over generic symptoms
- Return empty list if no specific biomedical named entities found"""


# ---------------------------------------------------------------------------
# MCQ-specific NER (MEDQA) — extracts from options + clinical clues from stem
# ---------------------------------------------------------------------------

NER_SYSTEM_MCQ = (
    "You are a biomedical named entity recognition system for multiple-choice questions. "
    "Your goal is to identify entities that enable knowledge graph retrieval to answer the question. "
    "Return valid JSON only."
)

NER_USER_MCQ = """\
Question stem: {query}

Answer options:
{options_text}

Extract TWO categories:

1. option_entities: The specific disease/drug/syndrome/mechanism names from the answer options.
   Use the full canonical name (e.g. "pheochromocytoma" not "Catecholamine-secreting mass").
   Include ALL options — these are the diagnostic candidates being compared.

2. clinical_clues: 2-3 SPECIFIC and DISTINCTIVE findings from the stem that differentiate the options.
   SKIP: generic symptoms (fever, fatigue, pain), patient age/sex, routine vitals,
         medications unrelated to the diagnostic question, non-specific lab values.
   INCLUDE: specific lab patterns (hypokalemia + metabolic alkalosis),
            specific exposures (DES in utero, travel to SE Asia),
            specific physical findings (negative tourniquet test, bronze skin discoloration),
            specific test results that point to a diagnosis.

Return JSON:
{{
  "option_entities": ["<entity from option A>", "<entity from option B>", ...],
  "clinical_clues": ["<specific clue 1>", "<specific clue 2>"]
}}

Examples:

Stem: "...travel to Vietnam/Cambodia, arthritis hands/wrists, maculopapular rash, leukopenia, thrombocytopenia, negative tourniquet test..."
Options: A=Chikungunya, B=Dengue fever, C=Epstein-Barr virus, D=Hepatitis A, E=Typhoid fever
→ {{"option_entities": ["Chikungunya", "Dengue fever", "Epstein-Barr virus", "Hepatitis A", "Typhoid fever"],
    "clinical_clues": ["arthritis hands wrists", "negative tourniquet test", "Southeast Asia travel"]}}

Stem: "...hypokalemia K+ 3.3, metabolic alkalosis HCO3 33, hypertension..."
Options: A=Aldosterone excess, B=Catecholamine-secreting mass, C=Cortisol excess, D=Impaired kidney perfusion, E=Increased peripheral vascular resistance
→ {{"option_entities": ["primary hyperaldosteronism", "pheochromocytoma", "Cushing syndrome", "renal artery stenosis"],
    "clinical_clues": ["hypokalemia metabolic alkalosis", "resistant hypertension"]}}

Stem: "...DES exposure in utero, polypoid mass anterior vaginal wall..."
Options: A=Clear cell adenocarcinoma, B=Melanoma, C=Botryoid sarcoma, D=Verrucous carcinoma, E=Squamous cell carcinoma
→ {{"option_entities": ["clear cell adenocarcinoma vagina", "vaginal melanoma", "botryoid sarcoma", "squamous cell carcinoma vagina"],
    "clinical_clues": ["diethylstilbestrol in utero exposure", "vaginal adenocarcinoma"]}}"""


# ---------------------------------------------------------------------------
# Intent classification (BioASQ only — MEDQA/PubMedQA are rule-based)
# ---------------------------------------------------------------------------

INTENT_SYSTEM = (
    "You are a biomedical question classifier. "
    "Classify the question type and identify the biomedical relationship being queried. "
    "Return valid JSON only."
)

INTENT_USER = """\
Question: {query}

Return JSON with exactly these keys:
{{
  "question_type": "<one of: yes_no | factoid | list | summary>",
  "relation_intents": ["<relationship type 1>", ...]
}}

question_type rules:
- yes_no: question can be answered with yes or no (e.g. "Is X associated with Y?")
- factoid: asks for a specific fact (e.g. "What gene causes X?", "What is the onset age of Y?")
- list: asks for multiple items (e.g. "What are the symptoms of X?", "List all drugs for Y")
- summary: asks for a description or explanation (e.g. "Describe the pathogenesis of X")

relation_intents: list the biomedical relationship types relevant to answering this question.
Examples: "gene-disease association", "drug indication", "disease phenotype",
"temporal onset", "disease progression", "drug mechanism", "biomarker", "pathway involvement"
Return 1-3 most relevant relation types."""


# ---------------------------------------------------------------------------
# Answer templates — 6 templates for BioASQ×4 + MEDQA + PubMedQA
# ---------------------------------------------------------------------------

_ANSWER_SYSTEM_BASE = (
    "You are a biomedical expert. "
    "Answer the question using the provided knowledge graph context when available. "
    "Base your answer on the context; use your knowledge only when context is insufficient. "
    "Return valid JSON only — no markdown, no extra text."
)

# Shared context block prefix
_CTX_HEADER = "Knowledge Graph Context:\n{context}\n\n"
_NO_CTX_HEADER = "No knowledge graph context available. Use your biomedical knowledge.\n\n"

ANSWER_TEMPLATES: dict[str, dict] = {

    # --- BioASQ Yes/No ---
    "yes_no": {
        "system": _ANSWER_SYSTEM_BASE,
        "user": (
            "{context_block}"
            "Question: {question}\n\n"
            'Return JSON: {{"answer": "yes" or "no", '
            '"explanation": "evidence-based explanation citing the context"}}'
        ),
    },

    # --- BioASQ Factoid ---
    "factoid": {
        "system": _ANSWER_SYSTEM_BASE,
        "user": (
            "{context_block}"
            "Question: {question}\n\n"
            'Return JSON: {{"answer": "precise factual answer (entity name, value, or short phrase)", '
            '"explanation": "brief supporting evidence from context"}}'
        ),
    },

    # --- BioASQ List ---
    "list": {
        "system": _ANSWER_SYSTEM_BASE,
        "user": (
            "{context_block}"
            "Question: {question}\n\n"
            'Return JSON: {{"answer": ["item 1", "item 2", ...], '
            '"explanation": "brief explanation of the list based on context"}}.\n'
            "Rules: be comprehensive but non-redundant; each item is a distinct entity or concept."
        ),
    },

    # --- BioASQ Summary ---
    "summary": {
        "system": _ANSWER_SYSTEM_BASE,
        "user": (
            "{context_block}"
            "Question: {question}\n\n"
            'Return JSON: {{"answer": "coherent summary paragraph", '
            '"key_points": ["point 1", "point 2", "point 3"]}}'
        ),
    },

    # --- MEDQA (USMLE-style MCQ) ---
    "mcq": {
        "system": (
            "You are a medical expert specializing in USMLE-style questions. "
            "When Knowledge Graph Context is provided, evaluate each answer option against it: "
            "if the context directly mentions an option's disease or drug with clinically relevant facts, cite it as evidence; "
            "if context entries are about unrelated entities (different disease, unrelated drug), ignore them and rely on clinical reasoning. "
            "Return valid JSON only."
        ),
        "user": (
            "{context_block}"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            'Return JSON: {{"answer": "<single letter A/B/C/D/E>", '
            '"explanation": "For the most likely option: cite supporting context if available, '
            'or give step-by-step clinical reasoning. State why other options are less likely."}}'
        ),
    },

    # --- PubMedQA (Yes/No/Maybe) ---
    "yes_no_maybe": {
        "system": (
            "You are a biomedical research expert. "
            "Answer PubMed research questions based on the provided evidence. "
            "Return valid JSON only."
        ),
        "user": (
            "{context_block}"
            "Research question: {question}\n\n"
            'Return JSON: {{"answer": "yes" or "no" or "maybe", '
            '"explanation": "evidence summary supporting the answer"}}.\n'
            'Use "maybe" when evidence is conflicting or insufficient.'
        ),
    },
}


def build_context_block(sentences: list[str]) -> str:
    """Format context sentences into the block injected into answer prompts."""
    if not sentences:
        return _NO_CTX_HEADER
    body = "\n".join(f"- {s}" for s in sentences)
    return _CTX_HEADER.format(context=body)


def format_mcq_options(options: dict) -> str:
    """Format MEDQA options dict → readable string."""
    return "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
