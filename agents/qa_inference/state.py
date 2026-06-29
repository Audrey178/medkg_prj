from typing import TypedDict, Optional, NotRequired


class MatchedNode(TypedDict):
    cui: str
    name: str
    confidence: float
    strategy: str           # "exact" | "abbreviation" | "fuzzy" | "semantic"
    source: NotRequired[str]  # "option" | "clue" | "stem" — set by entity_node


class QAState(TypedDict):
    # --- Input ---
    query_raw: str
    benchmark_type: str   # "bioasq" | "medqa" | "pubmedqa"
    mode: str             # "kg_rag" | "llm_only" | "kg_only"
    options: dict         # pass-through for per-request overrides

    # --- translate_node ---
    query_en: str
    lang_detected: str

    # --- entity_node ---
    extracted_entities: list[str]
    matched_nodes: list[MatchedNode]

    # --- intent_node ---
    question_type: str        # "yes_no" | "factoid" | "list" | "summary"
                              # | "mcq" | "yes_no_maybe"
    relation_intents: list[str]

    # --- retrieval_node ---
    raw_triples: list[dict]
    sources: list[str]        # PMIDs

    # --- answer_node ---
    answer: Optional[dict]
    kg_coverage: bool
    latency_ms: float
    tokens_used: int
    error: Optional[str]

    # --- localize_node ---
    lang_localized: bool
