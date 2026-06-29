"""
LangGraph StateGraph for KG-RAG QA inference.
Nodes are imported from agents/qa_inference/nodes/.
"""

import time
from pathlib import Path

import yaml
from langgraph.graph import StateGraph, END

from .state import QAState
from .nodes.translate_node import translate_node
from .nodes.intent_node import intent_node
from .nodes.retrieval_node import retrieval_node
from .nodes.answer_node import answer_node
from .nodes.localize_node import localize_node


_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "qa_inference.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_graph() -> StateGraph:
    graph = StateGraph(QAState)

    graph.add_node("translate", translate_node)
    graph.add_node("intent", intent_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("answer", answer_node)
    graph.add_node("localize", localize_node)

    graph.set_entry_point("translate")
    graph.add_edge("translate", "intent")
    graph.add_edge("intent", "retrieval")
    graph.add_edge("retrieval", "answer")
    graph.add_conditional_edges(
        "answer",
        lambda s: "localize" if s.get("lang_detected", "en") not in ("en", "unknown", "") else END,
        {"localize": "localize", END: END},
    )
    graph.add_edge("localize", END)

    return graph.compile()


class QAPipeline:
    """Public interface for KG-RAG QA inference."""

    def __init__(self, config: dict | None = None):
        self._config = config or _load_config()
        self._graph = build_graph()

    def run(
        self,
        query: str,
        benchmark_type: str = "bioasq",
        mode: str = "kg_rag",
        options: dict | None = None,
        question_type: str | None = None,
    ) -> dict:
        t0 = time.monotonic()

        initial_state: QAState = {
            "query_raw": query,
            "benchmark_type": benchmark_type,
            "mode": mode,
            "options": options or {},
            "query_en": "",
            "lang_detected": "",
            "extracted_entities": [],
            "matched_nodes": [],
            "question_type": question_type or "",
            "relation_intents": [],
            "raw_triples": [],
            "sources": [],
            "answer": None,
            "kg_coverage": False,
            "latency_ms": 0.0,
            "tokens_used": 0,
            "error": None,
            "lang_localized": False,
        }

        final_state = self._graph.invoke(initial_state)
        final_state["latency_ms"] = round((time.monotonic() - t0) * 1000, 1)
        

        matched_nodes = final_state.get("matched_nodes") or []
        return {
            "answer": final_state["answer"],
            "question_type": final_state["question_type"],
            "sources": final_state["sources"],
            "kg_coverage": final_state["kg_coverage"],
            "matched_entities": [n["name"] for n in matched_nodes],
            "lang_detected": final_state["lang_detected"],
            "lang_localized": final_state["lang_localized"],
            "latency_ms": final_state["latency_ms"],
            "tokens_used": final_state["tokens_used"],
            "error": final_state["error"],
            # Step-level data for per-question debugging / step logging
            "_debug": {
                "step_translate": {
                    "query_en": final_state.get("query_en", ""),
                    "lang_detected": final_state.get("lang_detected", ""),
                },
                "step_intent": {
                    "question_type": final_state.get("question_type", ""),
                    "relation_intents": final_state.get("relation_intents") or [],
                },
                "step_retrieval": {
                    "extracted_entities": final_state.get("extracted_entities") or [],
                    "matched_nodes": [
                        {
                            "name": n.get("name", ""),
                            "cui": n.get("cui", ""),
                            "confidence": n.get("confidence", 0.0),
                            "strategy": n.get("strategy", ""),
                        }
                        for n in matched_nodes
                    ],
                    "n_triples": len(final_state.get("raw_triples") or []),
                    "sources": final_state.get("sources") or [],
                    "kg_coverage": final_state.get("kg_coverage", False),
                    "raw_triples_preview": [
                        {k: t.get(k) for k in ("relation", "source_name", "target_name", "pmid", "credibility_score")}
                        for t in (final_state.get("raw_triples") or [])[:5]
                    ],
                },
                "step_answer": {
                    "answer": (final_state.get("answer") or {}).get("answer"),
                    "explanation": (final_state.get("answer") or {}).get("explanation", ""),
                    "reasoning_answer": (final_state.get("answer") or {}).get("reasoning_answer"),
                    "tokens_used": final_state.get("tokens_used", 0),
                    "latency_ms": final_state.get("latency_ms", 0.0),
                },
            },
        }
        
        
if __name__ == '__main__':
    graph = build_graph()
    png_data = graph.get_graph().draw_mermaid_png()
    with open('graph.png', 'wb') as f:
        f.write(png_data)
    print('Graph saved')
