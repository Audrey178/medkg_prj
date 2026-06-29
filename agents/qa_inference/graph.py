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
        

        return {
            "answer": final_state["answer"],
            "question_type": final_state["question_type"],
            "sources": final_state["sources"],
            "kg_coverage": final_state["kg_coverage"],
            "matched_entities": [n["name"] for n in final_state["matched_nodes"]],
            "lang_detected": final_state["lang_detected"],
            "lang_localized": final_state["lang_localized"],
            "latency_ms": final_state["latency_ms"],
            "tokens_used": final_state["tokens_used"],
            "error": final_state["error"],
        }
        
        
if __name__ == '__main__':
    graph = build_graph()
    png_data = graph.get_graph().draw_mermaid_png()
    with open('graph.png', 'wb') as f:
        f.write(png_data)
    print('Graph saved')
