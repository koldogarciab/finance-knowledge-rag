from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_retrieve import HybridRetriever
from src.ollama_adapter import OllamaAdapterError
from src.ollama_rag import OllamaGroundedRAG
from src.rag_answer import ABSTENTION_MESSAGE


SUPPORTED_QUERY = (
    "How is gross margin calculated and what target "
    "applied for the nine months ended 31 March 2026?"
)

UNSUPPORTED_QUERY = (
    "What was Harbour Retail Group's carbon emissions target?"
)


class StubAdapter:
    def __init__(
        self,
        content: str = "",
        error: Exception | None = None,
    ) -> None:
        self.model = "qwen3.5:4b"
        self.content = content
        self.error = error
        self.call_count = 0
        self.received_messages: list[
            dict[str, str]
        ] = []

    def chat(
        self,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        self.call_count += 1
        self.received_messages = messages

        if self.error is not None:
            raise self.error

        return {
            "content": self.content,
            "model": self.model,
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 100,
            "eval_count": 20,
            "total_duration": 1_000_000,
        }


def main() -> None:
    retriever = HybridRetriever(
        project_root=PROJECT_ROOT,
    )

    valid_adapter = StubAdapter(
        content=(
            "Gross margin is calculated as "
            "(revenue minus cost of goods sold) "
            "divided by revenue, and its target was "
            "at least 43.1%. [1]"
        )
    )

    valid_rag = OllamaGroundedRAG(
        retriever=retriever,
        adapter=valid_adapter,
        top_k=5,
    )

    valid_result = valid_rag.answer(
        SUPPORTED_QUERY
    )

    assert valid_result["generation_mode"] == "ollama"
    assert valid_result["abstained"] is False
    assert valid_result[
        "citation_validation"
    ]["valid"] is True

    assert valid_result["model_source_count"] == 1
    assert valid_result["cited_source_count"] == 1
    assert len(valid_result["model_sources"]) == 1

    assert valid_result[
        "model_sources"
    ][0]["chunk_id"] == (
        "DOC_JSON_001:kpi:KPI_FIN_003:chunk:01"
    )

    assert "0.431" in valid_result["model_context"]
    assert "42.4" not in valid_result["model_context"]

    sent_text = "\n".join(
        message["content"]
        for message
        in valid_adapter.received_messages
    )

    assert "0.431" in sent_text
    assert "42.4" not in sent_text
    assert valid_adapter.call_count == 1

    invalid_citation_adapter = StubAdapter(
        content=(
            "Gross margin has a target of 43.1%. [9]"
        )
    )

    invalid_citation_rag = OllamaGroundedRAG(
        retriever=retriever,
        adapter=invalid_citation_adapter,
        top_k=5,
    )

    invalid_citation_result = (
        invalid_citation_rag.answer(
            SUPPORTED_QUERY
        )
    )

    assert invalid_citation_result[
        "generation_mode"
    ] == "deterministic_fallback"

    assert invalid_citation_result[
        "citation_validation"
    ]["valid"] is True

    assert "[9]" not in invalid_citation_result[
        "answer"
    ]

    assert "unknown citations" in (
        invalid_citation_result[
            "fallback_reason"
        ].casefold()
    )

    missing_citation_adapter = StubAdapter(
        content=(
            "Gross margin has a target of 43.1%."
        )
    )

    missing_citation_rag = OllamaGroundedRAG(
        retriever=retriever,
        adapter=missing_citation_adapter,
        top_k=5,
    )

    missing_citation_result = (
        missing_citation_rag.answer(
            SUPPORTED_QUERY
        )
    )

    assert missing_citation_result[
        "generation_mode"
    ] == "deterministic_fallback"

    assert "must contain at least one citation" in (
        missing_citation_result[
            "fallback_reason"
        ].casefold()
    )

    error_adapter = StubAdapter(
        error=OllamaAdapterError(
            "simulated local service failure"
        )
    )

    error_rag = OllamaGroundedRAG(
        retriever=retriever,
        adapter=error_adapter,
        top_k=5,
    )

    error_result = error_rag.answer(
        SUPPORTED_QUERY
    )

    assert error_result[
        "generation_mode"
    ] == "deterministic_fallback"

    assert error_result["provider_error"] == (
        "simulated local service failure"
    )

    abstention_adapter = StubAdapter(
        content="This must never be returned."
    )

    abstention_rag = OllamaGroundedRAG(
        retriever=retriever,
        adapter=abstention_adapter,
        top_k=5,
    )

    abstention_result = abstention_rag.answer(
        UNSUPPORTED_QUERY
    )

    assert abstention_result[
        "generation_mode"
    ] == "retrieval_abstention"

    assert abstention_result["abstained"] is True
    assert abstention_result["answer"] == (
        ABSTENTION_MESSAGE
    )

    assert abstention_adapter.call_count == 0
    assert abstention_result["model_source_count"] == 0
    assert abstention_result["model_sources"] == []
    assert abstention_result["model_context"] == ""
    assert abstention_result["cited_sources"] == []

    print("=" * 80)
    print("LOCAL OLLAMA RAG VALIDATION PASSED")
    print("=" * 80)
    print("Valid local-model answer: PASS")
    print("Evidence-context restriction: PASS")
    print("Invented citation fallback: PASS")
    print("Missing citation fallback: PASS")
    print("Local service failure fallback: PASS")
    print("Pre-generation abstention: PASS")
    print("No model call on unsupported query: PASS")


if __name__ == "__main__":
    main()
