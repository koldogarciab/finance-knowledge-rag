from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_retrieve import HybridRetriever
from src.rag_answer import (
    ABSTENTION_MESSAGE,
    GroundedRAG,
    build_llm_messages,
    validate_answer_citations,
)


SUPPORTED_QUERY = (
    "How is gross margin calculated and what target "
    "applied for the nine months ended 31 March 2026?"
)

UNSUPPORTED_QUERY = (
    "What was Harbour Retail Group's carbon emissions target?"
)


def main() -> None:
    retriever = HybridRetriever(
        project_root=PROJECT_ROOT,
    )

    rag = GroundedRAG(
        retriever=retriever,
        top_k=5,
    )

    supported = rag.answer(SUPPORTED_QUERY)

    assert supported["abstained"] is False
    assert supported["citation_validation"]["valid"] is True
    assert supported["cited_source_count"] >= 1
    assert supported["cited_sources"]
    assert "[1]" in supported["answer"]

    supported_answer = supported["answer"].casefold()

    assert "revenue - cost of goods sold" in supported_answer
    assert "0.431" in supported_answer

    cited_numbers = {
        source["source_number"]
        for source in supported["cited_sources"]
    }

    assert cited_numbers == set(
        supported["citation_validation"]["used_citations"]
    )

    gross_margin_chunks = {
        source["chunk_id"]
        for source in supported["cited_sources"]
    }

    assert (
        "DOC_JSON_001:kpi:KPI_FIN_003:chunk:01"
        in gross_margin_chunks
    )

    unsupported = rag.answer(UNSUPPORTED_QUERY)

    assert unsupported["abstained"] is True
    assert unsupported["answer"] == ABSTENTION_MESSAGE
    assert unsupported["citation_validation"]["valid"] is True
    assert unsupported["cited_source_count"] == 0
    assert unsupported["cited_sources"] == []
    assert (
        unsupported["citation_validation"]["used_citations"]
        == []
    )

    valid_citations = validate_answer_citations(
        answer="Gross margin target was 43.1%. [1]",
        sources=[
            {
                "source_number": 1,
            }
        ],
        abstained=False,
    )

    assert valid_citations["valid"] is True

    unknown_citation = validate_answer_citations(
        answer="Gross margin target was 43.1%. [9]",
        sources=[
            {
                "source_number": 1,
            }
        ],
        abstained=False,
    )

    assert unknown_citation["valid"] is False
    assert unknown_citation["unknown_citations"] == [9]

    missing_citation = validate_answer_citations(
        answer="Gross margin target was 43.1%.",
        sources=[
            {
                "source_number": 1,
            }
        ],
        abstained=False,
    )

    assert missing_citation["valid"] is False

    abstention_with_citation = validate_answer_citations(
        answer=ABSTENTION_MESSAGE + " [1]",
        sources=[
            {
                "source_number": 1,
            }
        ],
        abstained=True,
    )

    assert abstention_with_citation["valid"] is False

    messages = build_llm_messages(
        query=SUPPORTED_QUERY,
        context_bundle={
            "context": supported["context"],
        },
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Numbered sources:" in messages[1]["content"]
    assert "[1]" in messages[1]["content"]

    impossible_filter = rag.answer(
        query="What was revenue?",
        filters={
            "document_id": "DOCUMENT_THAT_DOES_NOT_EXIST"
        },
    )

    assert impossible_filter["abstained"] is True
    assert impossible_filter["source_count"] == 0
    assert impossible_filter["cited_source_count"] == 0

    print("=" * 80)
    print("GROUNDED RAG CORE VALIDATION PASSED")
    print("=" * 80)
    print("Supported answer: PASS")
    print("Formula and target coverage: PASS")
    print("Valid citations: PASS")
    print("Unknown citation rejection: PASS")
    print("Missing citation rejection: PASS")
    print("Abstention: PASS")
    print("Provider-neutral LLM messages: PASS")
    print("Impossible metadata filter: PASS")


if __name__ == "__main__":
    main()
