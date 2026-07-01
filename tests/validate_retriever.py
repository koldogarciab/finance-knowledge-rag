from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieve import SemanticRetriever


QUERY = (
    "How did revenue perform against budget in the "
    "nine months ended 31 March 2026?"
)

ANSWER_BEARING_CHUNKS = {
    "DOC_PDF_001:page:01:chunk:01",
    "DOC_PDF_001:page:02:chunk:01",
    "DOC_PDF_001:page:03:chunk:01",
}

REQUIRED_FIELDS = {
    "rank",
    "score",
    "row_index",
    "chunk_id",
    "record_id",
    "document_id",
    "document_title",
    "document_name",
    "file_type",
    "source_path",
    "citation",
    "content",
    "metadata",
}


def assert_valid_results(
    results: list[dict[str, Any]],
    expected_length: int,
) -> None:
    assert len(results) == expected_length, (
        f"Expected {expected_length} results, found {len(results)}."
    )

    scores = [result["score"] for result in results]

    assert all(math.isfinite(score) for score in scores), (
        "All similarity scores must be finite."
    )

    assert scores == sorted(scores, reverse=True), (
        "Results are not ordered by descending score."
    )

    for expected_rank, result in enumerate(results, start=1):
        assert result["rank"] == expected_rank
        assert REQUIRED_FIELDS.issubset(result)
        assert result["chunk_id"]
        assert result["citation"]
        assert result["content"]


def contains_answer_chunk(
    results: list[dict[str, Any]],
) -> bool:
    return any(
        result["chunk_id"] in ANSWER_BEARING_CHUNKS
        for result in results
    )


def compact_result(
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "rank": result["rank"],
        "score": round(result["score"], 4),
        "chunk_id": result["chunk_id"],
        "file_type": result["file_type"],
        "citation": result["citation"],
    }


def main() -> None:
    retriever = SemanticRetriever(project_root=PROJECT_ROOT)

    global_top5 = retriever.search(
        query=QUERY,
        top_k=5,
    )

    pdf_top5 = retriever.search(
        query=QUERY,
        top_k=5,
        filters={"file_type": "pdf"},
    )

    document_top5 = retriever.search(
        query=QUERY,
        top_k=5,
        filters={"document_id": "DOC_PDF_001"},
    )

    impossible_filter = retriever.search(
        query=QUERY,
        top_k=5,
        filters={"document_id": "DOCUMENT_THAT_DOES_NOT_EXIST"},
    )

    all_ranked = retriever.search(
        query=QUERY,
        top_k=len(retriever.rows),
    )

    assert_valid_results(global_top5, expected_length=5)
    assert_valid_results(pdf_top5, expected_length=5)
    assert_valid_results(document_top5, expected_length=5)

    assert all(
        result["file_type"] == "pdf"
        for result in pdf_top5
    ), "The file_type filter was not applied correctly."

    assert all(
        result["document_id"] == "DOC_PDF_001"
        for result in document_top5
    ), "The document_id filter was not applied correctly."

    assert impossible_filter == [], (
        "An impossible filter should return an empty result list."
    )

    assert contains_answer_chunk(pdf_top5), (
        "PDF-filtered results do not contain an answer-bearing chunk."
    )

    assert contains_answer_chunk(document_top5), (
        "Document-filtered results do not contain an answer-bearing chunk."
    )

    relevant_global_results = [
        result
        for result in all_ranked
        if result["chunk_id"] in ANSWER_BEARING_CHUNKS
    ]

    best_relevant_global_rank = min(
        result["rank"]
        for result in relevant_global_results
    )

    report = {
        "status": "passed",
        "query": QUERY,
        "corpus_size": len(retriever.rows),
        "global_top5_answer_hit": contains_answer_chunk(global_top5),
        "pdf_top5_answer_hit": contains_answer_chunk(pdf_top5),
        "document_top5_answer_hit": contains_answer_chunk(
            document_top5
        ),
        "best_answer_bearing_global_rank": best_relevant_global_rank,
        "global_top5": [
            compact_result(result)
            for result in global_top5
        ],
        "pdf_top5": [
            compact_result(result)
            for result in pdf_top5
        ],
    }

    print("=" * 80)
    print("BASELINE RETRIEVER VALIDATION PASSED")
    print("=" * 80)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
