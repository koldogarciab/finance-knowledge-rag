from __future__ import annotations

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interface_utils import (
    build_filters,
    filter_options_from_rows,
    format_duration_ns,
    generation_mode_label,
    source_display_rows,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_app_structure() -> None:
    app_path = PROJECT_ROOT / "app.py"
    source = app_path.read_text(encoding="utf-8")
    ast.parse(source)

    required_fragments = (
        "OllamaGroundedRAG",
        "HybridRetriever",
        "st.chat_input",
        "st.chat_message",
        "st.cache_resource",
        "generation_mode",
        "cited_sources",
        "model_context",
    )

    for fragment in required_fragments:
        require(
            fragment in source,
            f"Missing app integration fragment: {fragment}",
        )


def validate_filter_helpers() -> None:
    require(
        build_filters(
            file_type="pdf",
            granularity="All",
            document_id="",
        )
        == {"file_type": "pdf"},
        "Filter construction is incorrect.",
    )

    rows = [
        {
            "file_type": "pdf",
            "document_id": "doc-b",
            "metadata": {"granularity": "page"},
        },
        {
            "file_type": "csv",
            "document_id": "doc-a",
            "metadata": {
                "granularity": "monthly_department_summary"
            },
        },
    ]

    file_types, granularities, document_ids = (
        filter_options_from_rows(rows)
    )

    require(
        file_types == ["csv", "pdf"],
        "File type options are not deterministic.",
    )
    require(
        granularities
        == ["monthly_department_summary", "page"],
        "Granularity options are not deterministic.",
    )
    require(
        document_ids == ["doc-a", "doc-b"],
        "Document options are not deterministic.",
    )


def validate_display_helpers() -> None:
    require(
        generation_mode_label("ollama_repair")
        == "Ollama repair",
        "Generation mode label is incorrect.",
    )
    require(
        format_duration_ns(2_500_000_000)
        == "2.50 s",
        "Duration formatting is incorrect.",
    )

    rows = source_display_rows(
        {
            "cited_sources": [
                {
                    "source_number": 2,
                    "chunk_id": "b",
                    "document_title": "Second",
                    "file_type": "pdf",
                    "citation": "Page 2",
                    "source_path": "b.pdf",
                },
                {
                    "source_number": 1,
                    "chunk_id": "a",
                    "document_title": "First",
                    "file_type": "csv",
                    "citation": "Row 1",
                    "source_path": "a.csv",
                },
            ]
        }
    )

    require(
        [row["source_number"] for row in rows]
        == [1, 2],
        "Cited sources are not ordered by citation number.",
    )


def main() -> None:
    validate_app_structure()
    validate_filter_helpers()
    validate_display_helpers()

    print("=" * 72)
    print("STREAMLIT INTERFACE VALIDATION PASSED")
    print("=" * 72)
    print("Core RAG integration: PASS")
    print("Chat interface elements: PASS")
    print("Retriever filter helpers: PASS")
    print("Generation mode display: PASS")
    print("Cited source display: PASS")
    print("Provider duration formatting: PASS")


if __name__ == "__main__":
    main()
