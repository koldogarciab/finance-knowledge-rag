from __future__ import annotations

from typing import Any, Iterable


ALL_VALUE = "All"


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None

    cleaned = str(value).strip()

    if not cleaned or cleaned.casefold() == ALL_VALUE.casefold():
        return None

    return cleaned


def build_filters(
    *,
    file_type: str | None = None,
    granularity: str | None = None,
    document_id: str | None = None,
) -> dict[str, str]:
    """Build filters accepted by the existing retriever API."""
    filters: dict[str, str] = {}

    cleaned_file_type = _clean_optional(file_type)
    cleaned_granularity = _clean_optional(granularity)
    cleaned_document_id = _clean_optional(document_id)

    if cleaned_file_type:
        filters["file_type"] = cleaned_file_type

    if cleaned_granularity:
        filters["granularity"] = cleaned_granularity

    if cleaned_document_id:
        filters["document_id"] = cleaned_document_id

    return filters


def filter_options_from_rows(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    """Return sorted file-type, granularity and document filter options."""
    file_types: set[str] = set()
    granularities: set[str] = set()
    document_ids: set[str] = set()

    for row in rows:
        file_type = str(row.get("file_type") or "").strip()
        document_id = str(row.get("document_id") or "").strip()
        metadata = row.get("metadata") or {}
        granularity = str(
            metadata.get("granularity") or ""
        ).strip()

        if file_type:
            file_types.add(file_type)

        if granularity:
            granularities.add(granularity)

        if document_id:
            document_ids.add(document_id)

    return (
        sorted(file_types, key=str.casefold),
        sorted(granularities, key=str.casefold),
        sorted(document_ids, key=str.casefold),
    )


GENERATION_MODE_LABELS = {
    "ollama": "Ollama",
    "ollama_repair": "Ollama repair",
    "retrieval_abstention": "Retrieval abstention",
    "ollama_abstention": "Ollama abstention",
    "deterministic_fallback": "Deterministic fallback",
    "context_extractive_fallback": "Extractive fallback",
    "safe_fallback_abstention": "Safe fallback abstention",
}


def generation_mode_label(mode: str) -> str:
    """Convert an internal mode name into a user-facing label."""
    cleaned = str(mode or "unknown").strip()

    return GENERATION_MODE_LABELS.get(
        cleaned,
        cleaned.replace("_", " ").title(),
    )


def format_duration_ns(value: Any) -> str | None:
    """Format an Ollama nanosecond duration for display."""
    if value is None:
        return None

    try:
        nanoseconds = float(value)
    except (TypeError, ValueError):
        return None

    if nanoseconds < 0:
        return None

    seconds = nanoseconds / 1_000_000_000

    if seconds < 1:
        return f"{seconds * 1_000:.0f} ms"

    return f"{seconds:.2f} s"


def source_display_rows(
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return stable, complete source rows for the interface."""
    rows: list[dict[str, Any]] = []

    for source in result.get("cited_sources") or []:
        rows.append(
            {
                "source_number": int(
                    source.get("source_number", 0)
                ),
                "chunk_id": str(
                    source.get("chunk_id") or ""
                ),
                "document_title": str(
                    source.get("document_title") or ""
                ),
                "file_type": str(
                    source.get("file_type") or ""
                ),
                "citation": str(
                    source.get("citation") or ""
                ),
                "source_path": str(
                    source.get("source_path") or ""
                ),
            }
        )

    return sorted(
        rows,
        key=lambda row: row["source_number"],
    )
