from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "corpus_chunks.jsonl"
)

ACCEPTANCE_PATH = (
    PROJECT_ROOT
    / "tests"
    / "baseline_acceptance_cases.json"
)

FULL_CATALOG_PATH = (
    PROJECT_ROOT
    / "reports"
    / "evaluation_source_catalog.jsonl"
)

SHORTLIST_JSON_PATH = (
    PROJECT_ROOT
    / "reports"
    / "evaluation_candidate_shortlist.json"
)

SHORTLIST_MD_PATH = (
    PROJECT_ROOT
    / "reports"
    / "evaluation_candidate_shortlist.md"
)

SHORTLIST_PER_FORMAT = 12
PREVIEW_CHARACTERS = 850

FINANCE_TERMS = {
    "actual",
    "action",
    "approval",
    "authorisation",
    "budget",
    "cash",
    "deadline",
    "decision",
    "ebitda",
    "expenditure",
    "forecast",
    "formula",
    "gross margin",
    "journal",
    "materiality",
    "month-end",
    "owner",
    "policy",
    "reconciliation",
    "revenue",
    "risk",
    "target",
    "threshold",
    "variance",
    "working capital",
}

LOW_VALUE_LOCATORS = {
    "front matter",
    "cover",
    "appendix",
    "attendees",
    "purpose",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}."
                ) from exc

    return rows


def load_reserved_chunk_ids() -> set[str]:
    if not ACCEPTANCE_PATH.exists():
        return set()

    with ACCEPTANCE_PATH.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        cases = json.load(file)

    return {
        chunk_id
        for case in cases
        for chunk_id in case.get("expected_chunk_ids", [])
    }


def compact(text: str, max_chars: int) -> str:
    value = " ".join(text.split())

    if len(value) <= max_chars:
        return value

    return value[: max_chars - 3].rstrip() + "..."


def information_score(chunk: dict[str, Any]) -> float:
    content = compact(chunk.get("content", ""), 100_000)
    lowered = content.casefold()
    metadata = chunk.get("metadata", {})

    word_count = len(content.split())
    number_count = len(
        re.findall(
            r"\b\d[\d,.%]*\b",
            content,
        )
    )
    term_count = sum(
        term in lowered
        for term in FINANCE_TERMS
    )

    score = 0.0
    score += min(word_count, 180) / 30
    score += min(number_count, 10) * 0.65
    score += min(term_count, 10) * 0.55

    granularity = metadata.get("granularity")

    granularity_bonus = {
        "kpi": 2.5,
        "monthly_department_summary": 2.5,
        "account_category_row": 1.0,
        "section": 1.8,
        "page": 1.5,
    }

    score += granularity_bonus.get(granularity, 0.0)

    locator = str(
        metadata.get("citation_locator", "")
    ).casefold()

    for phrase in LOW_VALUE_LOCATORS:
        if phrase in locator:
            score -= 2.5

    if word_count < 25:
        score -= 4.0

    return round(score, 4)


def diversity_key(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {})

    locator = metadata.get("citation_locator")

    if locator:
        return str(locator).casefold()

    chunk_id = chunk["chunk_id"]

    if ":chunk:" in chunk_id:
        return chunk_id.rsplit(":chunk:", maxsplit=1)[0]

    return chunk_id


def csv_dimensions(
    chunk: dict[str, Any],
) -> tuple[str | None, str | None]:
    chunk_id = chunk["chunk_id"]
    parts = chunk_id.split(":")

    if (
        len(parts) >= 6
        and parts[1] == "summary"
    ):
        return parts[2], parts[3]

    return None, None


def select_diverse_candidates(
    candidates: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    used_diversity_keys: set[str] = set()

    ranked = sorted(
        candidates,
        key=lambda item: (
            -item["candidate_score"],
            item["chunk_id"],
        ),
    )

    # Primera pasada: una sola selección por página, sección o KPI.
    for item in ranked:
        key = item["diversity_key"]

        if key in used_diversity_keys:
            continue

        selected.append(item)
        selected_ids.add(item["chunk_id"])
        used_diversity_keys.add(key)

        if len(selected) == limit:
            return selected

    # Segunda pasada: completar si el documento no tiene suficientes
    # localizadores diferentes.
    for item in ranked:
        if item["chunk_id"] in selected_ids:
            continue

        selected.append(item)
        selected_ids.add(item["chunk_id"])

        if len(selected) == limit:
            break

    return selected


def select_csv_candidates(
    candidates: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    summaries = [
        item
        for item in candidates
        if item["granularity"]
        == "monthly_department_summary"
    ]

    ranked = sorted(
        summaries,
        key=lambda item: (
            -item["candidate_score"],
            item["chunk_id"],
        ),
    )

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    used_periods: set[str] = set()
    used_departments: set[str] = set()

    while len(selected) < limit:
        available = [
            item
            for item in ranked
            if item["chunk_id"] not in selected_ids
        ]

        if not available:
            break

        def novelty_score(item: dict[str, Any]) -> tuple[float, float]:
            period, department = csv_dimensions(item)

            novelty = 0.0

            if period and period not in used_periods:
                novelty += 2.0

            if department and department not in used_departments:
                novelty += 3.0

            return (
                novelty + item["candidate_score"],
                item["candidate_score"],
            )

        chosen = max(
            available,
            key=novelty_score,
        )

        selected.append(chosen)
        selected_ids.add(chosen["chunk_id"])

        period, department = csv_dimensions(chosen)

        if period:
            used_periods.add(period)

        if department:
            used_departments.add(department)

    return selected


def markdown_shortlist(
    shortlist_by_type: dict[str, list[dict[str, Any]]],
    reserved_count: int,
) -> str:
    lines = [
        "# Formal evaluation candidate shortlist",
        "",
        "This catalogue excludes chunks already used as expected "
        "answers in the baseline acceptance tests.",
        "",
        f"- Reserved acceptance chunks: {reserved_count}",
        f"- Candidates per file type: {SHORTLIST_PER_FORMAT}",
        "",
    ]

    for file_type in sorted(shortlist_by_type):
        items = shortlist_by_type[file_type]

        lines.extend(
            [
                f"## {file_type.upper()}",
                "",
            ]
        )

        for position, item in enumerate(items, start=1):
            lines.extend(
                [
                    (
                        f"### {position}. `{item['chunk_id']}`"
                    ),
                    "",
                    f"- **Score:** {item['candidate_score']}",
                    f"- **Title:** {item['document_title']}",
                    f"- **Granularity:** {item['granularity']}",
                    f"- **Citation:** {item['citation']}",
                    "",
                    item["content_preview"],
                    "",
                ]
            )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    chunks = load_jsonl(CHUNKS_PATH)
    reserved_chunk_ids = load_reserved_chunk_ids()

    catalog: list[dict[str, Any]] = []
    eligible_by_type: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        reserved = chunk["chunk_id"] in reserved_chunk_ids

        item = {
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "document_title": chunk["document_title"],
            "document_name": chunk["document_name"],
            "file_type": chunk["file_type"],
            "citation": chunk["citation"],
            "source_path": chunk["source_path"],
            "granularity": metadata.get("granularity"),
            "citation_locator": metadata.get(
                "citation_locator"
            ),
            "word_count": metadata.get(
                "chunk_word_count",
                len(chunk.get("content", "").split()),
            ),
            "token_count": metadata.get(
                "chunk_token_count"
            ),
            "candidate_score": information_score(chunk),
            "reserved_for_acceptance": reserved,
            "diversity_key": diversity_key(chunk),
            "content_preview": compact(
                chunk.get("content", ""),
                PREVIEW_CHARACTERS,
            ),
        }

        catalog.append(item)

        if not reserved:
            eligible_by_type[item["file_type"]].append(item)

    shortlist_by_type: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for file_type, candidates in eligible_by_type.items():
        if file_type == "csv":
            selected = select_csv_candidates(
                candidates,
                SHORTLIST_PER_FORMAT,
            )
        else:
            selected = select_diverse_candidates(
                candidates,
                SHORTLIST_PER_FORMAT,
            )

        shortlist_by_type[file_type] = selected

    FULL_CATALOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with FULL_CATALOG_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for item in catalog:
            file.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    with SHORTLIST_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            shortlist_by_type,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    SHORTLIST_MD_PATH.write_text(
        markdown_shortlist(
            shortlist_by_type,
            reserved_count=len(reserved_chunk_ids),
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print("FORMAL EVALUATION CANDIDATE CATALOGUE")
    print("=" * 80)
    print(f"Corpus chunks: {len(chunks)}")
    print(f"Reserved acceptance chunks: {len(reserved_chunk_ids)}")
    print()

    for file_type in sorted(shortlist_by_type):
        print(
            f"{file_type:10} "
            f"{len(shortlist_by_type[file_type]):>3} candidates"
        )

    print()
    print(
        "Full catalogue: "
        f"{FULL_CATALOG_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Shortlist JSON: "
        f"{SHORTLIST_JSON_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Shortlist Markdown: "
        f"{SHORTLIST_MD_PATH.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
