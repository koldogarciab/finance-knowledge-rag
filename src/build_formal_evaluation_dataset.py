from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILES = [
    PROJECT_ROOT
    / "tests"
    / "formal_evaluation_cases_pdf_docx.json",
    PROJECT_ROOT
    / "tests"
    / "formal_evaluation_cases_json_markdown.json",
    PROJECT_ROOT
    / "tests"
    / "formal_evaluation_cases_csv.json",
]

ACCEPTANCE_PATH = (
    PROJECT_ROOT
    / "tests"
    / "baseline_acceptance_cases.json"
)

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "corpus_chunks.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "tests"
    / "formal_evaluation_cases.json"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "reports"
    / "formal_evaluation_manifest.json"
)

EXPECTED_FILE_TYPE_COUNTS = {
    "pdf": 6,
    "docx": 6,
    "json": 6,
    "markdown": 6,
    "csv": 6,
}

VALID_DIFFICULTIES = {
    "easy",
    "medium",
    "hard",
}

REQUIRED_FIELDS = {
    "case_id",
    "file_type",
    "document_id",
    "difficulty",
    "query_type",
    "tags",
    "query",
    "reference_answer",
    "expected_chunk_ids",
    "filters",
    "source_citations",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


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


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    formal_cases: list[dict[str, Any]] = []

    for source_path in SOURCE_FILES:
        if not source_path.exists():
            raise FileNotFoundError(
                f"Missing formal evaluation block: {source_path}"
            )

        block = load_json(source_path)

        if not isinstance(block, list):
            raise ValueError(
                f"{source_path} must contain a JSON list."
            )

        formal_cases.extend(block)

    chunks = load_jsonl(CHUNKS_PATH)

    available_chunks = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    acceptance_cases = load_json(ACCEPTANCE_PATH)

    acceptance_chunk_ids = {
        chunk_id
        for case in acceptance_cases
        for chunk_id in case["expected_chunk_ids"]
    }

    case_ids = [
        case.get("case_id")
        for case in formal_cases
    ]

    if len(formal_cases) != 30:
        raise ValueError(
            f"Expected 30 formal cases, found {len(formal_cases)}."
        )

    if len(case_ids) != len(set(case_ids)):
        duplicates = [
            case_id
            for case_id, count
            in Counter(case_ids).items()
            if count > 1
        ]

        raise ValueError(
            f"Duplicate case_id values: {duplicates}"
        )

    file_type_counts = Counter(
        case["file_type"]
        for case in formal_cases
    )

    if dict(file_type_counts) != EXPECTED_FILE_TYPE_COUNTS:
        raise ValueError(
            "Unexpected file-type distribution: "
            f"{dict(file_type_counts)}"
        )

    used_formal_chunks: set[str] = set()

    for case in formal_cases:
        missing_fields = REQUIRED_FIELDS - set(case)

        if missing_fields:
            raise ValueError(
                f"{case['case_id']} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        if case["difficulty"] not in VALID_DIFFICULTIES:
            raise ValueError(
                f"{case['case_id']} has invalid difficulty "
                f"{case['difficulty']!r}."
            )

        if not case["query"].strip():
            raise ValueError(
                f"{case['case_id']} has an empty query."
            )

        if not case["reference_answer"].strip():
            raise ValueError(
                f"{case['case_id']} has an empty reference answer."
            )

        if not case["expected_chunk_ids"]:
            raise ValueError(
                f"{case['case_id']} has no expected chunks."
            )

        if not case["tags"]:
            raise ValueError(
                f"{case['case_id']} has no tags."
            )

        for chunk_id in case["expected_chunk_ids"]:
            if chunk_id not in available_chunks:
                raise ValueError(
                    f"{case['case_id']} references missing "
                    f"chunk {chunk_id}."
                )

            if chunk_id in acceptance_chunk_ids:
                raise ValueError(
                    f"{case['case_id']} reuses acceptance "
                    f"chunk {chunk_id}."
                )

            if chunk_id in used_formal_chunks:
                raise ValueError(
                    f"Formal chunk reused across cases: {chunk_id}."
                )

            chunk = available_chunks[chunk_id]

            if chunk["file_type"] != case["file_type"]:
                raise ValueError(
                    f"{case['case_id']} has a file-type mismatch "
                    f"for {chunk_id}."
                )

            if chunk["document_id"] != case["document_id"]:
                raise ValueError(
                    f"{case['case_id']} has a document mismatch "
                    f"for {chunk_id}."
                )

            used_formal_chunks.add(chunk_id)

    formal_cases.sort(
        key=lambda case: (
            case["file_type"],
            case["case_id"],
        )
    )

    difficulty_counts = Counter(
        case["difficulty"]
        for case in formal_cases
    )

    query_type_counts = Counter(
        case["query_type"]
        for case in formal_cases
    )

    tag_counts = Counter(
        tag
        for case in formal_cases
        for tag in case["tags"]
    )

    manifest = {
        "dataset_name": "finance_rag_formal_retrieval_evaluation",
        "version": "1.0",
        "case_count": len(formal_cases),
        "acceptance_case_count": len(acceptance_cases),
        "acceptance_chunks_excluded": len(
            acceptance_chunk_ids
        ),
        "formal_expected_chunks": len(
            used_formal_chunks
        ),
        "file_type_counts": dict(
            sorted(file_type_counts.items())
        ),
        "difficulty_counts": dict(
            sorted(difficulty_counts.items())
        ),
        "query_type_counts": dict(
            sorted(query_type_counts.items())
        ),
        "tag_counts": dict(
            sorted(tag_counts.items())
        ),
        "evaluation_modes": {
            "primary": (
                "Global retrieval over all corpus chunks without filters."
            ),
            "secondary": (
                "Filtered retrieval using the metadata filters "
                "stored in each case."
            ),
        },
        "metrics": [
            "Hit@1",
            "Hit@3",
            "Hit@5",
            "Hit@10",
            "MRR",
            "mean rank",
            "median rank",
        ],
        "source_files": [
            str(path.relative_to(PROJECT_ROOT))
            for path in SOURCE_FILES
        ],
        "output_file": str(
            OUTPUT_PATH.relative_to(PROJECT_ROOT)
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            formal_cases,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    dataset_sha256 = hashlib.sha256(
        OUTPUT_PATH.read_bytes()
    ).hexdigest()
    manifest["dataset_sha256"] = dataset_sha256

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    print("=" * 88)
    print("FORMAL EVALUATION DATASET VALIDATION PASSED")
    print("=" * 88)
    print(f"Cases: {len(formal_cases)}")
    print(
        "File types:",
        dict(sorted(file_type_counts.items())),
    )
    print(
        "Difficulty:",
        dict(sorted(difficulty_counts.items())),
    )
    print(
        "Query types:",
        dict(sorted(query_type_counts.items())),
    )
    print(f"Formal expected chunks: {len(used_formal_chunks)}")
    print("Acceptance chunk overlap: 0")
    print()
    print(
        "Dataset: "
        f"{OUTPUT_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Manifest: "
        f"{MANIFEST_PATH.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
