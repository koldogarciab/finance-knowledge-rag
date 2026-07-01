from collections import Counter
from pathlib import Path
from statistics import mean
import json


RECORDS_PATH = Path("data/processed/corpus_records.jsonl")
CHUNKS_PATH = Path("data/processed/corpus_chunks.jsonl")
MANIFEST_PATH = Path("data/processed/chunk_manifest.json")

MAX_WORDS = 220
OVERLAP_WORDS = 40

ATOMIC_GRANULARITIES = {
    "account_category_row",
    "monthly_department_summary",
    "kpi"
}

DOCUMENT_TITLES = {
    "DOC_PDF_001": "Q3 Management Performance Report FY2025/26",
    "DOC_DOCX_001": "Finance Policies and Procedures",
    "DOC_CSV_001": "Monthly Budget vs Actual FY2025/26",
    "DOC_JSON_001": "Finance KPI Dictionary",
    "DOC_MD_001": "FP&A Forecast Meeting Notes - 10 April 2026"
}


def split_into_word_chunks(
    text: str,
    max_words: int = MAX_WORDS,
    overlap_words: int = OVERLAP_WORDS
) -> list[str]:
    """Create overlapping word windows for long narrative records."""
    words = text.split()

    if not words:
        return []

    if len(words) <= max_words:
        return [text.strip()]

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap_words

    return chunks


def build_locator(record: dict) -> str:
    metadata = record["metadata"]
    file_type = record["file_type"]
    granularity = metadata["granularity"]

    if file_type == "pdf":
        return (
            f"page {metadata['page_number']}: "
            f"{metadata['page_title']}"
        )

    if file_type in {"docx", "markdown"}:
        return (
            f"section {metadata['section_number']}: "
            f"{metadata['section_title']}"
        )

    if file_type == "json":
        return (
            f"KPI {metadata['kpi_id']}: "
            f"{metadata['kpi_name']}"
        )

    if file_type == "csv":
        month = metadata["month"]
        department = metadata["department"]

        if granularity == "monthly_department_summary":
            return f"{month}, {department}, department summary"

        return (
            f"{month}, {department}, "
            f"{metadata['account_category']}"
        )

    return granularity


def build_retrieval_text(
    record: dict,
    chunk_content: str
) -> str:
    metadata = record["metadata"]
    document_title = DOCUMENT_TITLES[record["document_id"]]

    header_lines = [
        f"Document: {document_title}",
        f"File type: {record['file_type']}",
        f"Location: {build_locator(record)}"
    ]

    if metadata.get("month"):
        header_lines.append(f"Month: {metadata['month']}")

    if metadata.get("department"):
        header_lines.append(
            f"Department: {metadata['department']}"
        )

    if metadata.get("account_category"):
        header_lines.append(
            f"Account category: {metadata['account_category']}"
        )

    if metadata.get("kpi_name"):
        header_lines.append(f"KPI: {metadata['kpi_name']}")

    if metadata.get("variance_status"):
        header_lines.append(
            f"Variance status: {metadata['variance_status']}"
        )

    return "\n".join(header_lines) + "\n\n" + chunk_content


def main() -> None:
    if not RECORDS_PATH.exists():
        raise FileNotFoundError(
            f"Processed records not found: {RECORDS_PATH}"
        )

    records: list[dict] = []

    with RECORDS_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

    chunks: list[dict] = []
    split_record_count = 0

    for record in records:
        granularity = record["metadata"]["granularity"]

        if granularity in ATOMIC_GRANULARITIES:
            content_parts = [record["content"]]
        else:
            content_parts = split_into_word_chunks(
                record["content"]
            )

        if len(content_parts) > 1:
            split_record_count += 1

        total_parts = len(content_parts)
        locator = build_locator(record)

        for chunk_index, chunk_content in enumerate(
            content_parts,
            start=1
        ):
            metadata = dict(record["metadata"])
            metadata.update(
                {
                    "source_record_id": record["record_id"],
                    "chunk_index": chunk_index,
                    "chunk_count": total_parts,
                    "chunk_word_count": len(
                        chunk_content.split()
                    ),
                    "citation_locator": locator
                }
            )

            chunk_id = (
                f"{record['record_id']}:chunk:"
                f"{chunk_index:02d}"
            )

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "record_id": record["record_id"],
                    "document_id": record["document_id"],
                    "document_title": DOCUMENT_TITLES[
                        record["document_id"]
                    ],
                    "document_name": record["document_name"],
                    "file_type": record["file_type"],
                    "source_path": record["source_path"],
                    "citation": (
                        f"{record['document_name']} - {locator}"
                    ),
                    "content": chunk_content,
                    "retrieval_text": build_retrieval_text(
                        record,
                        chunk_content
                    ),
                    "metadata": metadata
                }
            )

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        duplicates = [
            chunk_id
            for chunk_id, count in Counter(chunk_ids).items()
            if count > 1
        ]
        raise ValueError(f"Duplicate chunk IDs: {duplicates}")

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CHUNKS_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n"
    ) as file:
        for chunk in chunks:
            file.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False
                )
            )
            file.write("\n")

    chunks_by_file_type = Counter(
        chunk["file_type"]
        for chunk in chunks
    )

    chunks_by_granularity = Counter(
        chunk["metadata"]["granularity"]
        for chunk in chunks
    )

    word_counts = [
        chunk["metadata"]["chunk_word_count"]
        for chunk in chunks
    ]

    manifest = {
        "input_file": RECORDS_PATH.as_posix(),
        "output_file": CHUNKS_PATH.as_posix(),
        "source_record_count": len(records),
        "chunk_count": len(chunks),
        "split_record_count": split_record_count,
        "chunking_strategy": {
            "method": "overlapping_word_windows",
            "maximum_words": MAX_WORDS,
            "overlap_words": OVERLAP_WORDS,
            "atomic_granularities": sorted(
                ATOMIC_GRANULARITIES
            )
        },
        "chunks_by_file_type": dict(
            sorted(chunks_by_file_type.items())
        ),
        "chunks_by_granularity": dict(
            sorted(chunks_by_granularity.items())
        ),
        "chunk_word_statistics": {
            "minimum": min(word_counts),
            "maximum": max(word_counts),
            "average": round(mean(word_counts), 2)
        }
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n"
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False
        )
        file.write("\n")

    print("Corpus chunking completed successfully")
    print(f"Source records: {len(records)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Split narrative records: {split_record_count}")
    print(
        "Chunks by file type: "
        f"{dict(sorted(chunks_by_file_type.items()))}"
    )
    print(
        "Word count range: "
        f"{min(word_counts)}-{max(word_counts)}"
    )
    print(f"Average words per chunk: {mean(word_counts):.2f}")
    print(f"Created: {CHUNKS_PATH}")
    print(f"Created: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
