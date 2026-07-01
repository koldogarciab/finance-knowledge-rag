from collections import Counter, defaultdict
from pathlib import Path
import json
import sys


RECORDS_PATH = Path("data/processed/corpus_records.jsonl")
CHUNKS_PATH = Path("data/processed/corpus_chunks.jsonl")
MANIFEST_PATH = Path("data/processed/chunk_manifest.json")

EXPECTED_CHUNKS_BY_FILE_TYPE = {
    "csv": 252,
    "docx": 13,
    "json": 13,
    "markdown": 18,
    "pdf": 10
}

EXPECTED_CHUNKS_BY_GRANULARITY = {
    "account_category_row": 189,
    "kpi": 13,
    "monthly_department_summary": 63,
    "page": 10,
    "section": 31
}

ATOMIC_GRANULARITIES = {
    "account_category_row",
    "monthly_department_summary",
    "kpi"
}

MAX_WORDS = 220
OVERLAP_WORDS = 40


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


def load_jsonl(path: Path) -> list[dict]:
    items = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as error:
                fail(
                    f"{path}: invalid JSON on line "
                    f"{line_number}: {error}"
                )

    return items


for path in [RECORDS_PATH, CHUNKS_PATH, MANIFEST_PATH]:
    if not path.exists():
        fail(f"Missing file: {path}")

    if path.stat().st_size == 0:
        fail(f"File is empty: {path}")


records = load_jsonl(RECORDS_PATH)
chunks = load_jsonl(CHUNKS_PATH)

with MANIFEST_PATH.open("r", encoding="utf-8") as file:
    manifest = json.load(file)


# -------------------------------------------------------------------
# Overall counts
# -------------------------------------------------------------------

if len(records) != 305:
    fail(f"Expected 305 source records, found {len(records)}")

if len(chunks) != 306:
    fail(f"Expected 306 chunks, found {len(chunks)}")

if manifest.get("source_record_count") != len(records):
    fail(
        "Manifest source_record_count does not match: "
        f"{manifest.get('source_record_count')} vs "
        f"{len(records)}"
    )

if manifest.get("chunk_count") != len(chunks):
    fail(
        "Manifest chunk_count does not match: "
        f"{manifest.get('chunk_count')} vs {len(chunks)}"
    )

if manifest.get("split_record_count") != 1:
    fail(
        "Manifest should report one split record, found "
        f"{manifest.get('split_record_count')}"
    )


# -------------------------------------------------------------------
# Required fields and unique IDs
# -------------------------------------------------------------------

required_chunk_fields = {
    "chunk_id",
    "record_id",
    "document_id",
    "document_title",
    "document_name",
    "file_type",
    "source_path",
    "citation",
    "content",
    "retrieval_text",
    "metadata"
}

chunk_ids = []
record_ids = {
    record["record_id"]
    for record in records
}

for position, chunk in enumerate(chunks, start=1):
    missing_fields = required_chunk_fields - set(chunk)

    if missing_fields:
        fail(
            f"Chunk {position} is missing fields: "
            f"{sorted(missing_fields)}"
        )

    chunk_id = str(chunk["chunk_id"]).strip()

    if not chunk_id:
        fail(f"Chunk {position} has an empty chunk_id")

    if chunk["record_id"] not in record_ids:
        fail(
            f"{chunk_id}: unknown source record "
            f"{chunk['record_id']}"
        )

    if not str(chunk["content"]).strip():
        fail(f"{chunk_id}: content is empty")

    if not str(chunk["retrieval_text"]).strip():
        fail(f"{chunk_id}: retrieval_text is empty")

    if not str(chunk["citation"]).strip():
        fail(f"{chunk_id}: citation is empty")

    if not isinstance(chunk["metadata"], dict):
        fail(f"{chunk_id}: metadata is not an object")

    if "data/blueprint" in chunk["source_path"]:
        fail(f"{chunk_id}: blueprint must not be indexed")

    chunk_ids.append(chunk_id)


if len(chunk_ids) != len(set(chunk_ids)):
    duplicates = [
        chunk_id
        for chunk_id, count in Counter(chunk_ids).items()
        if count > 1
    ]

    fail(f"Duplicate chunk IDs detected: {duplicates}")


# -------------------------------------------------------------------
# Every source record must be represented
# -------------------------------------------------------------------

chunks_by_record = defaultdict(list)

for chunk in chunks:
    chunks_by_record[chunk["record_id"]].append(chunk)

if set(chunks_by_record) != record_ids:
    missing_records = record_ids - set(chunks_by_record)
    unexpected_records = set(chunks_by_record) - record_ids

    fail(
        "Chunk/source record coverage mismatch. "
        f"Missing: {sorted(missing_records)}. "
        f"Unexpected: {sorted(unexpected_records)}"
    )

split_records = {
    record_id: record_chunks
    for record_id, record_chunks in chunks_by_record.items()
    if len(record_chunks) > 1
}

if len(split_records) != 1:
    fail(
        f"Expected exactly one split record, "
        f"found {len(split_records)}"
    )

split_record_id, split_chunks = next(
    iter(split_records.items())
)

split_chunks = sorted(
    split_chunks,
    key=lambda item: item["metadata"]["chunk_index"]
)

if len(split_chunks) != 2:
    fail(
        f"{split_record_id}: expected 2 chunks, "
        f"found {len(split_chunks)}"
    )

if split_chunks[0]["file_type"] != "pdf":
    fail(
        f"{split_record_id}: split record should be a PDF page"
    )

if split_chunks[0]["metadata"]["granularity"] != "page":
    fail(
        f"{split_record_id}: split record should have page granularity"
    )


# -------------------------------------------------------------------
# Chunk numbering, sizes and metadata
# -------------------------------------------------------------------

for record_id, record_chunks in chunks_by_record.items():
    ordered_chunks = sorted(
        record_chunks,
        key=lambda item: item["metadata"]["chunk_index"]
    )

    expected_count = len(ordered_chunks)

    expected_indices = list(range(1, expected_count + 1))
    actual_indices = [
        chunk["metadata"]["chunk_index"]
        for chunk in ordered_chunks
    ]

    if actual_indices != expected_indices:
        fail(
            f"{record_id}: chunk indices should be "
            f"{expected_indices}, found {actual_indices}"
        )

    for chunk in ordered_chunks:
        metadata = chunk["metadata"]
        word_count = len(chunk["content"].split())

        if metadata.get("source_record_id") != record_id:
            fail(
                f"{chunk['chunk_id']}: source_record_id "
                "does not match record_id"
            )

        if metadata.get("chunk_count") != expected_count:
            fail(
                f"{chunk['chunk_id']}: chunk_count should be "
                f"{expected_count}"
            )

        if metadata.get("chunk_word_count") != word_count:
            fail(
                f"{chunk['chunk_id']}: stored word count "
                f"{metadata.get('chunk_word_count')} does not "
                f"match actual count {word_count}"
            )

        if word_count <= 0:
            fail(f"{chunk['chunk_id']}: word count is zero")

        if word_count > MAX_WORDS:
            fail(
                f"{chunk['chunk_id']}: word count "
                f"{word_count} exceeds {MAX_WORDS}"
            )

        if not metadata.get("citation_locator"):
            fail(
                f"{chunk['chunk_id']}: citation locator is missing"
            )

        expected_citation = (
            f"{chunk['document_name']} - "
            f"{metadata['citation_locator']}"
        )

        if chunk["citation"] != expected_citation:
            fail(
                f"{chunk['chunk_id']}: citation does not match "
                "the document name and locator"
            )

        required_retrieval_values = [
            f"Document: {chunk['document_title']}",
            f"File type: {chunk['file_type']}",
            f"Location: {metadata['citation_locator']}"
        ]

        for value in required_retrieval_values:
            if value not in chunk["retrieval_text"]:
                fail(
                    f"{chunk['chunk_id']}: retrieval_text "
                    f"is missing '{value}'"
                )

        if chunk["content"] not in chunk["retrieval_text"]:
            fail(
                f"{chunk['chunk_id']}: retrieval_text "
                "does not contain the chunk content"
            )


# -------------------------------------------------------------------
# Atomic records must not be split
# -------------------------------------------------------------------

for chunk in chunks:
    granularity = chunk["metadata"]["granularity"]

    if granularity in ATOMIC_GRANULARITIES:
        if chunk["metadata"]["chunk_count"] != 1:
            fail(
                f"{chunk['chunk_id']}: atomic granularity "
                f"'{granularity}' must not be split"
            )

        if chunk["metadata"]["chunk_index"] != 1:
            fail(
                f"{chunk['chunk_id']}: atomic chunk index "
                "must be 1"
            )


# -------------------------------------------------------------------
# Validate the 40-word overlap
# -------------------------------------------------------------------

first_words = split_chunks[0]["content"].split()
second_words = split_chunks[1]["content"].split()

if len(first_words) < OVERLAP_WORDS:
    fail("First split chunk is too short for the required overlap")

if len(second_words) < OVERLAP_WORDS:
    fail("Second split chunk is too short for the required overlap")

previous_tail = first_words[-OVERLAP_WORDS:]
next_head = second_words[:OVERLAP_WORDS]

if previous_tail != next_head:
    fail(
        f"{split_record_id}: expected a "
        f"{OVERLAP_WORDS}-word overlap"
    )


# -------------------------------------------------------------------
# Counts by file type and granularity
# -------------------------------------------------------------------

actual_by_file_type = Counter(
    chunk["file_type"]
    for chunk in chunks
)

actual_by_granularity = Counter(
    chunk["metadata"]["granularity"]
    for chunk in chunks
)

if dict(actual_by_file_type) != EXPECTED_CHUNKS_BY_FILE_TYPE:
    fail(
        "Unexpected chunks by file type. "
        f"Expected {EXPECTED_CHUNKS_BY_FILE_TYPE}, "
        f"found {dict(actual_by_file_type)}"
    )

if dict(actual_by_granularity) != (
    EXPECTED_CHUNKS_BY_GRANULARITY
):
    fail(
        "Unexpected chunks by granularity. "
        f"Expected {EXPECTED_CHUNKS_BY_GRANULARITY}, "
        f"found {dict(actual_by_granularity)}"
    )

if manifest["chunks_by_file_type"] != (
    EXPECTED_CHUNKS_BY_FILE_TYPE
):
    fail("Manifest chunks_by_file_type is incorrect")

if manifest["chunks_by_granularity"] != (
    EXPECTED_CHUNKS_BY_GRANULARITY
):
    fail("Manifest chunks_by_granularity is incorrect")


# -------------------------------------------------------------------
# Validate selected metadata examples
# -------------------------------------------------------------------

march_marketing = [
    chunk
    for chunk in chunks
    if (
        chunk["metadata"]["granularity"]
        == "monthly_department_summary"
        and chunk["metadata"]["month"] == "2026-03"
        and chunk["metadata"]["department"] == "Marketing"
    )
]

if len(march_marketing) != 1:
    fail(
        "Expected one March Marketing department-summary chunk"
    )

marketing_metadata = march_marketing[0]["metadata"]

if marketing_metadata["variance_aud"] != 185000:
    fail("March Marketing variance should be AUD 185,000")

if marketing_metadata["variance_status"] != "Unfavourable":
    fail("March Marketing status should be Unfavourable")

if "Month: 2026-03" not in march_marketing[0]["retrieval_text"]:
    fail("March Marketing retrieval_text is missing the month")

if (
    "Department: Marketing"
    not in march_marketing[0]["retrieval_text"]
):
    fail(
        "March Marketing retrieval_text is missing the department"
    )


gross_margin_kpi = [
    chunk
    for chunk in chunks
    if chunk["metadata"].get("kpi_id") == "KPI_FIN_003"
]

if len(gross_margin_kpi) != 1:
    fail("Expected one Gross margin KPI chunk")

if gross_margin_kpi[0]["metadata"]["kpi_name"] != "Gross margin":
    fail("Gross margin KPI metadata is incorrect")

if "KPI: Gross margin" not in (
    gross_margin_kpi[0]["retrieval_text"]
):
    fail("Gross margin retrieval_text is missing the KPI name")


pdf_page_numbers = sorted(
    {
        chunk["metadata"]["page_number"]
        for chunk in chunks
        if chunk["file_type"] == "pdf"
    }
)

if pdf_page_numbers != list(range(1, 10)):
    fail(
        f"PDF pages should cover 1 to 9, found "
        f"{pdf_page_numbers}"
    )


# -------------------------------------------------------------------
# Manifest strategy and word statistics
# -------------------------------------------------------------------

strategy = manifest.get("chunking_strategy", {})

if strategy.get("method") != "overlapping_word_windows":
    fail("Unexpected chunking method in manifest")

if strategy.get("maximum_words") != MAX_WORDS:
    fail("Manifest maximum_words is incorrect")

if strategy.get("overlap_words") != OVERLAP_WORDS:
    fail("Manifest overlap_words is incorrect")

if set(strategy.get("atomic_granularities", [])) != (
    ATOMIC_GRANULARITIES
):
    fail("Manifest atomic granularities are incorrect")

word_counts = [
    len(chunk["content"].split())
    for chunk in chunks
]

statistics = manifest.get("chunk_word_statistics", {})

if statistics.get("minimum") != min(word_counts):
    fail("Manifest minimum word count is incorrect")

if statistics.get("maximum") != max(word_counts):
    fail("Manifest maximum word count is incorrect")

expected_average = round(
    sum(word_counts) / len(word_counts),
    2
)

if statistics.get("average") != expected_average:
    fail(
        "Manifest average word count is incorrect: "
        f"expected {expected_average}, "
        f"found {statistics.get('average')}"
    )


print("[PASS] Chunk files exist and are non-empty")
print(f"[PASS] Source records: {len(records)}")
print(f"[PASS] Chunks: {len(chunks)}")
print("[PASS] Chunk IDs are unique")
print("[PASS] Every source record is represented")
print(
    f"[PASS] Split narrative record: {split_record_id}"
)
print("[PASS] Chunk numbering and word counts are correct")
print("[PASS] Atomic CSV and KPI records were not split")
print(
    f"[PASS] Split chunks preserve a "
    f"{OVERLAP_WORDS}-word overlap"
)
print("[PASS] File-type and granularity counts are correct")
print("[PASS] Citations and retrieval headers are complete")
print("[PASS] PDF, CSV and KPI metadata were preserved")
print("[PASS] Manifest matches the generated chunks")
print("[PASS] Corpus chunking is fully consistent")
