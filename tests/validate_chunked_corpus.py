from collections import Counter, defaultdict
from pathlib import Path
import json
import sys

from transformers import AutoTokenizer


RECORDS_PATH = Path("data/processed/corpus_records.jsonl")
CHUNKS_PATH = Path("data/processed/corpus_chunks.jsonl")
MANIFEST_PATH = Path("data/processed/chunk_manifest.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_TOKEN_LIMIT = 256
TARGET_MAX_TOKENS = 240
OVERLAP_WORDS = 40

EXPECTED_CHUNKS_BY_FILE_TYPE = {
    "csv": 252,
    "docx": 14,
    "json": 13,
    "markdown": 19,
    "pdf": 18
}

EXPECTED_CHUNKS_BY_GRANULARITY = {
    "account_category_row": 189,
    "kpi": 13,
    "monthly_department_summary": 63,
    "page": 18,
    "section": 33
}

ATOMIC_GRANULARITIES = {
    "account_category_row",
    "monthly_department_summary",
    "kpi"
}


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


def count_tokens(tokenizer, text: str) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False
    )

    return len(encoded["input_ids"])


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

if len(chunks) != 316:
    fail(f"Expected 316 chunks, found {len(chunks)}")

if manifest.get("source_record_count") != len(records):
    fail("Manifest source_record_count is incorrect")

if manifest.get("chunk_count") != len(chunks):
    fail("Manifest chunk_count is incorrect")

if manifest.get("split_record_count") != 10:
    fail(
        "Manifest should report 10 split records, found "
        f"{manifest.get('split_record_count')}"
    )


# -------------------------------------------------------------------
# Required fields and unique identifiers
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

record_ids = {
    record["record_id"]
    for record in records
}

chunk_ids = []

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
# Source-record coverage
# -------------------------------------------------------------------

chunks_by_record = defaultdict(list)

for chunk in chunks:
    chunks_by_record[chunk["record_id"]].append(chunk)

if set(chunks_by_record) != record_ids:
    missing = record_ids - set(chunks_by_record)
    unexpected = set(chunks_by_record) - record_ids

    fail(
        "Chunk/source record coverage mismatch. "
        f"Missing: {sorted(missing)}. "
        f"Unexpected: {sorted(unexpected)}"
    )

split_records = {
    record_id: record_chunks
    for record_id, record_chunks in chunks_by_record.items()
    if len(record_chunks) > 1
}

if len(split_records) != 10:
    fail(
        f"Expected 10 split records, "
        f"found {len(split_records)}"
    )


# -------------------------------------------------------------------
# Tokenizer and chunk-level validation
# -------------------------------------------------------------------

print(f"Loading tokenizer: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

all_word_counts = []
all_token_counts = []

for record_id, record_chunks in chunks_by_record.items():
    ordered_chunks = sorted(
        record_chunks,
        key=lambda item: item["metadata"]["chunk_index"]
    )

    expected_count = len(ordered_chunks)

    actual_indices = [
        chunk["metadata"]["chunk_index"]
        for chunk in ordered_chunks
    ]

    expected_indices = list(range(1, expected_count + 1))

    if actual_indices != expected_indices:
        fail(
            f"{record_id}: chunk indices should be "
            f"{expected_indices}, found {actual_indices}"
        )

    for chunk in ordered_chunks:
        metadata = chunk["metadata"]
        chunk_id = chunk["chunk_id"]

        word_count = len(chunk["content"].split())
        token_count = count_tokens(
            tokenizer,
            chunk["retrieval_text"]
        )

        all_word_counts.append(word_count)
        all_token_counts.append(token_count)

        if metadata.get("source_record_id") != record_id:
            fail(
                f"{chunk_id}: source_record_id does not "
                "match record_id"
            )

        if metadata.get("chunk_count") != expected_count:
            fail(
                f"{chunk_id}: chunk_count should be "
                f"{expected_count}"
            )

        if metadata.get("chunk_word_count") != word_count:
            fail(
                f"{chunk_id}: stored word count does not "
                f"match actual word count {word_count}"
            )

        if metadata.get("chunk_token_count") != token_count:
            fail(
                f"{chunk_id}: stored token count "
                f"{metadata.get('chunk_token_count')} does not "
                f"match actual token count {token_count}"
            )

        if word_count <= 0:
            fail(f"{chunk_id}: word count is zero")

        if token_count > TARGET_MAX_TOKENS:
            fail(
                f"{chunk_id}: {token_count} tokens exceeds "
                f"the target of {TARGET_MAX_TOKENS}"
            )

        if token_count > MODEL_TOKEN_LIMIT:
            fail(
                f"{chunk_id}: {token_count} tokens exceeds "
                f"the model limit of {MODEL_TOKEN_LIMIT}"
            )

        locator = metadata.get("citation_locator")

        if not locator:
            fail(f"{chunk_id}: citation locator is missing")

        expected_citation = (
            f"{chunk['document_name']} - {locator}"
        )

        if chunk["citation"] != expected_citation:
            fail(
                f"{chunk_id}: citation is inconsistent"
            )

        retrieval_requirements = [
            f"Document: {chunk['document_title']}",
            f"File type: {chunk['file_type']}",
            f"Location: {locator}"
        ]

        for required_value in retrieval_requirements:
            if required_value not in chunk["retrieval_text"]:
                fail(
                    f"{chunk_id}: retrieval_text is missing "
                    f"'{required_value}'"
                )

        if chunk["content"] not in chunk["retrieval_text"]:
            fail(
                f"{chunk_id}: retrieval_text does not "
                "contain the chunk content"
            )


# -------------------------------------------------------------------
# Atomic records must remain atomic
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
# Validate overlap for every split record
# -------------------------------------------------------------------

for record_id, record_chunks in split_records.items():
    ordered_chunks = sorted(
        record_chunks,
        key=lambda item: item["metadata"]["chunk_index"]
    )

    for previous, following in zip(
        ordered_chunks,
        ordered_chunks[1:]
    ):
        previous_words = previous["content"].split()
        following_words = following["content"].split()

        if len(previous_words) < OVERLAP_WORDS:
            fail(
                f"{previous['chunk_id']}: insufficient words "
                "for the expected overlap"
            )

        if len(following_words) < OVERLAP_WORDS:
            fail(
                f"{following['chunk_id']}: insufficient words "
                "for the expected overlap"
            )

        if (
            previous_words[-OVERLAP_WORDS:]
            != following_words[:OVERLAP_WORDS]
        ):
            fail(
                f"{record_id}: adjacent chunks do not "
                f"preserve the {OVERLAP_WORDS}-word overlap"
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
# Selected metadata checks
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
        "Expected one March Marketing summary chunk"
    )

marketing_metadata = march_marketing[0]["metadata"]

if marketing_metadata["variance_aud"] != 185000:
    fail("March Marketing variance should be AUD 185,000")

if marketing_metadata["variance_status"] != "Unfavourable":
    fail("March Marketing status should be Unfavourable")

if "Month: 2026-03" not in march_marketing[0]["retrieval_text"]:
    fail("March Marketing retrieval text is missing the month")

if (
    "Department: Marketing"
    not in march_marketing[0]["retrieval_text"]
):
    fail(
        "March Marketing retrieval text is missing "
        "the department"
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
    fail(
        "Gross margin retrieval text is missing "
        "the KPI name"
    )


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
# Manifest strategy and statistics
# -------------------------------------------------------------------

strategy = manifest.get("chunking_strategy", {})

expected_strategy_values = {
    "method": "token_aware_word_windows",
    "tokenizer": MODEL_NAME,
    "model_token_limit": MODEL_TOKEN_LIMIT,
    "target_max_tokens": TARGET_MAX_TOKENS,
    "maximum_words": 220,
    "overlap_words": OVERLAP_WORDS
}

for field, expected_value in expected_strategy_values.items():
    if strategy.get(field) != expected_value:
        fail(
            f"Manifest strategy field '{field}' should be "
            f"{expected_value}, found {strategy.get(field)}"
        )

if set(strategy.get("atomic_granularities", [])) != (
    ATOMIC_GRANULARITIES
):
    fail("Manifest atomic granularities are incorrect")

word_statistics = manifest.get(
    "chunk_word_statistics",
    {}
)

token_statistics = manifest.get(
    "chunk_token_statistics",
    {}
)

expected_average_words = round(
    sum(all_word_counts) / len(all_word_counts),
    2
)

expected_average_tokens = round(
    sum(all_token_counts) / len(all_token_counts),
    2
)

if word_statistics.get("minimum") != min(all_word_counts):
    fail("Manifest minimum word count is incorrect")

if word_statistics.get("maximum") != max(all_word_counts):
    fail("Manifest maximum word count is incorrect")

if word_statistics.get("average") != expected_average_words:
    fail("Manifest average word count is incorrect")

if token_statistics.get("minimum") != min(all_token_counts):
    fail("Manifest minimum token count is incorrect")

if token_statistics.get("maximum") != max(all_token_counts):
    fail("Manifest maximum token count is incorrect")

if token_statistics.get("average") != expected_average_tokens:
    fail("Manifest average token count is incorrect")

if token_statistics.get("texts_over_target") != 0:
    fail("Manifest should report zero texts above target")

if token_statistics.get("texts_over_model_limit") != 0:
    fail("Manifest should report zero texts above model limit")


print("[PASS] Chunk files exist and are non-empty")
print(f"[PASS] Source records: {len(records)}")
print(f"[PASS] Chunks: {len(chunks)}")
print("[PASS] Chunk IDs are unique")
print("[PASS] Every source record is represented")
print(f"[PASS] Split narrative records: {len(split_records)}")
print("[PASS] Chunk numbering and metadata are correct")
print("[PASS] Atomic CSV and KPI records were not split")
print(
    f"[PASS] Split chunks preserve a "
    f"{OVERLAP_WORDS}-word overlap"
)
print(
    f"[PASS] Maximum retrieval length: "
    f"{max(all_token_counts)} tokens"
)
print("[PASS] No retrieval text exceeds the token target")
print("[PASS] File-type and granularity counts are correct")
print("[PASS] Citations and retrieval headers are complete")
print("[PASS] PDF, CSV and KPI metadata were preserved")
print("[PASS] Manifest matches the token-aware chunks")
print("[PASS] Token-aware corpus chunking is fully consistent")
