from collections import Counter
from pathlib import Path
import json
import sys


RECORDS_PATH = Path("data/processed/corpus_records.jsonl")
MANIFEST_PATH = Path("data/processed/corpus_manifest.json")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


for path in [RECORDS_PATH, MANIFEST_PATH]:
    if not path.exists():
        fail(f"Missing file: {path}")

    if path.stat().st_size == 0:
        fail(f"File is empty: {path}")


records = []

with RECORDS_PATH.open("r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            fail(
                f"Invalid JSON on line {line_number}: {error}"
            )

        records.append(record)


with MANIFEST_PATH.open("r", encoding="utf-8") as file:
    manifest = json.load(file)


# -------------------------------------------------------------------
# Overall structure
# -------------------------------------------------------------------

if len(records) != 305:
    fail(f"Expected 305 records, found {len(records)}")

if manifest.get("record_count") != len(records):
    fail(
        "Manifest record count does not match JSONL: "
        f"{manifest.get('record_count')} vs {len(records)}"
    )

if manifest.get("document_count") != 5:
    fail(
        f"Manifest should contain 5 documents, "
        f"found {manifest.get('document_count')}"
    )

required_top_level_fields = {
    "record_id",
    "document_id",
    "document_name",
    "file_type",
    "source_path",
    "content",
    "metadata"
}

record_ids = []

for position, record in enumerate(records, start=1):
    missing_fields = (
        required_top_level_fields - set(record)
    )

    if missing_fields:
        fail(
            f"Record {position} is missing fields: "
            f"{sorted(missing_fields)}"
        )

    if not str(record["record_id"]).strip():
        fail(f"Record {position} has an empty record_id")

    if not str(record["content"]).strip():
        fail(
            f"{record['record_id']}: content is empty"
        )

    if not isinstance(record["metadata"], dict):
        fail(
            f"{record['record_id']}: metadata is not an object"
        )

    if not record["metadata"].get("granularity"):
        fail(
            f"{record['record_id']}: granularity is missing"
        )

    if "data/blueprint" in record["source_path"]:
        fail(
            f"{record['record_id']}: blueprint must not be indexed"
        )

    record_ids.append(record["record_id"])


if len(record_ids) != len(set(record_ids)):
    duplicate_ids = [
        record_id
        for record_id, count in Counter(record_ids).items()
        if count > 1
    ]

    fail(f"Duplicate record IDs found: {duplicate_ids}")


# -------------------------------------------------------------------
# Counts by format and granularity
# -------------------------------------------------------------------

expected_by_file_type = {
    "csv": 252,
    "docx": 13,
    "json": 13,
    "markdown": 18,
    "pdf": 9
}

expected_by_granularity = {
    "account_category_row": 189,
    "kpi": 13,
    "monthly_department_summary": 63,
    "page": 9,
    "section": 31
}

actual_by_file_type = Counter(
    record["file_type"]
    for record in records
)

actual_by_granularity = Counter(
    record["metadata"]["granularity"]
    for record in records
)

if dict(actual_by_file_type) != expected_by_file_type:
    fail(
        "Unexpected records by file type. "
        f"Expected {expected_by_file_type}, "
        f"found {dict(actual_by_file_type)}"
    )

if dict(actual_by_granularity) != expected_by_granularity:
    fail(
        "Unexpected records by granularity. "
        f"Expected {expected_by_granularity}, "
        f"found {dict(actual_by_granularity)}"
    )

if manifest["records_by_file_type"] != expected_by_file_type:
    fail("Manifest file-type counts are incorrect")

if manifest["records_by_granularity"] != expected_by_granularity:
    fail("Manifest granularity counts are incorrect")


# -------------------------------------------------------------------
# Document IDs and source paths
# -------------------------------------------------------------------

expected_documents = {
    "DOC_PDF_001": {
        "file_type": "pdf",
        "document_name": (
            "harbour_retail_q3_management_report_fy2026.pdf"
        ),
        "source_path": (
            "data/raw/pdf/"
            "harbour_retail_q3_management_report_fy2026.pdf"
        )
    },
    "DOC_DOCX_001": {
        "file_type": "docx",
        "document_name": (
            "finance_policies_and_procedures.docx"
        ),
        "source_path": (
            "data/raw/docx/"
            "finance_policies_and_procedures.docx"
        )
    },
    "DOC_CSV_001": {
        "file_type": "csv",
        "document_name": (
            "monthly_budget_vs_actual_fy2026.csv"
        ),
        "source_path": (
            "data/raw/csv/"
            "monthly_budget_vs_actual_fy2026.csv"
        )
    },
    "DOC_JSON_001": {
        "file_type": "json",
        "document_name": "finance_kpi_dictionary.json",
        "source_path": (
            "data/raw/json/finance_kpi_dictionary.json"
        )
    },
    "DOC_MD_001": {
        "file_type": "markdown",
        "document_name": (
            "fp_and_a_forecast_meeting_2026-04-10.md"
        ),
        "source_path": (
            "data/raw/markdown/"
            "fp_and_a_forecast_meeting_2026-04-10.md"
        )
    }
}

actual_document_ids = {
    record["document_id"]
    for record in records
}

if actual_document_ids != set(expected_documents):
    fail(
        "Unexpected document IDs. "
        f"Found {sorted(actual_document_ids)}"
    )

for record in records:
    expected = expected_documents[record["document_id"]]

    for field in [
        "file_type",
        "document_name",
        "source_path"
    ]:
        if record[field] != expected[field]:
            fail(
                f"{record['record_id']}: field '{field}' "
                f"should be '{expected[field]}', "
                f"found '{record[field]}'"
            )


# -------------------------------------------------------------------
# PDF metadata
# -------------------------------------------------------------------

pdf_records = [
    record
    for record in records
    if record["file_type"] == "pdf"
]

pdf_pages = sorted(
    record["metadata"]["page_number"]
    for record in pdf_records
)

if pdf_pages != list(range(1, 10)):
    fail(f"PDF page numbers are incorrect: {pdf_pages}")

for record in pdf_records:
    metadata = record["metadata"]

    if metadata.get("total_pages") != 9:
        fail(
            f"{record['record_id']}: total_pages should be 9"
        )

    if not metadata.get("page_title"):
        fail(
            f"{record['record_id']}: page title is missing"
        )


# -------------------------------------------------------------------
# DOCX and Markdown section metadata
# -------------------------------------------------------------------

for file_type, expected_count in [
    ("docx", 13),
    ("markdown", 18)
]:
    section_records = [
        record
        for record in records
        if record["file_type"] == file_type
    ]

    section_numbers = sorted(
        record["metadata"]["section_number"]
        for record in section_records
    )

    if section_numbers != list(
        range(1, expected_count + 1)
    ):
        fail(
            f"{file_type}: section numbering is incorrect: "
            f"{section_numbers}"
        )

    for record in section_records:
        if not record["metadata"].get("section_title"):
            fail(
                f"{record['record_id']}: section title is missing"
            )


# -------------------------------------------------------------------
# CSV metadata and March summaries
# -------------------------------------------------------------------

csv_detail_records = [
    record
    for record in records
    if record["metadata"]["granularity"]
    == "account_category_row"
]

csv_summary_records = [
    record
    for record in records
    if record["metadata"]["granularity"]
    == "monthly_department_summary"
]

detail_groups = Counter(
    (
        record["metadata"]["month"],
        record["metadata"]["department"]
    )
    for record in csv_detail_records
)

for group, count in detail_groups.items():
    if count != 3:
        fail(
            f"CSV detail group {group} should contain "
            f"3 categories, found {count}"
        )

if len(detail_groups) != 63:
    fail(
        f"Expected 63 month-department groups, "
        f"found {len(detail_groups)}"
    )

summary_keys = {
    (
        record["metadata"]["month"],
        record["metadata"]["department"]
    )
    for record in csv_summary_records
}

if summary_keys != set(detail_groups):
    fail(
        "CSV summary groups do not match detail groups"
    )

march_expected = {
    "Marketing": {
        "budget_aud": 620000,
        "actual_aud": 805000,
        "variance_aud": 185000
    },
    "Supply Chain": {
        "budget_aud": 1240000,
        "actual_aud": 1390000,
        "variance_aud": 150000
    },
    "Information Technology": {
        "budget_aud": 410000,
        "actual_aud": 365000,
        "variance_aud": -45000
    }
}

for department, expected_values in march_expected.items():
    matches = [
        record
        for record in csv_summary_records
        if (
            record["metadata"]["month"] == "2026-03"
            and record["metadata"]["department"] == department
        )
    ]

    if len(matches) != 1:
        fail(
            f"Expected one March summary for {department}, "
            f"found {len(matches)}"
        )

    metadata = matches[0]["metadata"]

    for field, expected_value in expected_values.items():
        if metadata[field] != expected_value:
            fail(
                f"March {department} {field} should be "
                f"{expected_value}, found {metadata[field]}"
            )


# -------------------------------------------------------------------
# JSON KPI metadata
# -------------------------------------------------------------------

kpi_records = [
    record
    for record in records
    if record["metadata"]["granularity"] == "kpi"
]

kpi_ids = [
    record["metadata"]["kpi_id"]
    for record in kpi_records
]

if len(kpi_ids) != len(set(kpi_ids)):
    fail("Duplicate KPI IDs detected")

if set(kpi_ids) != {
    f"KPI_FIN_{number:03d}"
    for number in range(1, 14)
}:
    fail("KPI IDs are incomplete or incorrect")

for record in kpi_records:
    metadata = record["metadata"]

    for field in [
        "kpi_name",
        "category",
        "frequency",
        "owner_name",
        "owner_title",
        "unit"
    ]:
        if not str(metadata.get(field, "")).strip():
            fail(
                f"{record['record_id']}: missing KPI metadata "
                f"field '{field}'"
            )


# -------------------------------------------------------------------
# Manifest source documents
# -------------------------------------------------------------------

manifest_sources = {
    item["document_id"]: item
    for item in manifest["source_documents"]
}

if set(manifest_sources) != set(expected_documents):
    fail("Manifest source-document IDs are incorrect")

for document_id, expected in expected_documents.items():
    source = manifest_sources[document_id]

    if source["file_type"] != expected["file_type"]:
        fail(
            f"Manifest file type is incorrect for {document_id}"
        )

    if source["path"] != expected["source_path"]:
        fail(
            f"Manifest source path is incorrect for {document_id}"
        )


print("[PASS] Processed corpus files exist and are non-empty")
print(f"[PASS] Records: {len(records)}")
print("[PASS] Record IDs are unique")
print("[PASS] All records contain content and metadata")
print("[PASS] Blueprint data was not indexed")
print("[PASS] File-type and granularity counts are correct")
print("[PASS] All five source documents are represented")
print("[PASS] PDF page metadata is complete")
print("[PASS] DOCX and Markdown sections are correctly numbered")
print("[PASS] CSV detail and summary records are consistent")
print("[PASS] March departmental summaries are correct")
print("[PASS] JSON KPI metadata is complete")
print("[PASS] Manifest matches the processed corpus")
print("[PASS] Multiformat corpus extraction is fully consistent")
