from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import re
from typing import Any, Iterator

from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader


ROOT = Path(".")
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_PATH = ROOT / "data" / "processed" / "corpus_records.jsonl"
MANIFEST_PATH = ROOT / "data" / "processed" / "corpus_manifest.json"

PDF_PATH = (
    RAW_DIR
    / "pdf"
    / "harbour_retail_q3_management_report_fy2026.pdf"
)

DOCX_PATH = (
    RAW_DIR
    / "docx"
    / "finance_policies_and_procedures.docx"
)

CSV_PATH = (
    RAW_DIR
    / "csv"
    / "monthly_budget_vs_actual_fy2026.csv"
)

JSON_PATH = (
    RAW_DIR
    / "json"
    / "finance_kpi_dictionary.json"
)

MARKDOWN_PATH = (
    RAW_DIR
    / "markdown"
    / "fp_and_a_forecast_meeting_2026-04-10.md"
)


DOCUMENT_IDS = {
    "pdf": "DOC_PDF_001",
    "docx": "DOC_DOCX_001",
    "csv": "DOC_CSV_001",
    "json": "DOC_JSON_001",
    "markdown": "DOC_MD_001"
}


def normalise_text(text: str) -> str:
    """Trim lines while retaining useful paragraph boundaries."""
    lines = [line.strip() for line in str(text).splitlines()]
    cleaned_lines: list[str] = []
    previous_blank = False

    for line in lines:
        if line:
            cleaned_lines.append(line)
            previous_blank = False
        elif not previous_blank and cleaned_lines:
            cleaned_lines.append("")
            previous_blank = True

    return "\n".join(cleaned_lines).strip()


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def relative_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def make_record(
    *,
    record_id: str,
    document_id: str,
    document_name: str,
    file_type: str,
    source_path: Path,
    content: str,
    metadata: dict[str, Any]
) -> dict[str, Any]:
    cleaned_content = normalise_text(content)

    if not cleaned_content:
        raise ValueError(f"Empty content for record {record_id}")

    return {
        "record_id": record_id,
        "document_id": document_id,
        "document_name": document_name,
        "file_type": file_type,
        "source_path": relative_path(source_path),
        "content": cleaned_content,
        "metadata": metadata
    }


# -------------------------------------------------------------------
# PDF
# -------------------------------------------------------------------

def clean_pdf_page(text: str, page_number: int) -> str:
    ignored_lines = {
        "Harbour Retail Group Q3 Management Performance Report FY2025/26",
        (
            "Synthetic business information for RAG evaluation "
            f"Page {page_number}"
        )
    }

    kept_lines = [
        line
        for line in text.splitlines()
        if line.strip() not in ignored_lines
    ]

    return normalise_text("\n".join(kept_lines))


def infer_pdf_page_title(text: str, page_number: int) -> str:
    if page_number == 1:
        return "Cover and headline metrics"

    for line in text.splitlines():
        line = line.strip()

        if re.match(r"^\d+\.\s+\S", line):
            return line

    return f"Page {page_number}"


def load_pdf(path: Path) -> list[dict[str, Any]]:
    reader = PdfReader(path)
    records: list[dict[str, Any]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        content = clean_pdf_page(extracted, page_number)
        page_title = infer_pdf_page_title(content, page_number)

        records.append(
            make_record(
                record_id=f"DOC_PDF_001:page:{page_number:02d}",
                document_id=DOCUMENT_IDS["pdf"],
                document_name=path.name,
                file_type="pdf",
                source_path=path,
                content=content,
                metadata={
                    "granularity": "page",
                    "page_number": page_number,
                    "page_title": page_title,
                    "total_pages": len(reader.pages)
                }
            )
        )

    return records


# -------------------------------------------------------------------
# DOCX
# -------------------------------------------------------------------

def iter_docx_blocks(
    document: DocxDocument
) -> Iterator[Paragraph | Table]:
    body = document.element.body

    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def table_to_text(table: Table) -> str:
    rows: list[str] = []

    for row in table.rows:
        values = [
            normalise_text(cell.text).replace("\n", " ")
            for cell in row.cells
        ]

        if any(values):
            rows.append(" | ".join(values))

    return "\n".join(rows)


def load_docx(path: Path) -> list[dict[str, Any]]:
    document = Document(path)
    records: list[dict[str, Any]] = []

    current_title = "Front matter"
    current_parts: list[str] = []
    section_number = 0

    def flush_section() -> None:
        nonlocal section_number, current_parts

        content = normalise_text("\n\n".join(current_parts))

        if not content:
            current_parts = []
            return

        section_number += 1

        records.append(
            make_record(
                record_id=(
                    f"DOC_DOCX_001:section:{section_number:02d}:"
                    f"{slugify(current_title)}"
                ),
                document_id=DOCUMENT_IDS["docx"],
                document_name=path.name,
                file_type="docx",
                source_path=path,
                content=content,
                metadata={
                    "granularity": "section",
                    "section_number": section_number,
                    "section_title": current_title
                }
            )
        )

        current_parts = []

    for block in iter_docx_blocks(document):
        if isinstance(block, Paragraph):
            text = normalise_text(block.text)

            if not text:
                continue

            style_name = (
                block.style.name
                if block.style is not None
                else ""
            )

            if style_name.startswith("Heading 1"):
                flush_section()
                current_title = text
                current_parts = [text]
            else:
                current_parts.append(text)

        elif isinstance(block, Table):
            table_text = table_to_text(block)

            if table_text:
                current_parts.append(table_text)

    flush_section()

    return records


# -------------------------------------------------------------------
# CSV
# -------------------------------------------------------------------

CSV_INTEGER_FIELDS = [
    "budget_aud",
    "actual_aud",
    "variance_aud",
    "forecast_aud",
    "forecast_variance_aud"
]


def format_aud(value: int) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}AUD {abs(value):,.0f}"


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    records: list[dict[str, Any]] = []

    for row in rows:
        numeric = {
            field: int(row[field])
            for field in CSV_INTEGER_FIELDS
        }

        variance_pct = float(row["variance_pct"])

        content = (
            f"In {row['month']}, {row['department']} "
            f"cost centre {row['cost_centre']} recorded "
            f"{row['account_category']} expenditure. "
            f"Budget was {format_aud(numeric['budget_aud'])}, "
            f"actual expenditure was "
            f"{format_aud(numeric['actual_aud'])}, and the "
            f"budget variance was "
            f"{format_aud(numeric['variance_aud'])} "
            f"({variance_pct * 100:.2f}%). "
            f"The variance was classified as "
            f"{row['variance_status']}. "
            f"The pre-close forecast was "
            f"{format_aud(numeric['forecast_aud'])}, producing "
            f"an actual-versus-forecast variance of "
            f"{format_aud(numeric['forecast_variance_aud'])}. "
            f"The responsible manager was "
            f"{row['responsible_manager']}."
        )

        records.append(
            make_record(
                record_id=f"DOC_CSV_001:row:{row['row_id']}",
                document_id=DOCUMENT_IDS["csv"],
                document_name=path.name,
                file_type="csv",
                source_path=path,
                content=content,
                metadata={
                    "granularity": "account_category_row",
                    "row_id": row["row_id"],
                    "month": row["month"],
                    "department": row["department"],
                    "cost_centre": row["cost_centre"],
                    "account_category": row["account_category"],
                    "budget_aud": numeric["budget_aud"],
                    "actual_aud": numeric["actual_aud"],
                    "variance_aud": numeric["variance_aud"],
                    "variance_pct": variance_pct,
                    "forecast_aud": numeric["forecast_aud"],
                    "forecast_variance_aud": (
                        numeric["forecast_variance_aud"]
                    ),
                    "variance_status": row["variance_status"],
                    "responsible_person_id": (
                        row["responsible_person_id"]
                    ),
                    "responsible_manager": (
                        row["responsible_manager"]
                    )
                }
            )
        )

    aggregates: dict[
        tuple[str, str],
        dict[str, Any]
    ] = defaultdict(
        lambda: {
            "budget_aud": 0,
            "actual_aud": 0,
            "variance_aud": 0,
            "forecast_aud": 0,
            "forecast_variance_aud": 0,
            "categories": [],
            "responsible_manager": "",
            "responsible_person_id": "",
            "cost_centre": ""
        }
    )

    for row in rows:
        key = (row["month"], row["department"])
        summary = aggregates[key]

        for field in CSV_INTEGER_FIELDS:
            summary[field] += int(row[field])

        summary["categories"].append(row["account_category"])
        summary["responsible_manager"] = row["responsible_manager"]
        summary["responsible_person_id"] = (
            row["responsible_person_id"]
        )
        summary["cost_centre"] = row["cost_centre"]

    for (month, department), summary in sorted(aggregates.items()):
        variance = summary["variance_aud"]

        status = (
            "Unfavourable"
            if variance > 0
            else "Favourable"
            if variance < 0
            else "On budget"
        )

        percentage = (
            variance / summary["budget_aud"]
            if summary["budget_aud"]
            else 0
        )

        category_list = ", ".join(summary["categories"])

        content = (
            f"In {month}, {department} had a total expense "
            f"budget of {format_aud(summary['budget_aud'])} "
            f"and actual expenditure of "
            f"{format_aud(summary['actual_aud'])}. "
            f"The total variance was "
            f"{format_aud(variance)} "
            f"({percentage * 100:.2f}%), classified as "
            f"{status}. The pre-close forecast was "
            f"{format_aud(summary['forecast_aud'])}, and the "
            f"actual-versus-forecast variance was "
            f"{format_aud(summary['forecast_variance_aud'])}. "
            f"The included categories were {category_list}. "
            f"The responsible manager was "
            f"{summary['responsible_manager']}."
        )

        records.append(
            make_record(
                record_id=(
                    "DOC_CSV_001:summary:"
                    f"{month}:{slugify(department)}"
                ),
                document_id=DOCUMENT_IDS["csv"],
                document_name=path.name,
                file_type="csv",
                source_path=path,
                content=content,
                metadata={
                    "granularity": "monthly_department_summary",
                    "month": month,
                    "department": department,
                    "cost_centre": summary["cost_centre"],
                    "budget_aud": summary["budget_aud"],
                    "actual_aud": summary["actual_aud"],
                    "variance_aud": variance,
                    "variance_pct": percentage,
                    "forecast_aud": summary["forecast_aud"],
                    "forecast_variance_aud": (
                        summary["forecast_variance_aud"]
                    ),
                    "variance_status": status,
                    "responsible_person_id": (
                        summary["responsible_person_id"]
                    ),
                    "responsible_manager": (
                        summary["responsible_manager"]
                    ),
                    "account_categories": summary["categories"]
                }
            )
        )

    return records


# -------------------------------------------------------------------
# JSON
# -------------------------------------------------------------------

def load_json_kpis(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        document = json.load(file)

    records: list[dict[str, Any]] = []

    for kpi in document["kpis"]:
        target = kpi["target"]

        target_parts = [
            f"period {target.get('period', 'not specified')}",
            f"operator {target['operator']}"
        ]

        if "value" in target:
            target_parts.append(f"value {target['value']}")

        if "minimum" in target:
            target_parts.append(f"minimum {target['minimum']}")

        if "maximum" in target:
            target_parts.append(f"maximum {target['maximum']}")

        target_text = ", ".join(target_parts)
        owner = kpi["owner"]

        content = (
            f"KPI: {kpi['name']}. "
            f"Category: {kpi['category']}. "
            f"Definition: {kpi['definition']} "
            f"Formula: {kpi['formula']}. "
            f"Frequency: {kpi['frequency']}. "
            f"Owner: {owner['name']}, {owner['title']}. "
            f"Data source: {kpi['data_source']}. "
            f"Unit: {kpi['unit']}. "
            f"Better direction: {kpi['better_direction']}. "
            f"Target: {target_text}."
        )

        records.append(
            make_record(
                record_id=f"DOC_JSON_001:kpi:{kpi['kpi_id']}",
                document_id=DOCUMENT_IDS["json"],
                document_name=path.name,
                file_type="json",
                source_path=path,
                content=content,
                metadata={
                    "granularity": "kpi",
                    "kpi_id": kpi["kpi_id"],
                    "kpi_name": kpi["name"],
                    "category": kpi["category"],
                    "frequency": kpi["frequency"],
                    "owner_person_id": kpi["owner_person_id"],
                    "owner_name": owner["name"],
                    "owner_title": owner["title"],
                    "unit": kpi["unit"],
                    "better_direction": kpi["better_direction"]
                }
            )
        )

    return records


# -------------------------------------------------------------------
# Markdown
# -------------------------------------------------------------------

def load_markdown(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    records: list[dict[str, Any]] = []
    current_title = "Front matter"
    current_level = 1
    current_lines: list[str] = []
    section_number = 0

    heading_pattern = re.compile(r"^(#{2,3})\s+(.+?)\s*$")

    def flush_section() -> None:
        nonlocal section_number, current_lines

        content = normalise_text("\n".join(current_lines))

        if not content:
            current_lines = []
            return

        section_number += 1

        records.append(
            make_record(
                record_id=(
                    f"DOC_MD_001:section:{section_number:02d}:"
                    f"{slugify(current_title)}"
                ),
                document_id=DOCUMENT_IDS["markdown"],
                document_name=path.name,
                file_type="markdown",
                source_path=path,
                content=content,
                metadata={
                    "granularity": "section",
                    "section_number": section_number,
                    "section_title": current_title,
                    "heading_level": current_level
                }
            )
        )

        current_lines = []

    for line in lines:
        match = heading_pattern.match(line)

        if match:
            flush_section()
            current_level = len(match.group(1))
            current_title = match.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    flush_section()

    return records


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    required_paths = [
        PDF_PATH,
        DOCX_PATH,
        CSV_PATH,
        JSON_PATH,
        MARKDOWN_PATH
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing source document: {path}")

    records: list[dict[str, Any]] = []

    records.extend(load_pdf(PDF_PATH))
    records.extend(load_docx(DOCX_PATH))
    records.extend(load_csv(CSV_PATH))
    records.extend(load_json_kpis(JSON_PATH))
    records.extend(load_markdown(MARKDOWN_PATH))

    record_ids = [record["record_id"] for record in records]

    if len(record_ids) != len(set(record_ids)):
        duplicates = [
            record_id
            for record_id, count in Counter(record_ids).items()
            if count > 1
        ]
        raise ValueError(f"Duplicate record IDs: {duplicates}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
            )
            file.write("\n")

    by_file_type = Counter(
        record["file_type"]
        for record in records
    )

    by_granularity = Counter(
        record["metadata"]["granularity"]
        for record in records
    )

    manifest = {
        "output_file": relative_path(OUTPUT_PATH),
        "record_count": len(records),
        "document_count": 5,
        "records_by_file_type": dict(sorted(by_file_type.items())),
        "records_by_granularity": dict(
            sorted(by_granularity.items())
        ),
        "source_documents": [
            {
                "document_id": DOCUMENT_IDS["pdf"],
                "file_type": "pdf",
                "path": relative_path(PDF_PATH)
            },
            {
                "document_id": DOCUMENT_IDS["docx"],
                "file_type": "docx",
                "path": relative_path(DOCX_PATH)
            },
            {
                "document_id": DOCUMENT_IDS["csv"],
                "file_type": "csv",
                "path": relative_path(CSV_PATH)
            },
            {
                "document_id": DOCUMENT_IDS["json"],
                "file_type": "json",
                "path": relative_path(JSON_PATH)
            },
            {
                "document_id": DOCUMENT_IDS["markdown"],
                "file_type": "markdown",
                "path": relative_path(MARKDOWN_PATH)
            }
        ]
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

    print("Multiformat corpus loaded successfully")
    print(f"Records: {len(records)}")
    print(f"By file type: {dict(sorted(by_file_type.items()))}")
    print(
        "By granularity: "
        f"{dict(sorted(by_granularity.items()))}"
    )
    print(f"Created: {OUTPUT_PATH}")
    print(f"Created: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
