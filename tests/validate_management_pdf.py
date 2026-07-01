from pathlib import Path
import json
import sys

from pypdf import PdfReader


BLUEPRINT_PATH = Path("data/blueprint/corpus_blueprint.json")
PDF_PATH = Path(
    "data/raw/pdf/harbour_retail_q3_management_report_fy2026.pdf"
)


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


if not PDF_PATH.exists():
    fail(f"PDF file is missing: {PDF_PATH}")

if PDF_PATH.stat().st_size == 0:
    fail("PDF file is empty")


with BLUEPRINT_PATH.open("r", encoding="utf-8") as file:
    blueprint = json.load(file)

try:
    reader = PdfReader(PDF_PATH)
except Exception as error:
    fail(f"PDF cannot be opened: {error}")


if reader.is_encrypted:
    fail("PDF should not be encrypted")

if len(reader.pages) != 9:
    fail(f"Expected 9 pages, found {len(reader.pages)}")


page_texts = []

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""

    if len(text.strip()) < 100:
        fail(
            f"Page {page_number} contains insufficient "
            "extractable text"
        )

    page_texts.append(text)

full_text = "\n".join(page_texts)


def normalize_text(value: str) -> str:
    return " ".join(str(value).split())


normalized_full_text = normalize_text(full_text)


required_by_page = {
    1: [
        "HARBOUR RETAIL GROUP",
        "Q3 Management",
        "Performance Report",
        "AUD 91.8m",
        "42.4%",
        "AUD 8.7m",
        "DOC_PDF_001"
    ],
    2: [
        "1. Executive summary",
        "Store sales growth",
        "E-commerce growth",
        "FY revenue forecast",
        "Immediate management focus"
    ],
    3: [
        "2. Financial performance",
        "Revenue",
        "Gross margin",
        "Adjusted EBITDA",
        "Performance against budget"
    ],
    4: [
        "3. Channel and commercial performance",
        "Physical stores",
        "E-commerce",
        "14.8%"
    ],
    5: [
        "4. March departmental cost performance",
        "Marketing",
        "Supply Chain",
        "Information Technology",
        "AUD 393,000"
    ],
    6: [
        "5. Full-year forecast",
        "AUD 121.8m",
        "AUD 124.0m",
        "AUD 11.3m",
        "Forecast assumptions"
    ],
    7: [
        "6. Principal business risks",
        "International freight costs",
        "High",
        "Risk response"
    ],
    8: [
        "7. Q4 priorities and agreed actions",
        "PRI_001",
        "ACT_001",
        "ACT_006",
        "27 April 2026"
    ],
    9: [
        "8. Appendix",
        "Selected KPI definitions",
        "Document sources",
        "Variance conventions",
        "End of Q3 Management Performance Report"
    ]
}


for page_number, required_values in required_by_page.items():
    page_text = page_texts[page_number - 1]

    for value in required_values:
        if value not in page_text:
            fail(
                f"Page {page_number}: required text "
                f"is missing: {value}"
            )


financials = {
    item["metric"]: item
    for item in blueprint["headline_financials"]
}

required_financial_values = [
    "AUD 91.8m",
    "AUD 93.0m",
    "42.4%",
    "43.1%",
    "AUD 8.7m",
    "AUD 9.4m",
    "AUD 121.8m",
    "AUD 124.0m",
    "AUD 11.3m",
    "AUD 12.2m"
]

for value in required_financial_values:
    if value not in full_text:
        fail(f"Financial value is missing: {value}")


for driver in blueprint["variance_drivers"]:
    required_values = [
        driver["department"],
        driver["primary_cause"]
    ]

    for value in required_values:
        if value not in full_text:
            fail(
                f"Variance driver value is missing: {value}"
            )


for risk in blueprint["business_risks"]:
    if normalize_text(risk["description"]) not in normalized_full_text:
        fail(
            f"Business risk is missing: "
            f"{risk['risk_id']}"
        )


for priority in blueprint["q4_priorities"]:
    if priority["priority_id"] not in full_text:
        fail(
            f"Q4 priority is missing: "
            f"{priority['priority_id']}"
        )


for action in blueprint["forecast_meeting"]["action_items"]:
    if action["action_id"] not in full_text:
        fail(
            f"Meeting action is missing: "
            f"{action['action_id']}"
        )


metadata = reader.metadata or {}

if metadata.get("/Title") != (
    "Q3 Management Performance Report FY2025/26"
):
    fail(
        "PDF title metadata is missing or incorrect: "
        f"{metadata.get('/Title')}"
    )


print("[PASS] Q3 management PDF opened successfully")
print(f"[PASS] File size: {PDF_PATH.stat().st_size:,} bytes")
print(f"[PASS] Pages: {len(reader.pages)}")
print("[PASS] Every page contains extractable text")
print("[PASS] All report sections are present")
print("[PASS] Headline financial figures are complete")
print("[PASS] March variance drivers match the blueprint")
print("[PASS] Business risks and Q4 priorities are complete")
print("[PASS] Forecast action items are complete")
print("[PASS] PDF metadata is correct")
print("[PASS] Q3 management PDF is fully consistent")
