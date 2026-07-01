from pathlib import Path
import csv
import json
import math
import sys

BLUEPRINT_PATH = Path("data/blueprint/corpus_blueprint.json")
CSV_PATH = Path("data/raw/csv/monthly_budget_vs_actual_fy2026.csv")
JSON_PATH = Path("data/raw/json/finance_kpi_dictionary.json")
MARKDOWN_PATH = Path(
    "data/raw/markdown/fp_and_a_forecast_meeting_2026-04-10.md"
)


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


def format_aud(value: int) -> str:
    return f"AUD {value:,.0f}"


for path in [
    BLUEPRINT_PATH,
    CSV_PATH,
    JSON_PATH,
    MARKDOWN_PATH
]:
    if not path.exists():
        fail(f"Missing file: {path}")

    if path.stat().st_size == 0:
        fail(f"File is empty: {path}")


with BLUEPRINT_PATH.open("r", encoding="utf-8") as file:
    blueprint = json.load(file)

people_by_id = {
    person["person_id"]: person
    for person in blueprint["people"]
}


# -------------------------------------------------------------------
# Validate CSV
# -------------------------------------------------------------------

expected_columns = [
    "row_id",
    "month",
    "department",
    "cost_centre",
    "account_category",
    "budget_aud",
    "actual_aud",
    "variance_aud",
    "variance_pct",
    "forecast_aud",
    "forecast_variance_aud",
    "variance_status",
    "responsible_person_id",
    "responsible_manager",
    "period_status"
]

with CSV_PATH.open(
    "r",
    encoding="utf-8",
    newline=""
) as file:
    reader = csv.DictReader(file)
    csv_columns = reader.fieldnames
    csv_rows = list(reader)

if csv_columns != expected_columns:
    fail(
        "CSV columns are incorrect.\n"
        f"Expected: {expected_columns}\n"
        f"Found: {csv_columns}"
    )

source_rows = blueprint["monthly_budget_actual"]["rows"]

if len(csv_rows) != 189:
    fail(f"CSV should contain 189 rows, found {len(csv_rows)}")

if len(csv_rows) != len(source_rows):
    fail("CSV row count does not match the blueprint")

source_by_id = {
    row["row_id"]: row
    for row in source_rows
}

csv_by_id = {
    row["row_id"]: row
    for row in csv_rows
}

if set(csv_by_id) != set(source_by_id):
    fail("CSV row IDs do not match the blueprint")

integer_fields = [
    "budget_aud",
    "actual_aud",
    "variance_aud",
    "forecast_aud",
    "forecast_variance_aud"
]

text_fields = [
    "month",
    "department",
    "cost_centre",
    "account_category",
    "variance_status",
    "responsible_person_id",
    "responsible_manager",
    "period_status"
]

for row_id, csv_row in csv_by_id.items():
    source_row = source_by_id[row_id]

    for field in integer_fields:
        if int(csv_row[field]) != source_row[field]:
            fail(
                f"{row_id}: CSV field '{field}' does not "
                "match the blueprint"
            )

    for field in text_fields:
        if csv_row[field] != source_row[field]:
            fail(
                f"{row_id}: CSV field '{field}' does not "
                "match the blueprint"
            )

    if not math.isclose(
        float(csv_row["variance_pct"]),
        float(source_row["variance_pct"]),
        abs_tol=0.000001
    ):
        fail(
            f"{row_id}: CSV variance_pct does not "
            "match the blueprint"
        )

march_rows = [
    row
    for row in csv_rows
    if row["month"] == "2026-03"
]

march_budget = sum(
    int(row["budget_aud"])
    for row in march_rows
)

march_actual = sum(
    int(row["actual_aud"])
    for row in march_rows
)

if march_budget != 6240000:
    fail(f"CSV March budget is incorrect: {march_budget}")

if march_actual != 6633000:
    fail(f"CSV March actual is incorrect: {march_actual}")


# -------------------------------------------------------------------
# Validate JSON KPI dictionary
# -------------------------------------------------------------------

with JSON_PATH.open("r", encoding="utf-8") as file:
    kpi_document = json.load(file)

if kpi_document["document_id"] != "DOC_JSON_001":
    fail("KPI JSON has an incorrect document ID")

if kpi_document["company"] != blueprint["company"]["legal_name"]:
    fail("KPI JSON company does not match the blueprint")

if kpi_document["financial_year"] != blueprint[
    "financial_year"
]["label"]:
    fail("KPI JSON financial year does not match the blueprint")

if kpi_document["kpi_count"] != 13:
    fail(
        f"KPI JSON should report 13 KPIs, "
        f"found {kpi_document['kpi_count']}"
    )

output_kpis = {
    kpi["kpi_id"]: kpi
    for kpi in kpi_document["kpis"]
}

source_kpis = {
    kpi["kpi_id"]: kpi
    for kpi in blueprint["kpi_dictionary"]
}

if set(output_kpis) != set(source_kpis):
    fail("KPI IDs in the JSON do not match the blueprint")

for kpi_id, source_kpi in source_kpis.items():
    output_kpi = output_kpis[kpi_id]

    for key, expected_value in source_kpi.items():
        if output_kpi.get(key) != expected_value:
            fail(
                f"{kpi_id}: JSON field '{key}' does not "
                "match the blueprint"
            )

    person = people_by_id[source_kpi["owner_person_id"]]

    expected_owner = {
        "person_id": person["person_id"],
        "name": person["name"],
        "title": person["title"],
        "department": person["department"]
    }

    if output_kpi.get("owner") != expected_owner:
        fail(f"{kpi_id}: expanded owner details are incorrect")


# -------------------------------------------------------------------
# Validate Markdown meeting notes
# -------------------------------------------------------------------

markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
meeting = blueprint["forecast_meeting"]

required_headings = [
    "# FP&A Forecast Meeting Notes",
    "## Purpose",
    "## Attendees",
    "## Full-year forecast summary",
    "## Discussion points",
    "## Decisions",
    "## Action items",
    "## Closing outlook"
]

for heading in required_headings:
    if heading not in markdown:
        fail(f"Markdown heading is missing: {heading}")

if blueprint["company"]["legal_name"] not in markdown:
    fail("Markdown company name is missing")

if meeting["date"] not in markdown:
    fail("Markdown meeting date is missing")

for person_id in meeting["attendee_person_ids"]:
    person = people_by_id[person_id]

    if person["name"] not in markdown:
        fail(
            f"Markdown attendee is missing: "
            f"{person['name']}"
        )

for item in meeting["forecast_summary"]:
    required_values = [
        item["metric"],
        format_aud(item["forecast_aud"]),
        format_aud(item["budget_aud"]),
        format_aud(item["variance_aud"]),
        item["status"]
    ]

    for value in required_values:
        if value not in markdown:
            fail(
                f"Markdown forecast value is missing: "
                f"{value}"
            )

for item in meeting["discussion_points"]:
    if item["topic"] not in markdown:
        fail(
            f"Markdown discussion topic is missing: "
            f"{item['topic']}"
        )

    if item["summary"] not in markdown:
        fail(
            f"Markdown discussion summary is missing: "
            f"{item['discussion_id']}"
        )

for item in meeting["decisions"]:
    if item["decision_id"] not in markdown:
        fail(
            f"Markdown decision ID is missing: "
            f"{item['decision_id']}"
        )

    if item["decision"] not in markdown:
        fail(
            f"Markdown decision text is missing: "
            f"{item['decision_id']}"
        )

for item in meeting["action_items"]:
    expected_action = item["action"].replace("|", "/")
    owner = people_by_id[item["owner_person_id"]]

    required_values = [
        item["action_id"],
        expected_action,
        owner["name"],
        item["deadline"],
        item["status"]
    ]

    for value in required_values:
        if value not in markdown:
            fail(
                f"Markdown action value is missing for "
                f"{item['action_id']}: {value}"
            )


print("[PASS] All structured corpus files exist and are non-empty")
print("[PASS] CSV contains the expected columns")
print(f"[PASS] CSV rows: {len(csv_rows)}")
print("[PASS] Every CSV row matches the blueprint")
print("[PASS] CSV March consolidated totals are correct")
print(f"[PASS] JSON KPIs: {len(output_kpis)}")
print("[PASS] Every JSON KPI matches the blueprint")
print("[PASS] KPI owner details were expanded correctly")
print("[PASS] Markdown structure is complete")
print("[PASS] Markdown attendees and forecast values are complete")
print("[PASS] Markdown discussions, decisions and actions are complete")
print("[PASS] Structured corpus documents are fully consistent")
