from pathlib import Path
import csv
import json

BLUEPRINT_PATH = Path("data/blueprint/corpus_blueprint.json")
CSV_PATH = Path("data/raw/csv/monthly_budget_vs_actual_fy2026.csv")
JSON_PATH = Path("data/raw/json/finance_kpi_dictionary.json")
MARKDOWN_PATH = Path(
    "data/raw/markdown/fp_and_a_forecast_meeting_2026-04-10.md"
)

with BLUEPRINT_PATH.open("r", encoding="utf-8") as file:
    data = json.load(file)

people_by_id = {
    person["person_id"]: person
    for person in data["people"]
}


def format_aud(value: int) -> str:
    return f"AUD {value:,.0f}"


def resolve_person(person_id: str) -> dict:
    if person_id not in people_by_id:
        raise ValueError(f"Unknown person ID: {person_id}")

    return people_by_id[person_id]


# -------------------------------------------------------------------
# 1. CSV - Monthly Budget vs Actual
# -------------------------------------------------------------------

CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

csv_rows = data["monthly_budget_actual"]["rows"]

csv_columns = [
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
    "w",
    encoding="utf-8",
    newline=""
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=csv_columns,
        extrasaction="ignore"
    )
    writer.writeheader()
    writer.writerows(csv_rows)


# -------------------------------------------------------------------
# 2. JSON - Finance KPI Dictionary
# -------------------------------------------------------------------

JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

kpi_output = {
    "document_id": "DOC_JSON_001",
    "title": "Finance KPI Dictionary",
    "company": data["company"]["legal_name"],
    "financial_year": data["financial_year"]["label"],
    "currency": data["project"]["currency"],
    "language": data["project"]["corpus_language"],
    "synthetic_data": True,
    "description": (
        "Definitions, formulas, ownership, data sources and targets "
        "for the principal financial and operational KPIs used by "
        "Harbour Retail Group."
    ),
    "kpi_count": len(data["kpi_dictionary"]),
    "kpis": []
}

for kpi in data["kpi_dictionary"]:
    owner = resolve_person(kpi["owner_person_id"])

    output_kpi = dict(kpi)
    output_kpi["owner"] = {
        "person_id": owner["person_id"],
        "name": owner["name"],
        "title": owner["title"],
        "department": owner["department"]
    }

    kpi_output["kpis"].append(output_kpi)

with JSON_PATH.open("w", encoding="utf-8", newline="\n") as file:
    json.dump(
        kpi_output,
        file,
        indent=2,
        ensure_ascii=False
    )
    file.write("\n")


# -------------------------------------------------------------------
# 3. Markdown - FP&A Forecast Meeting Notes
# -------------------------------------------------------------------

MARKDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)

meeting = data["forecast_meeting"]
chair = resolve_person(meeting["chair_person_id"])
minute_taker = resolve_person(meeting["minute_taker_person_id"])

lines = [
    "# FP&A Forecast Meeting Notes",
    "",
    f"**Company:** {data['company']['legal_name']}  ",
    f"**Date:** {meeting['date']}  ",
    (
        f"**Time:** {meeting['start_time']} - "
        f"{meeting['end_time']}  "
    ),
    f"**Location:** {meeting['location']}  ",
    f"**Chair:** {chair['name']}, {chair['title']}  ",
    (
        f"**Minute taker:** {minute_taker['name']}, "
        f"{minute_taker['title']}  "
    ),
    "",
    "> This document contains synthetic business information created "
    "for a reproducible RAG evaluation project.",
    "",
    "## Purpose",
    "",
    meeting["purpose"],
    "",
    "## Attendees",
    ""
]

for person_id in meeting["attendee_person_ids"]:
    person = resolve_person(person_id)
    lines.append(
        f"- **{person['name']}** - {person['title']} "
        f"({person['department']})"
    )

lines.extend(
    [
        "",
        "## Full-year forecast summary",
        "",
        "| Metric | Forecast | Budget | Variance | Status |",
        "|---|---:|---:|---:|---|"
    ]
)

for item in meeting["forecast_summary"]:
    lines.append(
        f"| {item['metric']} "
        f"| {format_aud(item['forecast_aud'])} "
        f"| {format_aud(item['budget_aud'])} "
        f"| {format_aud(item['variance_aud'])} "
        f"| {item['status']} |"
    )

lines.extend(
    [
        "",
        "## Discussion points",
        ""
    ]
)

for item in meeting["discussion_points"]:
    lead = resolve_person(item["lead_person_id"])
    linked_facts = ", ".join(item["linked_fact_ids"])

    lines.extend(
        [
            f"### {item['topic']}",
            "",
            item["summary"],
            "",
            f"**Discussion lead:** {lead['name']}, {lead['title']}  ",
            f"**Linked facts:** {linked_facts}",
            ""
        ]
    )

lines.extend(
    [
        "## Decisions",
        ""
    ]
)

for item in meeting["decisions"]:
    owner = resolve_person(item["owner_person_id"])
    priorities = ", ".join(item["linked_priority_ids"])

    lines.extend(
        [
            f"### {item['decision_id']}",
            "",
            item["decision"],
            "",
            f"**Owner:** {owner['name']}, {owner['title']}  ",
            f"**Linked priorities:** {priorities}",
            ""
        ]
    )

lines.extend(
    [
        "## Action items",
        "",
        (
            "| Action ID | Action | Owner | Deadline | "
            "Status | Linked decision |"
        ),
        "|---|---|---|---|---|---|"
    ]
)

for item in meeting["action_items"]:
    owner = resolve_person(item["owner_person_id"])
    linked_decisions = ", ".join(item["linked_decision_ids"])

    clean_action = item["action"].replace("|", "/")

    lines.append(
        f"| {item['action_id']} "
        f"| {clean_action} "
        f"| {owner['name']} "
        f"| {item['deadline']} "
        f"| {item['status']} "
        f"| {linked_decisions} |"
    )

lines.extend(
    [
        "",
        "## Closing outlook",
        "",
        (
            "Management agreed that protecting gross margin, "
            "controlling discretionary expenditure, reducing "
            "slow-moving inventory and refreshing the Q4 forecast "
            "were the immediate priorities before executive review."
        ),
        ""
    ]
)

MARKDOWN_PATH.write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print("Structured corpus documents generated successfully")
print(f"CSV rows: {len(csv_rows)}")
print(f"JSON KPIs: {len(kpi_output['kpis'])}")
print(
    "Markdown discussion points: "
    f"{len(meeting['discussion_points'])}"
)
print(f"Markdown decisions: {len(meeting['decisions'])}")
print(f"Markdown actions: {len(meeting['action_items'])}")
print(f"Created: {CSV_PATH}")
print(f"Created: {JSON_PATH}")
print(f"Created: {MARKDOWN_PATH}")
