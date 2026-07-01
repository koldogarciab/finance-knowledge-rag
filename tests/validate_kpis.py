from pathlib import Path
import json
import math
import sys

BLUEPRINT_PATH = Path("data/blueprint/corpus_blueprint.json")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


with BLUEPRINT_PATH.open("r", encoding="utf-8") as file:
    data = json.load(file)

kpis = data.get("kpi_dictionary", [])
people_ids = {
    person["person_id"]
    for person in data.get("people", [])
}

if len(kpis) != 13:
    fail(f"Expected 13 KPIs, found {len(kpis)}")

kpi_ids = [kpi["kpi_id"] for kpi in kpis]
kpi_names = [kpi["name"] for kpi in kpis]

if len(kpi_ids) != len(set(kpi_ids)):
    fail("Duplicate KPI IDs detected")

if len(kpi_names) != len(set(kpi_names)):
    fail("Duplicate KPI names detected")

allowed_operators = {">=", "<=", "between"}
allowed_directions = {
    "Higher",
    "Lower",
    "Context dependent",
    "Target range"
}

for kpi in kpis:
    kpi_id = kpi["kpi_id"]

    if kpi["owner_person_id"] not in people_ids:
        fail(
            f"{kpi_id}: unknown owner "
            f"{kpi['owner_person_id']}"
        )

    required_text_fields = [
        "name",
        "category",
        "definition",
        "formula",
        "frequency",
        "data_source",
        "unit"
    ]

    for field in required_text_fields:
        if not str(kpi.get(field, "")).strip():
            fail(f"{kpi_id}: missing or empty field '{field}'")

    if kpi["better_direction"] not in allowed_directions:
        fail(
            f"{kpi_id}: invalid better_direction "
            f"{kpi['better_direction']}"
        )

    target = kpi.get("target", {})
    operator = target.get("operator")

    if operator not in allowed_operators:
        fail(f"{kpi_id}: invalid target operator '{operator}'")

    if operator == "between":
        minimum = target.get("minimum")
        maximum = target.get("maximum")

        if minimum is None or maximum is None:
            fail(f"{kpi_id}: target range is incomplete")

        if minimum > maximum:
            fail(f"{kpi_id}: target minimum exceeds maximum")

    else:
        if "value" not in target:
            fail(f"{kpi_id}: target value is missing")

    if kpi["unit"] == "percentage":
        target_values = []

        if "value" in target:
            target_values.append(target["value"])

        if "minimum" in target:
            target_values.append(target["minimum"])

        if "maximum" in target:
            target_values.append(target["maximum"])

        for value in target_values:
            if not 0 <= value <= 1:
                fail(
                    f"{kpi_id}: percentage target "
                    f"{value} is outside 0 to 1"
                )

financials = {
    item["metric"]: item
    for item in data["headline_financials"]
}

revenue_kpi = next(
    kpi for kpi in kpis
    if kpi["name"] == "Revenue"
)

if revenue_kpi["target"]["value"] != financials[
    "Full-year revenue forecast"
]["budget"]:
    fail("Revenue KPI target does not match the annual revenue budget")

margin_kpi = next(
    kpi for kpi in kpis
    if kpi["name"] == "Gross margin"
)

if not math.isclose(
    margin_kpi["target"]["value"],
    financials["Gross margin"]["budget"],
    abs_tol=0.0001
):
    fail("Gross margin KPI target does not match the budget")

ebitda_kpi = next(
    kpi for kpi in kpis
    if kpi["name"] == "Adjusted EBITDA"
)

if ebitda_kpi["target"]["value"] != financials[
    "Full-year Adjusted EBITDA forecast"
]["budget"]:
    fail(
        "Adjusted EBITDA KPI target does not match "
        "the annual budget"
    )

print("[PASS] KPI dictionary loaded successfully")
print(f"[PASS] KPIs: {len(kpis)}")
print("[PASS] KPI IDs and names are unique")
print("[PASS] All KPI owners exist")
print("[PASS] Definitions, formulas and data sources are complete")
print("[PASS] Target structures and percentage values are valid")
print("[PASS] Revenue target matches the financial blueprint")
print("[PASS] Gross margin target matches the financial blueprint")
print("[PASS] Adjusted EBITDA target matches the financial blueprint")
