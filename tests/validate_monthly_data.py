from collections import Counter, defaultdict
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

monthly_data = data.get("monthly_budget_actual")

if not monthly_data:
    fail("Monthly budget vs actual section is missing")

rows = monthly_data.get("rows", [])

expected_months = {
    "2025-07",
    "2025-08",
    "2025-09",
    "2025-10",
    "2025-11",
    "2025-12",
    "2026-01",
    "2026-02",
    "2026-03"
}

expected_departments = set(data["departments"])

people_by_id = {
    person["person_id"]: person
    for person in data["people"]
}

if len(rows) != 189:
    fail(f"Expected 189 rows, found {len(rows)}")

if monthly_data["row_count"] != len(rows):
    fail(
        "Stored row_count does not match "
        f"the number of rows: {monthly_data['row_count']} "
        f"vs {len(rows)}"
    )

row_ids = [row["row_id"] for row in rows]

if len(row_ids) != len(set(row_ids)):
    fail("Duplicate row IDs detected")

actual_months = {row["month"] for row in rows}
actual_departments = {row["department"] for row in rows}

if actual_months != expected_months:
    fail(
        f"Unexpected months. Expected {expected_months}, "
        f"found {actual_months}"
    )

if actual_departments != expected_departments:
    fail(
        f"Unexpected departments. Expected {expected_departments}, "
        f"found {actual_departments}"
    )

rows_per_group = Counter(
    (row["month"], row["department"])
    for row in rows
)

for month in expected_months:
    for department in expected_departments:
        count = rows_per_group[(month, department)]

        if count != 3:
            fail(
                f"{month} / {department}: expected 3 rows, "
                f"found {count}"
            )

for row in rows:
    row_id = row["row_id"]

    expected_variance = (
        row["actual_aud"] - row["budget_aud"]
    )

    if row["variance_aud"] != expected_variance:
        fail(
            f"{row_id}: variance should be "
            f"{expected_variance}, found {row['variance_aud']}"
        )

    expected_forecast_variance = (
        row["actual_aud"] - row["forecast_aud"]
    )

    if (
        row["forecast_variance_aud"]
        != expected_forecast_variance
    ):
        fail(
            f"{row_id}: forecast variance should be "
            f"{expected_forecast_variance}, found "
            f"{row['forecast_variance_aud']}"
        )

    expected_pct = (
        round(expected_variance / row["budget_aud"], 4)
        if row["budget_aud"] != 0
        else 0
    )

    if not math.isclose(
        row["variance_pct"],
        expected_pct,
        abs_tol=0.0001
    ):
        fail(
            f"{row_id}: variance percentage should be "
            f"{expected_pct}, found {row['variance_pct']}"
        )

    expected_status = (
        "Unfavourable"
        if expected_variance > 0
        else "Favourable"
        if expected_variance < 0
        else "On budget"
    )

    if row["variance_status"] != expected_status:
        fail(
            f"{row_id}: status should be "
            f"{expected_status}, found "
            f"{row['variance_status']}"
        )

    person_id = row["responsible_person_id"]

    if person_id not in people_by_id:
        fail(
            f"{row_id}: unknown responsible person "
            f"{person_id}"
        )

    expected_manager = people_by_id[person_id]["name"]

    if row["responsible_manager"] != expected_manager:
        fail(
            f"{row_id}: manager name should be "
            f"{expected_manager}, found "
            f"{row['responsible_manager']}"
        )

    if row["period_status"] != "Closed":
        fail(
            f"{row_id}: expected period status Closed, "
            f"found {row['period_status']}"
        )

march_totals = defaultdict(
    lambda: {
        "budget_aud": 0,
        "actual_aud": 0,
        "variance_aud": 0
    }
)

for row in rows:
    if row["month"] == "2026-03":
        department = row["department"]

        march_totals[department]["budget_aud"] += (
            row["budget_aud"]
        )
        march_totals[department]["actual_aud"] += (
            row["actual_aud"]
        )
        march_totals[department]["variance_aud"] += (
            row["variance_aud"]
        )

variance_drivers = {
    item["department"]: item
    for item in data["variance_drivers"]
    if item["month"] == "2026-03"
}

for department in [
    "Marketing",
    "Supply Chain",
    "Information Technology"
]:
    csv_values = march_totals[department]
    source_values = variance_drivers[department]

    for field in [
        "budget_aud",
        "actual_aud",
        "variance_aud"
    ]:
        if csv_values[field] != source_values[field]:
            fail(
                f"{department}: March {field} does not "
                "match the variance driver"
            )

march_budget = sum(
    values["budget_aud"]
    for values in march_totals.values()
)

march_actual = sum(
    values["actual_aud"]
    for values in march_totals.values()
)

if march_budget != 6240000:
    fail(
        f"March budget should be 6240000, "
        f"found {march_budget}"
    )

if march_actual != 6633000:
    fail(
        f"March actual should be 6633000, "
        f"found {march_actual}"
    )

if march_actual - march_budget != 393000:
    fail("March total variance should be 393000")

print("[PASS] Monthly budget vs actual data loaded successfully")
print(f"[PASS] Rows: {len(rows)}")
print(f"[PASS] Months: {len(actual_months)}")
print(f"[PASS] Departments: {len(actual_departments)}")
print("[PASS] Every month and department has 3 categories")
print("[PASS] Row IDs are unique")
print("[PASS] Variance calculations and statuses are correct")
print("[PASS] Forecast variance calculations are correct")
print("[PASS] All responsible managers are consistent")
print("[PASS] March Marketing figures match the blueprint")
print("[PASS] March Supply Chain figures match the blueprint")
print("[PASS] March Information Technology figures match the blueprint")
print("[PASS] March consolidated totals are correct")
