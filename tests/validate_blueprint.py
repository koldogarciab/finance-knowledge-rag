from pathlib import Path
import json
import math
import sys

BLUEPRINT_PATH = Path("data/blueprint/corpus_blueprint.json")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


def check_unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        fail(f"Duplicate {label} detected")


with BLUEPRINT_PATH.open("r", encoding="utf-8") as file:
    data = json.load(file)

documents = data["documents"]
people = data["people"]
financials = data["headline_financials"]
variance_drivers = data["variance_drivers"]
risks = data["business_risks"]
priorities = data["q4_priorities"]

document_ids = {item["document_id"] for item in documents}
person_ids = {item["person_id"] for item in people}

check_unique(
    [item["document_id"] for item in documents],
    "document IDs"
)
check_unique(
    [item["person_id"] for item in people],
    "person IDs"
)
check_unique(
    [item["fact_id"] for item in financials],
    "financial fact IDs"
)
check_unique(
    [item["fact_id"] for item in variance_drivers],
    "variance fact IDs"
)
check_unique(
    [item["risk_id"] for item in risks],
    "risk IDs"
)
check_unique(
    [item["priority_id"] for item in priorities],
    "priority IDs"
)

expected_formats = {"pdf", "docx", "csv", "json", "markdown"}
actual_formats = {item["file_type"] for item in documents}

if len(documents) != 5:
    fail(f"Expected 5 corpus documents, found {len(documents)}")

if actual_formats != expected_formats:
    fail(
        f"Document formats do not match. "
        f"Expected {expected_formats}, found {actual_formats}"
    )

for item in variance_drivers:
    expected_variance = item["actual_aud"] - item["budget_aud"]

    if item["variance_aud"] != expected_variance:
        fail(
            f"{item['fact_id']}: variance should be "
            f"{expected_variance}, found {item['variance_aud']}"
        )

    expected_status = (
        "Unfavourable"
        if item["variance_aud"] > 0
        else "Favourable"
        if item["variance_aud"] < 0
        else "On budget"
    )

    if item["variance_status"] != expected_status:
        fail(
            f"{item['fact_id']}: status should be "
            f"{expected_status}, found {item['variance_status']}"
        )

    if item["responsible_person_id"] not in person_ids:
        fail(
            f"{item['fact_id']}: unknown responsible person "
            f"{item['responsible_person_id']}"
        )

    for document_id in item["referenced_documents"]:
        if document_id not in document_ids:
            fail(
                f"{item['fact_id']}: unknown referenced document "
                f"{document_id}"
            )

for item in risks:
    if item["owner_person_id"] not in person_ids:
        fail(
            f"{item['risk_id']}: unknown owner "
            f"{item['owner_person_id']}"
        )

for item in priorities:
    for person_id in item["owner_person_ids"]:
        if person_id not in person_ids:
            fail(
                f"{item['priority_id']}: unknown owner {person_id}"
            )

for item in financials:
    if {"actual", "budget", "variance"}.issubset(item):
        expected_variance = item["actual"] - item["budget"]

        if item["variance"] != expected_variance:
            fail(
                f"{item['fact_id']}: financial variance should be "
                f"{expected_variance}, found {item['variance']}"
            )

    if {"forecast", "budget", "variance"}.issubset(item):
        expected_variance = item["forecast"] - item["budget"]

        if item["variance"] != expected_variance:
            fail(
                f"{item['fact_id']}: forecast variance should be "
                f"{expected_variance}, found {item['variance']}"
            )

    if (
        "variance_pct" in item
        and "budget" in item
        and item["budget"] != 0
    ):
        numerator = item.get("variance")

        if numerator is not None:
            expected_pct = numerator / item["budget"]

            if not math.isclose(
                item["variance_pct"],
                expected_pct,
                abs_tol=0.0001
            ):
                fail(
                    f"{item['fact_id']}: percentage variance does not "
                    f"agree with the underlying values"
                )

print("[PASS] Blueprint JSON loaded successfully")
print(f"[PASS] Documents: {len(documents)}")
print(f"[PASS] Formats: {', '.join(sorted(actual_formats))}")
print(f"[PASS] People: {len(people)}")
print(f"[PASS] Financial facts: {len(financials)}")
print(f"[PASS] Variance drivers: {len(variance_drivers)}")
print(f"[PASS] Business risks: {len(risks)}")
print(f"[PASS] Q4 priorities: {len(priorities)}")
print("[PASS] All references and financial calculations are consistent")
