from pathlib import Path
from datetime import date
import json
import sys

BLUEPRINT_PATH = Path("data/blueprint/corpus_blueprint.json")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


with BLUEPRINT_PATH.open("r", encoding="utf-8") as file:
    data = json.load(file)

policies = data.get("finance_policies", [])
people_ids = {
    person["person_id"]
    for person in data.get("people", [])
}

if len(policies) != 10:
    fail(f"Expected 10 finance policies, found {len(policies)}")

policy_ids = [policy["policy_id"] for policy in policies]

if len(policy_ids) != len(set(policy_ids)):
    fail("Duplicate policy IDs detected")

policy_titles = [policy["title"] for policy in policies]

if len(policy_titles) != len(set(policy_titles)):
    fail("Duplicate policy titles detected")

for policy in policies:
    policy_id = policy["policy_id"]

    if policy["owner_person_id"] not in people_ids:
        fail(
            f"{policy_id}: unknown owner "
            f"{policy['owner_person_id']}"
        )

    try:
        date.fromisoformat(policy["effective_date"])
    except ValueError:
        fail(
            f"{policy_id}: invalid effective date "
            f"{policy['effective_date']}"
        )

    rules = policy.get("rules", [])
    approval_levels = policy.get("approval_levels", [])
    additional_rules = policy.get("additional_rules", [])

    if not rules and not approval_levels and not additional_rules:
        fail(f"{policy_id}: policy has no rules")

    for level in approval_levels:
        minimum = level["minimum_aud"]
        maximum = level["maximum_aud"]
        approvers = level["required_approvers"]

        if minimum < 0:
            fail(f"{policy_id}: negative approval minimum")

        if maximum is not None and maximum < minimum:
            fail(
                f"{policy_id}: maximum approval value is "
                f"below the minimum"
            )

        if not approvers:
            fail(f"{policy_id}: approval level has no approvers")

opex_policy = next(
    policy
    for policy in policies
    if policy["policy_id"] == "POL_004"
)

opex_highest = opex_policy["approval_levels"][-1]

if opex_highest["maximum_aud"] is not None:
    fail("POL_004: highest operating expenditure level must be open-ended")

if set(opex_highest["required_approvers"]) != {
    "Finance Director",
    "Chief Executive Officer"
}:
    fail("POL_004: highest approval level has incorrect approvers")

capex_policy = next(
    policy
    for policy in policies
    if policy["policy_id"] == "POL_006"
)

capex_highest = capex_policy["approval_levels"][-1]

if capex_highest["minimum_aud"] != 250001:
    fail("POL_006: Board approval threshold should begin at AUD 250,001")

if "Board of Directors" not in capex_highest["required_approvers"]:
    fail("POL_006: highest capex level must require Board approval")

print("[PASS] Finance policies loaded successfully")
print(f"[PASS] Policies: {len(policies)}")
print("[PASS] Policy IDs and titles are unique")
print("[PASS] All policy owners exist")
print("[PASS] Effective dates are valid")
print("[PASS] Approval levels are correctly structured")
print("[PASS] Operating expenditure approvals are consistent")
print("[PASS] Capital expenditure approvals are consistent")
