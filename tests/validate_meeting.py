from datetime import date, datetime
from pathlib import Path
import json
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

meeting = data.get("forecast_meeting")

if not meeting:
    fail("Forecast meeting section is missing")

people_ids = {
    person["person_id"]
    for person in data.get("people", [])
}

financial_fact_ids = {
    item["fact_id"]
    for item in data.get("headline_financials", [])
}

variance_fact_ids = {
    item["fact_id"]
    for item in data.get("variance_drivers", [])
}

risk_ids = {
    item["risk_id"]
    for item in data.get("business_risks", [])
}

priority_ids = {
    item["priority_id"]
    for item in data.get("q4_priorities", [])
}

valid_fact_ids = (
    financial_fact_ids
    | variance_fact_ids
    | risk_ids
)

attendee_ids = set(meeting.get("attendee_person_ids", []))

for person_id in attendee_ids:
    if person_id not in people_ids:
        fail(f"Unknown attendee: {person_id}")

if meeting["chair_person_id"] not in people_ids:
    fail("Meeting chair does not exist")

if meeting["minute_taker_person_id"] not in people_ids:
    fail("Minute taker does not exist")

if meeting["chair_person_id"] not in attendee_ids:
    fail("Meeting chair is not included in attendees")

if meeting["minute_taker_person_id"] not in attendee_ids:
    fail("Minute taker is not included in attendees")

try:
    meeting_date = date.fromisoformat(meeting["date"])
    start_time = datetime.strptime(
        meeting["start_time"],
        "%H:%M"
    ).time()
    end_time = datetime.strptime(
        meeting["end_time"],
        "%H:%M"
    ).time()
except ValueError as error:
    fail(f"Invalid meeting date or time: {error}")

if end_time <= start_time:
    fail("Meeting end time must be later than start time")

discussion_points = meeting.get("discussion_points", [])
decisions = meeting.get("decisions", [])
action_items = meeting.get("action_items", [])
forecast_summary = meeting.get("forecast_summary", [])

if len(discussion_points) != 5:
    fail(
        f"Expected 5 discussion points, "
        f"found {len(discussion_points)}"
    )

if len(decisions) != 5:
    fail(f"Expected 5 decisions, found {len(decisions)}")

if len(action_items) != 6:
    fail(f"Expected 6 action items, found {len(action_items)}")

check_unique(
    [item["discussion_id"] for item in discussion_points],
    "discussion IDs"
)

check_unique(
    [item["decision_id"] for item in decisions],
    "decision IDs"
)

check_unique(
    [item["action_id"] for item in action_items],
    "action IDs"
)

decision_ids = {
    item["decision_id"]
    for item in decisions
}

for item in discussion_points:
    discussion_id = item["discussion_id"]

    if item["lead_person_id"] not in people_ids:
        fail(
            f"{discussion_id}: unknown discussion lead "
            f"{item['lead_person_id']}"
        )

    if item["lead_person_id"] not in attendee_ids:
        fail(
            f"{discussion_id}: discussion lead is not "
            "included in attendees"
        )

    for fact_id in item.get("linked_fact_ids", []):
        if fact_id not in valid_fact_ids:
            fail(
                f"{discussion_id}: unknown linked fact "
                f"{fact_id}"
            )

for item in decisions:
    decision_id = item["decision_id"]

    if item["owner_person_id"] not in people_ids:
        fail(
            f"{decision_id}: unknown decision owner "
            f"{item['owner_person_id']}"
        )

    if item["owner_person_id"] not in attendee_ids:
        fail(
            f"{decision_id}: decision owner is not "
            "included in attendees"
        )

    for priority_id in item.get("linked_priority_ids", []):
        if priority_id not in priority_ids:
            fail(
                f"{decision_id}: unknown linked priority "
                f"{priority_id}"
            )

for item in action_items:
    action_id = item["action_id"]

    if item["owner_person_id"] not in people_ids:
        fail(
            f"{action_id}: unknown action owner "
            f"{item['owner_person_id']}"
        )

    if item["owner_person_id"] not in attendee_ids:
        fail(
            f"{action_id}: action owner is not "
            "included in attendees"
        )

    try:
        deadline = date.fromisoformat(item["deadline"])
    except ValueError:
        fail(
            f"{action_id}: invalid deadline "
            f"{item['deadline']}"
        )

    if deadline <= meeting_date:
        fail(
            f"{action_id}: deadline must be after "
            "the meeting date"
        )

    if item["status"] not in {"Open", "Completed", "Cancelled"}:
        fail(
            f"{action_id}: invalid status "
            f"{item['status']}"
        )

    for decision_id in item.get("linked_decision_ids", []):
        if decision_id not in decision_ids:
            fail(
                f"{action_id}: unknown linked decision "
                f"{decision_id}"
            )

    for fact_id in item.get("linked_fact_ids", []):
        if fact_id not in valid_fact_ids:
            fail(
                f"{action_id}: unknown linked fact "
                f"{fact_id}"
            )

financials_by_metric = {
    item["metric"]: item
    for item in data["headline_financials"]
}

if len(forecast_summary) != 2:
    fail(
        f"Expected 2 forecast summary metrics, "
        f"found {len(forecast_summary)}"
    )

for item in forecast_summary:
    metric = item["metric"]

    if metric not in financials_by_metric:
        fail(f"Unknown forecast summary metric: {metric}")

    source = financials_by_metric[metric]

    if item["forecast_aud"] != source["forecast"]:
        fail(f"{metric}: forecast does not match blueprint")

    if item["budget_aud"] != source["budget"]:
        fail(f"{metric}: budget does not match blueprint")

    expected_variance = (
        item["forecast_aud"] - item["budget_aud"]
    )

    if item["variance_aud"] != expected_variance:
        fail(f"{metric}: variance calculation is incorrect")

    expected_status = (
        "Below budget"
        if expected_variance < 0
        else "Above budget"
        if expected_variance > 0
        else "On budget"
    )

    if item["status"] != expected_status:
        fail(
            f"{metric}: expected status "
            f"{expected_status}, found {item['status']}"
        )

print("[PASS] Forecast meeting loaded successfully")
print(f"[PASS] Attendees: {len(attendee_ids)}")
print("[PASS] Meeting date and times are valid")
print(f"[PASS] Discussion points: {len(discussion_points)}")
print(f"[PASS] Decisions: {len(decisions)}")
print(f"[PASS] Action items: {len(action_items)}")
print("[PASS] All people, facts and priorities exist")
print("[PASS] All action deadlines follow the meeting date")
print("[PASS] Forecast summary matches the financial blueprint")
