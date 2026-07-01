from pathlib import Path
import json

BLUEPRINT_PATH = Path("data/blueprint/corpus_blueprint.json")

with BLUEPRINT_PATH.open("r", encoding="utf-8") as file:
    data = json.load(file)

data["forecast_meeting"] = {
    "meeting_id": "MTG_001",
    "title": "FP&A Forecast Meeting",
    "date": "2026-04-10",
    "start_time": "09:30",
    "end_time": "11:00",
    "location": "Sydney Head Office — Boardroom 2",
    "chair_person_id": "PER_003",
    "minute_taker_person_id": "PER_004",
    "attendee_person_ids": [
        "PER_002",
        "PER_003",
        "PER_004",
        "PER_005",
        "PER_006",
        "PER_007",
        "PER_008",
        "PER_009",
        "PER_010"
    ],
    "purpose": (
        "Review March performance, confirm the FY2025/26 outlook "
        "and agree actions to protect Q4 profitability and cash flow."
    ),
    "forecast_summary": [
        {
            "metric": "Full-year revenue forecast",
            "forecast_aud": 121800000,
            "budget_aud": 124000000,
            "variance_aud": -2200000,
            "status": "Below budget"
        },
        {
            "metric": "Full-year Adjusted EBITDA forecast",
            "forecast_aud": 11300000,
            "budget_aud": 12200000,
            "variance_aud": -900000,
            "status": "Below budget"
        }
    ],
    "discussion_points": [
        {
            "discussion_id": "DISC_001",
            "topic": "Marketing expenditure",
            "summary": (
                "March Marketing expenditure was AUD 185,000 above "
                "budget because of an unplanned digital customer "
                "acquisition campaign."
            ),
            "linked_fact_ids": ["VAR_001"],
            "lead_person_id": "PER_007"
        },
        {
            "discussion_id": "DISC_002",
            "topic": "Supply Chain expenditure",
            "summary": (
                "March Supply Chain expenditure was AUD 150,000 above "
                "budget because of freight surcharges and expedited "
                "shipments."
            ),
            "linked_fact_ids": ["VAR_002", "RISK_001", "RISK_002"],
            "lead_person_id": "PER_008"
        },
        {
            "discussion_id": "DISC_003",
            "topic": "Information Technology expenditure",
            "summary": (
                "March Information Technology expenditure was "
                "AUD 45,000 below budget because a planned vendor "
                "implementation milestone was delayed."
            ),
            "linked_fact_ids": ["VAR_003"],
            "lead_person_id": "PER_009"
        },
        {
            "discussion_id": "DISC_004",
            "topic": "E-commerce performance",
            "summary": (
                "E-commerce sales grew by 14.8%, but paid acquisition "
                "costs increased faster than online conversion."
            ),
            "linked_fact_ids": ["FIN_007", "RISK_003"],
            "lead_person_id": "PER_006"
        },
        {
            "discussion_id": "DISC_005",
            "topic": "Headcount assumptions",
            "summary": (
                "Non-critical recruitment will be deferred until the "
                "Q4 forecast refresh confirms sufficient funding."
            ),
            "linked_fact_ids": ["RISK_004"],
            "lead_person_id": "PER_010"
        }
    ],
    "decisions": [
        {
            "decision_id": "DEC_001",
            "decision": (
                "No additional unplanned Marketing campaigns may begin "
                "without Finance Director approval."
            ),
            "owner_person_id": "PER_007",
            "linked_priority_ids": ["PRI_003"]
        },
        {
            "decision_id": "DEC_002",
            "decision": (
                "The Q4 freight forecast will be updated using current "
                "surcharge rates and revised shipment assumptions."
            ),
            "owner_person_id": "PER_008",
            "linked_priority_ids": ["PRI_001"]
        },
        {
            "decision_id": "DEC_003",
            "decision": (
                "The delayed IT implementation will remain in the "
                "forecast, but timing and cash flow assumptions must "
                "be refreshed."
            ),
            "owner_person_id": "PER_009",
            "linked_priority_ids": ["PRI_004"]
        },
        {
            "decision_id": "DEC_004",
            "decision": (
                "Department Directors must identify additional Q4 "
                "savings before the executive forecast review."
            ),
            "owner_person_id": "PER_003",
            "linked_priority_ids": ["PRI_004"]
        },
        {
            "decision_id": "DEC_005",
            "decision": (
                "Non-critical recruitment will be paused until the "
                "Q4 forecast refresh is approved."
            ),
            "owner_person_id": "PER_010",
            "linked_priority_ids": ["PRI_004"]
        }
    ],
    "action_items": [
        {
            "action_id": "ACT_001",
            "action": (
                "Prepare a revised Marketing channel allocation showing "
                "expected spend, conversion and customer acquisition cost."
            ),
            "owner_person_id": "PER_007",
            "deadline": "2026-04-17",
            "status": "Open",
            "linked_decision_ids": ["DEC_001"],
            "linked_fact_ids": ["VAR_001", "RISK_003"]
        },
        {
            "action_id": "ACT_002",
            "action": (
                "Update the Q4 logistics forecast using current freight "
                "surcharges and revised shipment volumes."
            ),
            "owner_person_id": "PER_008",
            "deadline": "2026-04-15",
            "status": "Open",
            "linked_decision_ids": ["DEC_002"],
            "linked_fact_ids": ["VAR_002", "RISK_001"]
        },
        {
            "action_id": "ACT_003",
            "action": (
                "Refresh the IT implementation business case, timeline "
                "and cash flow profile."
            ),
            "owner_person_id": "PER_009",
            "deadline": "2026-04-24",
            "status": "Open",
            "linked_decision_ids": ["DEC_003"],
            "linked_fact_ids": ["VAR_003"]
        },
        {
            "action_id": "ACT_004",
            "action": (
                "Consolidate departmental savings and issue the updated "
                "Q4 forecast for executive review."
            ),
            "owner_person_id": "PER_003",
            "deadline": "2026-04-27",
            "status": "Open",
            "linked_decision_ids": ["DEC_004"],
            "linked_fact_ids": ["FIN_004", "FIN_005"]
        },
        {
            "action_id": "ACT_005",
            "action": (
                "Prepare a slow-moving inventory reduction plan with "
                "category-level targets."
            ),
            "owner_person_id": "PER_005",
            "deadline": "2026-04-22",
            "status": "Open",
            "linked_decision_ids": ["DEC_004"],
            "linked_fact_ids": ["RISK_002"]
        },
        {
            "action_id": "ACT_006",
            "action": (
                "Provide an updated list of vacancies classified as "
                "critical or non-critical."
            ),
            "owner_person_id": "PER_010",
            "deadline": "2026-04-16",
            "status": "Open",
            "linked_decision_ids": ["DEC_005"],
            "linked_fact_ids": ["RISK_004"]
        }
    ]
}

with BLUEPRINT_PATH.open("w", encoding="utf-8", newline="\n") as file:
    json.dump(data, file, indent=2, ensure_ascii=False)
    file.write("\n")

meeting = data["forecast_meeting"]

print("Forecast meeting facts added successfully")
print(f"Discussion points: {len(meeting['discussion_points'])}")
print(f"Decisions: {len(meeting['decisions'])}")
print(f"Action items: {len(meeting['action_items'])}")
