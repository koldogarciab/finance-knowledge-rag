from pathlib import Path
import json

BLUEPRINT_PATH = Path("data/blueprint/corpus_blueprint.json")

with BLUEPRINT_PATH.open("r", encoding="utf-8") as file:
    data = json.load(file)

data["finance_policies"] = [
    {
        "policy_id": "POL_001",
        "section": "Month-end close",
        "title": "Month-end close timetable",
        "effective_date": "2025-07-01",
        "owner_person_id": "PER_004",
        "rules": [
            "Departmental accrual submissions are due by 12:00 noon on business day 2.",
            "Standard and recurring journals must be posted by the end of business day 3.",
            "The general ledger must be closed by 17:00 on business day 5.",
            "Late journals require approval from the Financial Controller."
        ]
    },
    {
        "policy_id": "POL_002",
        "section": "Accruals",
        "title": "Accrual recognition and review",
        "effective_date": "2025-07-01",
        "owner_person_id": "PER_004",
        "rules": [
            "An accrual is required when goods or services have been received but the supplier invoice has not been recorded.",
            "The minimum accrual threshold is AUD 5,000 per individual item.",
            "Accruals must be supported by a purchase order, contract, supplier estimate or other reasonable calculation.",
            "Accruals must be reviewed each month and reversed when the related invoice is recorded."
        ]
    },
    {
        "policy_id": "POL_003",
        "section": "Prepayments",
        "title": "Prepayment recognition",
        "effective_date": "2025-07-01",
        "owner_person_id": "PER_004",
        "rules": [
            "A payment must be recorded as a prepayment when it is at least AUD 12,000 and the benefit extends for more than three months.",
            "Prepayments must be amortised monthly over the period in which the economic benefit is received.",
            "The supporting schedule must show the supplier, total amount, service period and monthly release."
        ]
    },
    {
        "policy_id": "POL_004",
        "section": "Operating expenditure",
        "title": "Operating expenditure approval",
        "effective_date": "2025-07-01",
        "owner_person_id": "PER_002",
        "approval_levels": [
            {
                "minimum_aud": 0,
                "maximum_aud": 5000,
                "required_approvers": [
                    "Department Manager"
                ]
            },
            {
                "minimum_aud": 5001,
                "maximum_aud": 25000,
                "required_approvers": [
                    "Department Director"
                ]
            },
            {
                "minimum_aud": 25001,
                "maximum_aud": 100000,
                "required_approvers": [
                    "Department Director",
                    "Finance Director"
                ]
            },
            {
                "minimum_aud": 100001,
                "maximum_aud": None,
                "required_approvers": [
                    "Finance Director",
                    "Chief Executive Officer"
                ]
            }
        ]
    },
    {
        "policy_id": "POL_005",
        "section": "Procurement",
        "title": "Purchase order requirements",
        "effective_date": "2025-07-01",
        "owner_person_id": "PER_002",
        "rules": [
            "A purchase order must be approved before committing Harbour Retail Group to goods or services above AUD 5,000.",
            "Splitting a purchase into smaller amounts to avoid the threshold is prohibited.",
            "Payroll, taxes, rent, regulated utilities and emergency expenditure approved by the Finance Director are exempt.",
            "Invoices without a valid purchase order may be returned to the requesting department."
        ]
    },
    {
        "policy_id": "POL_006",
        "section": "Capital expenditure",
        "title": "Capital expenditure approval",
        "effective_date": "2025-07-01",
        "owner_person_id": "PER_002",
        "approval_levels": [
            {
                "minimum_aud": 0,
                "maximum_aud": 50000,
                "required_approvers": [
                    "Department Director",
                    "Finance Director"
                ]
            },
            {
                "minimum_aud": 50001,
                "maximum_aud": 250000,
                "required_approvers": [
                    "Finance Director",
                    "Chief Executive Officer"
                ]
            },
            {
                "minimum_aud": 250001,
                "maximum_aud": None,
                "required_approvers": [
                    "Finance Director",
                    "Chief Executive Officer",
                    "Board of Directors"
                ]
            }
        ],
        "additional_rules": [
            "Capital expenditure above AUD 50,000 requires a documented business case.",
            "The business case must include expected benefits, implementation cost, timing, risks and financial return.",
            "Projects may not be divided into smaller components to avoid an approval threshold."
        ]
    },
    {
        "policy_id": "POL_007",
        "section": "Manual journals",
        "title": "Manual journal preparation and approval",
        "effective_date": "2025-07-01",
        "owner_person_id": "PER_004",
        "rules": [
            "Every manual journal must include a clear description, calculation and supporting documentation.",
            "The preparer and approver must be different people.",
            "A journal preparer may not approve their own journal.",
            "Journals above AUD 100,000 require approval from the Financial Controller."
        ]
    },
    {
        "policy_id": "POL_008",
        "section": "Balance sheet reconciliations",
        "title": "Balance sheet reconciliation timetable",
        "effective_date": "2025-07-01",
        "owner_person_id": "PER_004",
        "rules": [
            "All balance sheet accounts must be reconciled monthly.",
            "Preparers must complete reconciliations by business day 7.",
            "Reviewers must complete their review by business day 10.",
            "Reconciling items older than 60 days must include an owner and a documented resolution date."
        ]
    },
    {
        "policy_id": "POL_009",
        "section": "Document retention",
        "title": "Finance document retention",
        "effective_date": "2025-07-01",
        "owner_person_id": "PER_004",
        "rules": [
            "Accounting records and supporting documents must be retained for seven years.",
            "Electronic records must be stored in an approved company system.",
            "Documents subject to an audit, investigation or legal hold must not be destroyed."
        ]
    },
    {
        "policy_id": "POL_010",
        "section": "Forecasting",
        "title": "Departmental forecast submissions",
        "effective_date": "2025-07-01",
        "owner_person_id": "PER_003",
        "rules": [
            "Department Directors are responsible for the completeness and reasonableness of their forecasts.",
            "Forecast submissions must explain material changes to revenue, expenditure, headcount and capital expenditure.",
            "Material forecast movements above AUD 100,000 require written commentary.",
            "The Head of FP&A consolidates and challenges departmental submissions before executive review."
        ]
    }
]

with BLUEPRINT_PATH.open("w", encoding="utf-8", newline="\n") as file:
    json.dump(data, file, indent=2, ensure_ascii=False)
    file.write("\n")

print("Finance policies added successfully")
print(f"Policies: {len(data['finance_policies'])}")

