from pathlib import Path
import json

BLUEPRINT_PATH = Path("data/blueprint/corpus_blueprint.json")

with BLUEPRINT_PATH.open("r", encoding="utf-8") as file:
    data = json.load(file)

data["accounting_conventions"]["percentage_format"] = (
    "Rates are stored as decimal fractions. "
    "For example, 0.424 represents 42.4%."
)

data["people"] = [
    {
        "person_id": "PER_001",
        "name": "Amelia Hart",
        "title": "Chief Executive Officer",
        "department": "Executive"
    },
    {
        "person_id": "PER_002",
        "name": "Olivia Bennett",
        "title": "Finance Director",
        "department": "Finance"
    },
    {
        "person_id": "PER_003",
        "name": "Daniel Wu",
        "title": "Head of FP&A",
        "department": "Finance"
    },
    {
        "person_id": "PER_004",
        "name": "Priya Nair",
        "title": "Financial Controller",
        "department": "Finance"
    },
    {
        "person_id": "PER_005",
        "name": "James Carter",
        "title": "Retail Operations Director",
        "department": "Retail Operations"
    },
    {
        "person_id": "PER_006",
        "name": "Mia Thompson",
        "title": "E-commerce Director",
        "department": "E-commerce"
    },
    {
        "person_id": "PER_007",
        "name": "Sarah Mitchell",
        "title": "Marketing Director",
        "department": "Marketing"
    },
    {
        "person_id": "PER_008",
        "name": "Liam O'Connor",
        "title": "Supply Chain Director",
        "department": "Supply Chain"
    },
    {
        "person_id": "PER_009",
        "name": "Ethan Brooks",
        "title": "Information Technology Director",
        "department": "Information Technology"
    },
    {
        "person_id": "PER_010",
        "name": "Grace Lee",
        "title": "People & Culture Director",
        "department": "People & Culture"
    }
]

data["headline_financials"] = [
    {
        "fact_id": "FIN_001",
        "metric": "Revenue",
        "period": "Nine months ended 31 March 2026",
        "actual": 91800000,
        "budget": 93000000,
        "variance": -1200000,
        "variance_pct": -0.0129,
        "unit": "AUD"
    },
    {
        "fact_id": "FIN_002",
        "metric": "Gross margin",
        "period": "Nine months ended 31 March 2026",
        "actual": 0.424,
        "budget": 0.431,
        "variance_percentage_points": -0.7,
        "unit": "percentage"
    },
    {
        "fact_id": "FIN_003",
        "metric": "Adjusted EBITDA",
        "period": "Nine months ended 31 March 2026",
        "actual": 8700000,
        "budget": 9400000,
        "variance": -700000,
        "variance_pct": -0.0745,
        "unit": "AUD"
    },
    {
        "fact_id": "FIN_004",
        "metric": "Full-year revenue forecast",
        "period": "FY2025/26",
        "forecast": 121800000,
        "budget": 124000000,
        "variance": -2200000,
        "variance_pct": -0.0177,
        "unit": "AUD"
    },
    {
        "fact_id": "FIN_005",
        "metric": "Full-year Adjusted EBITDA forecast",
        "period": "FY2025/26",
        "forecast": 11300000,
        "budget": 12200000,
        "variance": -900000,
        "variance_pct": -0.0738,
        "unit": "AUD"
    },
    {
        "fact_id": "FIN_006",
        "metric": "Store sales growth",
        "period": "Nine months ended 31 March 2026",
        "actual": 0.019,
        "unit": "percentage"
    },
    {
        "fact_id": "FIN_007",
        "metric": "E-commerce sales growth",
        "period": "Nine months ended 31 March 2026",
        "actual": 0.148,
        "unit": "percentage"
    }
]

data["variance_drivers"] = [
    {
        "fact_id": "VAR_001",
        "month": "2026-03",
        "department": "Marketing",
        "budget_aud": 620000,
        "actual_aud": 805000,
        "variance_aud": 185000,
        "variance_status": "Unfavourable",
        "primary_cause": "Unplanned digital customer acquisition campaign",
        "responsible_person_id": "PER_007",
        "referenced_documents": [
            "DOC_CSV_001",
            "DOC_MD_001",
            "DOC_PDF_001"
        ]
    },
    {
        "fact_id": "VAR_002",
        "month": "2026-03",
        "department": "Supply Chain",
        "budget_aud": 1240000,
        "actual_aud": 1390000,
        "variance_aud": 150000,
        "variance_status": "Unfavourable",
        "primary_cause": "Freight surcharges and expedited shipments",
        "responsible_person_id": "PER_008",
        "referenced_documents": [
            "DOC_CSV_001",
            "DOC_MD_001",
            "DOC_PDF_001"
        ]
    },
    {
        "fact_id": "VAR_003",
        "month": "2026-03",
        "department": "Information Technology",
        "budget_aud": 410000,
        "actual_aud": 365000,
        "variance_aud": -45000,
        "variance_status": "Favourable",
        "primary_cause": "Delay in a planned vendor implementation milestone",
        "responsible_person_id": "PER_009",
        "referenced_documents": [
            "DOC_CSV_001",
            "DOC_MD_001"
        ]
    }
]

data["business_risks"] = [
    {
        "risk_id": "RISK_001",
        "category": "Supply Chain",
        "description": "International freight costs may remain above budget during Q4.",
        "financial_impact": "Pressure on gross margin and distribution expenses",
        "owner_person_id": "PER_008",
        "severity": "High"
    },
    {
        "risk_id": "RISK_002",
        "category": "Inventory",
        "description": "Late supplier deliveries may reduce availability of selected seasonal ranges.",
        "financial_impact": "Lost sales and increased use of expedited freight",
        "owner_person_id": "PER_005",
        "severity": "Medium"
    },
    {
        "risk_id": "RISK_003",
        "category": "Marketing",
        "description": "Paid digital acquisition costs have increased faster than online conversion.",
        "financial_impact": "Lower return on marketing investment",
        "owner_person_id": "PER_007",
        "severity": "Medium"
    },
    {
        "risk_id": "RISK_004",
        "category": "People",
        "description": "Wage cost pressure may increase store operating expenses.",
        "financial_impact": "Higher labour cost percentage",
        "owner_person_id": "PER_010",
        "severity": "Medium"
    }
]

data["q4_priorities"] = [
    {
        "priority_id": "PRI_001",
        "priority": "Recover gross margin through tighter promotional controls and supplier negotiations.",
        "owner_person_ids": ["PER_002", "PER_005", "PER_008"]
    },
    {
        "priority_id": "PRI_002",
        "priority": "Reduce slow-moving inventory and improve working capital.",
        "owner_person_ids": ["PER_003", "PER_005", "PER_008"]
    },
    {
        "priority_id": "PRI_003",
        "priority": "Reallocate marketing spend toward channels with measurable conversion.",
        "owner_person_ids": ["PER_003", "PER_007"]
    },
    {
        "priority_id": "PRI_004",
        "priority": "Complete the Q4 forecast refresh and identify actions to protect Adjusted EBITDA.",
        "owner_person_ids": ["PER_002", "PER_003"]
    }
]

with BLUEPRINT_PATH.open("w", encoding="utf-8", newline="\n") as file:
    json.dump(data, file, indent=2, ensure_ascii=False)
    file.write("\n")

print("Blueprint core facts added successfully")
print(f"People: {len(data['people'])}")
print(f"Financial facts: {len(data['headline_financials'])}")
print(f"Variance drivers: {len(data['variance_drivers'])}")
print(f"Business risks: {len(data['business_risks'])}")
print(f"Q4 priorities: {len(data['q4_priorities'])}")
