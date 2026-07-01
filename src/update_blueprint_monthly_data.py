from pathlib import Path
import json

BLUEPRINT_PATH = Path("data/blueprint/corpus_blueprint.json")

with BLUEPRINT_PATH.open("r", encoding="utf-8") as file:
    data = json.load(file)


def allocate_total(total: int, weights: list[float]) -> list[int]:
    """Allocate an integer total while preserving the exact sum."""
    allocated = [round(total * weight) for weight in weights[:-1]]
    allocated.append(total - sum(allocated))
    return allocated


months = [
    "2025-07",
    "2025-08",
    "2025-09",
    "2025-10",
    "2025-11",
    "2025-12",
    "2026-01",
    "2026-02",
    "2026-03"
]

department_config = {
    "Retail Operations": {
        "cost_centre": "RET100",
        "responsible_person_id": "PER_005",
        "responsible_manager": "James Carter",
        "categories": [
            "Store labour",
            "Occupancy",
            "Repairs and maintenance"
        ],
        "weights": [0.55, 0.30, 0.15]
    },
    "E-commerce": {
        "cost_centre": "ECOM200",
        "responsible_person_id": "PER_006",
        "responsible_manager": "Mia Thompson",
        "categories": [
            "Platform fees",
            "Online fulfilment",
            "Digital operations"
        ],
        "weights": [0.30, 0.45, 0.25]
    },
    "Marketing": {
        "cost_centre": "MKT300",
        "responsible_person_id": "PER_007",
        "responsible_manager": "Sarah Mitchell",
        "categories": [
            "Digital advertising",
            "Brand campaigns",
            "Agency and creative"
        ],
        "weights": [0.50, 0.30, 0.20]
    },
    "Supply Chain": {
        "cost_centre": "SC400",
        "responsible_person_id": "PER_008",
        "responsible_manager": "Liam O'Connor",
        "categories": [
            "Freight",
            "Warehousing",
            "Distribution labour"
        ],
        "weights": [0.50, 0.30, 0.20]
    },
    "Information Technology": {
        "cost_centre": "IT500",
        "responsible_person_id": "PER_009",
        "responsible_manager": "Ethan Brooks",
        "categories": [
            "Software and licences",
            "Infrastructure",
            "Technology projects"
        ],
        "weights": [0.40, 0.30, 0.30]
    },
    "Finance": {
        "cost_centre": "FIN600",
        "responsible_person_id": "PER_004",
        "responsible_manager": "Priya Nair",
        "categories": [
            "Finance salaries",
            "Audit and tax",
            "Shared finance services"
        ],
        "weights": [0.60, 0.20, 0.20]
    },
    "People & Culture": {
        "cost_centre": "PC700",
        "responsible_person_id": "PER_010",
        "responsible_manager": "Grace Lee",
        "categories": [
            "People team salaries",
            "Recruitment",
            "Training and wellbeing"
        ],
        "weights": [0.65, 0.20, 0.15]
    }
}

monthly_budgets = {
    "Retail Operations": [
        2150000, 2120000, 2180000, 2200000, 2350000,
        2700000, 2250000, 2200000, 2280000
    ],
    "E-commerce": [
        620000, 640000, 660000, 680000, 780000,
        950000, 690000, 680000, 710000
    ],
    "Marketing": [
        480000, 500000, 520000, 550000, 650000,
        700000, 500000, 540000, 620000
    ],
    "Supply Chain": [
        1050000, 1060000, 1100000, 1130000, 1280000,
        1420000, 1150000, 1180000, 1240000
    ],
    "Information Technology": [
        380000, 390000, 400000, 410000, 420000,
        430000, 400000, 410000, 410000
    ],
    "Finance": [
        430000, 440000, 450000, 460000, 470000,
        480000, 450000, 450000, 460000
    ],
    "People & Culture": [
        480000, 490000, 500000, 510000, 520000,
        540000, 510000, 510000, 520000
    ]
}

variance_rates = {
    "Retail Operations": [
        0.010, -0.005, 0.008, 0.015, 0.020,
        0.030, -0.010, 0.010
    ],
    "E-commerce": [
        -0.010, 0.005, 0.012, 0.020, 0.025,
        0.035, -0.005, 0.010
    ],
    "Marketing": [
        -0.020, 0.010, 0.015, 0.030, 0.040,
        0.050, 0.000, 0.020
    ],
    "Supply Chain": [
        0.010, 0.020, 0.025, 0.030, 0.040,
        0.060, 0.015, 0.025
    ],
    "Information Technology": [
        -0.030, -0.020, 0.000, 0.010, 0.020,
        -0.010, -0.020, -0.030
    ],
    "Finance": [
        -0.010, -0.005, 0.000, 0.005, 0.010,
        0.015, -0.010, -0.020
    ],
    "People & Culture": [
        0.000, 0.010, 0.015, 0.020, 0.025,
        0.030, 0.015, 0.020
    ]
}

march_actual_totals = {
    "Retail Operations": 2345000,
    "E-commerce": 735000,
    "Marketing": 805000,
    "Supply Chain": 1390000,
    "Information Technology": 365000,
    "Finance": 448000,
    "People & Culture": 545000
}

march_forecast_totals = {
    "Retail Operations": 2320000,
    "E-commerce": 730000,
    "Marketing": 760000,
    "Supply Chain": 1350000,
    "Information Technology": 380000,
    "Finance": 452000,
    "People & Culture": 540000
}

march_actual_by_category = {
    "Marketing": [455000, 210000, 140000],
    "Supply Chain": [745000, 390000, 255000],
    "Information Technology": [160000, 121000, 84000]
}

rows = []
row_number = 1

for department, config in department_config.items():
    categories = config["categories"]
    weights = config["weights"]

    for month_index, month in enumerate(months):
        budget_total = monthly_budgets[department][month_index]
        budget_values = allocate_total(budget_total, weights)

        if month == "2026-03":
            actual_total = march_actual_totals[department]
            forecast_total = march_forecast_totals[department]

            if department in march_actual_by_category:
                actual_values = march_actual_by_category[department]
            else:
                actual_values = allocate_total(actual_total, weights)

            forecast_values = allocate_total(forecast_total, weights)
        else:
            variance_rate = variance_rates[department][month_index]
            actual_total = round(budget_total * (1 + variance_rate))

            forecast_total = (
                budget_total
                + round((actual_total - budget_total) * 0.80)
            )

            actual_values = allocate_total(actual_total, weights)
            forecast_values = allocate_total(forecast_total, weights)

        for category_index, category in enumerate(categories):
            budget_aud = budget_values[category_index]
            actual_aud = actual_values[category_index]
            forecast_aud = forecast_values[category_index]

            variance_aud = actual_aud - budget_aud
            forecast_variance_aud = actual_aud - forecast_aud

            variance_pct = (
                round(variance_aud / budget_aud, 4)
                if budget_aud != 0
                else 0
            )

            variance_status = (
                "Unfavourable"
                if variance_aud > 0
                else "Favourable"
                if variance_aud < 0
                else "On budget"
            )

            rows.append(
                {
                    "row_id": f"BVA_{row_number:03d}",
                    "month": month,
                    "department": department,
                    "cost_centre": config["cost_centre"],
                    "account_category": category,
                    "budget_aud": budget_aud,
                    "actual_aud": actual_aud,
                    "variance_aud": variance_aud,
                    "variance_pct": variance_pct,
                    "forecast_aud": forecast_aud,
                    "forecast_variance_aud": forecast_variance_aud,
                    "variance_status": variance_status,
                    "responsible_person_id": (
                        config["responsible_person_id"]
                    ),
                    "responsible_manager": (
                        config["responsible_manager"]
                    ),
                    "period_status": "Closed"
                }
            )

            row_number += 1

data["accounting_conventions"]["forecast_definition"] = (
    "forecast_aud is the latest departmental pre-close forecast "
    "for the relevant month."
)

data["accounting_conventions"]["forecast_variance_formula"] = (
    "actual_aud - forecast_aud"
)

data["accounting_conventions"][
    "positive_forecast_variance_meaning"
] = "Actual expense is above the pre-close forecast"

data["monthly_budget_actual"] = {
    "document_id": "DOC_CSV_001",
    "grain": "One row per month, department and account category",
    "period_start": "2025-07",
    "period_end": "2026-03",
    "row_count": len(rows),
    "column_definitions": {
        "budget_aud": "Approved monthly budget in Australian dollars",
        "actual_aud": "Recorded monthly expense in Australian dollars",
        "variance_aud": "Actual expense less budget",
        "variance_pct": "Variance divided by budget",
        "forecast_aud": "Pre-close monthly forecast",
        "forecast_variance_aud": "Actual expense less forecast"
    },
    "rows": rows
}

with BLUEPRINT_PATH.open("w", encoding="utf-8", newline="\n") as file:
    json.dump(data, file, indent=2, ensure_ascii=False)
    file.write("\n")

march_rows = [
    row
    for row in rows
    if row["month"] == "2026-03"
]

march_budget = sum(row["budget_aud"] for row in march_rows)
march_actual = sum(row["actual_aud"] for row in march_rows)

print("Monthly budget vs actual facts added successfully")
print(f"Rows: {len(rows)}")
print(f"Months: {len(months)}")
print(f"Departments: {len(department_config)}")
print(f"March budget: AUD {march_budget:,}")
print(f"March actual: AUD {march_actual:,}")
print(f"March variance: AUD {march_actual - march_budget:,}")
