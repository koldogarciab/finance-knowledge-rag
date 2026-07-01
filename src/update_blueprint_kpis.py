from pathlib import Path
import json

BLUEPRINT_PATH = Path("data/blueprint/corpus_blueprint.json")

with BLUEPRINT_PATH.open("r", encoding="utf-8") as file:
    data = json.load(file)

data["kpi_dictionary"] = [
    {
        "kpi_id": "KPI_FIN_001",
        "name": "Revenue",
        "category": "Growth",
        "definition": "Total income generated from the sale of goods before deducting operating expenses.",
        "formula": "Gross sales less returns, discounts and sales taxes",
        "frequency": "Monthly",
        "owner_person_id": "PER_003",
        "data_source": "ERP general ledger and point-of-sale system",
        "unit": "AUD",
        "better_direction": "Higher",
        "target": {
            "period": "FY2025/26",
            "value": 124000000,
            "operator": ">="
        }
    },
    {
        "kpi_id": "KPI_FIN_002",
        "name": "Like-for-like sales growth",
        "category": "Growth",
        "definition": "Percentage change in sales from comparable stores that traded in both the current and prior periods.",
        "formula": "(Comparable store sales current period - comparable store sales prior period) / comparable store sales prior period",
        "frequency": "Monthly",
        "owner_person_id": "PER_005",
        "data_source": "Point-of-sale system",
        "unit": "percentage",
        "better_direction": "Higher",
        "target": {
            "period": "FY2025/26",
            "value": 0.03,
            "operator": ">="
        }
    },
    {
        "kpi_id": "KPI_FIN_003",
        "name": "Gross margin",
        "category": "Profitability",
        "definition": "Gross profit expressed as a percentage of revenue.",
        "formula": "(Revenue - cost of goods sold) / revenue",
        "frequency": "Monthly",
        "owner_person_id": "PER_003",
        "data_source": "ERP general ledger",
        "unit": "percentage",
        "better_direction": "Higher",
        "target": {
            "period": "Nine months ended 31 March 2026",
            "value": 0.431,
            "operator": ">="
        }
    },
    {
        "kpi_id": "KPI_FIN_004",
        "name": "Adjusted EBITDA",
        "category": "Profitability",
        "definition": "Earnings before interest, tax, depreciation and amortisation, adjusted for approved non-recurring items.",
        "formula": "Operating profit + depreciation + amortisation + approved adjustments",
        "frequency": "Monthly",
        "owner_person_id": "PER_002",
        "data_source": "ERP general ledger and FP&A adjustment schedule",
        "unit": "AUD",
        "better_direction": "Higher",
        "target": {
            "period": "FY2025/26",
            "value": 12200000,
            "operator": ">="
        }
    },
    {
        "kpi_id": "KPI_FIN_005",
        "name": "Operating expense ratio",
        "category": "Cost efficiency",
        "definition": "Operating expenses expressed as a percentage of revenue.",
        "formula": "Operating expenses / revenue",
        "frequency": "Monthly",
        "owner_person_id": "PER_003",
        "data_source": "ERP general ledger",
        "unit": "percentage",
        "better_direction": "Lower",
        "target": {
            "period": "FY2025/26",
            "value": 0.326,
            "operator": "<="
        }
    },
    {
        "kpi_id": "KPI_FIN_006",
        "name": "Inventory days",
        "category": "Working capital",
        "definition": "Estimated number of days that inventory is held before being sold.",
        "formula": "Average inventory / annualised cost of goods sold * 365",
        "frequency": "Monthly",
        "owner_person_id": "PER_008",
        "data_source": "ERP inventory and general ledger modules",
        "unit": "days",
        "better_direction": "Lower",
        "target": {
            "period": "FY2025/26",
            "value": 92,
            "operator": "<="
        }
    },
    {
        "kpi_id": "KPI_FIN_007",
        "name": "Stock availability",
        "category": "Operations",
        "definition": "Percentage of active products available for sale when measured.",
        "formula": "Available active product locations / total active product locations",
        "frequency": "Weekly",
        "owner_person_id": "PER_005",
        "data_source": "Inventory management system",
        "unit": "percentage",
        "better_direction": "Higher",
        "target": {
            "period": "FY2025/26",
            "value": 0.96,
            "operator": ">="
        }
    },
    {
        "kpi_id": "KPI_FIN_008",
        "name": "Online conversion rate",
        "category": "E-commerce",
        "definition": "Percentage of website sessions that result in a completed customer order.",
        "formula": "Completed online orders / website sessions",
        "frequency": "Weekly",
        "owner_person_id": "PER_006",
        "data_source": "E-commerce platform and web analytics system",
        "unit": "percentage",
        "better_direction": "Higher",
        "target": {
            "period": "FY2025/26",
            "value": 0.032,
            "operator": ">="
        }
    },
    {
        "kpi_id": "KPI_FIN_009",
        "name": "Average transaction value",
        "category": "Sales productivity",
        "definition": "Average revenue generated by each completed customer transaction.",
        "formula": "Revenue / number of completed transactions",
        "frequency": "Monthly",
        "owner_person_id": "PER_005",
        "data_source": "Point-of-sale and e-commerce systems",
        "unit": "AUD per transaction",
        "better_direction": "Higher",
        "target": {
            "period": "FY2025/26",
            "value": 84,
            "operator": ">="
        }
    },
    {
        "kpi_id": "KPI_FIN_010",
        "name": "Labour cost percentage",
        "category": "People",
        "definition": "Employee labour costs expressed as a percentage of revenue.",
        "formula": "Wages, salaries and on-costs / revenue",
        "frequency": "Monthly",
        "owner_person_id": "PER_010",
        "data_source": "Payroll system and ERP general ledger",
        "unit": "percentage",
        "better_direction": "Lower",
        "target": {
            "period": "FY2025/26",
            "value": 0.164,
            "operator": "<="
        }
    },
    {
        "kpi_id": "KPI_FIN_011",
        "name": "Forecast accuracy",
        "category": "Planning",
        "definition": "Accuracy of the most recently approved forecast compared with actual results.",
        "formula": "1 - absolute value of (actual - forecast) / actual",
        "frequency": "Monthly",
        "owner_person_id": "PER_003",
        "data_source": "FP&A planning model and ERP general ledger",
        "unit": "percentage",
        "better_direction": "Higher",
        "target": {
            "period": "FY2025/26",
            "value": 0.95,
            "operator": ">="
        }
    },
    {
        "kpi_id": "KPI_FIN_012",
        "name": "Net working capital",
        "category": "Working capital",
        "definition": "Short-term operating assets less short-term operating liabilities.",
        "formula": "Trade receivables + inventory - trade payables",
        "frequency": "Monthly",
        "owner_person_id": "PER_004",
        "data_source": "ERP general ledger",
        "unit": "AUD",
        "better_direction": "Context dependent",
        "target": {
            "period": "FY2025/26",
            "value": 18500000,
            "operator": "<="
        }
    },
    {
        "kpi_id": "KPI_FIN_013",
        "name": "Capital expenditure utilisation",
        "category": "Investment",
        "definition": "Approved capital expenditure used or committed as a percentage of the annual capital budget.",
        "formula": "(Capital expenditure incurred + committed expenditure) / approved capital budget",
        "frequency": "Monthly",
        "owner_person_id": "PER_002",
        "data_source": "ERP fixed asset module and capital expenditure register",
        "unit": "percentage",
        "better_direction": "Target range",
        "target": {
            "period": "FY2025/26",
            "minimum": 0.90,
            "maximum": 1.00,
            "operator": "between"
        }
    }
]

with BLUEPRINT_PATH.open("w", encoding="utf-8", newline="\n") as file:
    json.dump(data, file, indent=2, ensure_ascii=False)
    file.write("\n")

print("Finance KPI dictionary added successfully")
print(f"KPIs: {len(data['kpi_dictionary'])}")
