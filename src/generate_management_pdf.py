from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape
import json

from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle
)


BLUEPRINT_PATH = Path("data/blueprint/corpus_blueprint.json")
OUTPUT_PATH = Path(
    "data/raw/pdf/harbour_retail_q3_management_report_fy2026.pdf"
)

NAVY = HexColor("#17365D")
BLUE = HexColor("#2F75B5")
LIGHT_BLUE = HexColor("#D9EAF7")
PALE_BLUE = HexColor("#EEF5FA")
GREEN = HexColor("#548235")
LIGHT_GREEN = HexColor("#E2F0D9")
RED = HexColor("#C00000")
LIGHT_RED = HexColor("#FCE4D6")
ORANGE = HexColor("#C65911")
LIGHT_ORANGE = HexColor("#FCE4D6")
DARK_GREY = HexColor("#595959")
MID_GREY = HexColor("#A6A6A6")
LIGHT_GREY = HexColor("#F2F2F2")
WHITE = colors.white
BLACK = colors.black

PAGE_WIDTH, PAGE_HEIGHT = A4


def clean_text(value) -> str:
    """Replace typographic characters unsupported by standard PDF fonts."""
    return (
        str(value)
        .replace("—", "-")
        .replace("–", "-")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("…", "...")
    )


def safe(value) -> str:
    return escape(clean_text(value))


def format_aud(value: int) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}AUD {abs(value):,.0f}"


def format_millions(value: int) -> str:
    return f"AUD {value / 1_000_000:.1f}m"


def format_percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def variance_label(value: int) -> str:
    if value > 0:
        return "Unfavourable"
    if value < 0:
        return "Favourable"
    return "On budget"


def variance_colour(value: int):
    if value > 0:
        return RED
    if value < 0:
        return GREEN
    return DARK_GREY


with BLUEPRINT_PATH.open("r", encoding="utf-8") as file:
    data = json.load(file)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

people_by_id = {
    person["person_id"]: person
    for person in data["people"]
}

financials_by_metric = {
    item["metric"]: item
    for item in data["headline_financials"]
}

kpis_by_name = {
    item["name"]: item
    for item in data["kpi_dictionary"]
}


# -------------------------------------------------------------------
# Styles
# -------------------------------------------------------------------

sample_styles = getSampleStyleSheet()

styles = {
    "cover_title": ParagraphStyle(
        "CoverTitle",
        parent=sample_styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=27,
        leading=32,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceAfter=10
    ),
    "cover_subtitle": ParagraphStyle(
        "CoverSubtitle",
        parent=sample_styles["Normal"],
        fontName="Helvetica",
        fontSize=14,
        leading=18,
        textColor=DARK_GREY,
        alignment=TA_CENTER,
        spaceAfter=10
    ),
    "section_title": ParagraphStyle(
        "SectionTitle",
        parent=sample_styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=NAVY,
        spaceBefore=3,
        spaceAfter=9,
        keepWithNext=True
    ),
    "subheading": ParagraphStyle(
        "Subheading",
        parent=sample_styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=BLUE,
        spaceBefore=8,
        spaceAfter=5,
        keepWithNext=True
    ),
    "body": ParagraphStyle(
        "Body",
        parent=sample_styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.3,
        leading=13,
        textColor=BLACK,
        spaceAfter=6
    ),
    "body_bold": ParagraphStyle(
        "BodyBold",
        parent=sample_styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=9.3,
        leading=13,
        textColor=BLACK,
        spaceAfter=6
    ),
    "small": ParagraphStyle(
        "Small",
        parent=sample_styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.7,
        leading=10,
        textColor=DARK_GREY
    ),
    "small_white": ParagraphStyle(
        "SmallWhite",
        parent=sample_styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=7.8,
        leading=10,
        textColor=WHITE,
        alignment=TA_CENTER
    ),
    "table": ParagraphStyle(
        "TableText",
        parent=sample_styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.7,
        leading=9.5,
        textColor=BLACK
    ),
    "table_bold": ParagraphStyle(
        "TableBold",
        parent=sample_styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=7.7,
        leading=9.5,
        textColor=BLACK
    ),
    "callout": ParagraphStyle(
        "Callout",
        parent=sample_styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=NAVY,
        leftIndent=5,
        rightIndent=5,
        spaceAfter=0
    ),
    "metric_value": ParagraphStyle(
        "MetricValue",
        parent=sample_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=20,
        textColor=NAVY,
        alignment=TA_CENTER
    ),
    "metric_label": ParagraphStyle(
        "MetricLabel",
        parent=sample_styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=DARK_GREY,
        alignment=TA_CENTER
    ),
    "right": ParagraphStyle(
        "Right",
        parent=sample_styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        alignment=TA_RIGHT
    )
}


def paragraph(text, style="body"):
    return Paragraph(safe(text), styles[style])


def rich_paragraph(html, style="body"):
    return Paragraph(html, styles[style])


def section_heading(number: str, title: str):
    return Paragraph(
        f"{safe(number)}. {safe(title)}",
        styles["section_title"]
    )


def add_header_footer(canvas, document):
    canvas.saveState()

    page_number = canvas.getPageNumber()

    if page_number > 1:
        canvas.setStrokeColor(MID_GREY)
        canvas.setLineWidth(0.4)
        canvas.line(
            18 * mm,
            PAGE_HEIGHT - 15 * mm,
            PAGE_WIDTH - 18 * mm,
            PAGE_HEIGHT - 15 * mm
        )

        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(DARK_GREY)
        canvas.drawString(
            18 * mm,
            PAGE_HEIGHT - 11.5 * mm,
            "Harbour Retail Group"
        )
        canvas.drawRightString(
            PAGE_WIDTH - 18 * mm,
            PAGE_HEIGHT - 11.5 * mm,
            "Q3 Management Performance Report FY2025/26"
        )

    canvas.setStrokeColor(MID_GREY)
    canvas.setLineWidth(0.4)
    canvas.line(
        18 * mm,
        14 * mm,
        PAGE_WIDTH - 18 * mm,
        14 * mm
    )

    canvas.setFont("Helvetica", 7.2)
    canvas.setFillColor(DARK_GREY)
    canvas.drawString(
        18 * mm,
        9.5 * mm,
        "Synthetic business information for RAG evaluation"
    )
    canvas.drawRightString(
        PAGE_WIDTH - 18 * mm,
        9.5 * mm,
        f"Page {page_number}"
    )

    canvas.restoreState()


def metric_card(label: str, value: str, note: str = ""):
    content = [
        [Paragraph(safe(value), styles["metric_value"])],
        [Paragraph(safe(label), styles["metric_label"])]
    ]

    if note:
        content.append([Paragraph(safe(note), styles["metric_label"])])

    table = Table(content, colWidths=[49 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.7, BLUE),
                ("TOPPADDING", (0, 0), (-1, 0), 7),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 3),
                ("TOPPADDING", (0, 1), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
            ]
        )
    )
    return table


def callout_box(text: str, colour=LIGHT_BLUE):
    table = Table(
        [[Paragraph(safe(text), styles["callout"])]],
        colWidths=[170 * mm]
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colour),
                ("BOX", (0, 0), (-1, -1), 0.8, BLUE),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9)
            ]
        )
    )
    return table


def styled_table(
    rows,
    widths,
    numeric_columns=None,
    highlight_rows=None
):
    numeric_columns = numeric_columns or []
    highlight_rows = highlight_rows or {}

    table = Table(
        rows,
        colWidths=widths,
        repeatRows=1,
        hAlign="LEFT"
    )

    style_commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("LEADING", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.35, MID_GREY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4)
    ]

    for row_index in range(1, len(rows)):
        background = WHITE if row_index % 2 else LIGHT_GREY
        style_commands.append(
            ("BACKGROUND", (0, row_index), (-1, row_index), background)
        )

    for column in numeric_columns:
        style_commands.append(
            ("ALIGN", (column, 1), (column, -1), "RIGHT")
        )

    for row_index, colour in highlight_rows.items():
        style_commands.append(
            ("BACKGROUND", (0, row_index), (-1, row_index), colour)
        )

    table.setStyle(TableStyle(style_commands))
    return table


def progress_bar(label, value, maximum, display_value, colour=BLUE):
    width = 155 * mm
    height = 16 * mm
    drawing = Drawing(width, height)

    drawing.add(
        String(
            0,
            height - 9,
            clean_text(label),
            fontName="Helvetica-Bold",
            fontSize=8,
            fillColor=DARK_GREY
        )
    )

    bar_y = 2
    bar_height = 6
    bar_width = 120 * mm
    fill_width = max(0, min(bar_width, bar_width * value / maximum))

    drawing.add(
        Rect(
            32 * mm,
            bar_y,
            bar_width,
            bar_height,
            fillColor=LIGHT_GREY,
            strokeColor=MID_GREY,
            strokeWidth=0.4
        )
    )
    drawing.add(
        Rect(
            32 * mm,
            bar_y,
            fill_width,
            bar_height,
            fillColor=colour,
            strokeColor=colour
        )
    )
    drawing.add(
        String(
            width - 1,
            bar_y + 1,
            clean_text(display_value),
            fontName="Helvetica-Bold",
            fontSize=8,
            fillColor=DARK_GREY,
            textAnchor="end"
        )
    )

    return drawing


document = SimpleDocTemplate(
    str(OUTPUT_PATH),
    pagesize=A4,
    rightMargin=18 * mm,
    leftMargin=18 * mm,
    topMargin=20 * mm,
    bottomMargin=19 * mm,
    title="Q3 Management Performance Report FY2025/26",
    author="Harbour Retail Group",
    subject="Synthetic management performance report",
    keywords="finance, management reporting, FP&A, synthetic, RAG"
)

story = []


# -------------------------------------------------------------------
# Page 1 - Cover
# -------------------------------------------------------------------

story.append(Spacer(1, 24 * mm))
story.append(
    Paragraph(
        "HARBOUR RETAIL GROUP",
        styles["cover_subtitle"]
    )
)
story.append(Spacer(1, 5 * mm))
story.append(
    Paragraph(
        "Q3 Management<br/>Performance Report",
        styles["cover_title"]
    )
)
story.append(
    Paragraph(
        "FY2025/26 | Nine months ended 31 March 2026",
        styles["cover_subtitle"]
    )
)
story.append(Spacer(1, 12 * mm))
story.append(
    HRFlowable(
        width="75%",
        thickness=1.5,
        color=BLUE,
        spaceBefore=4,
        spaceAfter=12,
        hAlign="CENTER"
    )
)

cover_summary = Table(
    [
        [
            metric_card(
                "Revenue",
                "AUD 91.8m",
                "AUD 1.2m below budget"
            ),
            metric_card(
                "Gross margin",
                "42.4%",
                "0.7 percentage points below budget"
            ),
            metric_card(
                "Adjusted EBITDA",
                "AUD 8.7m",
                "AUD 0.7m below budget"
            )
        ]
    ],
    colWidths=[54 * mm, 54 * mm, 54 * mm],
    hAlign="CENTER"
)
cover_summary.setStyle(
    TableStyle(
        [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2)
        ]
    )
)
story.append(cover_summary)
story.append(Spacer(1, 18 * mm))

cover_metadata = Table(
    [
        ["Document ID", "DOC_PDF_001"],
        ["Prepared for", "Executive Leadership Team"],
        ["Reporting cutoff", "31 March 2026"],
        ["Classification", "Internal - synthetic training data"]
    ],
    colWidths=[42 * mm, 92 * mm],
    hAlign="CENTER"
)
cover_metadata.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.4, MID_GREY),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5)
        ]
    )
)
story.append(cover_metadata)
story.append(Spacer(1, 18 * mm))
story.append(
    callout_box(
        "This report contains synthetic business information created "
        "for a reproducible multiformat RAG evaluation project."
    )
)
story.append(PageBreak())


# -------------------------------------------------------------------
# Page 2 - Executive summary
# -------------------------------------------------------------------

story.append(section_heading("1", "Executive summary"))

story.append(
    callout_box(
        "Harbour Retail Group delivered revenue of AUD 91.8 million "
        "for the nine months ended 31 March 2026. Revenue remained "
        "AUD 1.2 million below budget, while gross margin and Adjusted "
        "EBITDA were affected by freight costs, promotional activity "
        "and higher digital customer acquisition expenditure."
    )
)
story.append(Spacer(1, 7 * mm))

summary_cards = Table(
    [
        [
            metric_card(
                "Store sales growth",
                "1.9%",
                "Positive but below online growth"
            ),
            metric_card(
                "E-commerce growth",
                "14.8%",
                "Strongest-performing channel"
            ),
            metric_card(
                "FY revenue forecast",
                "AUD 121.8m",
                "AUD 2.2m below budget"
            )
        ]
    ],
    colWidths=[54 * mm, 54 * mm, 54 * mm]
)
summary_cards.setStyle(
    TableStyle(
        [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2)
        ]
    )
)
story.append(summary_cards)
story.append(Spacer(1, 7 * mm))

story.append(Paragraph("Management assessment", styles["subheading"]))

assessment_points = [
    (
        "<b>Revenue:</b> Trading remained below the nine-month budget, "
        "with strong e-commerce growth partially offsetting softer "
        "performance in physical stores."
    ),
    (
        "<b>Gross margin:</b> The result of 42.4% was 0.7 percentage "
        "points below budget, reflecting freight surcharges, expedited "
        "shipments and promotional pressure."
    ),
    (
        "<b>Operating expenditure:</b> March costs were AUD 393,000 "
        "above budget, led by Marketing and Supply Chain."
    ),
    (
        "<b>Outlook:</b> The full-year forecast remains below budget. "
        "Management actions are focused on margin recovery, expenditure "
        "control and working-capital improvement."
    )
]

for point in assessment_points:
    story.append(
        Paragraph(
            f"- {point}",
            styles["body"]
        )
    )

story.append(Paragraph("Immediate management focus", styles["subheading"]))

focus_rows = [
    [
        Paragraph("Priority", styles["small_white"]),
        Paragraph("Management response", styles["small_white"])
    ],
    [
        paragraph("Gross margin recovery", "table_bold"),
        paragraph(
            "Tighter promotional controls, supplier negotiations and "
            "updated freight assumptions.",
            "table"
        )
    ],
    [
        paragraph("Marketing efficiency", "table_bold"),
        paragraph(
            "Reallocate spend toward channels with measurable conversion "
            "and customer acquisition returns.",
            "table"
        )
    ],
    [
        paragraph("Working capital", "table_bold"),
        paragraph(
            "Reduce slow-moving inventory and improve category-level "
            "inventory actions.",
            "table"
        )
    ],
    [
        paragraph("Forecast discipline", "table_bold"),
        paragraph(
            "Complete the Q4 refresh and identify savings required to "
            "protect Adjusted EBITDA.",
            "table"
        )
    ]
]

story.append(
    styled_table(
        focus_rows,
        widths=[45 * mm, 125 * mm]
    )
)
story.append(PageBreak())


# -------------------------------------------------------------------
# Page 3 - Financial performance
# -------------------------------------------------------------------

story.append(section_heading("2", "Financial performance"))

revenue = financials_by_metric["Revenue"]
gross_margin = financials_by_metric["Gross margin"]
ebitda = financials_by_metric["Adjusted EBITDA"]

financial_rows = [
    [
        Paragraph("Metric", styles["small_white"]),
        Paragraph("Actual", styles["small_white"]),
        Paragraph("Budget", styles["small_white"]),
        Paragraph("Variance", styles["small_white"]),
        Paragraph("Assessment", styles["small_white"])
    ],
    [
        paragraph("Revenue", "table_bold"),
        paragraph(format_millions(revenue["actual"]), "right"),
        paragraph(format_millions(revenue["budget"]), "right"),
        paragraph(format_aud(revenue["variance"]), "right"),
        paragraph("Below budget", "table")
    ],
    [
        paragraph("Gross margin", "table_bold"),
        paragraph(format_percentage(gross_margin["actual"]), "right"),
        paragraph(format_percentage(gross_margin["budget"]), "right"),
        paragraph(
            f"{gross_margin['variance_percentage_points']:.1f} pp",
            "right"
        ),
        paragraph("Below budget", "table")
    ],
    [
        paragraph("Adjusted EBITDA", "table_bold"),
        paragraph(format_millions(ebitda["actual"]), "right"),
        paragraph(format_millions(ebitda["budget"]), "right"),
        paragraph(format_aud(ebitda["variance"]), "right"),
        paragraph("Below budget", "table")
    ]
]

story.append(
    styled_table(
        financial_rows,
        widths=[
            41 * mm,
            30 * mm,
            30 * mm,
            32 * mm,
            37 * mm
        ],
        numeric_columns=[1, 2, 3]
    )
)
story.append(Spacer(1, 8 * mm))

story.append(Paragraph("Performance against budget", styles["subheading"]))
story.append(
    progress_bar(
        "Revenue",
        revenue["actual"],
        revenue["budget"],
        f"{format_millions(revenue['actual'])} / "
        f"{format_millions(revenue['budget'])}"
    )
)
story.append(
    progress_bar(
        "Gross margin",
        gross_margin["actual"],
        gross_margin["budget"],
        f"{format_percentage(gross_margin['actual'])} / "
        f"{format_percentage(gross_margin['budget'])}",
        colour=ORANGE
    )
)
story.append(
    progress_bar(
        "Adjusted EBITDA",
        ebitda["actual"],
        ebitda["budget"],
        f"{format_millions(ebitda['actual'])} / "
        f"{format_millions(ebitda['budget'])}",
        colour=GREEN
    )
)
story.append(Spacer(1, 6 * mm))

story.append(Paragraph("Key financial commentary", styles["subheading"]))

financial_commentary = [
    (
        "Revenue was 1.3% below budget. E-commerce provided the strongest "
        "growth contribution, while store performance was more moderate."
    ),
    (
        "Gross margin was 42.4%, compared with a budget of 43.1%. "
        "Freight surcharges, expedited shipments and promotional activity "
        "were the main sources of pressure."
    ),
    (
        "Adjusted EBITDA was AUD 8.7 million, AUD 700,000 below budget. "
        "The result reflects both the gross-margin shortfall and selected "
        "departmental overspends."
    )
]

for item in financial_commentary:
    story.append(paragraph(f"- {item}"))

story.append(Spacer(1, 4 * mm))
story.append(
    callout_box(
        "The financial result requires a stronger Q4 focus on gross-margin "
        "recovery and discretionary expenditure control.",
        colour=LIGHT_ORANGE
    )
)
story.append(PageBreak())


# -------------------------------------------------------------------
# Page 4 - Channel performance
# -------------------------------------------------------------------

story.append(section_heading("3", "Channel and commercial performance"))

store_growth = financials_by_metric["Store sales growth"]["actual"]
ecommerce_growth = financials_by_metric[
    "E-commerce sales growth"
]["actual"]

channel_cards = Table(
    [
        [
            metric_card(
                "Physical store sales growth",
                format_percentage(store_growth),
                "Positive growth across the store network"
            ),
            metric_card(
                "E-commerce sales growth",
                format_percentage(ecommerce_growth),
                "Strongest channel performance"
            )
        ]
    ],
    colWidths=[81 * mm, 81 * mm]
)
channel_cards.setStyle(
    TableStyle(
        [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3)
        ]
    )
)
story.append(channel_cards)
story.append(Spacer(1, 8 * mm))

story.append(Paragraph("Physical stores", styles["subheading"]))
story.append(
    paragraph(
        "Store sales increased by 1.9% for the nine-month period. "
        "Performance was positive but remained below the growth achieved "
        "by the online channel. Management continues to focus on stock "
        "availability, promotional discipline and labour productivity."
    )
)

story.append(Paragraph("E-commerce", styles["subheading"]))
story.append(
    paragraph(
        "E-commerce sales grew by 14.8%, making it the strongest-performing "
        "channel. However, paid digital acquisition costs increased faster "
        "than online conversion, reducing the incremental return from part "
        "of the Marketing investment."
    )
)

channel_rows = [
    [
        Paragraph("Channel", styles["small_white"]),
        Paragraph("Growth", styles["small_white"]),
        Paragraph("Main observation", styles["small_white"]),
        Paragraph("Q4 response", styles["small_white"])
    ],
    [
        paragraph("Physical stores", "table_bold"),
        paragraph("1.9%", "right"),
        paragraph(
            "Positive growth with inventory and labour-cost pressure.",
            "table"
        ),
        paragraph(
            "Improve availability and reduce slow-moving inventory.",
            "table"
        )
    ],
    [
        paragraph("E-commerce", "table_bold"),
        paragraph("14.8%", "right"),
        paragraph(
            "Strong sales growth but acquisition costs rose faster than "
            "conversion.",
            "table"
        ),
        paragraph(
            "Shift spend toward channels with measurable conversion.",
            "table"
        )
    ]
]

story.append(
    styled_table(
        channel_rows,
        widths=[31 * mm, 22 * mm, 61 * mm, 56 * mm],
        numeric_columns=[1]
    )
)
story.append(Spacer(1, 8 * mm))

story.append(
    callout_box(
        "Commercial growth remains concentrated in e-commerce. "
        "Management must improve digital marketing efficiency while "
        "protecting store availability and margin.",
        colour=LIGHT_BLUE
    )
)
story.append(PageBreak())


# -------------------------------------------------------------------
# Page 5 - March departmental performance
# -------------------------------------------------------------------

story.append(section_heading("4", "March departmental cost performance"))

march_totals = defaultdict(
    lambda: {
        "budget": 0,
        "actual": 0,
        "variance": 0,
        "forecast": 0
    }
)

for row in data["monthly_budget_actual"]["rows"]:
    if row["month"] != "2026-03":
        continue

    department = row["department"]
    march_totals[department]["budget"] += row["budget_aud"]
    march_totals[department]["actual"] += row["actual_aud"]
    march_totals[department]["variance"] += row["variance_aud"]
    march_totals[department]["forecast"] += row["forecast_aud"]

sorted_departments = sorted(
    march_totals.items(),
    key=lambda item: item[1]["variance"],
    reverse=True
)

march_rows = [
    [
        Paragraph("Department", styles["small_white"]),
        Paragraph("Budget", styles["small_white"]),
        Paragraph("Actual", styles["small_white"]),
        Paragraph("Variance", styles["small_white"]),
        Paragraph("Status", styles["small_white"])
    ]
]

highlight_rows = {}

for index, (department, values) in enumerate(
    sorted_departments,
    start=1
):
    variance = values["variance"]

    march_rows.append(
        [
            paragraph(department, "table_bold"),
            paragraph(format_aud(values["budget"]), "right"),
            paragraph(format_aud(values["actual"]), "right"),
            Paragraph(
                (
                    f'<font color="{variance_colour(variance).hexval()}">'
                    f'{safe(format_aud(variance))}</font>'
                ),
                styles["right"]
            ),
            paragraph(variance_label(variance), "table")
        ]
    )

    if department in {"Marketing", "Supply Chain"}:
        highlight_rows[index] = LIGHT_RED
    elif department == "Information Technology":
        highlight_rows[index] = LIGHT_GREEN

story.append(
    styled_table(
        march_rows,
        widths=[
            50 * mm,
            30 * mm,
            30 * mm,
            30 * mm,
            30 * mm
        ],
        numeric_columns=[1, 2, 3],
        highlight_rows=highlight_rows
    )
)
story.append(Spacer(1, 7 * mm))

story.append(
    callout_box(
        "March operating expenditure was AUD 6.633 million, "
        "AUD 393,000 above the total departmental budget of "
        "AUD 6.240 million.",
        colour=LIGHT_ORANGE
    )
)
story.append(Spacer(1, 6 * mm))

story.append(Paragraph("Principal variance drivers", styles["subheading"]))

for driver in data["variance_drivers"]:
    owner = people_by_id[driver["responsible_person_id"]]

    story.append(
        rich_paragraph(
            (
                f"<b>{safe(driver['department'])}:</b> "
                f"{safe(format_aud(driver['variance_aud']))} "
                f"{safe(driver['variance_status'].lower())}. "
                f"{safe(driver['primary_cause'])}. "
                f"<b>Owner:</b> {safe(owner['name'])}."
            )
        )
    )

story.append(Paragraph("Management interpretation", styles["subheading"]))
story.append(
    paragraph(
        "The March overspend was concentrated rather than broad-based. "
        "Marketing and Supply Chain represented AUD 335,000 of the total "
        "unfavourable variance. Information Technology partially offset "
        "the pressure because a vendor implementation milestone was delayed."
    )
)
story.append(PageBreak())


# -------------------------------------------------------------------
# Page 6 - Full-year forecast
# -------------------------------------------------------------------

story.append(section_heading("5", "Full-year forecast"))

revenue_forecast = financials_by_metric["Full-year revenue forecast"]
ebitda_forecast = financials_by_metric[
    "Full-year Adjusted EBITDA forecast"
]

forecast_rows = [
    [
        Paragraph("Metric", styles["small_white"]),
        Paragraph("Forecast", styles["small_white"]),
        Paragraph("Budget", styles["small_white"]),
        Paragraph("Variance", styles["small_white"]),
        Paragraph("Variance %", styles["small_white"])
    ],
    [
        paragraph("Revenue", "table_bold"),
        paragraph(format_millions(revenue_forecast["forecast"]), "right"),
        paragraph(format_millions(revenue_forecast["budget"]), "right"),
        paragraph(format_aud(revenue_forecast["variance"]), "right"),
        paragraph(
            format_percentage(revenue_forecast["variance_pct"]),
            "right"
        )
    ],
    [
        paragraph("Adjusted EBITDA", "table_bold"),
        paragraph(format_millions(ebitda_forecast["forecast"]), "right"),
        paragraph(format_millions(ebitda_forecast["budget"]), "right"),
        paragraph(format_aud(ebitda_forecast["variance"]), "right"),
        paragraph(
            format_percentage(ebitda_forecast["variance_pct"]),
            "right"
        )
    ]
]

story.append(
    styled_table(
        forecast_rows,
        widths=[48 * mm, 31 * mm, 31 * mm, 31 * mm, 29 * mm],
        numeric_columns=[1, 2, 3, 4]
    )
)
story.append(Spacer(1, 8 * mm))

forecast_cards = Table(
    [
        [
            metric_card(
                "Revenue recovery required",
                "AUD 2.2m",
                "Gap to the full-year budget"
            ),
            metric_card(
                "EBITDA recovery required",
                "AUD 0.9m",
                "Gap to the full-year budget"
            )
        ]
    ],
    colWidths=[81 * mm, 81 * mm]
)
forecast_cards.setStyle(
    TableStyle(
        [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3)
        ]
    )
)
story.append(forecast_cards)
story.append(Spacer(1, 8 * mm))

story.append(Paragraph("Forecast assumptions", styles["subheading"]))

assumptions = [
    (
        "Revenue forecast assumes continued e-commerce growth but no full "
        "recovery of the year-to-date sales shortfall."
    ),
    (
        "Freight costs remain above the original budget during Q4, using "
        "current surcharge rates and revised shipment assumptions."
    ),
    (
        "No additional unplanned Marketing campaigns are assumed without "
        "Finance Director approval."
    ),
    (
        "Non-critical recruitment remains paused until the Q4 forecast "
        "refresh has been approved."
    ),
    (
        "The delayed IT implementation remains in the forecast with "
        "updated timing and cash-flow assumptions."
    )
]

for assumption in assumptions:
    story.append(paragraph(f"- {assumption}"))

story.append(
    callout_box(
        "The updated forecast remains below budget. The executive review "
        "will focus on realistic savings and margin actions rather than "
        "relying on unsupported revenue recovery.",
        colour=LIGHT_ORANGE
    )
)
story.append(PageBreak())


# -------------------------------------------------------------------
# Page 7 - Risks
# -------------------------------------------------------------------

story.append(section_heading("6", "Principal business risks"))

risk_rows = [
    [
        Paragraph("Risk", styles["small_white"]),
        Paragraph("Description", styles["small_white"]),
        Paragraph("Financial impact", styles["small_white"]),
        Paragraph("Owner", styles["small_white"]),
        Paragraph("Severity", styles["small_white"])
    ]
]

risk_highlights = {}

for index, risk in enumerate(data["business_risks"], start=1):
    owner = people_by_id[risk["owner_person_id"]]

    risk_rows.append(
        [
            paragraph(risk["category"], "table_bold"),
            paragraph(risk["description"], "table"),
            paragraph(risk["financial_impact"], "table"),
            paragraph(owner["name"], "table"),
            paragraph(risk["severity"], "table")
        ]
    )

    if risk["severity"] == "High":
        risk_highlights[index] = LIGHT_RED

story.append(
    styled_table(
        risk_rows,
        widths=[
            25 * mm,
            58 * mm,
            47 * mm,
            25 * mm,
            15 * mm
        ],
        highlight_rows=risk_highlights
    )
)
story.append(Spacer(1, 8 * mm))

story.append(Paragraph("Risk response", styles["subheading"]))

risk_responses = [
    (
        "<b>Supply Chain:</b> update the Q4 logistics forecast, continue "
        "supplier negotiations and review opportunities to consolidate "
        "shipments."
    ),
    (
        "<b>Inventory:</b> prepare category-level targets for reducing "
        "slow-moving stock while protecting seasonal availability."
    ),
    (
        "<b>Marketing:</b> require measurable conversion and customer "
        "acquisition returns before reallocating discretionary spend."
    ),
    (
        "<b>People:</b> maintain the pause on non-critical recruitment "
        "and monitor wage-cost pressure through the monthly forecast."
    )
]

for response in risk_responses:
    story.append(Paragraph(f"- {response}", styles["body"]))

story.append(
    callout_box(
        "International freight cost remains the only risk currently "
        "assessed as High severity. It affects both gross margin and "
        "distribution expenditure.",
        colour=LIGHT_RED
    )
)
story.append(PageBreak())


# -------------------------------------------------------------------
# Page 8 - Priorities and actions
# -------------------------------------------------------------------

story.append(section_heading("7", "Q4 priorities and agreed actions"))

priority_rows = [
    [
        Paragraph("Priority", styles["small_white"]),
        Paragraph("Description", styles["small_white"]),
        Paragraph("Owners", styles["small_white"])
    ]
]

for priority in data["q4_priorities"]:
    owners = ", ".join(
        people_by_id[person_id]["name"]
        for person_id in priority["owner_person_ids"]
    )

    priority_rows.append(
        [
            paragraph(priority["priority_id"], "table_bold"),
            paragraph(priority["priority"], "table"),
            paragraph(owners, "table")
        ]
    )

story.append(
    styled_table(
        priority_rows,
        widths=[23 * mm, 103 * mm, 44 * mm]
    )
)
story.append(Spacer(1, 8 * mm))

story.append(Paragraph("Forecast meeting action plan", styles["subheading"]))

action_rows = [
    [
        Paragraph("Action", styles["small_white"]),
        Paragraph("Owner", styles["small_white"]),
        Paragraph("Deadline", styles["small_white"])
    ]
]

for action in data["forecast_meeting"]["action_items"]:
    owner = people_by_id[action["owner_person_id"]]

    action_rows.append(
        [
            paragraph(
                f"{action['action_id']}: {action['action']}",
                "table"
            ),
            paragraph(owner["name"], "table"),
            paragraph(action["deadline"], "table")
        ]
    )

story.append(
    styled_table(
        action_rows,
        widths=[112 * mm, 32 * mm, 26 * mm]
    )
)
story.append(Spacer(1, 7 * mm))

story.append(
    callout_box(
        "The Head of FP&A will consolidate departmental savings and "
        "issue the updated Q4 forecast for executive review by "
        "27 April 2026.",
        colour=LIGHT_BLUE
    )
)
story.append(PageBreak())


# -------------------------------------------------------------------
# Page 9 - Appendix
# -------------------------------------------------------------------

story.append(section_heading("8", "Appendix"))

story.append(Paragraph("Selected KPI definitions", styles["subheading"]))

selected_kpis = [
    "Revenue",
    "Gross margin",
    "Adjusted EBITDA",
    "Forecast accuracy"
]

kpi_rows = [
    [
        Paragraph("KPI", styles["small_white"]),
        Paragraph("Definition", styles["small_white"]),
        Paragraph("Formula", styles["small_white"]),
        Paragraph("Owner", styles["small_white"])
    ]
]

for kpi_name in selected_kpis:
    kpi = kpis_by_name[kpi_name]
    owner = people_by_id[kpi["owner_person_id"]]

    kpi_rows.append(
        [
            paragraph(kpi["name"], "table_bold"),
            paragraph(kpi["definition"], "table"),
            paragraph(kpi["formula"], "table"),
            paragraph(owner["name"], "table")
        ]
    )

story.append(
    styled_table(
        kpi_rows,
        widths=[31 * mm, 62 * mm, 52 * mm, 25 * mm]
    )
)
story.append(Spacer(1, 8 * mm))

story.append(Paragraph("Document sources", styles["subheading"]))

source_rows = [
    [
        Paragraph("Document ID", styles["small_white"]),
        Paragraph("Source", styles["small_white"]),
        Paragraph("Purpose", styles["small_white"])
    ]
]

for source in data["documents"]:
    source_rows.append(
        [
            paragraph(source["document_id"], "table_bold"),
            paragraph(source["title"], "table"),
            paragraph(source["purpose"], "table")
        ]
    )

story.append(
    styled_table(
        source_rows,
        widths=[30 * mm, 61 * mm, 79 * mm]
    )
)
story.append(Spacer(1, 8 * mm))

story.append(Paragraph("Variance conventions", styles["subheading"]))

conventions = [
    (
        "Expense variance equals actual expense less budget. A positive "
        "value is unfavourable because actual expenditure is above budget."
    ),
    (
        "Forecast variance equals actual expense less the departmental "
        "pre-close forecast."
    ),
    (
        "Rates are stored as decimal fractions in the source blueprint "
        "and displayed as percentages in this report."
    )
]

for convention in conventions:
    story.append(paragraph(f"- {convention}"))

story.append(Spacer(1, 7 * mm))
story.append(
    callout_box(
        "End of Q3 Management Performance Report FY2025/26"
    )
)


document.build(
    story,
    onFirstPage=add_header_footer,
    onLaterPages=add_header_footer
)

print("Q3 management report PDF generated successfully")
print(f"Created: {OUTPUT_PATH}")
print(f"File size: {OUTPUT_PATH.stat().st_size:,} bytes")

