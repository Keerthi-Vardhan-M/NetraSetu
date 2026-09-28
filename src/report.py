"""Doctor-reviewed PDF report generation."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from PIL import Image as PILImage

from src.grades import GRADE_LABELS


def build_report(case: dict) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CenteredSmall", parent=styles["BodyText"], alignment=TA_CENTER, fontSize=8))
    story = [
        Paragraph("NetraSetu Retinal Screening Report", styles["Title"]),
        Paragraph("Doctor-reviewed clinical decision-support record", styles["CenteredSmall"]),
        Spacer(1, 8 * mm),
    ]

    patient_rows = [
        ["Patient", case["patient_name"], "Local ID", case["local_patient_id"]],
        ["Age / sex", f'{case["age"]} / {case["sex"]}', "Village", case.get("village") or "-"],
        ["Screened", case["created_at"][:10], "Priority", case["priority"]],
    ]
    story.extend([_table(patient_rows), Spacer(1, 6 * mm)])

    image_cells = []
    for label, key in (("Right eye", "right_image"), ("Left eye", "left_image")):
        path = Path(case[key]) if case.get(key) else None
        if path and path.exists():
            image_cells.append([Paragraph(label, styles["Heading3"]), _report_image(path)])
        else:
            image_cells.append([Paragraph(label, styles["Heading3"]), Paragraph("Image unavailable", styles["BodyText"])])
    story.append(Table([[image_cells[0][1], image_cells[1][1]]], colWidths=[82 * mm, 82 * mm]))
    story.append(Spacer(1, 5 * mm))

    right_ai = case["right_result"].get("grade")
    left_ai = case["left_result"].get("grade")
    result_rows = [
        ["", "Right eye", "Left eye"],
        ["Preliminary AI grade", _grade(right_ai), _grade(left_ai)],
        ["Doctor-confirmed grade", _grade(case.get("reviewed_right_grade")), _grade(case.get("reviewed_left_grade"))],
    ]
    story.extend([_table(result_rows), Spacer(1, 5 * mm)])

    details = [
        ["Doctor", case.get("doctor_name") or "-"],
        ["Registration", case.get("doctor_registration") or "-"],
        ["Action", case.get("review_action") or "-"],
        ["Clinical notes", case.get("review_notes") or "-"],
        ["Prescription / instructions", case.get("prescription") or "-"],
        ["Follow-up date", case.get("follow_up_date") or "-"],
    ]
    story.extend([_table(details, widths=[45 * mm, 119 * mm]), Spacer(1, 8 * mm)])
    story.append(
        Paragraph(
            "Important: The AI result is preliminary decision support. The doctor-reviewed assessment governs care. "
            "This hackathon prototype is not independently validated as a medical device.",
            styles["CenteredSmall"],
        )
    )
    document.build(story)
    return buffer.getvalue()


def _grade(value: int | None) -> str:
    if value is None:
        return "Ungradable / not available"
    return f"{value} - {GRADE_LABELS[int(value)]}"


def _report_image(path: Path) -> Image:
    """Resize large fundus captures before embedding them in the PDF."""
    stream = BytesIO()
    with PILImage.open(path) as source:
        source = source.convert("RGB")
        source.thumbnail((1000, 700), PILImage.Resampling.LANCZOS)
        source.save(stream, format="JPEG", quality=86, optimize=True)
    stream.seek(0)
    return Image(stream, width=72 * mm, height=50 * mm)


def _table(rows: list[list[str]], widths: list[float] | None = None) -> Table:
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        name="TableBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#18332B"),
    )
    header_style = ParagraphStyle(
        name="TableHeader",
        parent=body_style,
        fontName="Helvetica-Bold",
    )
    wrapped_rows = []
    for row_index, row in enumerate(rows):
        wrapped_rows.append(
            [
                Paragraph(str(cell), header_style if row_index == 0 else body_style)
                if isinstance(cell, str)
                else cell
                for cell in row
            ]
        )
    table = Table(wrapped_rows, colWidths=widths, repeatRows=1 if len(rows) > 2 else 0)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7F3EF")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#18332B")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B8C8C3")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAF9")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table
