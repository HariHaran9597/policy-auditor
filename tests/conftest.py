"""Shared fixtures — build a small PDF via reportlab (same layout family as the
sample corpus) so parser/chunker tests run against real PDF structure."""

from pathlib import Path

import pytest
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Table,
    TableStyle,
)


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("pdfs") / "Test_Handbook.pdf"
    body = ParagraphStyle("b", fontName="Helvetica", fontSize=10, leading=14)
    h1 = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=14, leading=18)

    table = Table(
        [
            ["Leave type", "Days", "Notice"],
            ["Annual", "24", "14 days"],
            ["Sick", "12", "same day"],
        ],
        colWidths=[5 * cm, 3 * cm, 4 * cm],
    )
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )

    filler = " ".join(
        f"Policy paragraph number {i} containing ordinary compliance prose for the "
        f"purpose of exercising the chunker boundaries with realistic length."
        for i in range(40)
    )

    story = [
        Paragraph("1. Leave Entitlements", h1),
        Paragraph("Employees are entitled to the leave in the table below.", body),
        table,
        Paragraph("2. General Provisions", h1),
        Paragraph(filler, body),
        PageBreak(),
        Paragraph("3. Provisions on the Second Page", h1),
        Paragraph(filler, body),
    ]
    doc = SimpleDocTemplate(str(out), pagesize=A4)
    doc.build(story)
    return out
