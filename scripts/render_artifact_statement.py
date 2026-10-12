#!/usr/bin/env python3
"""Render the numbered release statement from its Markdown source (ReportLab)."""
from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak

ROOT = Path(__file__).resolve().parents[1]


def inline(value):
    value = escape(value)
    value = re.sub(r"\[([^]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#155e75">\1</link>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    return value


def main():
    target = ROOT / "output/pdf/artifact_statement.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("BodyRelease", fontName="Helvetica", fontSize=10,
                              leading=14, spaceAfter=8, allowWidows=0, allowOrphans=0,
                              textColor=colors.HexColor("#202b36")))
    styles.add(ParagraphStyle("TitleRelease", fontName="Helvetica-Bold", fontSize=21,
                              leading=25, spaceAfter=15, textColor=colors.HexColor("#173448")))
    styles.add(ParagraphStyle("SectionRelease", fontName="Helvetica-Bold", fontSize=13,
                              leading=17, spaceBefore=13, spaceAfter=8, keepWithNext=True,
                              textColor=colors.HexColor("#173448")))
    styles.add(ParagraphStyle("CellRelease", fontName="Helvetica", fontSize=8,
                              leading=10.5, alignment=TA_LEFT))
    lines = (ROOT / "docs/artifact_statement.md").read_text().splitlines()
    elements, paragraph = [], []

    def flush():
        if paragraph:
            elements.append(Paragraph(inline(" ".join(paragraph)), styles["BodyRelease"]))
            paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            flush()
        elif line.startswith("# "):
            flush()
            elements.append(Paragraph(inline(line[2:]), styles["TitleRelease"]))
        elif line.startswith("## "):
            flush()
            if line.startswith(("## S2.", "## S5.")):
                elements.append(PageBreak())
            elements.append(Paragraph(inline(line[3:]), styles["SectionRelease"]))
        elif line.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", c) for c in cells):
                    rows.append([Paragraph(inline(c), styles["CellRelease"]) for c in cells])
                i += 1
            table = Table(rows, colWidths=[110, 48, 65, 70, 67, 135], repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eff3")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, 0), .7, colors.HexColor("#859baa")),
                ("LINEBELOW", (0, 1), (-1, -1), .3, colors.HexColor("#d2dde3")),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
            # Keep the table's short introductory paragraph with its complete rows.
            if elements and isinstance(elements[-1], Paragraph):
                elements[-1].keepWithNext = True
            elements.extend([KeepTogether([table]), Spacer(1, 10)])
            continue
        else:
            paragraph.append(line)
        i += 1
    flush()

    def decorate_page(canvas, doc):
        canvas.setPageSize(A4)
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#5b6d79"))
        canvas.drawString(50, 815, "POISON DETECTION  /  ARTIFACT RELEASE STATEMENT")
        canvas.drawString(50, 30, "Prepared code-only distribution | 12 October 2026")
        canvas.drawRightString(A4[0] - 50, 30, f"S-{doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(str(target), pagesize=A4, leftMargin=50, rightMargin=50,
                            topMargin=45, bottomMargin=48,
                            title="Supplementary Artifact Release Statement",
                            author="Anonymous", subject="Licenses, privacy, content, documentation and data statistics")
    body = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                 leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id="release", frames=[body], pagesize=A4,
                                     onPage=decorate_page))
    doc.build(elements)
    print(target)


if __name__ == "__main__":
    main()
