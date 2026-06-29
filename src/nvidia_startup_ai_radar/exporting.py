"""Briefing export helpers for persisted radar runs."""

from __future__ import annotations

from html import escape
from io import BytesIO
from pathlib import Path
import re
from typing import Any, Literal


DEFAULT_EXPORT_DIR = Path("output") / "briefings"


def slugify(value: str, fallback: str = "startup") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug or fallback


def briefing_markdown(run: dict[str, Any]) -> str:
    """Return a complete Markdown briefing for one persisted run."""

    briefing = run.get("briefing_en") or run.get("briefing_pt") or ""
    metadata = [
        "<!-- NVIDIA Startup AI Radar export -->",
        f"<!-- run_id: {run.get('run_id')} -->",
        f"<!-- startup: {run.get('nome')} -->",
        f"<!-- classification: {run.get('classificacao')} -->",
        "",
    ]
    return "\n".join(metadata) + briefing.strip() + "\n"


def markdown_to_pdf_bytes(markdown_text: str, title: str = "NVIDIA Startup AI Radar") -> bytes:
    """Render briefing Markdown to a simple, readable PDF."""

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title=title,
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="RadarTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#1f2937"),
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="RadarHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#0f766e"),
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="RadarBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="RadarBullet",
            parent=styles["RadarBody"],
            leftIndent=12,
            firstLineIndent=-8,
        )
    )

    story: list[Any] = []
    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("<!--"):
            if story:
                story.append(Spacer(1, 4))
            continue
        if line.startswith("# "):
            story.append(Paragraph(escape(line[2:].strip()), styles["RadarTitle"]))
            continue
        if line.startswith("## "):
            story.append(Paragraph(escape(line[3:].strip()), styles["RadarHeading"]))
            continue
        if line.startswith("- "):
            story.append(Paragraph(f"- {escape(line[2:].strip())}", styles["RadarBullet"]))
            continue
        story.append(Paragraph(escape(line), styles["RadarBody"]))

    if not story:
        story.append(Paragraph("Briefing vazio.", styles["RadarBody"]))

    doc.build(story)
    return buffer.getvalue()


def export_run(
    run: dict[str, Any],
    export_dir: str | Path = DEFAULT_EXPORT_DIR,
    export_format: Literal["markdown", "pdf"] = "pdf",
) -> Path:
    """Write one persisted run to Markdown or PDF and return the file path."""

    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    filename = f"run-{run['run_id']}-{slugify(run.get('nome') or '')}"
    markdown = briefing_markdown(run)

    if export_format == "markdown":
        path = export_path / f"{filename}.md"
        path.write_text(markdown, encoding="utf-8")
        return path

    if export_format == "pdf":
        path = export_path / f"{filename}.pdf"
        path.write_bytes(markdown_to_pdf_bytes(markdown, title=str(run.get("nome") or filename)))
        return path

    raise ValueError(f"Unsupported export format: {export_format}")
