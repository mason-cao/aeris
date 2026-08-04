"""Deterministic PDF rendering of the expert labeling guide markdown.

The guide draft lives in ``docs/bracco`` as markdown; the PDF handed to the
expert is a committed artifact. Rendering it here — with the same reportlab
toolchain the review packet uses — makes the artifact reproducible from the
draft instead of depending on whatever converter happens to be installed.

CLI: ``python -m app.eval.guide_pdf --source <guide.md> --out <guide.pdf>``
"""

from __future__ import annotations

import argparse
import re
from html import escape
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

from app.eval.packet import invariant_canvas, unsafe_text_findings

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def _inline(text: str) -> str:
    """Escaped inline markup: ``**bold**`` becomes ``<b>``, ``\\*`` a literal."""
    rendered = _BOLD_RE.sub(r"<b>\1</b>", escape(text))
    return rendered.replace("\\*", "*")


def _blocks(markdown: str) -> list[tuple[str, str | list[str]]]:
    """(kind, content) blocks: heading1/heading2, bullets, numbers, paragraph."""
    blocks: list[tuple[str, str | list[str]]] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(("paragraph", " ".join(paragraph)))
            paragraph.clear()

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue
        if line.startswith("# "):
            flush_paragraph()
            blocks.append(("heading1", line[2:].strip()))
            continue
        if line.startswith("## "):
            flush_paragraph()
            blocks.append(("heading2", line[3:].strip()))
            continue
        if line.startswith("- "):
            flush_paragraph()
            if not blocks or blocks[-1][0] != "bullets":
                blocks.append(("bullets", []))
            items = blocks[-1][1]
            assert isinstance(items, list)
            items.append(line[2:].strip())
            continue
        numbered = re.match(r"(\d+)\.\s+(.*)", line)
        if numbered and not line.startswith(" "):
            flush_paragraph()
            if not blocks or blocks[-1][0] != "numbers":
                blocks.append(("numbers", []))
            items = blocks[-1][1]
            assert isinstance(items, list)
            items.append(numbered.group(2).strip())
            continue
        if line.startswith("  ") and blocks and blocks[-1][0] in (
            "bullets",
            "numbers",
        ) and not paragraph:
            items = blocks[-1][1]
            assert isinstance(items, list)
            items[-1] = f"{items[-1]} {stripped}"
            continue
        paragraph.append(stripped)
    flush_paragraph()
    return blocks


def build_guide_pdf(source: Path, out: Path) -> None:
    """Render the guide markdown to ``out``; refuses unsafe control chars."""
    markdown = source.read_text(encoding="utf-8")
    findings = unsafe_text_findings((("guide", markdown),))
    if findings:
        raise ValueError(f"unsafe control characters in guide: {findings}")

    base = getSampleStyleSheet()
    styles = {
        "heading1": ParagraphStyle(
            "guide_h1", parent=base["Heading1"], spaceAfter=10
        ),
        "heading2": ParagraphStyle(
            "guide_h2", parent=base["Heading2"], spaceBefore=12, spaceAfter=6
        ),
        "body": ParagraphStyle(
            "guide_body", parent=base["BodyText"], leading=14, spaceAfter=6
        ),
    }

    story: list = []
    for kind, content in _blocks(markdown):
        if kind in ("heading1", "heading2"):
            assert isinstance(content, str)
            story.append(Paragraph(_inline(content), styles[kind]))
            continue
        if kind == "paragraph":
            assert isinstance(content, str)
            story.append(Paragraph(_inline(content), styles["body"]))
            continue
        assert isinstance(content, list)
        bullet_type = "bullet" if kind == "bullets" else "1"
        story.append(
            ListFlowable(
                [
                    ListItem(Paragraph(_inline(item), styles["body"]))
                    for item in content
                ],
                bulletType=bullet_type,
                leftIndent=18,
            )
        )
        story.append(Spacer(1, 4))

    document = SimpleDocTemplate(
        str(out),
        pagesize=letter,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
        title="AERIS Expert Labeling Guide",
    )
    document.build(story, canvasmaker=invariant_canvas)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.guide_pdf",
        description="Render the labeling-guide markdown draft to PDF.",
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    build_guide_pdf(args.source, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
