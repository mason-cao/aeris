"""Reproducible labeling-guide PDF generation from the markdown draft."""

from pathlib import Path

import pytest

from app.eval.guide_pdf import build_guide_pdf
from app.eval.packet_audit import extract_pdf_text


def _write_guide(path: Path) -> None:
    path.write_text(
        "# AERIS Expert Labeling Guide\n"
        "\n"
        "**DRAFT**\n"
        "**Version 2**\n"
        "Intro paragraph with **bold text** inside, plus an escaped\n"
        "mean - 2\\*SD formula.\n"
        "\n"
        "## Label definitions\n"
        "\n"
        "- **Valid (V):** the claim holds up at the precision it states.\n"
        "- **Unsure (U):** you cannot tell from what is shown. This covers\n"
        "  thin evidence and ambiguous wording.\n"
        "\n"
        "1. Mark exactly one box per claim.\n"
        "2. To skip a claim, leave all three boxes blank.\n",
        encoding="utf-8",
    )


def test_build_guide_pdf_renders_headings_lists_and_bold(
    tmp_path: Path,
) -> None:
    source = tmp_path / "guide.md"
    _write_guide(source)
    out = tmp_path / "guide.pdf"

    build_guide_pdf(source, out)

    text = " ".join(extract_pdf_text(out).split())
    assert "AERIS Expert Labeling Guide" in text
    assert "Label definitions" in text
    assert "Valid (V): the claim holds up at the precision it states." in text
    assert "thin evidence and ambiguous wording" in text
    assert "Mark exactly one box per claim." in text
    assert "mean - 2*SD formula" in text


def test_build_guide_pdf_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "guide.md"
    _write_guide(source)
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"

    build_guide_pdf(source, first)
    build_guide_pdf(source, second)

    assert extract_pdf_text(first) == extract_pdf_text(second)
    # Bytes, not just text: ReportLab stamps a wall-clock CreationDate unless
    # the canvas is invariant, so a text-only check passed while every render
    # produced a different file and dirtied the tracked PDF for no reason.
    assert first.read_bytes() == second.read_bytes()


def test_build_guide_pdf_rejects_unsafe_control_characters(
    tmp_path: Path,
) -> None:
    source = tmp_path / "guide.md"
    source.write_text("# Title\n\nbad \x05 character\n", encoding="utf-8")
    out = tmp_path / "guide.pdf"

    with pytest.raises(ValueError, match="U\\+0005"):
        build_guide_pdf(source, out)
    assert not out.exists()
