"""Pre-send evidence-faithfulness and blinding audit for B4 packets."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from app.eval.packet import extract_plot_data, load_packet_source, summary_sha256

GENERIC_BLINDING_TERMS = (
    "confidence",
    "grounded",
    "grounding verdict",
    "unverified",
    "corroboration",
    "corroboration score",
    "per-source verdict",
    "per-channel verdict",
    "supporting vote",
    "contradicting vote",
)


class PacketAuditError(RuntimeError):
    """Raised when a packet fails a pre-send invariant."""


@dataclass(frozen=True)
class PacketAuditReport:
    pdf_sha256: str
    station_count: int
    vector_count: int
    blinding_leaks: tuple[str, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_pdf_metadata(pdf_path: Path) -> dict[str, str]:
    reader = PdfReader(str(pdf_path))
    metadata = reader.metadata or {}
    return {str(key): str(value) for key, value in metadata.items()}


def scan_blinding_leaks(
    text: str,
    metadata: dict[str, str],
    *,
    model_names: tuple[str, ...],
) -> tuple[str, ...]:
    searchable = "\n".join([text, *(f"{key}: {value}" for key, value in metadata.items())]).casefold()
    terms = [*GENERIC_BLINDING_TERMS, *(name for name in model_names if name)]
    return tuple(sorted({term for term in terms if term.casefold() in searchable}))


def audit_packet(
    pdf_path: Path,
    manifest_path: Path,
    summary: dict[str, Any],
    anomaly_timestamp: datetime,
    *,
    model_names: tuple[str, ...],
) -> PacketAuditReport:
    """Assert that a rendered packet matches stored plot data and stays blind."""
    if not pdf_path.is_file():
        raise PacketAuditError(f"packet PDF does not exist: {pdf_path}")
    if not manifest_path.is_file():
        raise PacketAuditError(f"packet audit manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise PacketAuditError("unsupported packet audit schema")

    actual_pdf_hash = _sha256_file(pdf_path)
    if manifest.get("pdf_sha256") != actual_pdf_hash:
        raise PacketAuditError("PDF hash does not match the audit manifest")
    if manifest.get("summary_sha256") != summary_sha256(summary):
        raise PacketAuditError("stored summary does not match the packet manifest")

    expected_plot_data = extract_plot_data(summary, anomaly_timestamp).to_dict()
    if manifest.get("plot_data") != expected_plot_data:
        raise PacketAuditError("plot data does not match the stored 72-hour context")

    text = extract_pdf_text(pdf_path)
    metadata = extract_pdf_metadata(pdf_path)
    leaks = scan_blinding_leaks(text, metadata, model_names=model_names)
    if leaks:
        raise PacketAuditError(f"blinding leak(s) found: {', '.join(leaks)}")

    return PacketAuditReport(
        pdf_sha256=actual_pdf_hash,
        station_count=len(expected_plot_data["stations"]),
        vector_count=len(expected_plot_data["vectors"]),
        blinding_leaks=leaks,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.packet_audit",
        description="Audit a packet against its stored enrichment context.",
    )
    parser.add_argument("--database", required=True, type=Path, help="read-only SQLite database")
    parser.add_argument("--anomaly-id", required=True, help="anomaly UUID")
    parser.add_argument("--pdf", required=True, type=Path, help="packet PDF")
    parser.add_argument("--manifest", required=True, type=Path, help="packet audit sidecar")
    return parser.parse_args(argv)


async def _amain(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    source = await load_packet_source(args.database, uuid.UUID(args.anomaly_id))
    report = audit_packet(
        args.pdf,
        args.manifest,
        source.summary,
        source.anomaly.timestamp,
        model_names=source.model_names,
    )
    print(
        f"audit passed: {report.station_count} stations, "
        f"{report.vector_count} vectors, sha256 {report.pdf_sha256}"
    )
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
