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

from app.eval.label_cli import ClaimGroup, presentation_order
from app.eval.packet import (
    calm_wind_claim_flags,
    evidence_detail_lines,
    extract_plot_data,
    load_packet_source,
    summary_sha256,
    unsafe_text_findings,
)

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
    calm_wind_flag_count: int
    evidence_detail_count: int
    blinding_leaks: tuple[str, ...]
    text_sanitation_findings: tuple[str, ...]


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
    claim_groups: tuple[ClaimGroup, ...],
) -> PacketAuditReport:
    """Assert that a rendered packet matches stored plot data and stays blind."""
    if not pdf_path.is_file():
        raise PacketAuditError(f"packet PDF does not exist: {pdf_path}")
    if not manifest_path.is_file():
        raise PacketAuditError(f"packet audit manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 3:
        raise PacketAuditError("unsupported packet audit schema")

    actual_pdf_hash = _sha256_file(pdf_path)
    if manifest.get("pdf_sha256") != actual_pdf_hash:
        raise PacketAuditError("PDF hash does not match the audit manifest")
    if manifest.get("summary_sha256") != summary_sha256(summary):
        raise PacketAuditError("stored summary does not match the packet manifest")

    expected_plot_data = extract_plot_data(summary, anomaly_timestamp).to_dict()
    if manifest.get("plot_data") != expected_plot_data:
        raise PacketAuditError("plot data does not match the stored 72-hour context")

    # The packet must carry every bucket and per-entity mean the model reasoned
    # from. When it did not, a labeler asked about one of those numbers could
    # only answer Unsure, which reads as model vagueness and corrupts the
    # headline agreement measurement.
    expected_detail = evidence_detail_lines(summary)
    if manifest.get("evidence_detail") != expected_detail:
        raise PacketAuditError(
            "evidence detail lines do not match the stored 72-hour context"
        )

    try:
        anomaly_id = uuid.UUID(str(manifest["anomaly_id"]))
        labeler = str(manifest["labeler"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PacketAuditError("invalid packet identity in audit manifest") from exc
    ordered_claims = presentation_order(list(claim_groups), anomaly_id, labeler)
    source_text_findings = unsafe_text_findings(
        tuple(
            (f"source claim {index}", claim.claim_text)
            for index, claim in enumerate(ordered_claims, start=1)
        )
    )
    if source_text_findings:
        raise PacketAuditError(
            "unsafe packet text character(s): " + "; ".join(source_text_findings)
        )
    expected_flags = [
        flag.to_dict()
        for flag in calm_wind_claim_flags(summary, ordered_claims)
    ]
    if manifest.get("calm_wind_flags") != expected_flags:
        raise PacketAuditError(
            "calm-wind flags do not match the stored 72-hour context"
        )

    text = extract_pdf_text(pdf_path)
    if expected_flags and "wind direction unstable under calm conditions" not in text:
        raise PacketAuditError("calm-wind flag text is missing from the PDF")
    # Manifest agreement proves the lines were computed; this proves they were
    # actually rendered. Checked on the bucket-means marker only, because PDF
    # text extraction rewraps long series lines unpredictably.
    if expected_detail and "h means" not in text:
        raise PacketAuditError("evidence detail rows are missing from the PDF")
    metadata = extract_pdf_metadata(pdf_path)
    rendered_text_findings = unsafe_text_findings(
        (
            ("extracted PDF text", text),
            *(
                surface
                for key, value in sorted(metadata.items())
                for surface in (
                    (f"PDF metadata key {key!r}", key),
                    (f"PDF metadata value for {key!r}", value),
                )
            ),
        )
    )
    if rendered_text_findings:
        raise PacketAuditError(
            "unsafe packet text character(s): " + "; ".join(rendered_text_findings)
        )
    leaks = scan_blinding_leaks(text, metadata, model_names=model_names)
    if leaks:
        raise PacketAuditError(f"blinding leak(s) found: {', '.join(leaks)}")

    return PacketAuditReport(
        pdf_sha256=actual_pdf_hash,
        station_count=len(expected_plot_data["stations"]),
        vector_count=len(expected_plot_data["vectors"]),
        calm_wind_flag_count=len(expected_flags),
        evidence_detail_count=len(expected_detail),
        blinding_leaks=leaks,
        text_sanitation_findings=(),
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
        claim_groups=source.claim_groups,
    )
    print(
        f"audit passed: {report.station_count} stations, "
        f"{report.vector_count} vectors, {report.calm_wind_flag_count} "
        f"calm-wind flags, {report.evidence_detail_count} evidence detail lines, "
        f"sha256 {report.pdf_sha256}"
    )
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
