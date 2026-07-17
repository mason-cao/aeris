"""Deterministic, blinded PDF packets for independent expert review (B4)."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import math
import os
import tempfile
import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aeris-matplotlib")
)

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import Anomaly, EnrichmentRecord, Explanation
from app.eval.label_cli import ClaimGroup, collect_claim_groups, presentation_order
from app.llm.corroboration import (
    CALM_WIND_FLOOR_STATUS,
    CalmWindDecision,
    ClaimType,
    calm_wind_source_decisions,
    classify_claim,
)

PACKET_SCHEMA_VERSION = 2
CALM_WIND_PACKET_NOTE = (
    "default Unsure — wind direction unstable under calm conditions."
)
STATION_SOURCES = ("asos", "epa_aqs", "openaq", "purpleair", "tceq")
WIND_SOURCES = ("asos", "noaa_gfs", "openweather")
SOURCE_COLORS = {
    "asos": "#176B87",
    "epa_aqs": "#4F6F52",
    "openaq": "#C06C3B",
    "purpleair": "#7B5EA7",
    "tceq": "#A13D3D",
}
ALLOWED_CONTROL_WHITESPACE = frozenset(("\t", "\n", "\r"))


@dataclass(frozen=True)
class StationPlotDatum:
    source: str
    entity_id: str
    lat: float
    lon: float

    def to_dict(self) -> dict[str, str | float]:
        return {
            "source": self.source,
            "entity_id": self.entity_id,
            "lat": self.lat,
            "lon": self.lon,
        }


@dataclass(frozen=True)
class WindVectorPlotDatum:
    source: str
    entity_id: str
    timestamp: str
    lat: float
    lon: float
    input_kind: str
    speed: float
    direction_from_degrees: float | None
    input_u_east: float | None
    input_v_north: float | None
    u_east: float
    v_north: float

    def to_dict(self) -> dict[str, str | float | None]:
        return {
            "source": self.source,
            "entity_id": self.entity_id,
            "timestamp": self.timestamp,
            "lat": self.lat,
            "lon": self.lon,
            "input_kind": self.input_kind,
            "speed": self.speed,
            "direction_from_degrees": self.direction_from_degrees,
            "input_u_east": self.input_u_east,
            "input_v_north": self.input_v_north,
            "u_east": self.u_east,
            "v_north": self.v_north,
        }


@dataclass(frozen=True)
class PacketPlotData:
    stations: tuple[StationPlotDatum, ...]
    vectors: tuple[WindVectorPlotDatum, ...]

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "stations": [station.to_dict() for station in self.stations],
            "vectors": [vector.to_dict() for vector in self.vectors],
        }


@dataclass(frozen=True)
class PacketBuildResult:
    pdf_path: Path
    manifest_path: Path
    plot_data: PacketPlotData
    claim_count: int
    calm_wind_flag_count: int
    pdf_sha256: str


@dataclass(frozen=True)
class PacketSource:
    anomaly: Anomaly
    summary: dict[str, Any]
    claim_groups: tuple[ClaimGroup, ...]
    model_names: tuple[str, ...]


def unsafe_text_findings(
    surfaces: Sequence[tuple[str, str]],
) -> tuple[str, ...]:
    """Identify code points that cannot enter an expert-facing packet."""
    findings: list[str] = []
    for surface, text in surfaces:
        for offset, character in enumerate(text):
            category = unicodedata.category(character)
            if category.startswith("C") and character not in ALLOWED_CONTROL_WHITESPACE:
                findings.append(
                    f"{surface}[{offset}]=U+{ord(character):04X} ({category})"
                )
    return tuple(findings)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _ensure_utc(parsed)


def _canonical_timestamp(value: str) -> str:
    return _parse_timestamp(value).isoformat()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary_sha256(summary: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(summary))


@dataclass(frozen=True)
class CalmWindClaimFlag:
    presentation_index: int
    claim_text_sha256: str
    source_decisions: tuple[CalmWindDecision, ...]
    standard_note: str = CALM_WIND_PACKET_NOTE
    floor_status: str = CALM_WIND_FLOOR_STATUS

    def to_dict(self) -> dict[str, Any]:
        return {
            "presentation_index": self.presentation_index,
            "claim_text_sha256": self.claim_text_sha256,
            "source_decisions": [
                decision.to_dict() for decision in self.source_decisions
            ],
            "standard_note": self.standard_note,
            "floor_status": self.floor_status,
        }


def calm_wind_claim_flags(
    summary: Mapping[str, Any],
    ordered_claims: Sequence[ClaimGroup],
) -> tuple[CalmWindClaimFlag, ...]:
    """Derive claim-level packet flags from the same B2 scorer decisions."""
    flags: list[CalmWindClaimFlag] = []
    for index, claim in enumerate(ordered_claims, start=1):
        primary = classify_claim(claim.claim_text)[0]
        if primary is ClaimType.TRANSPORT_DIRECTION:
            sources = ("noaa_gfs", "openweather", "asos")
        elif primary is ClaimType.POINT_SOURCE_ATTRIBUTION:
            sources = ("noaa_gfs", "openweather")
        else:
            continue
        decisions, _notes = calm_wind_source_decisions(summary, sources)
        calm = tuple(
            decisions[source]
            for source in sources
            if decisions[source].calm is True
        )
        if calm:
            flags.append(
                CalmWindClaimFlag(
                    presentation_index=index,
                    claim_text_sha256=_sha256_bytes(
                        claim.claim_text.encode("utf-8")
                    ),
                    source_decisions=calm,
                )
            )
    return tuple(flags)


def _entity_rows(metric: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for entity in metric.get("entities", []):
        entity_id = str(entity["entity_id"])
        if entity_id in rows:
            raise ValueError(f"duplicate entity {entity_id!r} in metric summary")
        rows[entity_id] = entity
    return rows


def _series_by_timestamp(entity: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in entity.get("series", []):
        if not isinstance(row, Sequence) or len(row) != 2:
            raise ValueError("each stored series row must contain timestamp and value")
        timestamp = _canonical_timestamp(str(row[0]))
        value = float(row[1])
        if not math.isfinite(value):
            raise ValueError("stored packet plot series contains a non-finite value")
        if timestamp in values:
            raise ValueError(f"duplicate series timestamp {timestamp!r}")
        values[timestamp] = value
    return values


def _coordinates(entity: Mapping[str, Any]) -> tuple[float, float]:
    lat = float(entity["lat"])
    lon = float(entity["lon"])
    if not math.isfinite(lat) or not math.isfinite(lon):
        raise ValueError("packet plot coordinates must be finite")
    return lat, lon


def _assert_same_coordinates(
    first: Mapping[str, Any], second: Mapping[str, Any], entity_id: str
) -> tuple[float, float]:
    first_coordinates = _coordinates(first)
    second_coordinates = _coordinates(second)
    if first_coordinates != second_coordinates:
        raise ValueError(f"inconsistent coordinates for entity {entity_id!r}")
    return first_coordinates


def _paired_event_values(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    anomaly_timestamp: datetime,
) -> tuple[str, float, float] | None:
    first_values = _series_by_timestamp(first)
    second_values = _series_by_timestamp(second)
    paired_timestamps = sorted(set(first_values) & set(second_values))
    if not paired_timestamps:
        return None
    event_time = _ensure_utc(anomaly_timestamp)
    selected = min(
        paired_timestamps,
        key=lambda timestamp: (
            abs((_parse_timestamp(timestamp) - event_time).total_seconds()),
            timestamp,
        ),
    )
    return selected, first_values[selected], second_values[selected]


def _zero_small(value: float) -> float:
    return 0.0 if abs(value) < 1e-12 else value


def _station_plot_data(summary: Mapping[str, Any]) -> tuple[StationPlotDatum, ...]:
    stations: dict[tuple[str, str], StationPlotDatum] = {}
    sources = summary.get("sources", {})
    for source in STATION_SOURCES:
        block = sources.get(source, {})
        for metric_name in sorted(block.get("metrics", {})):
            metric = block["metrics"][metric_name]
            for entity in metric.get("entities", []):
                entity_id = str(entity["entity_id"])
                lat, lon = _coordinates(entity)
                datum = StationPlotDatum(source, entity_id, lat, lon)
                key = (source, entity_id)
                previous = stations.get(key)
                if previous is not None and previous != datum:
                    raise ValueError(
                        f"inconsistent station coordinates for {source}/{entity_id}"
                    )
                stations[key] = datum
    return tuple(stations[key] for key in sorted(stations))


def _speed_direction_vectors(
    summary: Mapping[str, Any], source: str, anomaly_timestamp: datetime
) -> list[WindVectorPlotDatum]:
    metrics = summary.get("sources", {}).get(source, {}).get("metrics", {})
    speeds = _entity_rows(metrics.get("wind_speed", {}))
    directions = _entity_rows(metrics.get("wind_direction", {}))
    vectors: list[WindVectorPlotDatum] = []
    for entity_id in sorted(set(speeds) & set(directions)):
        speed_entity = speeds[entity_id]
        direction_entity = directions[entity_id]
        lat, lon = _assert_same_coordinates(speed_entity, direction_entity, entity_id)
        paired = _paired_event_values(
            speed_entity, direction_entity, anomaly_timestamp
        )
        if paired is None:
            continue
        timestamp, speed, direction = paired
        radians = math.radians(direction % 360.0)
        vectors.append(
            WindVectorPlotDatum(
                source=source,
                entity_id=entity_id,
                timestamp=timestamp,
                lat=lat,
                lon=lon,
                input_kind="speed_direction",
                speed=speed,
                direction_from_degrees=direction,
                input_u_east=None,
                input_v_north=None,
                u_east=_zero_small(-speed * math.sin(radians)),
                v_north=_zero_small(-speed * math.cos(radians)),
            )
        )
    return vectors


def _gfs_vectors(
    summary: Mapping[str, Any], anomaly_timestamp: datetime
) -> list[WindVectorPlotDatum]:
    metrics = summary.get("sources", {}).get("noaa_gfs", {}).get("metrics", {})
    eastward = _entity_rows(metrics.get("u_10m", {}))
    northward = _entity_rows(metrics.get("v_10m", {}))
    vectors: list[WindVectorPlotDatum] = []
    for entity_id in sorted(set(eastward) & set(northward)):
        u_entity = eastward[entity_id]
        v_entity = northward[entity_id]
        lat, lon = _assert_same_coordinates(u_entity, v_entity, entity_id)
        paired = _paired_event_values(u_entity, v_entity, anomaly_timestamp)
        if paired is None:
            continue
        timestamp, u_east, v_north = paired
        vectors.append(
            WindVectorPlotDatum(
                source="noaa_gfs",
                entity_id=entity_id,
                timestamp=timestamp,
                lat=lat,
                lon=lon,
                input_kind="uv",
                speed=math.hypot(u_east, v_north),
                direction_from_degrees=None,
                input_u_east=u_east,
                input_v_north=v_north,
                u_east=u_east,
                v_north=v_north,
            )
        )
    return vectors


def extract_plot_data(
    summary: Mapping[str, Any], anomaly_timestamp: datetime
) -> PacketPlotData:
    """Extract every plotted value directly from the stored 72-hour summary."""
    vectors = [
        *_speed_direction_vectors(summary, "asos", anomaly_timestamp),
        *_gfs_vectors(summary, anomaly_timestamp),
        *_speed_direction_vectors(summary, "openweather", anomaly_timestamp),
    ]
    vectors.sort(key=lambda item: (item.source, item.entity_id, item.timestamp))
    return PacketPlotData(
        stations=_station_plot_data(summary),
        vectors=tuple(vectors),
    )


def _plot_limits(
    lons: Sequence[float], lats: Sequence[float], anomaly: Anomaly
) -> tuple[tuple[float, float], tuple[float, float]]:
    all_lons = [*lons, float(anomaly.lon)]
    all_lats = [*lats, float(anomaly.lat)]
    min_lon, max_lon = min(all_lons), max(all_lons)
    min_lat, max_lat = min(all_lats), max(all_lats)
    lon_pad = max((max_lon - min_lon) * 0.08, 0.02)
    lat_pad = max((max_lat - min_lat) * 0.08, 0.02)
    return (
        (min_lon - lon_pad, max_lon + lon_pad),
        (min_lat - lat_pad, max_lat + lat_pad),
    )


def _station_figure(anomaly: Anomaly, plot_data: PacketPlotData) -> bytes:
    figure, axis = plt.subplots(figsize=(7.2, 4.1), constrained_layout=True)
    for source in STATION_SOURCES:
        rows = [station for station in plot_data.stations if station.source == source]
        if not rows:
            continue
        axis.scatter(
            [row.lon for row in rows],
            [row.lat for row in rows],
            s=22,
            color=SOURCE_COLORS[source],
            edgecolors="white",
            linewidths=0.4,
            label=f"{source} ({len(rows)})",
            zorder=2,
        )
    axis.scatter(
        [anomaly.lon],
        [anomaly.lat],
        marker="*",
        s=150,
        color="#111111",
        label="anomaly",
        zorder=4,
    )
    lons = [station.lon for station in plot_data.stations]
    lats = [station.lat for station in plot_data.stations]
    xlim, ylim = _plot_limits(lons, lats, anomaly)
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_xlabel("Longitude (degrees east)")
    axis.set_ylabel("Latitude (degrees north)")
    axis.set_title("Ground observation locations in the stored 72-hour context")
    axis.grid(color="#D8D8D8", linewidth=0.5, alpha=0.8)
    axis.legend(loc="best", fontsize=7, frameon=False, ncol=2)
    axis.set_aspect("equal", adjustable="box")
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=180,
        metadata={"Software": "AERIS"},
    )
    plt.close(figure)
    return buffer.getvalue()


def _wind_figure(anomaly: Anomaly, plot_data: PacketPlotData) -> bytes:
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), constrained_layout=True)
    for axis, source in zip(axes, WIND_SOURCES, strict=True):
        rows = [vector for vector in plot_data.vectors if vector.source == source]
        if not rows:
            axis.text(0.5, 0.5, "No paired wind values", ha="center", va="center")
            axis.set_axis_off()
            axis.set_title(source)
            continue
        lons = [row.lon for row in rows]
        lats = [row.lat for row in rows]
        xlim, ylim = _plot_limits(lons, lats, anomaly)
        extent = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        max_speed = max(row.speed for row in rows)
        component_scale = 0.18 * extent / max(max_speed, 1e-12)
        axis.scatter(lons, lats, s=14, color="#767676", zorder=1)
        axis.quiver(
            lons,
            lats,
            [row.u_east * component_scale for row in rows],
            [row.v_north * component_scale for row in rows],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.009,
            color="#176B87",
            zorder=3,
        )
        axis.scatter(
            [anomaly.lon], [anomaly.lat], marker="*", s=70, color="#111111", zorder=4
        )
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_title(f"{source}\nlongest arrow = {max_speed:.2f} m/s", fontsize=8)
        axis.grid(color="#E0E0E0", linewidth=0.4)
        axis.tick_params(labelsize=6)
        axis.set_aspect("equal", adjustable="box")
    figure.suptitle(
        "Event-nearest wind vectors from stored values (arrow points downwind)",
        fontsize=10,
    )
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=180,
        metadata={"Software": "AERIS"},
    )
    plt.close(figure)
    return buffer.getvalue()


def _format_number(value: Any) -> str:
    if value is None:
        return "missing"
    number = float(value)
    if not math.isfinite(number):
        return "non-finite"
    return f"{number:.6g}"


def _format_time(value: Any) -> str:
    if not value:
        return "missing"
    return _parse_timestamp(str(value)).strftime("%Y-%m-%d %H:%MZ")


def _evidence_rows(summary: Mapping[str, Any]) -> list[list[str]]:
    rows = [["Source", "Metric", "Unit", "Entities / points", "Range; mean", "Nearest to event"]]
    for source in sorted(summary.get("sources", {})):
        metrics = summary["sources"][source].get("metrics", {})
        for metric_name in sorted(metrics):
            metric = metrics[metric_name]
            value_range = metric.get("value_range", {})
            nearest = metric.get("nearest_in_time", {})
            rows.append(
                [
                    source,
                    metric_name,
                    str(metric.get("unit") or "unit not recorded"),
                    f"{metric.get('n_entities', 0)} / {metric.get('n_points', 0)}",
                    (
                        f"{_format_number(value_range.get('min'))} to "
                        f"{_format_number(value_range.get('max'))}; "
                        f"mean {_format_number(value_range.get('mean'))}"
                    ),
                    (
                        f"{_format_number(nearest.get('v'))} at "
                        f"{_format_time(nearest.get('t'))}; "
                        f"dt {_format_number(nearest.get('dt_minutes'))} min"
                    ),
                ]
            )
    return rows


def _packet_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "PacketTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#153243"),
            alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "PacketSubtitle",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#8A3F2B"),
            spaceAfter=10,
        ),
        "heading": ParagraphStyle(
            "PacketHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#153243"),
            spaceBefore=8,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "PacketBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "PacketSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
        ),
        "claim_index": ParagraphStyle(
            "ClaimIndex",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12,
            textColor=colors.HexColor("#8A3F2B"),
            spaceAfter=4,
        ),
        "claim": ParagraphStyle(
            "ClaimText",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            borderColor=colors.HexColor("#D2D8DC"),
            borderWidth=0.6,
            borderPadding=8,
            backColor=colors.HexColor("#F7F9FA"),
            spaceAfter=8,
        ),
        "calm_flag": ParagraphStyle(
            "CalmWindFlag",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#713B18"),
            borderColor=colors.HexColor("#D7A15D"),
            borderWidth=0.6,
            borderPadding=6,
            backColor=colors.HexColor("#FFF5E6"),
            spaceAfter=8,
        ),
        "center": ParagraphStyle(
            "PacketCenter",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
        ),
    }


def _invariant_canvas(filename: str, **kwargs: Any) -> canvas.Canvas:
    kwargs["invariant"] = 1
    return canvas.Canvas(filename, **kwargs)


def _set_pdf_metadata(pdf_canvas: canvas.Canvas, _document: Any) -> None:
    pdf_canvas.setTitle("AERIS Expert Review Packet")
    pdf_canvas.setAuthor("Mason Cao")
    pdf_canvas.setSubject("Blinded scientific claim review")
    pdf_canvas.setCreator("AERIS evaluation packet")
    pdf_canvas.setKeywords("expert review, atmospheric evidence")


def _anomaly_table(anomaly: Anomaly, styles: Mapping[str, ParagraphStyle]) -> Table:
    timestamp = _ensure_utc(anomaly.timestamp).strftime("%Y-%m-%d %H:%M UTC")
    expected = _format_number(anomaly.expected_value)
    detectors = ", ".join(str(item) for item in anomaly.methods_triggered)
    rows = [
        ["Source / metric", f"{anomaly.source} / {anomaly.metric}"],
        ["Observed / expected", f"{_format_number(anomaly.value)} / {expected}"],
        ["Time", timestamp],
        ["Location", f"{anomaly.lat:.4f}, {anomaly.lon:.4f}"],
        ["Severity", anomaly.severity],
        ["Detectors", detectors],
    ]
    table = Table(
        [[Paragraph(escape(key), styles["small"]), Paragraph(escape(value), styles["small"])] for key, value in rows],
        colWidths=[1.35 * inch, 5.15 * inch],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF2F4")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1C2B33")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D0D5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _evidence_table(summary: Mapping[str, Any], styles: Mapping[str, ParagraphStyle]) -> Table:
    rows = _evidence_rows(summary)
    cells = [
        [Paragraph(escape(str(value)), styles["small"]) for value in row]
        for row in rows
    ]
    table = Table(
        cells,
        colWidths=[0.72 * inch, 1.05 * inch, 0.72 * inch, 0.72 * inch, 1.43 * inch, 1.86 * inch],
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#153243")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F6F7")]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C9D0D4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _claim_block(
    index: int,
    total: int,
    claim: ClaimGroup,
    styles: Mapping[str, ParagraphStyle],
    calm_wind_flag: CalmWindClaimFlag | None,
) -> KeepTogether:
    marking = Table(
        [["Mark exactly one:", "[   ] V", "[   ] I", "[   ] U"]],
        colWidths=[1.75 * inch, 1.3 * inch, 1.3 * inch, 1.3 * inch],
        hAlign="LEFT",
    )
    marking.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (-1, 0), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#9DA8AE")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D5DADD")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    note = Table(
        [["Note:", ""], ["", ""], ["", ""]],
        colWidths=[0.55 * inch, 5.95 * inch],
        rowHeights=[0.24 * inch, 0.24 * inch, 0.24 * inch],
        hAlign="LEFT",
    )
    note.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LINEBELOW", (1, 0), (1, -1), 0.4, colors.HexColor("#AEB7BC")),
                ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
            ]
        )
    )
    content: list[Any] = [
        Paragraph(
            f"Claim {index} of {total} - presentation index {index}",
            styles["claim_index"],
        ),
        Paragraph(escape(claim.claim_text), styles["claim"]),
    ]
    if calm_wind_flag is not None:
        details = "; ".join(
            (
                f"{decision.source}: event {decision.event_speed_ms} m/s, "
                f"raw cutoff {decision.raw_cutoff_ms} m/s, effective cutoff "
                f"{decision.effective_cutoff_ms} m/s, n={decision.window_n}"
            )
            for decision in calm_wind_flag.source_decisions
        )
        status = (
            calm_wind_flag.floor_status.replace("_", " ")
            .replace("bracco", "Bracco")
        )
        content.append(
            Paragraph(
                escape(
                    f"{calm_wind_flag.standard_note} Flagged source(s): "
                    f"{details}. Floor status: {status}."
                ),
                styles["calm_flag"],
            )
        )
    content.extend((marking, Spacer(1, 5), note, Spacer(1, 12)))
    return KeepTogether(content)


def _build_story(
    anomaly: Anomaly,
    summary: Mapping[str, Any],
    ordered_claims: Sequence[ClaimGroup],
    plot_data: PacketPlotData,
    calm_wind_flags: Sequence[CalmWindClaimFlag],
    *,
    example: bool,
) -> list[Any]:
    styles = _packet_styles()
    story: list[Any] = [Paragraph("AERIS Expert Review Packet", styles["title"])]
    if example:
        story.append(
            Paragraph(
                "DRAFT EXAMPLE - format review only; do not return as an official label.",
                styles["subtitle"],
            )
        )
    story.extend(
        [
            Paragraph("Anomaly", styles["heading"]),
            _anomaly_table(anomaly, styles),
            Spacer(1, 8),
            Paragraph("Annotation instructions", styles["heading"]),
            Paragraph(
                "For each numbered claim, mark exactly one box: V (Valid), I (Invalid), or U (Unsure). "
                "Leave all boxes blank if you do not answer a claim; an unanswered claim is recorded as missing, "
                "not as Unsure. Add a short note for Invalid or Unsure when the reason is not obvious; a note on "
                "Valid is optional.",
                styles["body"],
            ),
            Paragraph(
                "After return, the checksummed annotated PDF is the primary artifact. Mason transcribes it twice "
                "in independent sessions; the second pass is blind to the first. A mechanical diff identifies "
                "mismatches for resolution against that PDF. Ambiguous marks are never interpreted; they are sent "
                "back as an enumerated clarification list before ingestion.",
                styles["body"],
            ),
            Paragraph(
                "Valid: scientifically defensible at the precision stated and supported by the evidence shown. "
                "Invalid: contradicted by the evidence, misstates a measurement, or uses indefensible physical "
                "reasoning. Unsure: the displayed evidence is insufficient, the measurements cannot resolve the "
                "claim, the wording is ambiguous, or the claim falls outside what you can judge confidently. "
                "Missing or insufficient evidence is Unsure.",
                styles["body"],
            ),
            Paragraph(
                "The location plot uses only coordinates stored in the 72-hour context. Each wind panel uses the "
                "timestamp-paired observation nearest the event for that entity. Arrows point downwind; direction "
                "encodes motion and length encodes speed in m/s within each panel. No basemap or external data is "
                "included.",
                styles["body"],
            ),
            Paragraph("Stored evidence summary", styles["heading"]),
            _evidence_table(summary, styles),
            Spacer(1, 10),
            Image(io.BytesIO(_station_figure(anomaly, plot_data)), width=6.5 * inch, height=3.7 * inch),
            Spacer(1, 6),
            Image(io.BytesIO(_wind_figure(anomaly, plot_data)), width=6.5 * inch, height=2.53 * inch),
            PageBreak(),
            Paragraph("Claims", styles["heading"]),
            Paragraph(
                "Claims are shown once in a deterministic labeler-specific order. The presentation index is the "
                "number printed with each claim.",
                styles["body"],
            ),
        ]
    )
    total = len(ordered_claims)
    flags_by_index = {
        flag.presentation_index: flag for flag in calm_wind_flags
    }
    story.extend(
        _claim_block(index, total, claim, styles, flags_by_index.get(index))
        for index, claim in enumerate(ordered_claims, start=1)
    )
    story.extend(
        [
            PageBreak(),
            Paragraph("Optional event-level note", styles["heading"]),
            Paragraph(
                "Most likely true cause, in your own words. It is acceptable to write: insufficient evidence to "
                "determine a single cause.",
                styles["body"],
            ),
            Table(
                [[""], [""], [""]],
                colWidths=[6.5 * inch],
                rowHeights=[0.35 * inch] * 3,
                style=TableStyle(
                    [("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#AEB7BC"))]
                ),
            ),
        ]
    )
    return story


def _write_manifest(
    path: Path,
    anomaly: Anomaly,
    labeler: str,
    summary: Mapping[str, Any],
    plot_data: PacketPlotData,
    ordered_claims: Sequence[ClaimGroup],
    calm_wind_flags: Sequence[CalmWindClaimFlag],
    pdf_sha256: str,
    *,
    example: bool,
) -> None:
    payload = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "packet_kind": "example" if example else "official",
        "anomaly_id": str(anomaly.id),
        "labeler": labeler,
        "summary_sha256": summary_sha256(summary),
        "pdf_sha256": pdf_sha256,
        "plot_data": plot_data.to_dict(),
        "calm_wind_flags": [flag.to_dict() for flag in calm_wind_flags],
        "claims": [
            {
                "presentation_index": index,
                "claim_text_sha256": _sha256_bytes(claim.claim_text.encode("utf-8")),
            }
            for index, claim in enumerate(ordered_claims, start=1)
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def render_packet(
    anomaly: Anomaly,
    summary: Mapping[str, Any],
    claim_groups: Sequence[ClaimGroup],
    labeler: str,
    output_path: Path,
    manifest_path: Path,
    *,
    example: bool = True,
) -> PacketBuildResult:
    """Render one deterministic packet and its evidence-audit sidecar."""
    if not claim_groups:
        raise ValueError("packet requires at least one unique claim")
    ordered_claims = presentation_order(list(claim_groups), anomaly.id, labeler)
    text_findings = unsafe_text_findings(
        tuple(
            (f"source claim {index}", claim.claim_text)
            for index, claim in enumerate(ordered_claims, start=1)
        )
    )
    if text_findings:
        raise ValueError("unsafe packet text character(s): " + "; ".join(text_findings))
    output_path = output_path.resolve()
    manifest_path = manifest_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    plot_data = extract_plot_data(summary, anomaly.timestamp)
    calm_wind_flags = calm_wind_claim_flags(summary, ordered_claims)
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        title="AERIS Expert Review Packet",
        author="Mason Cao",
        subject="Blinded scientific claim review",
        creator="AERIS evaluation packet",
        pageCompression=1,
    )
    document.build(
        _build_story(
            anomaly,
            summary,
            ordered_claims,
            plot_data,
            calm_wind_flags,
            example=example,
        ),
        onFirstPage=_set_pdf_metadata,
        onLaterPages=_set_pdf_metadata,
        canvasmaker=_invariant_canvas,
    )
    pdf_hash = _sha256_file(output_path)
    _write_manifest(
        manifest_path,
        anomaly,
        labeler,
        summary,
        plot_data,
        ordered_claims,
        calm_wind_flags,
        pdf_hash,
        example=example,
    )
    return PacketBuildResult(
        pdf_path=output_path,
        manifest_path=manifest_path,
        plot_data=plot_data,
        claim_count=len(ordered_claims),
        calm_wind_flag_count=len(calm_wind_flags),
        pdf_sha256=pdf_hash,
    )


def _readonly_sqlite_url(database_path: Path) -> str:
    resolved = database_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"database does not exist: {resolved}")
    return f"sqlite+aiosqlite:///file:{resolved}?mode=ro&uri=true"


async def load_packet_source(database_path: Path, anomaly_id: uuid.UUID) -> PacketSource:
    """Read packet inputs from SQLite without opening a writable connection."""
    engine = create_async_engine(_readonly_sqlite_url(database_path), echo=False)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with session_factory() as session:
            anomaly = await session.get(Anomaly, anomaly_id)
            if anomaly is None:
                raise ValueError(f"no anomaly with id {anomaly_id}")
            record = (
                await session.execute(
                    select(EnrichmentRecord)
                    .where(EnrichmentRecord.anomaly_id == anomaly_id)
                    .order_by(EnrichmentRecord.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if record is None:
                raise ValueError(f"anomaly {anomaly_id} has no enrichment record")
            groups = await collect_claim_groups(session, anomaly_id)
            if not groups:
                raise ValueError(f"anomaly {anomaly_id} has no claims")
            model_names = tuple(
                sorted(
                    str(name)
                    for name in (
                        await session.execute(
                            select(Explanation.model_name)
                            .where(Explanation.anomaly_id == anomaly_id)
                            .distinct()
                        )
                    ).scalars()
                )
            )
            summary = dict(record.cross_source_summary_json)
            return PacketSource(anomaly, summary, tuple(groups), model_names)
    finally:
        await engine.dispose()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.packet",
        description="Render one deterministic blinded expert-review PDF packet.",
    )
    parser.add_argument("--database", required=True, type=Path, help="read-only SQLite database")
    parser.add_argument("--anomaly-id", required=True, help="anomaly UUID")
    parser.add_argument("--labeler", required=True, help="labeler seed/name")
    parser.add_argument("--out", required=True, type=Path, help="output PDF path")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="audit sidecar path (default: <out>.audit.json)",
    )
    parser.add_argument(
        "--official",
        action="store_true",
        help="omit the example-only warning (prohibited until the freeze gate passes)",
    )
    return parser.parse_args(argv)


async def _amain(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.official:
        raise ValueError("official packet generation remains blocked by the freeze gate")
    source = await load_packet_source(args.database, uuid.UUID(args.anomaly_id))
    manifest_path = args.manifest or args.out.with_suffix(".audit.json")
    result = render_packet(
        source.anomaly,
        source.summary,
        source.claim_groups,
        args.labeler,
        args.out,
        manifest_path,
        example=True,
    )
    print(
        f"rendered {result.claim_count} unique claims to {result.pdf_path} "
        f"(sha256 {result.pdf_sha256})"
    )
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
