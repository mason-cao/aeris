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
    Flowable,
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
from app.llm.explain import entity_terms, render_bucket_means, render_entity_means

# 3: evidence rows carry the bucket and per-entity mean lines the model was
# given, and the manifest records them so the audit can detect drift.
PACKET_SCHEMA_VERSION = 3
CALM_WIND_PACKET_NOTE = "wind direction unstable under calm conditions."
CALM_WIND_FLOOR_DISPLAY = {
    "bracco_confirmed": "confirmed 2026-07-24",
    "proposed_pending_bracco_amendment": "proposed, not yet confirmed",
    "not_configured": "not configured",
}
STATION_SOURCES = ("asos", "epa_aqs", "openaq", "purpleair", "tceq")
WIND_SOURCES = ("asos", "noaa_gfs", "openweather")
# Distance rings on the station map. Without them a labeler cannot check a
# distance claim without measuring off the page, which sends checkable claims
# to Unsure. Spaced to span the 50 km enrichment radius.
DISTANCE_RING_KM = (10.0, 20.0, 30.0, 40.0, 50.0)
_RING_VERTICES = 181
_EARTH_RADIUS_KM = 6371.0
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


def ring_coordinates(
    lat: float, lon: float, radius_km: float
) -> tuple[list[float], list[float]]:
    """Closed (lons, lats) polygon of points exactly ``radius_km`` from a centre.

    Spherical destination-point formula on the same 6371 km sphere
    ``geo.distance_km`` measures with, so a plotted ring is the true locus of
    that distance rather than a small-angle circle drawn in degree space —
    which would be wrong by the cos(latitude) factor along the east-west axis.
    """
    phi1 = math.radians(lat)
    lambda1 = math.radians(lon)
    delta = radius_km / _EARTH_RADIUS_KM
    lons: list[float] = []
    lats: list[float] = []
    for index in range(_RING_VERTICES):
        theta = 2.0 * math.pi * index / (_RING_VERTICES - 1)
        phi2 = math.asin(
            math.sin(phi1) * math.cos(delta)
            + math.cos(phi1) * math.sin(delta) * math.cos(theta)
        )
        lambda2 = lambda1 + math.atan2(
            math.sin(theta) * math.sin(delta) * math.cos(phi1),
            math.cos(delta) - math.sin(phi1) * math.sin(phi2),
        )
        lats.append(math.degrees(phi2))
        lons.append(math.degrees(lambda2))
    return lons, lats


def _station_figure(anomaly: Anomaly, plot_data: PacketPlotData) -> bytes:
    figure, axis = plt.subplots(figsize=(7.2, 4.1), constrained_layout=True)
    lons = [station.lon for station in plot_data.stations]
    lats = [station.lat for station in plot_data.stations]
    xlim, ylim = _plot_limits(lons, lats, anomaly)
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    # Rings first so stations draw over them. A ring whose northern label would
    # fall outside the view is skipped entirely rather than drawn unlabelled:
    # an unlabelled ring is a distance scale the labeler cannot read.
    for radius_km in DISTANCE_RING_KM:
        ring_lons, ring_lats = ring_coordinates(
            float(anomaly.lat), float(anomaly.lon), radius_km
        )
        north = max(ring_lats)
        if north > ylim[1]:
            continue
        axis.plot(
            ring_lons,
            ring_lats,
            color="#9AA6AD",
            linewidth=0.5,
            linestyle=(0, (4, 3)),
            zorder=1,
        )
        axis.annotate(
            f"{radius_km:.0f} km",
            xy=(float(anomaly.lon), north),
            xytext=(1.5, 1.5),
            textcoords="offset points",
            fontsize=6,
            color="#6B767D",
            # Opaque backing: ring labels sit over the densest part of the
            # station cloud whenever the event is on the edge of the network,
            # which is exactly when reading distances off the map matters most.
            bbox={
                "boxstyle": "square,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
            zorder=3,
        )
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
    axis.set_xlabel("Longitude (degrees east)")
    axis.set_ylabel("Latitude (degrees north)")
    axis.set_title(
        "Ground observation locations in the stored 72-hour context\n"
        "dashed rings are great-circle distance from the anomaly"
    )
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


# The stored metric names are the pipeline's, and a claim that quotes one quotes
# it in this form, so the name stays printed. The label in front of it is for the
# reader: `s5p_aer_ai_granule_available` is not a phrase anyone should have to
# decode mid-judgment.
METRIC_LABELS: dict[str, str] = {
    "cloud_cover": "Cloud cover",
    "co": "Carbon monoxide",
    "gh_500": "500-hPa geopotential height",
    "humidity": "Relative humidity",
    "no2": "Nitrogen dioxide",
    "ozone": "Ozone",
    "pbl_height": "Planetary boundary layer height",
    "pm10": "PM10",
    "pm25": "PM2.5",
    "precipitable_water": "Precipitable water",
    "precipitation": "Precipitation rate",
    "pressure": "Surface pressure",
    # Formulas for the satellite rows: standard notation, and short enough that
    # the label does not wrap ahead of the stored name beside it.
    "s5p_aer_ai_granule_available": "Aerosol index granules",
    "s5p_ch4_granule_available": "CH4 granules available",
    "s5p_co_column": "CO column",
    "s5p_co_granule_available": "CO granules available",
    "s5p_hcho_column": "HCHO column",
    "s5p_hcho_granule_available": "HCHO granules available",
    "s5p_no2_column": "NO2 column",
    "s5p_no2_granule_available": "NO2 granules available",
    "s5p_o3_granule_available": "O3 granules available",
    "s5p_so2_column": "SO2 column",
    "s5p_so2_granule_available": "SO2 granules available",
    "so2": "Sulfur dioxide",
    "surface_pressure": "Surface pressure",
    "t_850": "850-hPa temperature",
    "temperature": "Air temperature",
    "u_10m": "10-m eastward wind",
    "v_10m": "10-m northward wind",
    "wind_direction": "Wind direction",
    "wind_speed": "Wind speed",
}


def metric_label(metric: str) -> str:
    """Reader's label plus the stored name. Unmapped metrics print bare."""
    label = METRIC_LABELS.get(metric)
    return f"{label} ({metric})" if label else metric


# The metric column carries a label as well as the stored name now, and the last
# column spells its offset out in words, so both take room from the three columns
# holding short fixed-width values. The metric column's 1.55in is set by the
# longest stored name, `(s5p_aer_ai_granule_available)`: any narrower and the
# identifier wraps mid-token, and that identifier is the string she matches a
# claim against. `test_metric_column_fits_the_longest_stored_name` pins it.
EVIDENCE_COL_WIDTHS: tuple[float, ...] = (0.69, 1.55, 0.48, 0.52, 0.75, 2.51)
SERIES_COL_WIDTHS: tuple[float, ...] = (0.69, 1.55, 4.26)
# `_grid_table` sets 3pt of padding on each side of every cell.
GRID_CELL_PADDING_IN = 6.0 / 72.0


def _format_event_offset(
    dt_minutes: Any, observed_at: Any, event: datetime
) -> str:
    """How far a reading sits from the event, in words and with a direction.

    A satellite overpass can be hours from the anomaly, and `dt 356.4 min` in a
    table cell reads as a small number. Spelling out the gap and which side of
    the event it falls on is what stops a distant overpass from being taken for
    a contemporaneous measurement.
    """
    if dt_minutes is None:
        return "offset not recorded"
    minutes = float(dt_minutes)
    if not math.isfinite(minutes):
        return "offset not recorded"
    total = int(round(abs(minutes)))
    if total == 0:
        return "at event time"
    hours, remainder = divmod(total, 60)
    if hours and remainder:
        magnitude = f"{hours} h {remainder} min"
    elif hours:
        magnitude = f"{hours} h"
    else:
        magnitude = f"{remainder} min"
    if not observed_at:
        return f"{magnitude} from event"
    observed = _parse_timestamp(str(observed_at))
    side = "after" if observed > _ensure_utc(event) else "before"
    return f"{magnitude} {side} event"


# Mirrors `consensus._resolve_expected_value`: Z-score's baseline wins over
# STL's, and isolation forest supplies none. The stored anomaly keeps only
# `methods_triggered`, so that is what the precedence reads here.
BASELINE_METHODS: dict[str, str] = {
    "zscore": "mean of the preceding 7 days of this series (Z-score detector)",
    "stl": "seasonal-trend fit for this time of day (STL detector)",
}


def baseline_method(methods_triggered: Sequence[Any]) -> str:
    methods = [str(method) for method in methods_triggered]
    for detector in ("zscore", "stl"):
        if detector in methods:
            return BASELINE_METHODS[detector]
    return "not recorded for this event"


# The series the models were given ride in their own tables rather than as
# spanned rows inside the evidence table. A full-width row breaks the column
# grid, and Bracco signed off on that grid on 2026-07-24; a sibling table in the
# same style adds the numbers without touching a layout she has already read.
SERIES_SECTIONS: tuple[tuple[int, str, str], ...] = (
    (
        0,
        "Six-hour averages",
        "Each metric averaged in 6-hour blocks across the stored 72 hours, in UTC. "
        "These averages, and the per-site ones below, are what the models were shown.",
    ),
    (
        1,
        "Per-site averages",
        "Each metric averaged at each site, with that site's distance from the "
        "event in brackets.",
    ),
)


def _metric_series(
    source: str, metric: Mapping[str, Any]
) -> tuple[str | None, str | None]:
    """(bucket means, per-entity means) for one metric, either possibly absent.

    Delegates to the prompt renderers rather than reformatting the summary, so
    the packet cannot drift from what ``render_enrichment_text`` showed the model.
    Positional, because a metric can have entity means and no bucket means.
    """
    _entity_label, means_label = entity_terms(source)
    return render_bucket_means(metric), render_entity_means(metric, means_label)


def _metric_detail_lines(source: str, metric: Mapping[str, Any]) -> list[str]:
    """The series lines present for this metric, in section order."""
    return [line for line in _metric_series(source, metric) if line]


def _series_value(line: str) -> str:
    """Drop the renderer's own label, which the column header already carries."""
    _label, _, values = line.partition(": ")
    return values or line


def evidence_detail_lines(summary: Mapping[str, Any]) -> list[str]:
    """Every detail line the packet must print, source/metric-qualified.

    Recorded in the audit manifest and recomputed by ``packet_audit`` so a packet
    that silently drops the model's series can never pass the audit.
    """
    lines: list[str] = []
    for source in sorted(summary.get("sources", {})):
        metrics = summary["sources"][source].get("metrics", {})
        for metric_name in sorted(metrics):
            lines.extend(
                f"{source} {metric_name}: {line}"
                for line in _metric_detail_lines(source, metrics[metric_name])
            )
    return lines


def _evidence_rows(
    summary: Mapping[str, Any], event: datetime
) -> list[list[str]]:
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
                    metric_label(metric_name),
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
                        + _format_event_offset(
                            nearest.get("dt_minutes"), nearest.get("t"), event
                        )
                    ),
                ]
            )
    return rows


def _series_rows(summary: Mapping[str, Any], position: int) -> list[list[str]]:
    """Rows for one series table: source, metric, and that metric's series."""
    rows = [["Source", "Metric", "Values"]]
    for source in sorted(summary.get("sources", {})):
        metrics = summary["sources"][source].get("metrics", {})
        for metric_name in sorted(metrics):
            line = _metric_series(source, metrics[metric_name])[position]
            if line:
                rows.append([source, metric_label(metric_name), _series_value(line)])
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
            fontSize=11,
            leading=13,
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
            # borderPadding draws outside the layout box, so the visible gap is
            # space minus padding; keep space > padding or the box touches text.
            spaceBefore=12,
            spaceAfter=18,
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
            spaceBefore=6,
            spaceAfter=16,
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


def invariant_canvas(filename: str, **kwargs: Any) -> canvas.Canvas:
    """A canvas that omits wall-clock timestamps, so renders are byte-stable.

    Public because every expert-facing PDF needs it, not just the packet: a
    tracked artifact that changes hash on every re-render cannot be checked
    against a provenance manifest.
    """
    kwargs["invariant"] = 1
    return canvas.Canvas(filename, **kwargs)


def _set_pdf_metadata(pdf_canvas: canvas.Canvas, _document: Any) -> None:
    pdf_canvas.setTitle("AERIS Expert Review Packet")
    pdf_canvas.setAuthor("Mason Cao")
    pdf_canvas.setSubject("Blinded scientific claim review")
    pdf_canvas.setCreator("AERIS evaluation packet")
    pdf_canvas.setKeywords("expert review, atmospheric evidence")


def _anomaly_unit(summary: Mapping[str, Any], anomaly: Anomaly) -> str:
    """The unit recorded for the anomaly's own metric, or '' when absent."""
    metrics = summary.get("sources", {}).get(anomaly.source, {}).get("metrics", {})
    return str(metrics.get(anomaly.metric, {}).get("unit") or "").strip()


def _anomaly_table(
    anomaly: Anomaly, summary: Mapping[str, Any], styles: Mapping[str, ParagraphStyle]
) -> Table:
    timestamp = _ensure_utc(anomaly.timestamp).strftime("%Y-%m-%d %H:%M UTC")
    expected = _format_number(anomaly.expected_value)
    detectors = ", ".join(str(item) for item in anomaly.methods_triggered)
    unit = _anomaly_unit(summary, anomaly)
    suffix = f" {unit}" if unit else ""
    rows = [
        ["Source / metric", f"{anomaly.source} / {metric_label(anomaly.metric)}"],
        [
            "Observed / expected baseline",
            f"{_format_number(anomaly.value)} / {expected}{suffix}",
        ],
        ["Baseline method", baseline_method(anomaly.methods_triggered)],
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


def _grid_table(
    rows: Sequence[Sequence[str]],
    styles: Mapping[str, ParagraphStyle],
    col_widths: Sequence[float],
) -> Table:
    """The packet's one table style: header band, full grid, alternating rows."""
    cells = [
        [Paragraph(escape(str(value)), styles["small"]) for value in row]
        for row in rows
    ]
    table = Table(
        cells,
        colWidths=[width * inch for width in col_widths],
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


def _evidence_table(
    summary: Mapping[str, Any], event: datetime, styles: Mapping[str, ParagraphStyle]
) -> Table:
    return _grid_table(_evidence_rows(summary, event), styles, EVIDENCE_COL_WIDTHS)


def _series_table(
    rows: Sequence[Sequence[str]], styles: Mapping[str, ParagraphStyle]
) -> Table:
    return _grid_table(rows, styles, SERIES_COL_WIDTHS)


# Verbatim from the labeling guide's "Label definitions" and the two rules that
# follow it. Duplicated onto the claim pages so the definitions are in front of
# her while she marks, rather than in a second document she has to hold open.
# `test_packet_reminder_matches_the_labeling_guide` fails if these ever drift
# from `docs/bracco/labeling-guide.md`.
LABEL_DEFINITIONS: tuple[str, ...] = (
    "Valid (V): the claim holds up at the precision it states, and the evidence "
    "in the packet supports it.",
    "Invalid (I): the evidence in the packet contradicts the claim, the claim "
    "misstates a measurement, or the physical reasoning does not hold.",
    "Unsure (U): you cannot tell from what is shown. This covers thin evidence, "
    "measurements that cannot resolve the claim, ambiguous wording, and anything "
    "outside what you are comfortable judging.",
    "Missing or thin evidence means Unsure, not Invalid.",
    "A cause that is plausible but unproven stays Unsure unless the evidence "
    "actually contradicts it.",
    "Some claims bundle several statements into one sentence. Judge those by their "
    "parts. If every part earns the same verdict, use that verdict.",
)


class _CheckboxField(Flowable):
    """One AcroForm checkbox, sized to sit inside a marking-table cell.

    The drawn box beside it is kept: a viewer that ignores form fields leaves the
    packet exactly as markable by hand as it was before, so the field can only
    add a way to answer, never take one away.
    """

    def __init__(self, name: str, size: float = 9.0) -> None:
        super().__init__()
        self.name = name
        self.width = size
        self.height = size

    def draw(self) -> None:
        # Stroked into the page content, not left to the widget's appearance
        # stream, so the box survives printing and any viewer that declines to
        # render form fields at all.
        self.canv.setStrokeColor(colors.HexColor("#5A6B75"))
        self.canv.setLineWidth(0.6)
        self.canv.rect(0, 0, self.width, self.height, stroke=1, fill=0)
        # acroForm takes absolute page coordinates and ignores the canvas
        # transform, so the flowable's own origin has to be resolved first.
        # Passing 0, 0 here stacks every widget in the page corner, which
        # leaves the drawn box unclickable in every reader.
        origin_x, origin_y = self.canv.absolutePosition(0, 0)
        self.canv.acroForm.checkbox(
            name=self.name,
            x=origin_x,
            y=origin_y,
            size=self.width,
            buttonStyle="check",
            borderWidth=0,
            textColor=colors.HexColor("#1C2B33"),
            checked=False,
        )


class _TextField(Flowable):
    """A multi-line AcroForm text field over ruled lines for the handwritten path."""

    def __init__(
        self, name: str, width: float, height: float, rules: int = 3
    ) -> None:
        super().__init__()
        self.name = name
        self.width = width
        self.height = height
        self.rules = rules

    def draw(self) -> None:
        self.canv.setStrokeColor(colors.HexColor("#AEB7BC"))
        self.canv.setLineWidth(0.4)
        for line in range(self.rules):
            y = self.height * line / self.rules
            self.canv.line(0, y, self.width, y)
        origin_x, origin_y = self.canv.absolutePosition(0, 0)
        self.canv.acroForm.textfield(
            name=self.name,
            x=origin_x,
            y=origin_y,
            width=self.width,
            height=self.height,
            borderWidth=0,
            textColor=colors.HexColor("#1C2B33"),
            fontSize=8,
            fieldFlags="multiline",
        )


def _reminder_table(styles: Mapping[str, ParagraphStyle]) -> Table:
    """The guide's label definitions, restated where the marking happens."""
    rows = [
        [Paragraph(escape(definition), styles["small"])]
        for definition in LABEL_DEFINITIONS
    ]
    table = Table(
        [[Paragraph("<b>Label definitions, from the labeling guide</b>", styles["small"])]]
        + rows,
        colWidths=[6.5 * inch],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#EEF2F4")),
                ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#9DA8AE")),
                ("LINEBELOW", (0, 0), (0, 0), 0.35, colors.HexColor("#C8D0D5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
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
        [
            [
                "Mark exactly one:",
                _CheckboxField(f"claim{index:03d}_V"),
                "V",
                _CheckboxField(f"claim{index:03d}_I"),
                "I",
                _CheckboxField(f"claim{index:03d}_U"),
                "U",
            ]
        ],
        colWidths=[
            1.75 * inch,
            0.22 * inch,
            1.08 * inch,
            0.22 * inch,
            1.08 * inch,
            0.22 * inch,
            1.08 * inch,
        ],
        hAlign="LEFT",
    )
    marking.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (-1, 0), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#9DA8AE")),
                # Separators only between the three V/I/U groups; a full inner
                # grid would now rule between each box and its own letter.
                ("LINEAFTER", (0, 0), (0, 0), 0.25, colors.HexColor("#D5DADD")),
                ("LINEAFTER", (2, 0), (2, 0), 0.25, colors.HexColor("#D5DADD")),
                ("LINEAFTER", (4, 0), (4, 0), 0.25, colors.HexColor("#D5DADD")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    note = Table(
        [
            [
                "Note (optional):",
                _TextField(f"claim{index:03d}_note", 5.45 * inch, 0.72 * inch),
            ]
        ],
        colWidths=[0.95 * inch, 5.55 * inch],
        hAlign="LEFT",
    )
    note.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (1, 0), (1, 0), 0),
            ]
        )
    )
    content: list[Any] = [
        Paragraph(f"Claim {index} of {total}", styles["claim_index"]),
        Paragraph(escape(claim.claim_text), styles["claim"]),
    ]
    if calm_wind_flag is not None:
        details = "; ".join(
            (
                f"{decision.source}: event wind {decision.event_speed_ms} m/s, "
                f"below its cutoff of {decision.effective_cutoff_ms} m/s "
                f"(mean - 2*SD gave {decision.raw_cutoff_ms} m/s; "
                f"n={decision.window_n} wind readings)"
            )
            for decision in calm_wind_flag.source_decisions
        )
        status = CALM_WIND_FLOOR_DISPLAY.get(
            calm_wind_flag.floor_status,
            calm_wind_flag.floor_status.replace("_", " "),
        )
        content.append(
            Paragraph(
                escape(
                    f"Calm-wind flag: {calm_wind_flag.standard_note} "
                    f"{details}. Cutoff floor: {status}."
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
        story.append(Paragraph("DRAFT", styles["subtitle"]))
    story.extend(
        [
            Paragraph("Anomaly", styles["heading"]),
            _anomaly_table(anomaly, summary, styles),
        ]
    )
    if (anomaly.expected_value or 0) < 0:
        story.append(
            Paragraph(
                "The baseline is a mean of this series' own recent history, so it "
                "can fall below zero for a metric the network reports near or "
                "below its detection limit.",
                styles["body"],
            )
        )
    story.extend(
        [
            Spacer(1, 8),
            Paragraph("Stored evidence summary", styles["heading"]),
            Paragraph(
                "The three tables in this section are look-up tables. You never "
                "have to read them through: they are here for the moment a claim "
                "quotes a number and you want to check it.",
                styles["body"],
            ),
            _evidence_table(summary, anomaly.timestamp, styles),
        ]
    )
    for position, title, blurb in SERIES_SECTIONS:
        rows = _series_rows(summary, position)
        if len(rows) < 2:
            continue
        story.extend(
            [
                Spacer(1, 10),
                Paragraph(title, styles["heading"]),
                Paragraph(blurb, styles["body"]),
                _series_table(rows, styles),
            ]
        )
    story.extend(
        [
            Spacer(1, 10),
            Image(io.BytesIO(_station_figure(anomaly, plot_data)), width=6.5 * inch, height=3.7 * inch),
            Spacer(1, 6),
            Image(io.BytesIO(_wind_figure(anomaly, plot_data)), width=6.5 * inch, height=2.53 * inch),
            PageBreak(),
            Paragraph("Claims", styles["heading"]),
            Paragraph(
                "Mark exactly one box per claim: V (Valid), I (Invalid), or U (Unsure). To skip a "
                "claim, leave all three boxes blank; a blank is recorded as missing and never turned "
                "into Unsure. Each claim appears exactly once, and the number printed next to it is "
                "how I'll refer to that claim if I need to ask you about it.",
                styles["body"],
            ),
            _reminder_table(styles),
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
            Paragraph("Optional: most likely cause", styles["heading"]),
            Paragraph(
                "Your best guess at the true cause of this event, in your own words. \"Insufficient "
                "evidence to determine a single cause\" is a perfectly fine answer.",
                styles["body"],
            ),
            _TextField("most_likely_cause", 6.5 * inch, 1.05 * inch),
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
        "evidence_detail": evidence_detail_lines(summary),
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
        canvasmaker=invariant_canvas,
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
        help=(
            "drop the DRAFT marker and record the packet as official; the "
            "freeze gates were declared passed on 2026-08-10"
        ),
    )
    return parser.parse_args(argv)


async def _amain(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    source = await load_packet_source(args.database, uuid.UUID(args.anomaly_id))
    manifest_path = args.manifest or args.out.with_suffix(".audit.json")
    result = render_packet(
        source.anomaly,
        source.summary,
        source.claim_groups,
        args.labeler,
        args.out,
        manifest_path,
        example=not args.official,
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
