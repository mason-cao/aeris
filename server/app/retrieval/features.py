"""Physical analog features for one anomaly and its stored evidence summary.

An event is described by the weather around it, the time of day, where it
happened, and how many detectors fired. Weather values are read from each
block's ``nearest_in_time`` observation, the same one the explanation prompt
prints.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

SCHEMA_VERSION = "analog-v1"

FEATURE_NAMES: tuple[str, ...] = (
    "wind_u",
    "wind_v",
    "ln_pbl_height",
    "temperature",
    "humidity",
    "cloud_cover",
    "ln1p_precipitation",
    "hour_sin",
    "hour_cos",
    "lat",
    "lon",
    "severity_tier",
)

# ASOS is the only source with gaps in the corpus (0.2% per metric).
IMPUTABLE_FEATURES: frozenset[str] = frozenset(
    {"wind_u", "wind_v", "temperature", "humidity"}
)

SEVERITY_TIER: dict[str, float] = {"minor": 1.0, "moderate": 2.0, "severe": 3.0}

# GFS fields kept out of the vector so they can test it.
HELD_OUT_METRICS: tuple[str, ...] = ("t_850", "surface_pressure", "precipitable_water")


class FeatureError(ValueError):
    """Raised instead of guessing a value the evidence does not contain."""


@dataclass(frozen=True)
class EventFeatures:
    """Raw features aligned with FEATURE_NAMES; None marks an imputable gap."""

    values: tuple[float | None, ...]

    @property
    def imputed(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, value in zip(FEATURE_NAMES, self.values, strict=True)
            if value is None
        )


def to_utc(moment: datetime) -> datetime:
    """UTC-aware copy; a naive value is taken as UTC, since SQLite drops the offset."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def epoch_seconds(moment: datetime, *, round_up: bool) -> int:
    """Whole epoch seconds, rounded in the direction that keeps the time guard strict."""
    seconds = to_utc(moment).timestamp()
    return math.ceil(seconds) if round_up else math.floor(seconds)


def nearest_value(summary: Mapping, source: str, metric: str) -> float | None:
    block = summary.get("sources", {}).get(source, {}).get("metrics", {}).get(metric)
    if not block:
        return None
    value = block.get("nearest_in_time", {}).get("v")
    return None if value is None else float(value)


def _required(summary: Mapping, source: str, metric: str) -> float:
    value = nearest_value(summary, source, metric)
    if value is None:
        raise FeatureError(f"{source}/{metric} is missing and may not be imputed")
    return value


def wind_components(speed: float, direction_deg: float) -> tuple[float, float]:
    """(u, v) in m/s for wind blowing from ``direction_deg``; calm is (0, 0)."""
    if speed == 0.0:
        return 0.0, 0.0
    theta = math.radians(direction_deg)
    return -speed * math.sin(theta), -speed * math.cos(theta)


def raw_features(
    *,
    timestamp: datetime,
    lat: float,
    lon: float,
    severity: str,
    summary: Mapping,
) -> EventFeatures:
    speed = nearest_value(summary, "asos", "wind_speed")
    direction = nearest_value(summary, "asos", "wind_direction")
    wind_u: float | None = None
    wind_v: float | None = None
    if speed is not None and direction is not None:
        wind_u, wind_v = wind_components(speed, direction)
    pbl = _required(summary, "noaa_gfs", "pbl_height")
    if pbl <= 0.0:
        raise FeatureError(f"pbl_height must be positive, got {pbl}")
    precipitation = _required(summary, "openweather", "precipitation")
    if precipitation < 0.0:
        raise FeatureError(f"precipitation must be non-negative, got {precipitation}")
    if severity not in SEVERITY_TIER:
        raise FeatureError(f"unknown severity {severity!r}")
    moment = to_utc(timestamp)
    angle = 2.0 * math.pi * (moment.hour + moment.minute / 60.0) / 24.0
    values: tuple[float | None, ...] = (
        wind_u,
        wind_v,
        math.log(pbl),
        nearest_value(summary, "asos", "temperature"),
        nearest_value(summary, "asos", "humidity"),
        _required(summary, "openweather", "cloud_cover"),
        math.log1p(precipitation),
        math.sin(angle),
        math.cos(angle),
        float(lat),
        float(lon),
        SEVERITY_TIER[severity],
    )
    for name, value in zip(FEATURE_NAMES, values, strict=True):
        if value is not None and not math.isfinite(value):
            raise FeatureError(f"{name} is not finite")
    return EventFeatures(values)


def held_out_values(summary: Mapping) -> tuple[float, ...]:
    values = tuple(_required(summary, "noaa_gfs", metric) for metric in HELD_OUT_METRICS)
    if not all(math.isfinite(value) for value in values):
        raise FeatureError("a held-out value is not finite")
    return values


@dataclass(frozen=True)
class Scaling:
    schema_version: str
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    n_events: int

    def to_json(self) -> str:
        payload = {
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "n_events": self.n_events,
            "schema_version": self.schema_version,
            "stds": list(self.stds),
        }
        return json.dumps(payload, indent=1, sort_keys=True) + "\n"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, text: str) -> Scaling:
        data = json.loads(text)
        scaling = cls(
            schema_version=str(data["schema_version"]),
            feature_names=tuple(str(name) for name in data["feature_names"]),
            means=tuple(float(value) for value in data["means"]),
            stds=tuple(float(value) for value in data["stds"]),
            n_events=int(data["n_events"]),
        )
        if scaling.schema_version != SCHEMA_VERSION or scaling.feature_names != FEATURE_NAMES:
            raise FeatureError("scaling fixture does not match the feature schema")
        return scaling


def fit_scaling(rows: Sequence[EventFeatures]) -> Scaling:
    """Mean and population standard deviation per feature, over present values."""
    if not rows:
        raise FeatureError("cannot fit scaling on zero events")
    means: list[float] = []
    stds: list[float] = []
    for index, name in enumerate(FEATURE_NAMES):
        present = [row.values[index] for row in rows if row.values[index] is not None]
        if not present:
            raise FeatureError(f"{name} has no values")
        mean = math.fsum(present) / len(present)  # type: ignore[arg-type]
        spread = math.sqrt(math.fsum((value - mean) ** 2 for value in present) / len(present))  # type: ignore[operator]
        if spread == 0.0:
            raise FeatureError(f"{name} has zero spread")
        means.append(mean)
        stds.append(spread)
    return Scaling(SCHEMA_VERSION, FEATURE_NAMES, tuple(means), tuple(stds), len(rows))


def standardize(features: EventFeatures, scaling: Scaling) -> tuple[float, ...]:
    """z-scores; an imputable gap becomes 0.0, the corpus mean."""
    vector: list[float] = []
    for value, mean, spread in zip(features.values, scaling.means, scaling.stds, strict=True):
        z = 0.0 if value is None else (value - mean) / spread
        if not math.isfinite(z):
            raise FeatureError("standardized value is not finite")
        vector.append(z)
    return tuple(vector)
