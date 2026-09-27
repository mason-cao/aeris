from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

DEFAULT_VALUES: dict[tuple[str, str], float] = {
    ("asos", "wind_speed"): 5.0,
    ("asos", "wind_direction"): 360.0,
    ("asos", "temperature"): 30.0,
    ("asos", "humidity"): 60.0,
    ("noaa_gfs", "pbl_height"): 1000.0,
    ("noaa_gfs", "t_850"): 290.0,
    ("noaa_gfs", "surface_pressure"): 101000.0,
    ("noaa_gfs", "precipitable_water"): 40.0,
    ("openweather", "cloud_cover"): 20.0,
    ("openweather", "precipitation"): 0.0,
}


def varied_values(variant: int) -> dict[tuple[str, str], float]:
    """Values that differ from every other variant on every field."""
    return {
        ("asos", "wind_speed"): 2.0 + variant,
        ("asos", "wind_direction"): (10.0 + 90.0 * variant) % 360.0,
        ("asos", "temperature"): 25.0 + variant,
        ("asos", "humidity"): 50.0 + 5.0 * variant,
        ("noaa_gfs", "pbl_height"): 500.0 + 300.0 * variant,
        ("noaa_gfs", "t_850"): 285.0 + variant,
        ("noaa_gfs", "surface_pressure"): 100000.0 + 100.0 * variant,
        ("noaa_gfs", "precipitable_water"): 30.0 + 5.0 * variant,
        ("openweather", "cloud_cover"): 10.0 + 20.0 * variant,
        ("openweather", "precipitation"): 0.5 * variant,
    }


def summary_with(
    overrides: Mapping[tuple[str, str], float | None] | None = None,
    *,
    window_end: datetime | None = None,
    variant: int | None = None,
) -> dict:
    """A minimal evidence summary; an override of None removes that block."""
    values: dict[tuple[str, str], float | None] = dict(
        DEFAULT_VALUES if variant is None else varied_values(variant)
    )
    values.update(overrides or {})
    sources: dict[str, dict] = {}
    for (source, metric), value in values.items():
        if value is None:
            continue
        block = sources.setdefault(source, {"metrics": {}})
        block["metrics"][metric] = {"nearest_in_time": {"v": value}}
    summary: dict = {"sources": sources}
    if window_end is not None:
        summary["window"] = {"end": window_end.isoformat(), "hours_after": 36.0}
    return summary
