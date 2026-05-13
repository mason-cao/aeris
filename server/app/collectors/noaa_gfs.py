import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.collectors.base import BaseCollector, DataPointCreate
from app.collectors.geo import target_bounding_box, within_target_radius
from app.config import settings

logger = logging.getLogger(__name__)

OPENDAP_BASE = "https://nomads.ncep.noaa.gov/dods/gfs_0p25"
CYCLE_HOURS = (0, 6, 12, 18)
CYCLE_FALLBACK = 4
BBOX_BUFFER_KM = 20.0
DATASET_OPEN_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class GFSVariable:
    gfs_name: str
    metric: str
    unit: str
    level: float | None
    convert: Callable[[float], float]


def _identity(value: float) -> float:
    return value


def _kelvin_to_celsius(value: float) -> float:
    return value - 273.15


def _pa_to_hpa(value: float) -> float:
    return value / 100.0


VARIABLES: tuple[GFSVariable, ...] = (
    GFSVariable("hgtprs", "gh_500", "m", 500.0, _identity),
    GFSVariable("tmpprs", "t_850", "degC", 850.0, _kelvin_to_celsius),
    GFSVariable("ugrd10m", "u_10m", "m/s", None, _identity),
    GFSVariable("vgrd10m", "v_10m", "m/s", None, _identity),
    GFSVariable("pressfc", "surface_pressure", "hPa", None, _pa_to_hpa),
    GFSVariable("pwatclm", "precipitable_water", "mm", None, _identity),
    GFSVariable("hpblsfc", "pbl_height", "m", None, _identity),
)

VARIABLES_BY_METRIC: dict[str, GFSVariable] = {var.metric: var for var in VARIABLES}


def cycle_candidates(now: datetime) -> list[datetime]:
    """GFS cycle times to try, newest first, walking back up to CYCLE_FALLBACK cycles."""
    now_utc = now.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    cycle_hour = (now_utc.hour // 6) * 6
    latest = now_utc.replace(hour=cycle_hour)
    return [latest - timedelta(hours=6 * offset) for offset in range(CYCLE_FALLBACK)]


def dataset_url(cycle: datetime) -> str:
    return (
        f"{OPENDAP_BASE}/gfs{cycle.strftime('%Y%m%d')}"
        f"/gfs_0p25_{cycle.hour:02d}z"
    )


def to_gfs_longitude(lon: float) -> float:
    """Convert WGS84 longitude (-180..180) to GFS 0..360 convention."""
    return lon % 360.0


def from_gfs_longitude(lon: float) -> float:
    """Convert GFS 0..360 longitude back to WGS84 (-180..180)."""
    return lon if lon <= 180.0 else lon - 360.0


def parse_cycle_time(value: str) -> datetime:
    text = value.replace("Z", "+00:00") if value.endswith("Z") else value
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class NOAAGFSCollector(BaseCollector):
    source_name = "noaa_gfs"
    collect_interval_minutes = 360

    async def fetch(self) -> dict[str, Any]:
        """Open the freshest reachable GFS analysis cycle and extract target-area grid values."""
        candidates = cycle_candidates(datetime.now(timezone.utc))
        last_error: Exception | None = None
        for cycle in candidates:
            try:
                grid = await asyncio.wait_for(
                    asyncio.to_thread(self._read_cycle, cycle),
                    timeout=DATASET_OPEN_TIMEOUT_S,
                )
            except Exception as exc:
                logger.warning(
                    "GFS cycle unreachable",
                    extra={"cycle": cycle.isoformat(), "error": str(exc)},
                )
                last_error = exc
                continue

            if not grid:
                last_error = RuntimeError("GFS bounding-box subset returned no grid cells")
                logger.warning(
                    "GFS cycle returned empty grid",
                    extra={"cycle": cycle.isoformat()},
                )
                continue

            return {
                "cycle_time": cycle.isoformat(),
                "cycle_date": cycle.strftime("%Y%m%d"),
                "cycle_hour": cycle.hour,
                "grid": grid,
            }

        raise RuntimeError(
            f"No GFS cycle reachable in the last {CYCLE_FALLBACK * 6} hours: {last_error}"
        )

    def _read_cycle(self, cycle: datetime) -> list[dict[str, Any]]:
        """Synchronous OPeNDAP open + variable extraction for one cycle."""
        import xarray as xr

        url = dataset_url(cycle)
        bbox = target_bounding_box(settings.aeris_target_radius_km + BBOX_BUFFER_KM)
        lon_lo = to_gfs_longitude(bbox.min_lon)
        lon_hi = to_gfs_longitude(bbox.max_lon)

        with xr.open_dataset(url, engine="pydap", decode_times=False) as ds:
            subset = ds.sel(
                lat=slice(bbox.min_lat, bbox.max_lat),
                lon=slice(lon_lo, lon_hi),
            ).isel(time=0)
            return self._extract_grid(subset)

    def _extract_grid(self, ds: Any) -> list[dict[str, Any]]:
        """Walk the subset dataset and emit one dict per (lat, lon)."""
        lats = [float(v) for v in ds["lat"].values]
        lons = [float(v) for v in ds["lon"].values]

        arrays: dict[str, Any] = {}
        for var in VARIABLES:
            if var.gfs_name not in ds.variables:
                raise RuntimeError(f"GFS variable missing from dataset: {var.gfs_name}")
            arr = ds[var.gfs_name]
            if var.level is not None:
                arr = arr.sel(lev=var.level)
            arrays[var.gfs_name] = arr.values

        grid: list[dict[str, Any]] = []
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                values = {
                    var.metric: float(arrays[var.gfs_name][i, j]) for var in VARIABLES
                }
                grid.append(
                    {
                        "lat": lat,
                        "lon": from_gfs_longitude(lon),
                        "values": values,
                    }
                )
        return grid

    def normalize(self, raw_data: dict[str, Any]) -> list[DataPointCreate]:
        cycle_time_str = raw_data.get("cycle_time")
        if not cycle_time_str:
            return []
        timestamp = parse_cycle_time(cycle_time_str)

        points: list[DataPointCreate] = []
        for cell in raw_data.get("grid", []):
            try:
                lat = float(cell["lat"])
                lon = float(cell["lon"])
            except (KeyError, TypeError, ValueError):
                continue

            if not within_target_radius(lat, lon):
                continue

            values = cell.get("values") or {}
            for metric, raw_val in values.items():
                var = VARIABLES_BY_METRIC.get(metric)
                if var is None or raw_val is None:
                    continue
                try:
                    numeric = float(raw_val)
                except (TypeError, ValueError):
                    continue

                points.append(
                    DataPointCreate(
                        timestamp=timestamp,
                        lat=lat,
                        lon=lon,
                        metric=var.metric,
                        value=var.convert(numeric),
                        unit=var.unit,
                        source=self.source_name,
                        source_entity_id=f"gfs:{lat:.2f},{lon:.2f}",
                        raw_json={
                            "cycle_time": raw_data.get("cycle_time"),
                            "cycle_date": raw_data.get("cycle_date"),
                            "cycle_hour": raw_data.get("cycle_hour"),
                            "gfs_variable": var.gfs_name,
                            "raw_value": numeric,
                        },
                    )
                )
        return points
