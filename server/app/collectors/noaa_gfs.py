import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from app.collectors.base import BaseCollector, DataPointCreate
from app.collectors.geo import target_bounding_box, within_target_radius
from app.config import settings

logger = logging.getLogger(__name__)

# NOMADS retired the OPeNDAP/GrADS Data Server (SCN 25-81). The GRIB filter is
# the supported successor: it subsets pgrb2 files server-side by variable,
# level, and bounding box, so each cycle download stays a few KB.
FILTER_BASE = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
CYCLE_FALLBACK = 4
BBOX_BUFFER_KM = 20.0
FETCH_TIMEOUT_S = 60.0

# GRIB filter request flags. The filter takes the cross product of var_* and
# lev_*, so the response carries some fields beyond the seven below; the GRIB
# parser keeps only the messages that match a known variable.
FILTER_VARS = ("HGT", "TMP", "UGRD", "VGRD", "PRES", "PWAT", "HPBL")
FILTER_LEVELS = (
    "500_mb",
    "850_mb",
    "10_m_above_ground",
    "surface",
    "entire_atmosphere_(considered_as_a_single_layer)",
)


def _identity(value: float) -> float:
    return value


def _kelvin_to_celsius(value: float) -> float:
    return value - 273.15


def _pa_to_hpa(value: float) -> float:
    return value / 100.0


@dataclass(frozen=True)
class GFSVariable:
    metric: str
    grib_name: str
    unit: str
    convert: Callable[[float], float]
    # GRIB key/value pairs that uniquely identify the message. Most fields key
    # off shortName; HPBL is an NCEP-local parameter eccodes does not name, so
    # it keys off the numeric parameter code instead.
    match_keys: dict[str, Any] = field(default_factory=dict)


VARIABLES: tuple[GFSVariable, ...] = (
    GFSVariable(
        "gh_500", "gh", "m", _identity,
        {"shortName": "gh", "typeOfLevel": "isobaricInhPa", "level": 500},
    ),
    GFSVariable(
        "t_850", "t", "degC", _kelvin_to_celsius,
        {"shortName": "t", "typeOfLevel": "isobaricInhPa", "level": 850},
    ),
    GFSVariable(
        "u_10m", "10u", "m/s", _identity,
        {"shortName": "10u", "typeOfLevel": "heightAboveGround", "level": 10},
    ),
    GFSVariable(
        "v_10m", "10v", "m/s", _identity,
        {"shortName": "10v", "typeOfLevel": "heightAboveGround", "level": 10},
    ),
    GFSVariable(
        "surface_pressure", "sp", "hPa", _pa_to_hpa,
        {"shortName": "sp", "typeOfLevel": "surface"},
    ),
    GFSVariable(
        "precipitable_water", "pwat", "mm", _identity,
        {"shortName": "pwat", "typeOfLevel": "atmosphereSingleLayer"},
    ),
    GFSVariable(
        "pbl_height", "hpbl", "m", _identity,
        {"parameterCategory": 3, "parameterNumber": 196, "typeOfLevel": "surface"},
    ),
)

VARIABLES_BY_METRIC: dict[str, GFSVariable] = {var.metric: var for var in VARIABLES}


def cycle_candidates(now: datetime) -> list[datetime]:
    """GFS cycle times to try, newest first, walking back up to CYCLE_FALLBACK cycles."""
    now_utc = now.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    cycle_hour = (now_utc.hour // 6) * 6
    latest = now_utc.replace(hour=cycle_hour)
    return [latest - timedelta(hours=6 * offset) for offset in range(CYCLE_FALLBACK)]


def filter_params(cycle: datetime) -> dict[str, str]:
    """Build the NOMADS GRIB-filter query for a cycle's f000 analysis over the target bbox."""
    bbox = target_bounding_box(settings.aeris_target_radius_km + BBOX_BUFFER_KM)
    params: dict[str, str] = {
        "dir": f"/gfs.{cycle.strftime('%Y%m%d')}/{cycle.hour:02d}/atmos",
        "file": f"gfs.t{cycle.hour:02d}z.pgrb2.0p25.f000",
        "subregion": "",
        "toplat": f"{bbox.max_lat:.4f}",
        "bottomlat": f"{bbox.min_lat:.4f}",
        "leftlon": f"{bbox.min_lon:.4f}",
        "rightlon": f"{bbox.max_lon:.4f}",
    }
    for var in FILTER_VARS:
        params[f"var_{var}"] = "on"
    for level in FILTER_LEVELS:
        params[f"lev_{level}"] = "on"
    return params


def from_gfs_longitude(lon: float) -> float:
    """Convert GFS 0..360 longitude back to WGS84 (-180..180)."""
    return lon if lon <= 180.0 else lon - 360.0


def parse_cycle_time(value: str) -> datetime:
    text = value.replace("Z", "+00:00") if value.endswith("Z") else value
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _message_matches(message: Any, keys: dict[str, Any]) -> bool:
    for key, expected in keys.items():
        try:
            actual = message[key]
        except Exception:
            return False
        if actual != expected:
            return False
    return True


def _match_variable(message: Any) -> GFSVariable | None:
    for variable in VARIABLES:
        if _message_matches(message, variable.match_keys):
            return variable
    return None


class NOAAGFSCollector(BaseCollector):
    source_name = "noaa_gfs"
    collect_interval_minutes = 360

    async def fetch(self) -> dict[str, Any]:
        """Download the freshest reachable GFS analysis cycle from the NOMADS GRIB filter."""
        candidates = cycle_candidates(datetime.now(timezone.utc))
        last_error: Exception | None = None
        for cycle in candidates:
            try:
                grid = await self._load_cycle(cycle)
            except Exception as exc:
                logger.warning(
                    "GFS cycle unreachable",
                    extra={"cycle": cycle.isoformat(), "error": str(exc)},
                )
                last_error = exc
                continue

            if not grid:
                last_error = RuntimeError("GFS filter returned no usable grid")
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

    async def _load_cycle(self, cycle: datetime) -> list[dict[str, Any]]:
        """Fetch and parse one GFS cycle; empty list means the cycle is unavailable."""
        data = await self._fetch_cycle_grib(cycle)
        if data is None:
            return []
        return self._parse_grib(data)

    async def _fetch_cycle_grib(self, cycle: datetime) -> bytes | None:
        """GET one cycle's GRIB subset; None if the cycle is not published yet."""
        client = await self._get_client()
        response = await client.get(
            FILTER_BASE,
            params=filter_params(cycle),
            timeout=FETCH_TIMEOUT_S,
        )
        if response.status_code != 200:
            logger.warning(
                "GFS filter returned non-200",
                extra={"cycle": cycle.isoformat(), "status": response.status_code},
            )
            return None

        content = response.content
        if not content.startswith(b"GRIB"):
            logger.warning(
                "GFS filter returned a non-GRIB body",
                extra={"cycle": cycle.isoformat()},
            )
            return None
        return content

    def _parse_grib(self, data: bytes) -> list[dict[str, Any]]:
        """Extract one (lat, lon, values) cell per grid point from a GRIB2 subset."""
        if not data.startswith(b"GRIB"):
            return []

        import pygrib

        with tempfile.NamedTemporaryFile(suffix=".grib2") as tmp:
            tmp.write(data)
            tmp.flush()
            grbs = pygrib.open(tmp.name)
            try:
                return self._extract_grid(grbs)
            finally:
                grbs.close()

    def _extract_grid(self, grbs: Any) -> list[dict[str, Any]]:
        columns: dict[str, Any] = {}
        lats: Any = None
        lons: Any = None
        for message in grbs:
            variable = _match_variable(message)
            if variable is None:
                continue
            if lats is None:
                lats, lons = message.latlons()
            columns[variable.metric] = message.values

        if lats is None or not columns:
            return []

        grid: list[dict[str, Any]] = []
        rows, cols = lats.shape
        for i in range(rows):
            for j in range(cols):
                grid.append(
                    {
                        "lat": float(lats[i, j]),
                        "lon": from_gfs_longitude(float(lons[i, j])),
                        "values": {
                            metric: float(array[i, j])
                            for metric, array in columns.items()
                        },
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
                            "gfs_variable": var.grib_name,
                            "raw_value": numeric,
                        },
                    )
                )
        return points
