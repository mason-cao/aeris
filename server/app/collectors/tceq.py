"""TCEQ CAMS hourly data — near-real-time ground NO2/SO2/CO for Houston.

This is the one source with no documented API: data comes from the public
"Hourly Data (All Parameters)" CGI report
(``/cgi-bin/compliance/monops/daily_summary.pl``), reverse-engineered and
verified live on 2026-06-24. AERIS uses it as the near-real-time path to in-situ
NO2/SO2/CO for the Ship-Channel petrochemical corridor and the ground leg of a
satellite(S5P)-versus-ground comparison, since the audited
OpenAQ/AirNow coverage does not surface those species in the target area and
EPA AQS is configured as historical-only.

Verified contract (daily_summary.pl):
* POST form: first_look=no, select_date=today|yesterday|user (+ user_month/
  user_day/user_year), select_site=``site|<name>|<aqs>|<cams>``, time_format=24hr
* response: HTML; a parameter x hour matrix, rows = full parameter names
  ("Nitrogen Dioxide"/"Sulfur Dioxide"/"Carbon Monoxide"), columns 0000..2300
* **times are CST (UTC-6, fixed, no DST)** — the page states "All times shown
  are in CST"; converted to UTC on ingest
* units (physically unambiguous + TCEQ/EPA regulatory standard): NO2/SO2 ppb,
  CO ppm
* **data is preliminary/uncertified** ("may change... not official until
  certified"); EPA AQS is the historical comparison archive used by AERIS

Operational caveats this is the fragile source: undocumented, no published
rate limit or terms, HTML-scraped. Hence: a conservative shared limiter, a
circuit breaker that stops hammering after consecutive failures, per-site
isolation, and parsing that fails to zero points rather than emitting garbage.

The site registry is baked, not discovered at runtime: the 14 in-radius
monitors below were cross-referenced against the TCEQ AQS_ACTIVE_SITES GIS
layer (verified WGS84 coordinates) and each confirmed live-reporting NO2/SO2/CO
on 2026-06-24. The CGI returns no coordinates, so they must come from here.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx

from app.collectors.base import BaseCollector, DataPointCreate
from app.collectors.ratelimit import AsyncRateLimiter

logger = logging.getLogger(__name__)

SOURCE_NAME = "tceq"
API_URL = "https://www.tceq.texas.gov/cgi-bin/compliance/monops/daily_summary.pl"
USER_AGENT = "aeris-collector/1.0 (environmental research)"

# TCEQ reports in Local Standard Time year-round (no DST): "All times shown are
# in CST" == UTC-6, fixed.
CST = timezone(timedelta(hours=-6))

# Undocumented service with no published limit; stay gentle. 10/min = 6s
# spacing, shared by collector + backfill.
TCEQ_MAX_REQUESTS_PER_MINUTE = 10
TCEQ_LIMITER = AsyncRateLimiter(TCEQ_MAX_REQUESTS_PER_MINUTE)

# Stop hammering an apparently-down service after this many consecutive failures.
DEFAULT_BREAKER_THRESHOLD = 5

# TCEQ parameter label -> (canonical metric, unit). Scoped to the petrochemical
# species that fill the independence gap; units match OpenAQ/EPA AQS canon.
PARAM_MAP: dict[str, tuple[str, str]] = {
    "Nitrogen Dioxide": ("no2", "ppb"),
    "Sulfur Dioxide": ("so2", "ppb"),
    "Carbon Monoxide": ("co", "ppm"),
}

# Verified Houston NO2/SO2/CO monitors (in-radius, live-confirmed 2026-06-24).
# Coordinates from the TCEQ AQS_ACTIVE_SITES GIS layer (WGS84). select_site is
# the exact daily_summary.pl form value; aqs_cd is the dedup entity id.
TCEQ_SITES: list[dict[str, Any]] = [
    {"name": "Houston North Loop", "aqs_cd": "482011052", "lat": 29.81439, "lon": -95.387817, "select_site": "site|Houston North Loop C1052|48_201_1052|1052"},
    {"name": "Park Place", "aqs_cd": "482010416", "lat": 29.686298, "lon": -95.294732, "select_site": "site|Park Place C416|48_201_0416|416"},
    {"name": "Clinton", "aqs_cd": "482011035", "lat": 29.733729, "lon": -95.257603, "select_site": "site|Clinton C403/C304/AH113 |48_201_1035|113"},
    {"name": "Houston Southwest Freeway", "aqs_cd": "482011066", "lat": 29.721618, "lon": -95.492655, "select_site": "site|Houston Southwest Freeway C1066|48_201_1066|1066"},
    {"name": "Lang", "aqs_cd": "482010047", "lat": 29.834206, "lon": -95.48912, "select_site": "site|Lang C408|48_201_0047|408"},
    {"name": "Houston Bayland Park", "aqs_cd": "482010055", "lat": 29.695747, "lon": -95.499222, "select_site": "site|Houston Bayland Park C53/A146 |48_201_0055|53"},
    {"name": "Houston Aldine", "aqs_cd": "482010024", "lat": 29.901031, "lon": -95.326125, "select_site": "site|Houston Aldine C8/AF108/X150 |48_201_0024|8"},
    {"name": "Houston Croquet", "aqs_cd": "482010051", "lat": 29.62398, "lon": -95.474347, "select_site": "site|Houston Croquet C409|48_201_0051|409"},
    {"name": "Channelview", "aqs_cd": "482010026", "lat": 29.802694, "lon": -95.12551, "select_site": "site|Channelview C15/AH115 |48_201_0026|15"},
    {"name": "Houston Deer Park #2", "aqs_cd": "482011039", "lat": 29.669736, "lon": -95.128525, "select_site": "site|Hou.DeerPrk2 C35/235/1001/AFH139/FP239|48_201_1039|35"},
    {"name": "Manvel Croix Park", "aqs_cd": "480391004", "lat": 29.520454, "lon": -95.392512, "select_site": "site|Manvel Croix Park C84|48_039_1004|84"},
    {"name": "Lynchburg Ferry", "aqs_cd": "482011015", "lat": 29.758947, "lon": -95.079341, "select_site": "site|Lynchburg Ferry C1015/A165 |48_201_1015|165"},
    {"name": "Seabrook Friendship Park", "aqs_cd": "482011050", "lat": 29.583054, "lon": -95.01554, "select_site": "site|Seabrook Friendship Park C45|48_201_1050|45"},
    {"name": "Northwest Harris County", "aqs_cd": "482010029", "lat": 30.039525, "lon": -95.673947, "select_site": "site|Northwest Harris Co. C26/A110/X154|48_201_0029|26"},
]

_MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11,
    "December": 12,
}

# Weekday-anchored so it matches the report heading ("...for Tuesday, June 23,
# 2026...") and not incidental dates elsewhere on the page.
_DATE_RE = re.compile(
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+"
    r"(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+(\d{1,2}),\s+(\d{4})",
    re.IGNORECASE,
)
_NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_TAG_RE = re.compile(r"<[^>]+>")
# Hour-column header label, e.g. "0000".."2300".
_HOUR_LABEL_RE = re.compile(r"^([01]\d|2[0-3])00$")


def parse_report_date(html: str) -> date | None:
    """The CST report date from the daily_summary heading, or None."""
    match = _DATE_RE.search(html)
    if not match:
        return None
    month = _MONTHS.get(match.group(1).capitalize())
    if month is None:
        return None
    try:
        return date(int(match.group(3)), month, int(match.group(2)))
    except ValueError:
        return None


def _clean(cell: str) -> str:
    text = _TAG_RE.sub(" ", cell).replace("\xa0", " ").replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", text).strip()


def _row_cells(tr: str) -> list[str]:
    return [
        _clean(c)
        for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.IGNORECASE | re.DOTALL)
    ]


def _hour_columns(rows: list[list[str]]) -> dict[int, int]:
    """Map ``{cell_index: hour}`` from the header row of hour labels.

    Returns the header with the most hour-label cells, so a full day (24
    columns) and a partial "today" (only elapsed hours) are both handled.
    """
    best: dict[int, int] = {}
    for cells in rows:
        hmap = {
            idx: int(cell.strip()) // 100
            for idx, cell in enumerate(cells)
            if _HOUR_LABEL_RE.match(cell.strip())
        }
        if len(hmap) > len(best):
            best = hmap
    return best


def parse_tceq_rows(html: str) -> list[tuple[str, dict[int, str], str]]:
    """Extract ``(parameter, {hour: value_str}, poc)`` rows for target species.

    Header-aware: each value is keyed by its column's hour label, so partial
    "today" reports (fewer hour columns) parse the same as full days. Only rows
    whose first cell is a known parameter are returned; met/PM/layout ignored.
    """
    all_cells = [
        _row_cells(tr)
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.IGNORECASE | re.DOTALL)
    ]
    hour_cols = _hour_columns(all_cells)
    if not hour_cols:
        return []

    rows: list[tuple[str, dict[int, str], str]] = []
    for cells in all_cells:
        if not cells or cells[0] not in PARAM_MAP:
            continue
        hour_values = {
            hour: cells[idx] for idx, hour in hour_cols.items() if idx < len(cells)
        }
        poc = cells[-1] if cells else ""
        rows.append((cells[0], hour_values, poc))
    return rows


def tceq_html_to_points(
    html: str,
    *,
    site: dict[str, Any],
    source_name: str,
    expected_date: date | None = None,
) -> list[DataPointCreate]:
    """Parse one site's daily_summary HTML into DataPointCreate, CST -> UTC.

    Returns ``[]`` (never raises) when the date heading or table can't be read,
    so a layout change degrades to zero rows instead of corrupt data.

    When ``expected_date`` is given (the backfill knows exactly which day it
    requested), a report whose heading is a *different* date is rejected to
    ``[]`` rather than ingested — defense-in-depth against a request landing on
    the wrong day (the class of bug the 0-based-month error caused). The live
    collector passes no ``expected_date`` (today/yesterday are trusted).
    """
    report_date = parse_report_date(html)
    if report_date is None:
        logger.warning("TCEQ: no report date in response for %s", site.get("aqs_cd"))
        return []
    if expected_date is not None and report_date != expected_date:
        logger.warning(
            "TCEQ: report date %s != requested %s for %s; dropping",
            report_date.isoformat(),
            expected_date.isoformat(),
            site.get("aqs_cd"),
        )
        return []

    points: list[DataPointCreate] = []
    for parameter, hour_values, poc in parse_tceq_rows(html):
        metric, unit = PARAM_MAP[parameter]
        for hour, raw_value in hour_values.items():
            token = raw_value.strip()
            if not _NUMERIC_RE.match(token):
                continue
            ts = datetime(
                report_date.year,
                report_date.month,
                report_date.day,
                hour,
                tzinfo=CST,
            ).astimezone(timezone.utc)
            points.append(
                DataPointCreate(
                    timestamp=ts,
                    lat=site["lat"],
                    lon=site["lon"],
                    metric=metric,
                    value=float(token),
                    unit=unit,
                    source=source_name,
                    source_entity_id=str(site["aqs_cd"]),
                    raw_json={
                        "aqs_cd": site["aqs_cd"],
                        "parameter": parameter,
                        "hour_cst": hour,
                        "poc": poc,
                    },
                )
            )
    return points


def site_post_data(select_site: str, *, select_date: str = "today") -> dict[str, str]:
    """Form body for a daily_summary.pl POST (today/yesterday)."""
    return {
        "first_look": "no",
        "select_date": select_date,
        "select_site": select_site,
        "time_format": "24hr",
    }


def user_date_post_data(select_site: str, day: date) -> dict[str, str]:
    """Form body for a daily_summary.pl POST for a specific CST date.

    ``user_month`` is **0-based**: the CGI indexes its own month array
    ``month[0]="January" ... month[11]="December"``, so June (calendar month 6)
    must be sent as ``5``. Sending the 1-based value silently requests the next
    month. Verified live 2026-06-25.
    """
    return {
        "first_look": "no",
        "select_date": "user",
        "user_month": str(day.month - 1),
        "user_day": str(day.day),
        "user_year": str(day.year),
        "select_site": select_site,
        "time_format": "24hr",
    }


class TCEQCollector(BaseCollector):
    source_name = SOURCE_NAME
    collect_interval_minutes = 60

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        *,
        rate_limiter: AsyncRateLimiter | None = None,
        sites: list[dict[str, Any]] | None = None,
        breaker_threshold: int = DEFAULT_BREAKER_THRESHOLD,
    ) -> None:
        super().__init__(http_client)
        self._limiter = rate_limiter or TCEQ_LIMITER
        self._sites = sites if sites is not None else TCEQ_SITES
        self._breaker_threshold = breaker_threshold

    async def fetch(self) -> dict[str, Any]:
        """POST today's report per site; isolate failures, trip on a streak."""
        client = await self._get_client()
        results: list[dict[str, Any]] = []
        consecutive_failures = 0

        for site in self._sites:
            await self._limiter.acquire()
            try:
                response = await client.post(
                    API_URL,
                    data=site_post_data(site["select_site"], select_date="today"),
                    headers={"User-Agent": USER_AGENT},
                )
                response.raise_for_status()
            except Exception as exc:  # noqa: BLE001 - isolate one site's failure
                consecutive_failures += 1
                logger.warning(
                    "TCEQ site fetch failed",
                    extra={"aqs_cd": site.get("aqs_cd"), "error": str(exc)},
                )
                if consecutive_failures >= self._breaker_threshold:
                    logger.error(
                        "TCEQ circuit breaker tripped after %d consecutive failures",
                        consecutive_failures,
                    )
                    break
                continue

            consecutive_failures = 0
            results.append({"site": site, "html": response.text})

        return {"sites": results}

    def normalize(self, raw_data: dict[str, Any]) -> list[DataPointCreate]:
        points: list[DataPointCreate] = []
        for entry in raw_data.get("sites", []):
            points.extend(
                tceq_html_to_points(
                    entry["html"], site=entry["site"], source_name=self.source_name
                )
            )
        return points
