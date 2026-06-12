import asyncio
import logging
import os
import re
import tempfile
import time
from datetime import datetime, timedelta, timezone
from math import isfinite, isnan
from typing import Any, Sequence

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.collectors.base import BaseCollector, CollectionResult, DataPointCreate
from app.collectors.geo import BoundingBox, target_bounding_box
from app.config import settings
from app.db.models import DataPoint

logger = logging.getLogger(__name__)

API_BASE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
COLLECTION_NAME = "SENTINEL-5P"
LOOKBACK_HOURS = 48
RESULT_LIMIT = 200

CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)
CDSE_CLIENT_ID = "cdse-public"
GRANULE_DOWNLOAD_TIMEOUT_S = 300.0
TOKEN_REQUEST_TIMEOUT_S = 30.0
# CDSE access tokens expire after ~600s; a multi-granule download run easily
# outlives one, so refresh ahead of the deadline.
TOKEN_MAX_AGE_S = 540.0
DOWNLOAD_MAX_REDIRECTS = 5
MIN_VALID_PIXELS = 10
COLUMN_UNIT = "mol/m^2"

# S5P L2 product codes (from filename) → AERIS metric prefix
PRODUCT_TYPE_MAP: dict[str, str] = {
    "NO2": "s5p_no2",
    "SO2": "s5p_so2",
    "CO": "s5p_co",
    "O3": "s5p_o3",
    "HCHO": "s5p_hcho",
    "CH4": "s5p_ch4",
    "AER_AI": "s5p_aer_ai",
}

# Products we actually open and extract column densities for (v1 Houston scope)
PRODUCT_NETCDF_VARIABLE: dict[str, str] = {
    "NO2": "nitrogendioxide_tropospheric_column",
    "SO2": "sulfurdioxide_total_vertical_column",
    "CO": "carbonmonoxide_total_column",
    "HCHO": "formaldehyde_tropospheric_vertical_column",
}

PRODUCT_QA_THRESHOLD: dict[str, float] = {
    "NO2": 0.75,
    "SO2": 0.50,
    "CO": 0.50,
    "HCHO": 0.50,
}

COLUMN_PRODUCTS: frozenset[str] = frozenset(PRODUCT_NETCDF_VARIABLE)

PRODUCT_PATTERN = re.compile(r"S5P_\w+_L2__([A-Z0-9_]+?)_+\d{8}T")


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip().rstrip("Z")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def extract_product_code(name: str | None) -> str | None:
    if not name:
        return None
    match = PRODUCT_PATTERN.match(name)
    if not match:
        return None
    return match.group(1).rstrip("_")


def attribute_value(attributes: list[dict[str, Any]], name: str) -> Any:
    for attribute in attributes or []:
        if attribute.get("Name") == name:
            return attribute.get("Value")
    return None


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def area_polygon() -> str:
    bbox = target_bounding_box()
    return (
        f"POLYGON(({bbox.min_lon:.6f} {bbox.min_lat:.6f},"
        f"{bbox.max_lon:.6f} {bbox.min_lat:.6f},"
        f"{bbox.max_lon:.6f} {bbox.max_lat:.6f},"
        f"{bbox.min_lon:.6f} {bbox.max_lat:.6f},"
        f"{bbox.min_lon:.6f} {bbox.min_lat:.6f}))"
    )


def odata_filter(now: datetime | None = None) -> str:
    end = now or datetime.now(tz=timezone.utc)
    start = end - timedelta(hours=LOOKBACK_HOURS)
    iso_start = start.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    polygon = area_polygon()
    return (
        f"Collection/Name eq '{COLLECTION_NAME}' "
        f"and ContentDate/Start gt {iso_start} "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{polygon}')"
    )


def granule_download_url(product_id: str) -> str:
    return f"{API_BASE}({product_id})/$value"


async def ids_with_stored_columns(
    session: AsyncSession,
    candidate_ids: Sequence[str],
) -> set[str]:
    """Product ids that already have a ``*_column`` DataPoint row.

    Checked before download, not just at insert: granules run hundreds of MB,
    and both the hourly collector (rolling 48h window) and a re-run backfill
    would otherwise re-fetch payloads whose values are already stored.
    """
    if not candidate_ids:
        return set()
    column_metrics = [
        f"{PRODUCT_TYPE_MAP[code]}_column" for code in COLUMN_PRODUCTS
    ]
    rows = await session.execute(
        select(DataPoint.source_entity_id).where(
            DataPoint.source == "sentinel5p",
            DataPoint.metric.in_(column_metrics),
            DataPoint.source_entity_id.in_([str(c) for c in candidate_ids]),
        )
    )
    return {str(entity_id) for entity_id in rows.scalars()}


async def fetch_access_token(
    client: httpx.AsyncClient,
    username: str,
    password: str,
) -> str:
    """Exchange CDSE password credentials for a short-lived bearer token."""
    response = await client.post(
        CDSE_TOKEN_URL,
        data={
            "grant_type": "password",
            "client_id": CDSE_CLIENT_ID,
            "username": username,
            "password": password,
        },
        timeout=TOKEN_REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    payload = response.json()
    token = payload.get("access_token")
    if not token:
        raise RuntimeError("CDSE token response missing access_token")
    return str(token)


def filter_and_mean(
    values: Sequence[float | None],
    qa: Sequence[float | None],
    lats: Sequence[float | None],
    lons: Sequence[float | None],
    bbox: BoundingBox,
    qa_threshold: float,
    min_pixels: int = MIN_VALID_PIXELS,
) -> float | None:
    """Mean of pixel values that survive QA + bbox + finite-value masking."""
    if not (len(values) == len(qa) == len(lats) == len(lons)):
        return None

    total = 0.0
    count = 0
    for value, q, lat, lon in zip(values, qa, lats, lons):
        if value is None or q is None or lat is None or lon is None:
            continue
        try:
            v = float(value)
            qv = float(q)
            la = float(lat)
            lo = float(lon)
        except (TypeError, ValueError):
            continue
        if not isfinite(v) or isnan(v):
            continue
        if qv < qa_threshold:
            continue
        if not (bbox.min_lat <= la <= bbox.max_lat):
            continue
        if not (bbox.min_lon <= lo <= bbox.max_lon):
            continue
        total += v
        count += 1

    if count < min_pixels:
        return None
    return total / count


def _load_granule_arrays(
    path: str,
    product_code: str,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Open a downloaded S5P granule and return flat (value, qa, lat, lon) lists."""
    import xarray as xr

    variable_name = PRODUCT_NETCDF_VARIABLE[product_code]
    with xr.open_dataset(path, engine="netcdf4", group="PRODUCT") as ds:
        if variable_name not in ds.variables:
            raise RuntimeError(
                f"S5P granule missing expected variable {variable_name}"
            )
        values = ds[variable_name].values.ravel().tolist()
        qa = ds["qa_value"].values.ravel().tolist()
        lat = ds["latitude"].values.ravel().tolist()
        lon = ds["longitude"].values.ravel().tolist()
    return values, qa, lat, lon


class Sentinel5PCollector(BaseCollector):
    """Collects Sentinel-5P L2 granules and extracts target-area column densities.

    Catalog query records granule availability and metadata-level cloud cover.
    For mapped products (NO2, SO2, CO, HCHO) the collector also downloads the
    granule payload, applies QA + bbox masking, and emits a mean column-density
    DataPoint. Granules are streamed to a temp file and deleted after parsing.
    """

    source_name = "sentinel5p"
    collect_interval_minutes = 360

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        super().__init__(http_client)
        self._session: AsyncSession | None = None

    async def collect(
        self, session: AsyncSession, max_retries: int = 3
    ) -> CollectionResult:
        # fetch() needs the session to skip granules whose columns are
        # already stored, but BaseCollector's fetch() signature doesn't
        # carry one.
        self._session = session
        try:
            return await super().collect(session, max_retries=max_retries)
        finally:
            self._session = None

    async def fetch(self) -> dict[str, Any]:
        client = await self._get_client()
        catalog = await self._fetch_catalog(client)

        if not (settings.cdse_username and settings.cdse_password):
            logger.info(
                "CDSE credentials not set; running Sentinel-5P in catalog-only mode"
            )
            return catalog

        to_extract = await self._granules_needing_columns(catalog)
        if not to_extract:
            return {**catalog, "extracted_columns": {}}

        try:
            token = await fetch_access_token(
                client, settings.cdse_username, settings.cdse_password
            )
        except Exception as exc:
            logger.warning(
                "CDSE token fetch failed; falling back to catalog-only mode",
                extra={"error": str(exc)},
            )
            return catalog

        extracted = await self._extract_columns(
            client, {"value": to_extract}, token
        )
        return {**catalog, "extracted_columns": extracted}

    async def _granules_needing_columns(
        self, catalog: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Column-product records minus those already extracted to the DB."""
        candidates = [
            record
            for record in catalog.get("value", [])
            if record.get("Id")
            and extract_product_code(record.get("Name")) in COLUMN_PRODUCTS
        ]
        if not candidates or self._session is None:
            return candidates
        stored = await ids_with_stored_columns(
            self._session, [str(record["Id"]) for record in candidates]
        )
        if stored:
            logger.info(
                "Sentinel-5P: %d of %d column granules already stored; skipping",
                len(stored),
                len(candidates),
            )
        return [r for r in candidates if str(r["Id"]) not in stored]

    async def _fetch_catalog(self, client: httpx.AsyncClient) -> dict[str, Any]:
        params = {
            "$filter": odata_filter(),
            "$top": str(RESULT_LIMIT),
            "$orderby": "ContentDate/Start desc",
            "$expand": "Attributes",
        }
        response = await client.get(API_BASE, params=params)
        response.raise_for_status()
        return response.json()

    async def _extract_columns(
        self,
        client: httpx.AsyncClient,
        catalog: dict[str, Any],
        token: str,
    ) -> dict[str, float]:
        bbox = target_bounding_box()
        results: dict[str, float] = {}
        token_fetched_at = time.monotonic()

        for record in catalog.get("value", []):
            product_id = record.get("Id")
            name = record.get("Name")
            if not product_id or not name:
                continue
            product_code = extract_product_code(name)
            if product_code not in COLUMN_PRODUCTS:
                continue

            if time.monotonic() - token_fetched_at > TOKEN_MAX_AGE_S:
                token = await fetch_access_token(
                    client, settings.cdse_username, settings.cdse_password
                )
                token_fetched_at = time.monotonic()

            try:
                mean = await self._download_and_extract(
                    client, str(product_id), product_code, token, bbox
                )
            except Exception as exc:
                logger.warning(
                    "S5P granule extraction failed",
                    extra={
                        "product_id": product_id,
                        "product_code": product_code,
                        "error": str(exc),
                    },
                )
                continue

            if mean is not None:
                results[str(product_id)] = mean
        return results

    async def _download_and_extract(
        self,
        client: httpx.AsyncClient,
        product_id: str,
        product_code: str,
        token: str,
        bbox: BoundingBox,
    ) -> float | None:
        fd, path = tempfile.mkstemp(suffix=".nc")
        try:
            await self._download_granule(client, product_id, token, fd)
            arrays = await asyncio.to_thread(
                _load_granule_arrays, path, product_code
            )
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

        values, qa, lats, lons = arrays
        threshold = PRODUCT_QA_THRESHOLD[product_code]
        return filter_and_mean(values, qa, lats, lons, bbox, threshold)

    async def _download_granule(
        self,
        client: httpx.AsyncClient,
        product_id: str,
        token: str,
        fd: int,
    ) -> None:
        """Stream a granule to ``fd``, following redirects by hand.

        The OData ``$value`` endpoint 301s to a separate download host, and
        httpx drops the Authorization header on cross-host redirects (so
        auto-following yields a 401). The header has to be re-attached on
        every hop, per the CDSE download documentation.
        """
        url = granule_download_url(product_id)
        headers = {"Authorization": f"Bearer {token}"}
        with os.fdopen(fd, "wb") as out:
            for _ in range(DOWNLOAD_MAX_REDIRECTS + 1):
                async with client.stream(
                    "GET", url, headers=headers, timeout=GRANULE_DOWNLOAD_TIMEOUT_S
                ) as response:
                    if response.is_redirect:
                        url = str(
                            httpx.URL(url).join(response.headers["location"])
                        )
                        continue
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            out.write(chunk)
                    return
        raise RuntimeError(
            f"S5P granule download exceeded {DOWNLOAD_MAX_REDIRECTS} redirects"
        )

    def normalize(self, raw_data: dict[str, Any]) -> list[DataPointCreate]:
        if not isinstance(raw_data, dict):
            return []
        records = raw_data.get("value", [])
        extracted = raw_data.get("extracted_columns") or {}
        points: list[DataPointCreate] = []
        for record in records:
            points.extend(self._normalize_record(record, extracted))
        return points

    def _normalize_record(
        self,
        record: dict[str, Any],
        extracted: dict[str, float],
    ) -> list[DataPointCreate]:
        product_id = record.get("Id")
        name = record.get("Name")
        if not product_id or not name:
            return []

        product_code = extract_product_code(name)
        metric_prefix = PRODUCT_TYPE_MAP.get(product_code or "")
        if metric_prefix is None:
            logger.debug("Skipping unmapped S5P product: %s", name)
            return []

        content_date = record.get("ContentDate") or {}
        timestamp = parse_iso_datetime(content_date.get("Start"))
        if timestamp is None:
            return []

        attributes = record.get("Attributes") or []
        readings: list[tuple[str, float, str]] = [
            (f"{metric_prefix}_granule_available", 1.0, "count"),
        ]

        cloud_cover = safe_float(attribute_value(attributes, "cloudCover"))
        if cloud_cover is not None:
            readings.append((f"{metric_prefix}_cloud_cover", cloud_cover, "percent"))

        column_value = extracted.get(str(product_id))
        if column_value is not None and product_code in COLUMN_PRODUCTS:
            readings.append((f"{metric_prefix}_column", column_value, COLUMN_UNIT))

        points: list[DataPointCreate] = []
        for metric, value, unit in readings:
            points.append(
                DataPointCreate(
                    timestamp=timestamp,
                    lat=settings.aeris_target_lat,
                    lon=settings.aeris_target_lon,
                    metric=metric,
                    value=value,
                    unit=unit,
                    source=self.source_name,
                    source_entity_id=str(product_id),
                    raw_json=record,
                )
            )
        return points
