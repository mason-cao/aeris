from datetime import datetime, timezone

import httpx
import pytest
from sqlalchemy import func, select

from app.collectors.backfill import EPAAQSBackfill
from app.collectors.epa_aqs import (
    AQS_PARAM_TO_METRIC,
    SOURCE_NAME,
    aqs_records_to_points,
    aqs_year_chunks,
    normalize_aqs_unit,
    parse_aqs_datetime,
)
from app.collectors.ratelimit import AsyncRateLimiter
from app.config import settings
from app.db.models import DataPoint

# A monitor inside the Houston target radius (Clinton, Ship Channel).
IN_LAT, IN_LON = 29.7335, -95.2577
FAR_LAT, FAR_LON = 32.7, -96.8  # Dallas, outside radius


def fast_limiter() -> AsyncRateLimiter:
    return AsyncRateLimiter(6_000_000)


def aqs_record(
    *,
    parameter_code: str = "42602",
    sample_measurement: float | None = 12.3,
    unit: str = "Parts per billion",
    date_gmt: str = "2024-06-01",
    time_gmt: str = "13:00",
    lat: float = IN_LAT,
    lon: float = IN_LON,
    state: str = "48",
    county: str = "201",
    site: str = "0057",
    poc: float = 1.0,
) -> dict:
    return {
        "state_code": state,
        "county_code": county,
        "site_number": site,
        "parameter_code": parameter_code,
        "poc": poc,
        "latitude": lat,
        "longitude": lon,
        "parameter_name": "Nitrogen dioxide (NO2)",
        "sample_measurement": sample_measurement,
        "unit_of_measure": unit,
        "date_gmt": date_gmt,
        "time_gmt": time_gmt,
        "datum": "WGS84",
    }


class TestAQSHelpers:
    def test_param_map_targets_petrochemical_species(self) -> None:
        assert AQS_PARAM_TO_METRIC == {"42602": "no2", "42401": "so2", "42101": "co"}

    def test_parse_datetime_combines_gmt_fields_as_utc(self) -> None:
        assert parse_aqs_datetime("2024-06-01", "13:00") == datetime(
            2024, 6, 1, 13, 0, tzinfo=timezone.utc
        )

    def test_parse_datetime_rejects_garbage(self) -> None:
        assert parse_aqs_datetime("", "13:00") is None
        assert parse_aqs_datetime("2024-06-01", "") is None
        assert parse_aqs_datetime("nope", "nope") is None

    def test_normalize_unit_maps_to_canon(self) -> None:
        assert normalize_aqs_unit("Parts per billion") == "ppb"
        assert normalize_aqs_unit("Parts per million") == "ppm"
        assert normalize_aqs_unit("Micrograms/cubic meter (LC)") == "ug/m3"

    def test_year_chunks_splits_on_calendar_year(self) -> None:
        # AQS requires edate in the same year as bdate.
        since = datetime(2024, 6, 1, tzinfo=timezone.utc)
        until = datetime(2025, 2, 1, tzinfo=timezone.utc)
        assert aqs_year_chunks(since, until) == [
            ("20240601", "20241231"),
            ("20250101", "20250201"),
        ]

    def test_year_chunks_single_year(self) -> None:
        since = datetime(2024, 3, 1, tzinfo=timezone.utc)
        until = datetime(2024, 9, 1, tzinfo=timezone.utc)
        assert aqs_year_chunks(since, until) == [("20240301", "20240901")]

    def test_year_chunks_empty_when_reversed(self) -> None:
        since = datetime(2024, 9, 1, tzinfo=timezone.utc)
        until = datetime(2024, 3, 1, tzinfo=timezone.utc)
        assert aqs_year_chunks(since, until) == []


class TestAQSRecordsToPoints:
    def test_maps_species_metric_and_unit(self) -> None:
        records = [
            aqs_record(parameter_code="42602", unit="Parts per billion"),
            aqs_record(parameter_code="42401", unit="Parts per billion"),
            aqs_record(parameter_code="42101", unit="Parts per million"),
        ]
        points = aqs_records_to_points(records, source_name=SOURCE_NAME)
        by_metric = {p.metric: p for p in points}
        assert set(by_metric) == {"no2", "so2", "co"}
        assert by_metric["no2"].unit == "ppb"
        assert by_metric["co"].unit == "ppm"

    def test_sets_source_entity_and_timestamp(self) -> None:
        point = aqs_records_to_points([aqs_record()], source_name=SOURCE_NAME)[0]
        assert point.source == "epa_aqs"
        # Monitor identity = state-county-site-poc.
        assert point.source_entity_id == "48-201-0057-1"
        assert point.timestamp == datetime(2024, 6, 1, 13, 0, tzinfo=timezone.utc)
        assert point.value == pytest.approx(12.3)

    def test_skips_null_measurement(self) -> None:
        assert aqs_records_to_points(
            [aqs_record(sample_measurement=None)], source_name=SOURCE_NAME
        ) == []

    def test_skips_unknown_parameter(self) -> None:
        assert aqs_records_to_points(
            [aqs_record(parameter_code="88101")], source_name=SOURCE_NAME
        ) == []

    def test_filters_out_of_radius(self) -> None:
        assert aqs_records_to_points(
            [aqs_record(lat=FAR_LAT, lon=FAR_LON)], source_name=SOURCE_NAME
        ) == []

    def test_empty(self) -> None:
        assert aqs_records_to_points([], source_name=SOURCE_NAME) == []


T_SINCE = datetime(2024, 6, 1, tzinfo=timezone.utc)
T_UNTIL = datetime(2024, 6, 30, tzinfo=timezone.utc)


async def _count(session) -> int:
    return await session.scalar(select(func.count()).select_from(DataPoint))


class TestEPAAQSBackfill:
    @pytest.mark.asyncio
    async def test_stores_points_and_sends_expected_params(
        self, db_session, monkeypatch
    ) -> None:
        monkeypatch.setattr(settings, "aqs_email", "me@example.org")
        monkeypatch.setattr(settings, "aqs_api_key", "k")
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            for k, v in request.url.params.items():
                seen[k] = v
            return httpx.Response(
                200,
                json={
                    "Header": [{"status": "Success", "rows": 2}],
                    "Data": [
                        aqs_record(parameter_code="42602"),
                        aqs_record(parameter_code="42401"),
                    ],
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        strategy = EPAAQSBackfill(http_client=client, rate_limiter=fast_limiter())
        try:
            result = await strategy.backfill(db_session, since=T_SINCE, until=T_UNTIL)
        finally:
            await client.aclose()

        assert result.source == "epa_aqs"
        assert result.error is None
        assert result.records == 2
        assert await _count(db_session) == 2
        # All three target species requested, plus the bbox + auth params.
        assert seen["param"] == "42602,42401,42101"
        assert seen["email"] == "me@example.org"
        assert seen["key"] == "k"
        assert {"minlat", "maxlat", "minlon", "maxlon"} <= set(seen)
        assert seen["bdate"] == "20240601" and seen["edate"] == "20240630"

    @pytest.mark.asyncio
    async def test_idempotent_rerun(self, db_session, monkeypatch) -> None:
        monkeypatch.setattr(settings, "aqs_email", "me@example.org")
        monkeypatch.setattr(settings, "aqs_api_key", "k")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "Header": [{"status": "Success"}],
                    "Data": [aqs_record(parameter_code="42602")],
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        strategy = EPAAQSBackfill(http_client=client, rate_limiter=fast_limiter())
        try:
            await strategy.backfill(db_session, since=T_SINCE, until=T_UNTIL)
            await strategy.backfill(db_session, since=T_SINCE, until=T_UNTIL)
        finally:
            await client.aclose()

        assert await _count(db_session) == 1

    @pytest.mark.asyncio
    async def test_skipped_without_credentials(self, db_session, monkeypatch) -> None:
        monkeypatch.setattr(settings, "aqs_email", "")
        monkeypatch.setattr(settings, "aqs_api_key", "")
        # No transport should be hit; pass a client that fails if used.
        strategy = EPAAQSBackfill(rate_limiter=fast_limiter())
        result = await strategy.backfill(db_session, since=T_SINCE, until=T_UNTIL)
        assert result.skipped is True
        assert result.records == 0
        assert result.notes is not None

    @pytest.mark.asyncio
    async def test_one_request_per_calendar_year(self, db_session, monkeypatch) -> None:
        monkeypatch.setattr(settings, "aqs_email", "me@example.org")
        monkeypatch.setattr(settings, "aqs_api_key", "k")
        requests = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            requests["n"] += 1
            return httpx.Response(200, json={"Header": [{"status": "Success"}], "Data": []})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        strategy = EPAAQSBackfill(http_client=client, rate_limiter=fast_limiter())
        try:
            await strategy.backfill(
                db_session,
                since=datetime(2023, 6, 1, tzinfo=timezone.utc),
                until=datetime(2025, 2, 1, tzinfo=timezone.utc),
            )
        finally:
            await client.aclose()

        assert requests["n"] == 3  # 2023, 2024, 2025
