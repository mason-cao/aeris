import logging
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from sqlalchemy import func, select

from app.collectors.asos import (
    KNOTS_TO_MS,
    ASOSCollector,
    clear_stations_cache,
    fahrenheit_to_celsius,
    parse_asos_csv,
    parse_asos_datetime,
)
from app.collectors.backfill import ASOSBackfill
from app.collectors.ratelimit import AsyncRateLimiter
from app.db.models import DataPoint


@pytest.fixture(autouse=True)
def _fresh_stations_cache():
    clear_stations_cache()
    yield
    clear_stations_cache()


def fast_limiter() -> AsyncRateLimiter:
    return AsyncRateLimiter(6_000_000)


@pytest.fixture
def collector() -> ASOSCollector:
    return ASOSCollector()


# A station inside the Houston target radius (Bush Intercontinental).
IAH_LAT, IAH_LON = 29.9844, -95.3607
# A Texas station far outside the 50km radius (Dallas/Fort Worth area).
FAR_LAT, FAR_LON = 33.6017, -97.7756

HEADER = "station,valid,lon,lat,drct,sknt,tmpf,relh"


def csv_text(rows: list[str], header: str = HEADER) -> str:
    return "\n".join([header, *rows]) + "\n"


def wind_row(
    valid: str,
    *,
    station: str = "IAH",
    lat: float = IAH_LAT,
    lon: float = IAH_LON,
    drct: str = "160.00",
    sknt: str = "12.00",
    tmpf: str = "M",
    relh: str = "M",
) -> str:
    return f"{station},{valid},{lon},{lat},{drct},{sknt},{tmpf},{relh}"


def raw(text: str) -> dict:
    return {"csv": text}


class TestASOSHelpers:
    def test_parse_datetime_is_utc(self) -> None:
        assert parse_asos_datetime("2026-06-23 00:53") == datetime(
            2026, 6, 23, 0, 53, tzinfo=timezone.utc
        )

    def test_parse_datetime_tolerates_seconds(self) -> None:
        assert parse_asos_datetime("2026-06-23 00:53:00") == datetime(
            2026, 6, 23, 0, 53, tzinfo=timezone.utc
        )

    def test_parse_datetime_rejects_garbage(self) -> None:
        assert parse_asos_datetime("not-a-date") is None
        assert parse_asos_datetime("") is None

    def test_knots_to_ms_constant(self) -> None:
        assert KNOTS_TO_MS == pytest.approx(0.514444, abs=1e-6)

    def test_fahrenheit_to_celsius(self) -> None:
        assert fahrenheit_to_celsius(32.0) == pytest.approx(0.0)
        assert fahrenheit_to_celsius(87.0) == pytest.approx(30.5556, abs=1e-3)

    def test_parse_csv_reads_rows_and_skips_header(self) -> None:
        text = csv_text([wind_row("2026-06-23 00:00")])
        rows = parse_asos_csv(text)
        assert len(rows) == 1
        assert rows[0]["station"] == "IAH"
        assert rows[0]["drct"] == "160.00"

    def test_parse_csv_empty_input(self) -> None:
        assert parse_asos_csv("") == []
        assert parse_asos_csv(HEADER + "\n") == []


class TestASOSNormalize:
    def test_routine_metar_yields_all_four_metrics(
        self, collector: ASOSCollector
    ) -> None:
        # The :53 routine METAR carries wind + temp + humidity.
        text = csv_text(
            [wind_row("2026-06-23 00:53", drct="170.00", sknt="9.00", tmpf="87.00", relh="69.99")]
        )
        points = collector.normalize(raw(text))
        by_metric = {p.metric: p for p in points}
        assert set(by_metric) == {
            "wind_direction",
            "wind_speed",
            "temperature",
            "humidity",
        }
        assert by_metric["wind_direction"].unit == "degree"
        assert by_metric["wind_speed"].unit == "m/s"
        assert by_metric["temperature"].unit == "degC"
        assert by_metric["humidity"].unit == "percent"

    def test_unit_conversions(self, collector: ASOSCollector) -> None:
        text = csv_text(
            [wind_row("2026-06-23 00:53", drct="170.00", sknt="9.00", tmpf="87.00", relh="70.00")]
        )
        by_metric = {p.metric: p for p in collector.normalize(raw(text))}
        assert by_metric["wind_direction"].value == pytest.approx(170.0)
        assert by_metric["wind_speed"].value == pytest.approx(9.0 * KNOTS_TO_MS)
        assert by_metric["temperature"].value == pytest.approx(30.5556, abs=1e-3)
        assert by_metric["humidity"].value == pytest.approx(70.0)

    def test_missing_values_are_skipped(self, collector: ASOSCollector) -> None:
        # A 5-minute wind ob has tmpf=M, relh=M -> only wind metrics emitted.
        text = csv_text([wind_row("2026-06-23 00:05", drct="160.00", sknt="11.00")])
        metrics = {p.metric for p in collector.normalize(raw(text))}
        assert metrics == {"wind_direction", "wind_speed"}

    def test_downsamples_to_one_obs_per_metric_per_hour(
        self, collector: ASOSCollector
    ) -> None:
        # Many 5-minute wind obs in hour 00 plus the :53 routine METAR; one
        # wind_direction point for the hour, taken from the latest valid ob.
        rows = [
            wind_row("2026-06-23 00:00", drct="160.00", sknt="12.00"),
            wind_row("2026-06-23 00:05", drct="165.00", sknt="11.00"),
            wind_row("2026-06-23 00:53", drct="170.00", sknt="9.00", tmpf="87.00", relh="70.00"),
        ]
        points = collector.normalize(raw(csv_text(rows)))
        wind_dirs = [p for p in points if p.metric == "wind_direction"]
        assert len(wind_dirs) == 1
        # Latest valid ob in the hour is the :53 METAR.
        assert wind_dirs[0].value == pytest.approx(170.0)
        assert wind_dirs[0].timestamp == datetime(2026, 6, 23, 0, 53, tzinfo=timezone.utc)

    def test_separate_hours_produce_separate_points(
        self, collector: ASOSCollector
    ) -> None:
        rows = [
            wind_row("2026-06-23 00:53", drct="170.00", sknt="9.00"),
            wind_row("2026-06-23 01:53", drct="180.00", sknt="6.00"),
        ]
        wind_dirs = [
            p for p in collector.normalize(raw(csv_text(rows))) if p.metric == "wind_direction"
        ]
        assert len(wind_dirs) == 2

    def test_sets_source_and_entity_id(self, collector: ASOSCollector) -> None:
        text = csv_text([wind_row("2026-06-23 00:53", station="HOU")])
        point = collector.normalize(raw(text))[0]
        assert point.source == "asos"
        assert point.source_entity_id == "HOU"

    def test_timestamps_are_utc_aware(self, collector: ASOSCollector) -> None:
        text = csv_text([wind_row("2026-06-23 00:53")])
        point = collector.normalize(raw(text))[0]
        assert point.timestamp.tzinfo is not None
        assert point.timestamp.utcoffset().total_seconds() == 0

    def test_filters_out_of_radius_station(self, collector: ASOSCollector) -> None:
        # A station outside the target radius must be dropped even if returned.
        text = csv_text(
            [wind_row("2026-06-23 00:53", station="DFW", lat=FAR_LAT, lon=FAR_LON)]
        )
        assert collector.normalize(raw(text)) == []

    def test_skips_unparseable_rows(self, collector: ASOSCollector) -> None:
        rows = [
            wind_row("not-a-date"),  # bad timestamp
            wind_row("2026-06-23 00:53", drct="170.00", sknt="9.00"),
        ]
        points = collector.normalize(raw(csv_text(rows)))
        assert all(p.timestamp == datetime(2026, 6, 23, 0, 53, tzinfo=timezone.utc) for p in points)
        assert len(points) == 2  # wind_direction + wind_speed from the good row

    def test_bad_numeric_value_skipped(self, collector: ASOSCollector) -> None:
        text = csv_text([wind_row("2026-06-23 00:53", drct="oops", sknt="9.00")])
        metrics = {p.metric for p in collector.normalize(raw(text))}
        assert metrics == {"wind_speed"}

    def test_empty_csv(self, collector: ASOSCollector) -> None:
        assert collector.normalize(raw("")) == []
        assert collector.normalize({}) == []


def _geojson(features: list[dict]) -> dict:
    return {"type": "FeatureCollection", "features": features}


def _feature(sid: str, lat: float, lon: float, *, online: bool = True) -> dict:
    return {
        "type": "Feature",
        "id": sid,
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {"sid": sid, "sname": f"{sid} name", "online": online},
    }


class TestASOSFetch:
    @pytest.mark.asyncio
    async def test_discovers_stations_then_requests_data(self) -> None:
        seen_paths: list[str] = []
        data_params: dict[str, list[str]] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen_paths.append(request.url.path)
            if request.url.path.endswith(".geojson"):
                return httpx.Response(
                    200,
                    json=_geojson(
                        [
                            _feature("IAH", IAH_LAT, IAH_LON),
                            _feature("DFW", FAR_LAT, FAR_LON),  # out of radius
                        ]
                    ),
                )
            # data request
            for key in request.url.params.multi_items():
                data_params.setdefault(key[0], []).append(key[1])
            return httpx.Response(200, text=csv_text([wind_row("2026-06-23 00:53")]))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = ASOSCollector(http_client=client, rate_limiter=fast_limiter())
        result = await collector.fetch()
        await client.aclose()

        assert any(p.endswith(".geojson") for p in seen_paths)
        assert any("asos.py" in p for p in seen_paths)
        # Only the in-radius station is requested.
        assert data_params.get("station") == ["IAH"]
        # UTC + comma format requested.
        assert data_params.get("tz") == ["UTC"]
        assert data_params.get("format") == ["onlycomma"]
        assert "csv" in result

    @pytest.mark.asyncio
    async def test_fetch_to_normalize_roundtrip(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith(".geojson"):
                return httpx.Response(
                    200, json=_geojson([_feature("IAH", IAH_LAT, IAH_LON)])
                )
            return httpx.Response(
                200,
                text=csv_text(
                    [wind_row("2026-06-23 00:53", drct="170.00", sknt="9.00", tmpf="87.00", relh="70.00")]
                ),
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = ASOSCollector(http_client=client, rate_limiter=fast_limiter())
        raw_data = await collector.fetch()
        points = collector.normalize(raw_data)
        await client.aclose()

        assert {p.metric for p in points} == {
            "wind_direction",
            "wind_speed",
            "temperature",
            "humidity",
        }

    @pytest.mark.asyncio
    async def test_offline_stations_excluded(self) -> None:
        requested: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith(".geojson"):
                return httpx.Response(
                    200,
                    json=_geojson(
                        [
                            _feature("IAH", IAH_LAT, IAH_LON, online=True),
                            _feature("MCJ", 29.81, -95.40, online=False),
                        ]
                    ),
                )
            requested.extend(request.url.params.get_list("station"))
            return httpx.Response(200, text=csv_text([wind_row("2026-06-23 00:53")]))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = ASOSCollector(http_client=client, rate_limiter=fast_limiter())
        await collector.fetch()
        await client.aclose()

        assert "MCJ" not in requested
        assert "IAH" in requested


T0 = datetime(2026, 6, 20, 0, 0, tzinfo=timezone.utc)


async def _count_points(session) -> int:
    return await session.scalar(select(func.count()).select_from(DataPoint))


def _backfill_handler(rows_by_request: list[str]):
    """A handler that serves the station geojson then one CSV per data request."""
    state = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(".geojson"):
            return httpx.Response(
                200, json=_geojson([_feature("IAH", IAH_LAT, IAH_LON)])
            )
        i = state["i"]
        state["i"] += 1
        body = rows_by_request[i] if i < len(rows_by_request) else csv_text([])
        return httpx.Response(200, text=body)

    return handler, state


class TestASOSBackfill:
    @pytest.mark.asyncio
    async def test_stores_points_in_window(self, db_session) -> None:
        body = csv_text(
            [
                wind_row("2026-06-20 00:53", drct="170.00", sknt="9.00", tmpf="87.00", relh="70.00"),
                wind_row("2026-06-20 01:53", drct="180.00", sknt="6.00", tmpf="85.00", relh="72.00"),
            ]
        )
        handler, _ = _backfill_handler([body])
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        strategy = ASOSBackfill(http_client=client, rate_limiter=fast_limiter())
        try:
            result = await strategy.backfill(
                db_session, since=T0, until=T0 + timedelta(hours=3)
            )
        finally:
            await client.aclose()

        assert result.source == "asos"
        assert result.error is None
        # 2 hours x 4 metrics = 8 points.
        assert result.records == 8
        assert await _count_points(db_session) == 8

    @pytest.mark.asyncio
    async def test_idempotent_rerun_does_not_duplicate(self, db_session) -> None:
        body = csv_text(
            [wind_row("2026-06-20 00:53", drct="170.00", sknt="9.00", tmpf="87.00", relh="70.00")]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith(".geojson"):
                return httpx.Response(
                    200, json=_geojson([_feature("IAH", IAH_LAT, IAH_LON)])
                )
            return httpx.Response(200, text=body)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        strategy = ASOSBackfill(http_client=client, rate_limiter=fast_limiter())
        try:
            await strategy.backfill(db_session, since=T0, until=T0 + timedelta(hours=1))
            await strategy.backfill(db_session, since=T0, until=T0 + timedelta(hours=1))
        finally:
            await client.aclose()

        # Re-running the same window re-fetches identical obs; dedup keeps 4.
        assert await _count_points(db_session) == 4

    @pytest.mark.asyncio
    async def test_chunks_long_window(self, db_session) -> None:
        data_requests = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith(".geojson"):
                return httpx.Response(
                    200, json=_geojson([_feature("IAH", IAH_LAT, IAH_LON)])
                )
            data_requests["n"] += 1
            return httpx.Response(200, text=csv_text([]))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        # 65-day window, 30-day chunks, one station -> 3 data requests.
        strategy = ASOSBackfill(
            http_client=client, rate_limiter=fast_limiter(), chunk_days=30
        )
        try:
            await strategy.backfill(
                db_session, since=T0, until=T0 + timedelta(days=65)
            )
        finally:
            await client.aclose()

        assert data_requests["n"] == 3

    @pytest.mark.asyncio
    async def test_no_stations_returns_skipped(self, db_session) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith(".geojson"):
                # Only an out-of-radius station exists.
                return httpx.Response(
                    200, json=_geojson([_feature("DFW", FAR_LAT, FAR_LON)])
                )
            return httpx.Response(200, text=csv_text([]))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        strategy = ASOSBackfill(http_client=client, rate_limiter=fast_limiter())
        try:
            result = await strategy.backfill(
                db_session, since=T0, until=T0 + timedelta(hours=1)
            )
        finally:
            await client.aclose()

        assert result.records == 0
        assert result.skipped is True
