import gzip
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import delete, select

from app.collectors.backfill import (
    OpenAQArchiveBackfill,
    OpenAQBackfill,
    archive_key,
    available_strategies,
    db_location_ids,
    parse_archive_csv,
)
from app.collectors.ratelimit import AsyncRateLimiter
from app.db.models import DataPoint

T0 = datetime(2026, 5, 1, tzinfo=timezone.utc)

CSV_HEADER = (
    '"location_id","sensors_id","location","datetime","lat","lon",'
    '"parameter","units","value"'
)


def archive_csv(rows: list[str]) -> bytes:
    return gzip.compress(("\n".join([CSV_HEADER, *rows]) + "\n").encode())


HOUSTON_ROW = (
    '42,3919,"Houston Monitor","2026-05-01T01:00:00-06:00",'
    '"29.7604","-95.3698","pm25","µg/m³","32.0"'
)


@pytest_asyncio.fixture(autouse=True)
async def _clean_data_points(db_session):
    await db_session.execute(delete(DataPoint))
    await db_session.commit()


class TestArchiveKey:
    def test_matches_bucket_layout(self) -> None:
        assert archive_key(2178, date(2026, 5, 1)) == (
            "records/csv.gz/locationid=2178/year=2026/month=05/"
            "location-2178-20260501.csv.gz"
        )


class TestParseArchiveCsv:
    def test_maps_rows_to_data_points(self) -> None:
        points = parse_archive_csv(archive_csv([HOUSTON_ROW]))

        assert len(points) == 1
        point = points[0]
        assert point.source == "openaq"
        assert point.metric == "pm25"
        assert point.unit == "ug/m3"
        assert point.value == 32.0
        # -06:00 offset converted to UTC.
        assert point.timestamp == datetime(2026, 5, 1, 7, 0, tzinfo=timezone.utc)
        assert point.source_entity_id == "3919"
        assert point.lat == pytest.approx(29.7604)
        assert point.lon == pytest.approx(-95.3698)

    def test_skips_unknown_parameter_and_out_of_radius_rows(self) -> None:
        albuquerque = (
            '2178,1,"Del Norte","2026-05-01T01:00:00-06:00",'
            '"35.1353","-106.584702","pm25","µg/m³","15.0"'
        )
        unknown = (
            '42,2,"Houston Monitor","2026-05-01T01:00:00-06:00",'
            '"29.7604","-95.3698","temperature","c","21.0"'
        )

        points = parse_archive_csv(archive_csv([albuquerque, unknown, HOUSTON_ROW]))

        assert [p.source_entity_id for p in points] == ["3919"]


class TestOpenAQArchiveBackfill:
    @pytest.mark.asyncio
    async def test_fetches_each_location_day_and_stores(self, db_session) -> None:
        day1 = archive_csv([HOUSTON_ROW])
        day3 = archive_csv(
            [
                '42,3919,"Houston Monitor","2026-05-03T01:00:00-06:00",'
                '"29.7604","-95.3698","pm25","µg/m³","12.0"'
            ]
        )
        requested: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested.append(request.url.path)
            if request.url.path.endswith("20260501.csv.gz"):
                return httpx.Response(200, content=day1)
            if request.url.path.endswith("20260503.csv.gz"):
                return httpx.Response(200, content=day3)
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        strategy = OpenAQArchiveBackfill(
            http_client=client, location_ids=[42], request_delay_s=0
        )

        result = await strategy.backfill(
            db_session, since=T0, until=T0 + timedelta(days=2)
        )

        assert result.error is None
        assert result.records == 2
        assert len(requested) == 3  # one object per day, 404 day included
        rows = (await db_session.execute(select(DataPoint))).scalars().all()
        assert {r.value for r in rows} == {32.0, 12.0}

    @pytest.mark.asyncio
    async def test_no_known_location_ids_is_an_error(self, db_session) -> None:
        strategy = OpenAQArchiveBackfill(location_ids=None, request_delay_s=0)

        result = await strategy.backfill(
            db_session, since=T0, until=T0 + timedelta(days=1)
        )

        assert result.error is not None
        assert "location" in result.error


class TestDbLocationIds:
    @pytest.mark.asyncio
    async def test_distinct_ids_from_openaq_raw_json(self, db_session) -> None:
        def point(entity: str, location_id: int, source: str = "openaq") -> DataPoint:
            return DataPoint(
                timestamp=T0,
                lat=29.76,
                lon=-95.37,
                metric="pm25",
                value=1.0,
                unit="ug/m3",
                source=source,
                source_entity_id=entity,
                raw_json={"location": {"id": location_id}},
            )

        for p in [
            point("s1", 7),
            point("s2", 7),
            point("s3", 9),
            point("w1", 99, source="openweather"),
        ]:
            db_session.add(p)
            await db_session.flush()
        await db_session.commit()

        assert await db_location_ids(db_session) == [7, 9]


class TestDefaultStrategies:
    def test_openaq_default_is_the_archive(self) -> None:
        openaq = [s for s in available_strategies() if s.source_name == "openaq"]
        assert len(openaq) == 1
        assert isinstance(openaq[0], OpenAQArchiveBackfill)


class TestApiBackfillRateLimit:
    @pytest.mark.asyncio
    async def test_api_backfill_acquires_limiter_per_request(self, db_session) -> None:
        acquired = 0

        class CountingLimiter(AsyncRateLimiter):
            def __init__(self) -> None:
                super().__init__(6_000_000)

            async def acquire(self) -> None:
                nonlocal acquired
                acquired += 1
                await super().acquire()

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v3/locations":
                return httpx.Response(200, json={"results": []})
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        strategy = OpenAQBackfill(http_client=client, rate_limiter=CountingLimiter())

        await strategy.backfill(db_session, since=T0, until=T0 + timedelta(days=1))

        assert acquired == 1
