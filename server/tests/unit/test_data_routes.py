from collections.abc import AsyncGenerator
from datetime import datetime, timezone

import httpx
import pytest
import pytest_asyncio

from app.db.models import DataPoint, DataSource
from app.db.session import get_session
from app.main import app


@pytest_asyncio.fixture
async def api_client(db_session) -> AsyncGenerator[httpx.AsyncClient, None]:
    async def override_get_session():
        yield db_session

    app.dependency_overrides[get_session] = override_get_session
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        yield client
    app.dependency_overrides.clear()


async def seed_data_point(
    db_session,
    *,
    source: str,
    metric: str,
    value: float,
    timestamp: datetime,
    lat: float = 34.0515,
    lon: float = -84.0713,
    source_entity_id: str = "entity-1",
) -> None:
    db_session.add(
        DataPoint(
            timestamp=timestamp,
            lat=lat,
            lon=lon,
            metric=metric,
            value=value,
            unit="ug/m3",
            source=source,
            source_entity_id=source_entity_id,
            raw_json={"seed": True},
        )
    )
    await db_session.commit()


class TestDataSourceRoutes:
    @pytest.mark.asyncio
    async def test_list_data_sources_documented_route(self, api_client, db_session) -> None:
        db_session.add_all(
            [
                DataSource(
                    name="api_source_beta",
                    source_type="collector",
                    status="active",
                    error_count=0,
                ),
                DataSource(
                    name="api_source_alpha",
                    source_type="collector",
                    status="error",
                    error_count=2,
                ),
            ]
        )
        await db_session.commit()

        response = await api_client.get("/api/data/sources")

        assert response.status_code == 200
        data = response.json()
        names = [item["name"] for item in data if item["name"].startswith("api_source_")]
        assert names == ["api_source_alpha", "api_source_beta"]
        alpha = next(item for item in data if item["name"] == "api_source_alpha")
        assert alpha["status"] == "error"
        assert alpha["error_count"] == 2

    @pytest.mark.asyncio
    async def test_list_data_sources_legacy_route(self, api_client, db_session) -> None:
        db_session.add(
            DataSource(
                name="api_legacy_source",
                source_type="collector",
                status="active",
                error_count=0,
            )
        )
        await db_session.commit()

        response = await api_client.get("/api/data")

        assert response.status_code == 200
        names = [item["name"] for item in response.json()]
        assert "api_legacy_source" in names


class TestDataPointRoutes:
    @pytest.mark.asyncio
    async def test_get_data_by_source_includes_unit_and_entity_id(
        self,
        api_client,
        db_session,
    ) -> None:
        await seed_data_point(
            db_session,
            source="api_data_source",
            metric="pm25",
            value=12.5,
            timestamp=datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc),
            source_entity_id="sensor-123",
        )

        response = await api_client.get("/api/data/api_data_source")

        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 1
        assert payload["items"][0]["unit"] == "ug/m3"
        assert payload["items"][0]["source_entity_id"] == "sensor-123"
        assert payload["items"][0]["raw_json"] == {"seed": True}

    @pytest.mark.asyncio
    async def test_get_data_by_source_filters_metric_time_and_radius(
        self,
        api_client,
        db_session,
    ) -> None:
        source = "api_filter_source"
        await seed_data_point(
            db_session,
            source=source,
            metric="pm25",
            value=10.0,
            timestamp=datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc),
            source_entity_id="sensor-near",
        )
        await seed_data_point(
            db_session,
            source=source,
            metric="pm10",
            value=22.0,
            timestamp=datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc),
            source_entity_id="sensor-wrong-metric",
        )
        await seed_data_point(
            db_session,
            source=source,
            metric="pm25",
            value=30.0,
            timestamp=datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
            source_entity_id="sensor-too-old",
        )
        await seed_data_point(
            db_session,
            source=source,
            metric="pm25",
            value=40.0,
            timestamp=datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc),
            lat=35.0,
            lon=-84.0713,
            source_entity_id="sensor-too-far",
        )

        response = await api_client.get(
            f"/api/data/{source}",
            params={
                "metric": "pm25",
                "start": "2026-04-20T00:00:00Z",
                "end": "2026-04-22T00:00:00Z",
                "lat": 34.0515,
                "lon": -84.0713,
                "radius_km": 5,
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 1
        assert payload["items"][0]["source_entity_id"] == "sensor-near"

    @pytest.mark.asyncio
    async def test_get_data_by_source_paginates(self, api_client, db_session) -> None:
        source = "api_pagination_source"
        for index in range(3):
            await seed_data_point(
                db_session,
                source=source,
                metric="pm25",
                value=float(index),
                timestamp=datetime(2026, 4, 21, 12 + index, tzinfo=timezone.utc),
                source_entity_id=f"sensor-{index}",
            )

        response = await api_client.get(
            f"/api/data/{source}",
            params={"limit": 1, "offset": 1},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 3
        assert payload["limit"] == 1
        assert payload["offset"] == 1
        assert len(payload["items"]) == 1
        assert payload["items"][0]["source_entity_id"] == "sensor-1"
