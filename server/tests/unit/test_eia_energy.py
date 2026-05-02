from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from app.collectors.eia_energy import (
    EIAEnergyCollector,
    SERIES_MAP,
    lookback_window,
    parse_eia_period,
)
from app.config import settings


@pytest.fixture
def collector() -> EIAEnergyCollector:
    return EIAEnergyCollector()


def make_record(**overrides: Any) -> dict[str, Any]:
    base = {
        "period": "2026-04-30T12",
        "respondent": "SOCO",
        "respondent-name": "Southern Company Services, Inc. - Trans",
        "type": "D",
        "type-name": "Demand",
        "value": 18420,
        "value-units": "megawatthours",
    }
    base.update(overrides)
    return base


def make_payload(*records: dict[str, Any]) -> dict[str, Any]:
    return {"response": {"data": list(records), "total": len(records)}}


class TestEIAEnergyNormalize:
    def test_normalize_maps_known_series(self, collector: EIAEnergyCollector) -> None:
        payload = make_payload(
            make_record(type="D", value=18420),
            make_record(type="NG", value=17800),
            make_record(type="TI", value=-300),
        )

        points = collector.normalize(payload)

        assert {point.metric for point in points} == {
            "grid_demand",
            "grid_net_generation",
            "grid_total_interchange",
        }

    def test_normalize_skips_unknown_series(
        self, collector: EIAEnergyCollector
    ) -> None:
        payload = make_payload(make_record(type="UNKNOWN"))

        assert collector.normalize(payload) == []

    def test_normalize_skips_record_without_value(
        self, collector: EIAEnergyCollector
    ) -> None:
        payload = make_payload(make_record(value=None))

        assert collector.normalize(payload) == []

    def test_normalize_uses_target_coordinates(
        self, collector: EIAEnergyCollector
    ) -> None:
        payload = make_payload(make_record())

        point = collector.normalize(payload)[0]

        assert point.lat == settings.aeris_target_lat
        assert point.lon == settings.aeris_target_lon

    def test_normalize_sets_source_and_entity_id(
        self, collector: EIAEnergyCollector
    ) -> None:
        payload = make_payload(make_record())

        point = collector.normalize(payload)[0]

        assert point.source == "eia_energy"
        assert point.source_entity_id == "SOCO:D"
        assert point.unit == "MWh"

    def test_normalize_parses_period_to_utc(
        self, collector: EIAEnergyCollector
    ) -> None:
        payload = make_payload(make_record(period="2026-04-30T12"))

        point = collector.normalize(payload)[0]

        assert point.timestamp == datetime(2026, 4, 30, 12, tzinfo=timezone.utc)


class TestEIAEnergyFetch:
    @pytest.mark.asyncio
    async def test_fetch_requires_api_key(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "eia_api_key", "")
        collector = EIAEnergyCollector()

        with pytest.raises(RuntimeError, match="EIA_API_KEY"):
            await collector.fetch()

    @pytest.mark.asyncio
    async def test_fetch_passes_api_key_and_facets(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "eia_api_key", "test-key")
        seen_params: dict[str, list[str]] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            for key in request.url.params.keys():
                seen_params[key] = request.url.params.get_list(key)
            return httpx.Response(200, json=make_payload(make_record()))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = EIAEnergyCollector(http_client=client)

        raw = await collector.fetch()

        assert seen_params["api_key"] == ["test-key"]
        assert seen_params["facets[respondent][]"] == ["SOCO"]
        assert seen_params["frequency"] == ["hourly"]
        assert raw["response"]["data"][0]["respondent"] == "SOCO"
        await client.aclose()

    @pytest.mark.asyncio
    async def test_fetch_supports_custom_respondent(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "eia_api_key", "test-key")
        seen_params: dict[str, list[str]] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen_params["facets[respondent][]"] = request.url.params.get_list(
                "facets[respondent][]"
            )
            return httpx.Response(200, json=make_payload(make_record(respondent="DUK")))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = EIAEnergyCollector(respondent="DUK", http_client=client)

        await collector.fetch()

        assert seen_params["facets[respondent][]"] == ["DUK"]
        await client.aclose()


class TestEIAEnergyHelpers:
    def test_series_map_covers_core_types(self) -> None:
        assert SERIES_MAP["D"] == ("grid_demand", "MWh")
        assert SERIES_MAP["NG"] == ("grid_net_generation", "MWh")

    def test_parse_eia_period_handles_hour_format(self) -> None:
        assert parse_eia_period("2026-04-30T12") == datetime(
            2026, 4, 30, 12, tzinfo=timezone.utc
        )

    def test_parse_eia_period_handles_minute_format(self) -> None:
        assert parse_eia_period("2026-04-30T12:30") == datetime(
            2026, 4, 30, 12, 30, tzinfo=timezone.utc
        )

    def test_parse_eia_period_returns_none_for_garbage(self) -> None:
        assert parse_eia_period("not-a-date") is None
        assert parse_eia_period(None) is None

    def test_lookback_window_is_72_hours(self) -> None:
        now = datetime(2026, 4, 30, 12, tzinfo=timezone.utc)

        start, end = lookback_window(now=now)

        assert start == "2026-04-27T12"
        assert end == "2026-04-30T12"
