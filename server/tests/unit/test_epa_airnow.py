from datetime import datetime, timezone

import pytest

from app.collectors.epa_airnow import EPAAirNowCollector, PARAMETER_MAP


@pytest.fixture
def collector():
    return EPAAirNowCollector()


@pytest.fixture
def sample_api_response() -> dict:
    """Realistic EPA AirNow API response for Atlanta area."""
    return {
        "observations": [
            {
                "DateObserved": "2026-04-05",
                "HourObserved": 14,
                "LocalTimeZone": "EST",
                "ReportingArea": "Atlanta",
                "StateCode": "GA",
                "Latitude": 33.7490,
                "Longitude": -84.3880,
                "ParameterName": "PM2.5",
                "AQI": 42,
                "Category": {"Number": 1, "Name": "Good"},
            },
            {
                "DateObserved": "2026-04-05",
                "HourObserved": 14,
                "LocalTimeZone": "EST",
                "ReportingArea": "Atlanta",
                "StateCode": "GA",
                "Latitude": 33.7490,
                "Longitude": -84.3880,
                "ParameterName": "OZONE",
                "AQI": 35,
                "Category": {"Number": 1, "Name": "Good"},
            },
            {
                "DateObserved": "2026-04-05",
                "HourObserved": 14,
                "LocalTimeZone": "EST",
                "ReportingArea": "Gwinnett",
                "StateCode": "GA",
                "Latitude": 34.0515,
                "Longitude": -84.0713,
                "ParameterName": "PM2.5",
                "AQI": 55,
                "Category": {"Number": 2, "Name": "Moderate"},
            },
        ]
    }


class TestEPAAirNowNormalize:
    def test_normalize_returns_correct_count(
        self, collector: EPAAirNowCollector, sample_api_response: dict
    ) -> None:
        points = collector.normalize(sample_api_response)
        assert len(points) == 3

    def test_normalize_maps_parameters(
        self, collector: EPAAirNowCollector, sample_api_response: dict
    ) -> None:
        points = collector.normalize(sample_api_response)
        metrics = {p.metric for p in points}
        assert metrics == {"pm25", "ozone"}

    def test_normalize_preserves_coordinates(
        self, collector: EPAAirNowCollector, sample_api_response: dict
    ) -> None:
        points = collector.normalize(sample_api_response)
        gwinnett_point = next(p for p in points if p.lat == 34.0515)
        assert gwinnett_point.lon == -84.0713
        assert gwinnett_point.value == 55

    def test_normalize_parses_timestamp(
        self, collector: EPAAirNowCollector, sample_api_response: dict
    ) -> None:
        points = collector.normalize(sample_api_response)
        expected = datetime(2026, 4, 5, 14, 0, tzinfo=timezone.utc)
        assert all(p.timestamp == expected for p in points)

    def test_normalize_sets_source(
        self, collector: EPAAirNowCollector, sample_api_response: dict
    ) -> None:
        points = collector.normalize(sample_api_response)
        assert all(p.source == "epa_airnow" for p in points)

    def test_normalize_preserves_raw_json(
        self, collector: EPAAirNowCollector, sample_api_response: dict
    ) -> None:
        points = collector.normalize(sample_api_response)
        assert all(p.raw_json is not None for p in points)
        assert points[0].raw_json["ReportingArea"] == "Atlanta"

    def test_normalize_skips_unknown_parameters(
        self, collector: EPAAirNowCollector
    ) -> None:
        raw = {
            "observations": [
                {
                    "DateObserved": "2026-04-05",
                    "HourObserved": 14,
                    "ParameterName": "UNKNOWN_PARAM",
                    "AQI": 99,
                    "Latitude": 34.0,
                    "Longitude": -84.0,
                },
            ]
        }
        points = collector.normalize(raw)
        assert len(points) == 0

    def test_normalize_skips_missing_aqi(
        self, collector: EPAAirNowCollector
    ) -> None:
        raw = {
            "observations": [
                {
                    "DateObserved": "2026-04-05",
                    "HourObserved": 14,
                    "ParameterName": "PM2.5",
                    "AQI": None,
                    "Latitude": 34.0,
                    "Longitude": -84.0,
                },
            ]
        }
        points = collector.normalize(raw)
        assert len(points) == 0

    def test_normalize_empty_response(
        self, collector: EPAAirNowCollector
    ) -> None:
        points = collector.normalize({"observations": []})
        assert len(points) == 0

    def test_normalize_handles_bad_timestamp(
        self, collector: EPAAirNowCollector
    ) -> None:
        raw = {
            "observations": [
                {
                    "DateObserved": "not-a-date",
                    "HourObserved": 14,
                    "ParameterName": "PM2.5",
                    "AQI": 42,
                    "Latitude": 34.0,
                    "Longitude": -84.0,
                },
            ]
        }
        points = collector.normalize(raw)
        assert len(points) == 0


class TestParameterMap:
    def test_all_expected_parameters_mapped(self) -> None:
        expected = {"PM2.5", "PM10", "OZONE", "O3", "NO2", "SO2", "CO"}
        assert set(PARAMETER_MAP.keys()) == expected

    def test_ozone_aliases_resolve_same(self) -> None:
        assert PARAMETER_MAP["OZONE"] == PARAMETER_MAP["O3"] == "ozone"
