import pytest

from app.collectors.base import BaseCollector
from app.collectors.registry import (
    COLLECTOR_REGISTRY,
    collector_names,
    create_collector,
    create_collectors,
    get_collector_class,
    source_choices,
)


class TestCollectorRegistry:
    def test_registry_contains_all_collectors(self) -> None:
        assert set(COLLECTOR_REGISTRY) == {
            "noaa_gfs",
            "openaq",
            "openweather",
            "sentinel5p",
        }

    def test_collector_names_are_sorted(self) -> None:
        names = collector_names()

        assert names == sorted(names)

    def test_source_choices_match_collector_names(self) -> None:
        assert tuple(collector_names()) == source_choices()

    def test_get_collector_class_returns_registered_class(self) -> None:
        collector_class = get_collector_class("sentinel5p")

        assert collector_class.source_name == "sentinel5p"

    def test_get_collector_class_rejects_unknown_source(self) -> None:
        with pytest.raises(ValueError, match="Unknown collector source"):
            get_collector_class("missing")

    def test_create_collector_returns_instance(self) -> None:
        collector = create_collector("openaq")

        assert isinstance(collector, BaseCollector)
        assert collector.source_name == "openaq"

    def test_create_collectors_filters_to_one_source(self) -> None:
        collectors = create_collectors("openweather")

        assert len(collectors) == 1
        assert collectors[0].source_name == "openweather"

    def test_create_collectors_returns_all_registered_sources(self) -> None:
        collectors = create_collectors()

        assert [collector.source_name for collector in collectors] == collector_names()
