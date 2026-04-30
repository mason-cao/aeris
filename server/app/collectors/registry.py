from collections.abc import Sequence
from typing import TypeAlias

from app.collectors.base import BaseCollector
from app.collectors.epa_airnow import EPAAirNowCollector
from app.collectors.nasa_firms import NASAFIRMSCollector
from app.collectors.noaa_weather import OpenWeatherCollector
from app.collectors.openaq import OpenAQCollector
from app.collectors.purpleair import PurpleAirCollector

CollectorClass: TypeAlias = type[BaseCollector]

COLLECTOR_REGISTRY: dict[str, CollectorClass] = {
    EPAAirNowCollector.source_name: EPAAirNowCollector,
    OpenAQCollector.source_name: OpenAQCollector,
    PurpleAirCollector.source_name: PurpleAirCollector,
    OpenWeatherCollector.source_name: OpenWeatherCollector,
    NASAFIRMSCollector.source_name: NASAFIRMSCollector,
}


def collector_names() -> list[str]:
    return sorted(COLLECTOR_REGISTRY)


def get_collector_class(source: str) -> CollectorClass:
    try:
        return COLLECTOR_REGISTRY[source]
    except KeyError as exc:
        available = ", ".join(collector_names())
        raise ValueError(f"Unknown collector source '{source}'. Available: {available}") from exc


def create_collector(source: str) -> BaseCollector:
    return get_collector_class(source)()


def create_collectors(source: str | None = None) -> list[BaseCollector]:
    if source is not None:
        return [create_collector(source)]

    return [COLLECTOR_REGISTRY[name]() for name in collector_names()]


def source_choices() -> Sequence[str]:
    return tuple(collector_names())
