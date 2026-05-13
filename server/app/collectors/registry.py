from collections.abc import Sequence
from typing import TypeAlias

from app.collectors.base import BaseCollector
from app.collectors.noaa_gfs import NOAAGFSCollector
from app.collectors.openweather import OpenWeatherCollector
from app.collectors.openaq import OpenAQCollector
from app.collectors.sentinel5p import Sentinel5PCollector

CollectorClass: TypeAlias = type[BaseCollector]

COLLECTOR_REGISTRY: dict[str, CollectorClass] = {
    NOAAGFSCollector.source_name: NOAAGFSCollector,
    OpenAQCollector.source_name: OpenAQCollector,
    OpenWeatherCollector.source_name: OpenWeatherCollector,
    Sentinel5PCollector.source_name: Sentinel5PCollector,
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
