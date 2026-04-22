import math
from dataclasses import dataclass

from app.config import settings


@dataclass(frozen=True)
class BoundingBox:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    @property
    def nw_lat(self) -> float:
        return self.max_lat

    @property
    def nw_lon(self) -> float:
        return self.min_lon

    @property
    def se_lat(self) -> float:
        return self.min_lat

    @property
    def se_lon(self) -> float:
        return self.max_lon

    def as_csv(self) -> str:
        """Return min lon, min lat, max lon, max lat."""
        return (
            f"{self.min_lon:.4f},{self.min_lat:.4f},"
            f"{self.max_lon:.4f},{self.max_lat:.4f}"
        )


def target_bounding_box(radius_km: float | None = None) -> BoundingBox:
    lat = settings.aeris_target_lat
    lon = settings.aeris_target_lon
    radius = radius_km if radius_km is not None else settings.aeris_target_radius_km

    lat_delta = radius / 111.32
    lon_delta = radius / (111.32 * max(math.cos(math.radians(lat)), 0.01))

    return BoundingBox(
        min_lat=lat - lat_delta,
        max_lat=lat + lat_delta,
        min_lon=lon - lon_delta,
        max_lon=lon + lon_delta,
    )


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 coordinates."""
    earth_radius_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * earth_radius_km * math.asin(math.sqrt(a))


def within_target_radius(lat: float, lon: float, radius_km: float | None = None) -> bool:
    radius = radius_km if radius_km is not None else settings.aeris_target_radius_km
    return (
        distance_km(settings.aeris_target_lat, settings.aeris_target_lon, lat, lon)
        <= radius
    )
