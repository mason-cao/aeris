import math
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DataPoint, DataSource
from app.db.session import get_session

router = APIRouter(tags=["data"])


def _to_utc(value: datetime) -> datetime:
    """Coerce a query-param datetime to tz-aware UTC for range filtering.

    The ``timestamp`` column is tz-aware; a naive bound (client omitted the
    offset) raises or mis-compares on asyncpg. The store is all-UTC, so a
    naive bound is assumed UTC. SQLite ignores tzinfo, so this is a no-op
    there.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


class DataPointResponse(BaseModel):
    id: UUID
    timestamp: datetime
    lat: float
    lon: float
    metric: str
    value: float
    unit: str
    source: str
    source_entity_id: str
    raw_json: dict[str, Any] | None
    collected_at: datetime


class DataSourceResponse(BaseModel):
    name: str
    source_type: str
    status: str
    last_collected_at: datetime | None
    error_count: int


class PaginatedDataPoints(BaseModel):
    items: list[DataPointResponse]
    total: int
    limit: int
    offset: int


async def fetch_data_sources(session: AsyncSession) -> list[DataSourceResponse]:
    result = await session.execute(select(DataSource).order_by(DataSource.name))
    sources = result.scalars().all()
    return [
        DataSourceResponse.model_validate(source, from_attributes=True)
        for source in sources
    ]


@router.get("/data/sources", response_model=list[DataSourceResponse])
async def list_data_sources(
    session: AsyncSession = Depends(get_session),
) -> list[DataSourceResponse]:
    return await fetch_data_sources(session)


@router.get("/data/{source}", response_model=PaginatedDataPoints)
async def get_data_by_source(
    source: str,
    metric: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    lat: float | None = None,
    lon: float | None = None,
    radius_km: float | None = Query(default=None, gt=0),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> PaginatedDataPoints:
    query = select(DataPoint).where(DataPoint.source == source)
    count_query = (
        select(func.count()).select_from(DataPoint).where(DataPoint.source == source)
    )

    if metric:
        query = query.where(DataPoint.metric == metric)
        count_query = count_query.where(DataPoint.metric == metric)
    if start:
        start = _to_utc(start)
        query = query.where(DataPoint.timestamp >= start)
        count_query = count_query.where(DataPoint.timestamp >= start)
    if end:
        end = _to_utc(end)
        query = query.where(DataPoint.timestamp <= end)
        count_query = count_query.where(DataPoint.timestamp <= end)
    if lat is not None and lon is not None and radius_km is not None:
        lat_delta = radius_km / 111.32
        lon_scale = 111.32 * max(math.cos(math.radians(lat)), 0.01)
        lon_delta = radius_km / lon_scale
        query = query.where(
            DataPoint.lat.between(lat - lat_delta, lat + lat_delta),
            DataPoint.lon.between(lon - lon_delta, lon + lon_delta),
        )
        count_query = count_query.where(
            DataPoint.lat.between(lat - lat_delta, lat + lat_delta),
            DataPoint.lon.between(lon - lon_delta, lon + lon_delta),
        )

    total_result = await session.execute(count_query)
    total = total_result.scalar_one()

    query = query.order_by(DataPoint.timestamp.desc()).limit(limit).offset(offset)
    result = await session.execute(query)
    rows = result.scalars().all()

    return PaginatedDataPoints(
        items=[DataPointResponse.model_validate(row, from_attributes=True) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/data", response_model=list[DataSourceResponse])
async def list_data_sources_legacy(
    session: AsyncSession = Depends(get_session),
) -> list[DataSourceResponse]:
    return await fetch_data_sources(session)
