import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, Field
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DataPoint, DataSource

logger = logging.getLogger(__name__)


class DataPointCreate(BaseModel):
    """Normalized data point schema — all collectors output this format."""

    timestamp: datetime
    lat: float
    lon: float
    metric: str
    value: float
    source: str
    raw_json: dict[str, Any] | None = None


class CollectionResult(BaseModel):
    """Result of a single collection run."""

    source: str
    success: bool
    record_count: int = 0
    duration_ms: float = 0
    errors: list[str] = Field(default_factory=list)


class BaseCollector(ABC):
    """Abstract base class for all data source collectors.

    Subclasses must implement:
      - fetch(): retrieve raw data from the external API
      - normalize(): transform raw data into DataPointCreate objects

    The collect() method orchestrates: fetch → normalize → store,
    with retry logic, timing, and structured logging built in.
    """

    source_name: str
    collect_interval_minutes: int

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._client = http_client
        self._owns_client = http_client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @abstractmethod
    async def fetch(self) -> dict[str, Any]:
        """Fetch raw data from the external API."""

    @abstractmethod
    def normalize(self, raw_data: dict[str, Any]) -> list[DataPointCreate]:
        """Transform raw API response into normalized DataPointCreate records."""

    async def collect(self, session: AsyncSession, max_retries: int = 3) -> CollectionResult:
        """Orchestrate fetch → normalize → store with retry and logging."""
        start = time.monotonic()
        errors: list[str] = []

        for attempt in range(1, max_retries + 1):
            try:
                raw_data = await self.fetch()
                points = self.normalize(raw_data)

                if points:
                    await self._store(session, points)

                duration_ms = (time.monotonic() - start) * 1000
                result = CollectionResult(
                    source=self.source_name,
                    success=True,
                    record_count=len(points),
                    duration_ms=round(duration_ms, 1),
                )

                await self._update_source_status(session, success=True)

                logger.info(
                    "Collection complete",
                    extra={
                        "source": self.source_name,
                        "records": len(points),
                        "duration_ms": result.duration_ms,
                        "attempt": attempt,
                    },
                )
                return result

            except Exception as e:
                errors.append(f"Attempt {attempt}: {type(e).__name__}: {e}")
                logger.warning(
                    "Collection attempt failed",
                    extra={
                        "source": self.source_name,
                        "attempt": attempt,
                        "error": str(e),
                    },
                )

                if attempt < max_retries:
                    backoff = 30 * (2 ** (attempt - 1))  # 30s, 60s, 120s
                    import asyncio

                    await asyncio.sleep(backoff)

        # All retries exhausted
        duration_ms = (time.monotonic() - start) * 1000
        await self._update_source_status(session, success=False)

        logger.error(
            "Collection failed after all retries",
            extra={"source": self.source_name, "errors": errors},
        )

        return CollectionResult(
            source=self.source_name,
            success=False,
            duration_ms=round(duration_ms, 1),
            errors=errors,
        )

    async def _store(self, session: AsyncSession, points: list[DataPointCreate]) -> None:
        """Bulk insert normalized data points."""
        stmt = insert(DataPoint).values(
            [point.model_dump() for point in points]
        )
        await session.execute(stmt)
        await session.commit()

    async def _update_source_status(
        self, session: AsyncSession, *, success: bool
    ) -> None:
        """Update the DataSource record with collection status."""
        from sqlalchemy import select, update
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)

        # Upsert the data source record
        stmt = pg_insert(DataSource).values(
            name=self.source_name,
            source_type=self.source_name,
            status="active" if success else "error",
            last_collected_at=now if success else None,
            error_count=0 if success else 1,
        ).on_conflict_do_update(
            index_elements=["name"],
            set_={
                "status": "active" if success else "error",
                "last_collected_at": now if success else DataSource.last_collected_at,
                "error_count": (
                    0 if success else DataSource.error_count + 1
                ),
            },
        )
        await session.execute(stmt)
        await session.commit()
