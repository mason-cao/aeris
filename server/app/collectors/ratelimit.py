import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

import httpx

logger = logging.getLogger(__name__)

# Default wait when a 429 arrives without a Retry-After header.
DEFAULT_RETRY_AFTER_S = 60.0

Clock = Callable[[], float]
Sleeper = Callable[[float], Awaitable[None]]


class AsyncRateLimiter:
    """Spaces requests evenly so a shared API key stays under its rate limit.

    One instance per API budget: every caller that spends requests against the
    same key must go through the same limiter, or the budget means nothing.
    """

    def __init__(
        self,
        max_per_minute: float,
        *,
        clock: Clock = time.monotonic,
        sleeper: Sleeper = asyncio.sleep,
    ) -> None:
        self._interval = 60.0 / max_per_minute
        self._clock = clock
        self._sleeper = sleeper
        self._next_at = float("-inf")
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = self._clock()
            wait = self._next_at - now
            self._next_at = max(now, self._next_at) + self._interval
        if wait > 0:
            await self._sleeper(wait)


def _retry_after_seconds(response: httpx.Response) -> float:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return DEFAULT_RETRY_AFTER_S
    try:
        return max(float(raw), 0.0)
    except ValueError:
        # HTTP-date form; not worth parsing for this client.
        return DEFAULT_RETRY_AFTER_S


async def rate_limited_get(
    client: httpx.AsyncClient,
    url: str,
    *,
    limiter: AsyncRateLimiter,
    max_attempts: int = 3,
    sleeper: Sleeper = asyncio.sleep,
    **request_kwargs,
) -> httpx.Response:
    """GET through the limiter, honoring Retry-After on 429 responses."""
    response: httpx.Response | None = None
    for attempt in range(1, max_attempts + 1):
        await limiter.acquire()
        response = await client.get(url, **request_kwargs)
        if response.status_code != 429:
            response.raise_for_status()
            return response

        if attempt < max_attempts:
            wait = _retry_after_seconds(response)
            logger.warning(
                "Rate limited (429); waiting %.0fs before retry",
                wait,
                extra={"url": url, "attempt": attempt},
            )
            await sleeper(wait)

    assert response is not None
    response.raise_for_status()
    return response
