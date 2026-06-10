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

    async def defer(self, seconds: float) -> None:
        """Push the next acquire at least ``seconds`` into the future.

        Used when an API's rate-limit headers say the budget for the current
        period is spent: stop spending before a 429 happens, not after.
        """
        async with self._lock:
            self._next_at = max(self._next_at, self._clock() + seconds)


def _header_seconds(response: httpx.Response, name: str) -> float | None:
    raw = response.headers.get(name)
    if raw is None:
        return None
    try:
        return max(float(raw), 0.0)
    except ValueError:
        # e.g. an HTTP-date Retry-After; not worth parsing for this client.
        return None


def _retry_wait_seconds(response: httpx.Response) -> float:
    # OpenAQ sends x-ratelimit-reset (seconds until the period restarts) and
    # no Retry-After; other APIs do the opposite. Prefer the precise one.
    for header in ("x-ratelimit-reset", "Retry-After"):
        seconds = _header_seconds(response, header)
        if seconds is not None:
            return seconds
    return DEFAULT_RETRY_AFTER_S


async def _defer_if_budget_spent(
    response: httpx.Response, limiter: AsyncRateLimiter
) -> None:
    remaining = _header_seconds(response, "x-ratelimit-remaining")
    if remaining is not None and remaining <= 0:
        reset = _header_seconds(response, "x-ratelimit-reset")
        if reset:
            await limiter.defer(reset)


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
            await _defer_if_budget_spent(response, limiter)
            response.raise_for_status()
            return response

        if attempt < max_attempts:
            wait = _retry_wait_seconds(response)
            logger.warning(
                "Rate limited (429); waiting %.0fs before retry",
                wait,
                extra={"url": url, "attempt": attempt},
            )
            await sleeper(wait)

    assert response is not None
    response.raise_for_status()
    return response
