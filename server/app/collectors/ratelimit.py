import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

import httpx

logger = logging.getLogger(__name__)

# Default wait when a 429 arrives without a Retry-After header.
DEFAULT_RETRY_AFTER_S = 60.0

# Ceiling on any header-derived wait. A rate-limit reset is at most one period
# away; a value far beyond that is an absolute epoch sent where a duration was
# expected, which would otherwise park the client for years.
MAX_RETRY_WAIT_S = 300.0

# Transient (retry-worthy) failures: connection-level transport errors and the
# explicitly-temporary server statuses. A single blip must not abort a weeks-long
# unattended backfill. 4xx is deterministic and never retried.
TRANSIENT_STATUS = frozenset({502, 503, 504})
TRANSIENT_BACKOFF_BASE_S = 1.0
TRANSIENT_BACKOFF_CAP_S = 30.0

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
            return min(seconds, MAX_RETRY_WAIT_S)
    return DEFAULT_RETRY_AFTER_S


async def _defer_if_budget_spent(
    response: httpx.Response, limiter: AsyncRateLimiter
) -> None:
    remaining = _header_seconds(response, "x-ratelimit-remaining")
    if remaining is not None and remaining <= 0:
        reset = _header_seconds(response, "x-ratelimit-reset")
        if reset:
            await limiter.defer(min(reset, MAX_RETRY_WAIT_S))


async def _backoff(sleeper: Sleeper, attempt: int, url: str, reason: str) -> None:
    wait = min(
        TRANSIENT_BACKOFF_BASE_S * 2 ** (attempt - 1), TRANSIENT_BACKOFF_CAP_S
    )
    logger.warning(
        "transient failure (%s); backing off %.0fs before retry",
        reason,
        wait,
        extra={"url": url, "attempt": attempt},
    )
    await sleeper(wait)


async def rate_limited_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    limiter: AsyncRateLimiter,
    max_attempts: int = 3,
    sleeper: Sleeper = asyncio.sleep,
    **request_kwargs,
) -> httpx.Response:
    """Request through the limiter, retrying 429s and transient failures.

    Retries: HTTP 429 (honoring Retry-After / x-ratelimit-reset), transient
    transport errors (``httpx.TransportError`` — ReadError/ReadTimeout/
    ConnectError/...), and transient server statuses (502/503/504). A 4xx or any
    other status is returned to ``raise_for_status`` immediately — retrying a
    deterministic client error is pointless. The exception/status from the final
    attempt propagates so the caller still sees the failure.
    """
    response: httpx.Response | None = None
    for attempt in range(1, max_attempts + 1):
        await limiter.acquire()
        try:
            response = await client.request(method, url, **request_kwargs)
        except httpx.TransportError as exc:
            if attempt >= max_attempts:
                raise
            await _backoff(sleeper, attempt, url, type(exc).__name__)
            continue

        if response.status_code == 429 and attempt < max_attempts:
            wait = _retry_wait_seconds(response)
            # Push the shared budget so concurrent callers back off too, not
            # just this one.
            await limiter.defer(wait)
            logger.warning(
                "Rate limited (429); waiting %.0fs before retry",
                wait,
                extra={"url": url, "attempt": attempt},
            )
            await sleeper(wait)
            continue
        if response.status_code in TRANSIENT_STATUS and attempt < max_attempts:
            await _backoff(sleeper, attempt, url, f"HTTP {response.status_code}")
            continue

        # Success, a non-retryable status, or the last attempt: let
        # raise_for_status surface any 429/5xx that exhausted its retries.
        await _defer_if_budget_spent(response, limiter)
        response.raise_for_status()
        return response

    assert response is not None  # max_attempts >= 1, so the loop ran
    response.raise_for_status()
    return response


async def rate_limited_get(
    client: httpx.AsyncClient,
    url: str,
    *,
    limiter: AsyncRateLimiter,
    max_attempts: int = 3,
    sleeper: Sleeper = asyncio.sleep,
    **request_kwargs,
) -> httpx.Response:
    """GET through the limiter, with 429 + transient-failure retry."""
    return await rate_limited_request(
        client, "GET", url, limiter=limiter, max_attempts=max_attempts,
        sleeper=sleeper, **request_kwargs,
    )


async def rate_limited_post(
    client: httpx.AsyncClient,
    url: str,
    *,
    limiter: AsyncRateLimiter,
    max_attempts: int = 3,
    sleeper: Sleeper = asyncio.sleep,
    **request_kwargs,
) -> httpx.Response:
    """POST through the limiter, with 429 + transient-failure retry."""
    return await rate_limited_request(
        client, "POST", url, limiter=limiter, max_attempts=max_attempts,
        sleeper=sleeper, **request_kwargs,
    )
