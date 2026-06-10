import httpx
import pytest

from app.collectors.ratelimit import AsyncRateLimiter, rate_limited_get


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class SleepRecorder:
    def __init__(self, clock: FakeClock | None = None) -> None:
        self.calls: list[float] = []
        self._clock = clock

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        if self._clock is not None:
            self._clock.now += seconds


class TestAsyncRateLimiter:
    @pytest.mark.asyncio
    async def test_spaces_back_to_back_acquires(self) -> None:
        clock = FakeClock()
        sleeper = SleepRecorder(clock)
        limiter = AsyncRateLimiter(30, clock=clock, sleeper=sleeper)

        await limiter.acquire()
        await limiter.acquire()

        # 30/min = one request every 2s; the second acquire must wait.
        assert sleeper.calls == [pytest.approx(2.0)]

    @pytest.mark.asyncio
    async def test_no_wait_when_naturally_spaced(self) -> None:
        clock = FakeClock()
        sleeper = SleepRecorder()
        limiter = AsyncRateLimiter(30, clock=clock, sleeper=sleeper)

        await limiter.acquire()
        clock.now += 5.0
        await limiter.acquire()

        assert sleeper.calls == []


class TestRateLimitedGet:
    @pytest.mark.asyncio
    async def test_honors_retry_after_on_429(self) -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(429, headers={"Retry-After": "7"})
            return httpx.Response(200, json={"ok": True})

        sleeper = SleepRecorder()
        limiter = AsyncRateLimiter(6000, clock=FakeClock(), sleeper=sleeper)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            response = await rate_limited_get(
                client,
                "https://api.example.org/x",
                limiter=limiter,
                sleeper=sleeper,
            )

        assert response.status_code == 200
        assert attempts == 2
        assert 7.0 in sleeper.calls

    @pytest.mark.asyncio
    async def test_persistent_429_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"Retry-After": "1"})

        sleeper = SleepRecorder()
        limiter = AsyncRateLimiter(6000, clock=FakeClock(), sleeper=sleeper)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await rate_limited_get(
                    client,
                    "https://api.example.org/x",
                    limiter=limiter,
                    sleeper=sleeper,
                    max_attempts=3,
                )

    @pytest.mark.asyncio
    async def test_missing_retry_after_uses_default(self) -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(429)
            return httpx.Response(200, json={})

        sleeper = SleepRecorder()
        limiter = AsyncRateLimiter(6000, clock=FakeClock(), sleeper=sleeper)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            response = await rate_limited_get(
                client,
                "https://api.example.org/x",
                limiter=limiter,
                sleeper=sleeper,
            )

        assert response.status_code == 200
        assert 60.0 in sleeper.calls


class TestOpenAQRateLimitHeaders:
    @pytest.mark.asyncio
    async def test_429_uses_x_ratelimit_reset_when_no_retry_after(self) -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(429, headers={"x-ratelimit-reset": "13"})
            return httpx.Response(200, json={})

        sleeper = SleepRecorder()
        limiter = AsyncRateLimiter(6000, clock=FakeClock(), sleeper=sleeper)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            response = await rate_limited_get(
                client,
                "https://api.example.org/x",
                limiter=limiter,
                sleeper=sleeper,
            )

        assert response.status_code == 200
        assert 13.0 in sleeper.calls
        assert 60.0 not in sleeper.calls

    @pytest.mark.asyncio
    async def test_exhausted_budget_defers_next_acquire(self) -> None:
        # A 200 whose headers say the budget is spent must push the next
        # request past the reset, even though nothing failed.
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={},
                headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "9"},
            )

        clock = FakeClock()
        sleeper = SleepRecorder(clock)
        limiter = AsyncRateLimiter(6000, clock=clock, sleeper=sleeper)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            await rate_limited_get(
                client, "https://api.example.org/x", limiter=limiter, sleeper=sleeper
            )
            await rate_limited_get(
                client, "https://api.example.org/x", limiter=limiter, sleeper=sleeper
            )

        assert any(call >= 9.0 for call in sleeper.calls)

    @pytest.mark.asyncio
    async def test_remaining_budget_does_not_defer(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={},
                headers={"x-ratelimit-remaining": "42", "x-ratelimit-reset": "9"},
            )

        clock = FakeClock()
        sleeper = SleepRecorder(clock)
        limiter = AsyncRateLimiter(6000, clock=clock, sleeper=sleeper)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            await rate_limited_get(
                client, "https://api.example.org/x", limiter=limiter, sleeper=sleeper
            )
            await rate_limited_get(
                client, "https://api.example.org/x", limiter=limiter, sleeper=sleeper
            )

        assert all(call < 9.0 for call in sleeper.calls)
