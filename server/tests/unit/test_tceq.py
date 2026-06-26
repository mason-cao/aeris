from datetime import date, datetime, timedelta, timezone

import httpx
import pytest
from sqlalchemy import func, select

from app.collectors.backfill import TCEQBackfill
from app.collectors.ratelimit import AsyncRateLimiter
from app.collectors.tceq import (
    PARAM_MAP,
    TCEQ_SITES,
    TCEQCollector,
    parse_report_date,
    parse_tceq_rows,
    tceq_html_to_points,
    user_date_post_data,
)
from app.db.models import DataPoint

# A real in-radius site from the verified registry.
SITE = {
    "name": "Clinton",
    "aqs_cd": "482011035",
    "lat": 29.733729,
    "lon": -95.257603,
    "select_site": "site|Clinton C403/C304/AH113 |48_201_1035|113",
}


def fast_limiter() -> AsyncRateLimiter:
    return AsyncRateLimiter(6_000_000)


def blanks(n: int = 24) -> list[str]:
    return [""] * n


def make_html(
    rows: list[tuple[str, list[str]]],
    heading: str = "Tuesday, June 23, 2026",
    n_hours: int = 24,
) -> str:
    hour_labels = [f"{h:02d}00" for h in range(n_hours)]
    header = (
        "<tr><th>Parameter Measured</th>"
        + "".join(f"<th>{h}</th>" for h in hour_labels)
        + "<th>Parameter Measured</th><th>POC</th></tr>"
    )
    trs = [header]
    for param, vals in rows:
        assert len(vals) == n_hours
        cells = (
            f"<td>{param}</td>"
            + "".join(f"<td>{v}</td>" for v in vals)
            + f"<td>{param}</td><td>1 R</td>"
        )
        trs.append(f"<tr>{cells}</tr>")
    return (
        "<html><body>"
        f"<p>Clinton C403 for {heading}. All times shown are in CST.</p>"
        f"<table>{''.join(trs)}</table></body></html>"
    )


class TestTCEQParsing:
    def test_parse_report_date_from_heading(self) -> None:
        html = make_html([("Nitrogen Dioxide", blanks())])
        assert parse_report_date(html) == date(2026, 6, 23)

    def test_parse_report_date_none_when_absent(self) -> None:
        assert parse_report_date("<html>no date here</html>") is None

    def test_parse_rows_extracts_species_and_hour_map(self) -> None:
        vals = [str(i) for i in range(24)]
        rows = parse_tceq_rows(make_html([("Nitrogen Dioxide", vals)]))
        assert len(rows) == 1
        name, hour_values, _poc = rows[0]
        assert name == "Nitrogen Dioxide"
        assert len(hour_values) == 24
        # hour-keyed mapping: hour 5 -> "5"
        assert hour_values[5] == "5"


class TestTCEQToPoints:
    def test_cst_hour0_maps_to_utc_plus_six(self) -> None:
        vals = blanks()
        vals[0] = "7.5"
        points = tceq_html_to_points(
            make_html([("Nitrogen Dioxide", vals)]), site=SITE, source_name="tceq"
        )
        assert len(points) == 1
        p = points[0]
        # 0000 CST on Jun 23 == 0600 UTC on Jun 23.
        assert p.timestamp == datetime(2026, 6, 23, 6, 0, tzinfo=timezone.utc)
        assert p.metric == "no2"
        assert p.unit == "ppb"
        assert p.value == pytest.approx(7.5)
        assert p.source == "tceq"
        assert p.source_entity_id == "482011035"
        assert p.lat == pytest.approx(29.733729)

    def test_partial_today_with_fewer_hour_columns(self) -> None:
        # "today" renders only elapsed hours -> rows have <24 columns.
        html = make_html(
            [("Nitrogen Dioxide", ["7.5", "8.0", ""])],
            heading="Wednesday, June 24, 2026",
            n_hours=3,
        )
        points = tceq_html_to_points(html, site=SITE, source_name="tceq")
        assert len(points) == 2  # hours 0 and 1; hour 2 is blank
        assert {p.timestamp for p in points} == {
            datetime(2026, 6, 24, 6, 0, tzinfo=timezone.utc),
            datetime(2026, 6, 24, 7, 0, tzinfo=timezone.utc),
        }

    def test_cst_hour23_rolls_into_next_utc_day(self) -> None:
        vals = blanks()
        vals[23] = "9.0"
        p = tceq_html_to_points(
            make_html([("Nitrogen Dioxide", vals)]), site=SITE, source_name="tceq"
        )[0]
        # 2300 CST Jun 23 == 0500 UTC Jun 24.
        assert p.timestamp == datetime(2026, 6, 24, 5, 0, tzinfo=timezone.utc)

    def test_species_metric_and_unit_mapping(self) -> None:
        v_no2, v_so2, v_co = blanks(), blanks(), blanks()
        v_no2[0], v_so2[0], v_co[0] = "8.0", "-0.3", "0.2"
        points = tceq_html_to_points(
            make_html(
                [
                    ("Nitrogen Dioxide", v_no2),
                    ("Sulfur Dioxide", v_so2),
                    ("Carbon Monoxide", v_co),
                ]
            ),
            site=SITE,
            source_name="tceq",
        )
        by = {p.metric: p for p in points}
        assert by["no2"].unit == "ppb"
        assert by["so2"].unit == "ppb"
        assert by["co"].unit == "ppm"
        # Near-MDL negative SO2 is a real reading; keep it.
        assert by["so2"].value == pytest.approx(-0.3)

    def test_skips_blank_and_nonnumeric_cells(self) -> None:
        vals = blanks()
        vals[0] = "5.0"
        vals[1] = ""  # missing
        vals[2] = "N/A"  # non-numeric flag
        points = tceq_html_to_points(
            make_html([("Nitrogen Dioxide", vals)]), site=SITE, source_name="tceq"
        )
        assert len(points) == 1
        assert points[0].timestamp == datetime(2026, 6, 23, 6, 0, tzinfo=timezone.utc)

    def test_ignores_non_target_parameters(self) -> None:
        vals = blanks()
        vals[0] = "42.0"
        points = tceq_html_to_points(
            make_html([("Ozone", vals), ("Outdoor Temperature", vals)]),
            site=SITE,
            source_name="tceq",
        )
        assert points == []

    def test_no_date_yields_no_points(self) -> None:
        vals = blanks()
        vals[0] = "5.0"
        html = "<table><tr><td>Nitrogen Dioxide</td>" + "".join(
            f"<td>{v}</td>" for v in vals
        ) + "<td>Nitrogen Dioxide</td><td>1</td></tr></table>"
        assert tceq_html_to_points(html, site=SITE, source_name="tceq") == []

    def test_registry_is_verified_and_in_radius(self) -> None:
        from app.collectors.geo import within_target_radius

        assert len(TCEQ_SITES) >= 10
        for site in TCEQ_SITES:
            assert within_target_radius(site["lat"], site["lon"])
            assert site["select_site"].startswith("site|")
            assert {"name", "aqs_cd", "lat", "lon", "select_site"} <= set(site)

    def test_param_map_targets_petrochemical_species(self) -> None:
        assert PARAM_MAP["Nitrogen Dioxide"] == ("no2", "ppb")
        assert PARAM_MAP["Sulfur Dioxide"] == ("so2", "ppb")
        assert PARAM_MAP["Carbon Monoxide"] == ("co", "ppm")


class TestUserDatePostData:
    """The daily_summary.pl CGI indexes months 0-based (its own page embeds
    month[0]="January" ... month[5]="June"). Sending the 1-based month requests
    the NEXT month — for a backfill of June that means future July dates, which
    return "no data available" and silently zero records. Verified live."""

    def test_month_is_zero_based(self) -> None:
        d = user_date_post_data(SITE["select_site"], date(2026, 6, 23))
        assert d["user_month"] == "5"  # June -> index 5, not 6
        assert d["user_day"] == "23"
        assert d["user_year"] == "2026"
        assert d["select_date"] == "user"

    def test_january_is_index_zero(self) -> None:
        assert user_date_post_data("s", date(2026, 1, 15))["user_month"] == "0"

    def test_december_is_index_eleven(self) -> None:
        assert user_date_post_data("s", date(2026, 12, 5))["user_month"] == "11"


class TestTCEQDateGuard:
    """Defense-in-depth: when the caller knows the requested date, a report whose
    heading is a DIFFERENT date must not be ingested (it would store the wrong
    day's data, or — with the old month bug — silently nothing)."""

    def test_rejects_when_report_date_differs_from_expected(self) -> None:
        vals = blanks()
        vals[0] = "7.5"
        html = make_html([("Nitrogen Dioxide", vals)], heading="Tuesday, June 23, 2026")
        assert (
            tceq_html_to_points(
                html, site=SITE, source_name="tceq", expected_date=date(2026, 6, 1)
            )
            == []
        )

    def test_accepts_when_report_date_matches_expected(self) -> None:
        vals = blanks()
        vals[0] = "7.5"
        html = make_html([("Nitrogen Dioxide", vals)], heading="Tuesday, June 23, 2026")
        points = tceq_html_to_points(
            html, site=SITE, source_name="tceq", expected_date=date(2026, 6, 23)
        )
        assert len(points) == 1

    def test_no_expected_date_skips_the_check(self) -> None:
        # Live path (today/yesterday) passes no expected_date and is unaffected.
        vals = blanks()
        vals[0] = "7.5"
        html = make_html([("Nitrogen Dioxide", vals)])
        assert len(tceq_html_to_points(html, site=SITE, source_name="tceq")) == 1


class TestTCEQFetch:
    @pytest.mark.asyncio
    async def test_posts_per_site_and_returns_html(self) -> None:
        vals = blanks()
        vals[0] = "7.5"
        seen = {"method": None, "fields": {}}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["method"] = request.method
            body = request.content.decode()
            for pair in body.split("&"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    seen["fields"][k] = v
            return httpx.Response(200, text=make_html([("Nitrogen Dioxide", vals)]))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = TCEQCollector(
            http_client=client, rate_limiter=fast_limiter(), sites=[SITE]
        )
        raw = await collector.fetch()
        points = collector.normalize(raw)
        await client.aclose()

        assert seen["method"] == "POST"
        assert seen["fields"].get("select_date") == "today"
        assert "select_site" in seen["fields"]
        assert len(points) == 1
        assert points[0].metric == "no2"

    @pytest.mark.asyncio
    async def test_circuit_breaker_trips_on_consecutive_failures(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(503)

        sites = [dict(SITE, aqs_cd=str(i)) for i in range(10)]
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = TCEQCollector(
            http_client=client,
            rate_limiter=fast_limiter(),
            sites=sites,
            breaker_threshold=3,
        )
        raw = await collector.fetch()
        await client.aclose()

        # Breaker stops hammering after 3 consecutive failures, not all 10.
        assert calls["n"] == 3
        assert raw["sites"] == []

    @pytest.mark.asyncio
    async def test_one_bad_site_does_not_sink_the_rest(self) -> None:
        vals = blanks()
        vals[0] = "5.0"
        good_site = {
            "name": "Houston Aldine",
            "aqs_cd": "482010024",
            "lat": 29.901031,
            "lon": -95.326125,
            "select_site": "site|Houston Aldine C8/AF108/X150 |48_201_0024|8",
        }

        def handler(request: httpx.Request) -> httpx.Response:
            # Clinton's request fails; Aldine's succeeds.
            if b"Clinton" in request.content:
                return httpx.Response(500)
            return httpx.Response(200, text=make_html([("Nitrogen Dioxide", vals)]))

        sites = [SITE, good_site]
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = TCEQCollector(
            http_client=client, rate_limiter=fast_limiter(), sites=sites
        )
        raw = await collector.fetch()
        points = collector.normalize(raw)
        await client.aclose()

        assert len(raw["sites"]) == 1
        assert points
        assert all(p.source_entity_id == "482010024" for p in points)


T0 = datetime(2026, 6, 20, tzinfo=timezone.utc)


async def _count(session) -> int:
    return await session.scalar(select(func.count()).select_from(DataPoint))


def _echo_date_handler(vals: list[str]):
    """Mock that decodes the posted user-date and heads the report with it.

    Mirrors the live CGI (0-based user_month) so the backfill's date-match guard
    is exercised: a request for day D returns a report whose heading is day D.
    """
    from urllib.parse import unquote_plus

    def handler(request: httpx.Request) -> httpx.Response:
        fields = {
            k: unquote_plus(v)
            for pair in request.content.decode().split("&")
            if "=" in pair
            for k, v in [pair.split("=", 1)]
        }
        day = date(
            int(fields["user_year"]),
            int(fields["user_month"]) + 1,  # CGI is 0-based; decode back
            int(fields["user_day"]),
        )
        heading = day.strftime("%A, %B %d, %Y")
        return httpx.Response(
            200, text=make_html([("Nitrogen Dioxide", vals)], heading=heading)
        )

    return handler


class TestTCEQBackfill:
    @pytest.mark.asyncio
    async def test_stores_points_per_day(self, db_session) -> None:
        vals = blanks()
        vals[0] = "7.5"

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(_echo_date_handler(vals))
        )
        strategy = TCEQBackfill(
            http_client=client, rate_limiter=fast_limiter(), sites=[SITE]
        )
        try:
            result = await strategy.backfill(
                db_session, since=T0, until=T0 + timedelta(days=1)
            )
        finally:
            await client.aclose()

        assert result.source == "tceq"
        assert result.error is None
        # Two days requested -> two distinct hourly timestamps stored.
        assert result.records == 2
        assert await _count(db_session) == 2

    @pytest.mark.asyncio
    async def test_idempotent_rerun(self, db_session) -> None:
        vals = blanks()
        vals[0] = "7.5"

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(_echo_date_handler(vals))
        )
        strategy = TCEQBackfill(
            http_client=client, rate_limiter=fast_limiter(), sites=[SITE]
        )
        try:
            await strategy.backfill(db_session, since=T0, until=T0)
            await strategy.backfill(db_session, since=T0, until=T0)
        finally:
            await client.aclose()

        assert await _count(db_session) == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_trips_on_consecutive_failures(self, db_session) -> None:
        # The undocumented service must not be hammered for 14 sites x ~85 days
        # when it is down; abort after a streak, like the live collector.
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(500)

        sites = [dict(SITE, aqs_cd=str(i)) for i in range(10)]
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        strategy = TCEQBackfill(
            http_client=client,
            rate_limiter=fast_limiter(),
            sites=sites,
            breaker_threshold=3,
        )
        try:
            result = await strategy.backfill(
                db_session, since=T0, until=T0 + timedelta(days=4)
            )
        finally:
            await client.aclose()

        assert calls["n"] == 3  # stops after 3 straight failures, not ~50
        assert result.notes is not None and "breaker" in result.notes.lower()

    @pytest.mark.asyncio
    async def test_backfill_retries_transient_post_error(self, db_session) -> None:
        # TCEQ POSTs go through the retry path: a transient blip is recovered,
        # not dropped as a silent missing day.
        vals = blanks()
        vals[0] = "7.5"
        base = _echo_date_handler(vals)
        state = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            state["n"] += 1
            if state["n"] == 1:
                raise httpx.ReadError("transient blip")
            return base(request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        strategy = TCEQBackfill(
            http_client=client, rate_limiter=fast_limiter(), sites=[SITE]
        )
        try:
            result = await strategy.backfill(db_session, since=T0, until=T0)
        finally:
            await client.aclose()

        assert state["n"] == 2  # retried after the transient error
        assert result.records == 1  # the day's data was recovered
