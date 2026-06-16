import math
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.db.models import Anomaly, DataPoint, EnrichmentRecord
from app.detection.enrichment import (
    DEFAULT_CONFIG,
    EXPECTED_SOURCES,
    SCHEMA_VERSION,
    EnrichmentConfig,
    EnrichmentSummary,
    _format_summary,
    _parse_args,
    anomaly_bounding_box,
    build_cross_source_summary,
    context_window,
    enrich_anomaly,
    enrich_pending_anomalies,
    load_context_points,
    persist_enrichment,
)


HOUSTON_LAT = 29.7604
HOUSTON_LON = -95.3698
# Anomaly timestamp used as the window centre across the pure-function tests.
T0 = datetime(2026, 5, 16, 14, 0, tzinfo=timezone.utc)


def _anomaly(**overrides) -> Anomaly:
    defaults = dict(
        timestamp=T0,
        lat=HOUSTON_LAT,
        lon=HOUSTON_LON,
        metric="pm25",
        source="openaq",
        value=180.0,
        expected_value=22.0,
        z_score=5.1,
        methods_triggered=["zscore", "stl"],
        severity="moderate",
    )
    defaults.update(overrides)
    return Anomaly(**defaults)


def _dp(
    *,
    ts: datetime,
    value: float,
    metric: str = "pm25",
    source: str = "openaq",
    entity: str = "openaq-1",
    lat: float = HOUSTON_LAT,
    lon: float = HOUSTON_LON,
    unit: str = "ug/m3",
) -> DataPoint:
    return DataPoint(
        timestamp=ts,
        lat=lat,
        lon=lon,
        metric=metric,
        value=value,
        unit=unit,
        source=source,
        source_entity_id=entity,
        raw_json=None,
    )


def _summary(
    anomaly: Anomaly,
    points: list[DataPoint],
    config: EnrichmentConfig = DEFAULT_CONFIG,
) -> dict:
    """build_cross_source_summary with the window derived from the anomaly."""
    start, end = context_window(anomaly.timestamp, config)
    return build_cross_source_summary(
        anomaly, points, window_start=start, window_end=end, config=config
    )


async def _seed(db_session, points: list[DataPoint]) -> None:
    # Commit per row: SQLAlchemy's bulk-insert sentinel path does not
    # round-trip the (UUID, timestamp) composite PK cleanly on SQLite.
    for point in points:
        db_session.add(point)
        await db_session.commit()


async def _persist_anomaly(db_session, **overrides) -> Anomaly:
    anomaly = _anomaly(**overrides)
    db_session.add(anomaly)
    await db_session.commit()
    return anomaly


class TestEnrichmentConfig:
    def test_defaults_are_72h_centered_and_50km(self) -> None:
        assert DEFAULT_CONFIG.hours_before == 36.0
        assert DEFAULT_CONFIG.hours_after == 36.0
        assert DEFAULT_CONFIG.spatial_radius_km == 50.0


class TestContextWindow:
    def test_default_window_is_72h_centered(self) -> None:
        start, end = context_window(T0)
        assert start == T0 - timedelta(hours=36)
        assert end == T0 + timedelta(hours=36)
        assert end - start == timedelta(hours=72)

    def test_custom_hours_before_and_after(self) -> None:
        config = EnrichmentConfig(hours_before=24.0, hours_after=12.0)
        start, end = context_window(T0, config)
        assert start == T0 - timedelta(hours=24)
        assert end == T0 + timedelta(hours=12)

    def test_naive_timestamp_coerced_to_utc(self) -> None:
        naive = datetime(2026, 5, 16, 14, 0)
        start, end = context_window(naive)
        assert start.tzinfo is not None
        assert end.tzinfo is not None
        assert start == T0 - timedelta(hours=36)


class TestAnomalyBoundingBox:
    def test_box_brackets_the_anomaly(self) -> None:
        box = anomaly_bounding_box(HOUSTON_LAT, HOUSTON_LON, 50.0)
        assert box.min_lat < HOUSTON_LAT < box.max_lat
        assert box.min_lon < HOUSTON_LON < box.max_lon

    def test_latitude_span_matches_radius(self) -> None:
        box = anomaly_bounding_box(HOUSTON_LAT, HOUSTON_LON, 50.0)
        # 1 deg latitude ~ 111.32 km, so a 50 km radius is ~0.449 deg each way.
        assert box.max_lat - box.min_lat == pytest.approx(
            2 * 50.0 / 111.32, abs=1e-6
        )

    def test_larger_radius_widens_box(self) -> None:
        small = anomaly_bounding_box(HOUSTON_LAT, HOUSTON_LON, 25.0)
        large = anomaly_bounding_box(HOUSTON_LAT, HOUSTON_LON, 75.0)
        assert large.max_lat - large.min_lat > small.max_lat - small.min_lat
        assert large.max_lon - large.min_lon > small.max_lon - small.min_lon


class TestBuildCrossSourceSummary:
    def test_includes_schema_version_and_anomaly_echo(self) -> None:
        summary = _summary(_anomaly(), [_dp(ts=T0, value=180.0)])
        assert summary["schema_version"] == SCHEMA_VERSION
        echo = summary["anomaly"]
        assert echo["metric"] == "pm25"
        assert echo["source"] == "openaq"
        assert echo["value"] == pytest.approx(180.0)
        assert echo["severity"] == "moderate"
        assert echo["lat"] == pytest.approx(HOUSTON_LAT)
        assert echo["timestamp"].startswith("2026-05-16T14:00:00")

    def test_window_echo_matches_inputs(self) -> None:
        window = _summary(_anomaly(), [_dp(ts=T0, value=10.0)])["window"]
        assert window["hours_before"] == 36.0
        assert window["hours_after"] == 36.0
        assert window["spatial_radius_km"] == 50.0
        assert window["start"].startswith("2026-05-15T02:00:00")
        assert window["end"].startswith("2026-05-18T02:00:00")

    def test_groups_points_by_source(self) -> None:
        points = [
            _dp(ts=T0, value=180.0, source="openaq", metric="pm25"),
            _dp(ts=T0, value=3.0, source="noaa_gfs", metric="u_10m",
                entity="gfs:29.76,-95.37", unit="m/s"),
        ]
        summary = _summary(_anomaly(), points)
        assert set(summary["sources"]) == {"openaq", "noaa_gfs"}

    def test_groups_points_by_metric_within_source(self) -> None:
        points = [
            _dp(ts=T0, value=180.0, metric="pm25", entity="openaq-1"),
            _dp(ts=T0, value=44.0, metric="no2", entity="openaq-2", unit="ppb"),
        ]
        summary = _summary(_anomaly(), points)
        assert set(summary["sources"]["openaq"]["metrics"]) == {"pm25", "no2"}

    def test_series_grouped_by_entity_and_sorted_by_time(self) -> None:
        points = [
            _dp(ts=T0 + timedelta(hours=2), value=30.0, entity="openaq-1"),
            _dp(ts=T0 - timedelta(hours=2), value=10.0, entity="openaq-1"),
            _dp(ts=T0, value=20.0, entity="openaq-1"),
        ]
        summary = _summary(_anomaly(), points)
        entities = summary["sources"]["openaq"]["metrics"]["pm25"]["entities"]
        assert len(entities) == 1
        series = entities[0]["series"]
        assert [row[0] for row in series] == sorted(row[0] for row in series)
        assert [row[1] for row in series] == [10.0, 20.0, 30.0]

    def test_distance_km_computed_per_entity(self) -> None:
        # Entity ~22 km north of the anomaly (0.2 deg latitude).
        points = [_dp(ts=T0, value=20.0, lat=HOUSTON_LAT + 0.2, entity="openaq-1")]
        summary = _summary(_anomaly(), points)
        entity = summary["sources"]["openaq"]["metrics"]["pm25"]["entities"][0]
        assert entity["distance_km"] == pytest.approx(22.26, abs=0.5)

    def test_entities_sorted_by_distance(self) -> None:
        points = [
            _dp(ts=T0, value=99.0, lat=HOUSTON_LAT + 0.3, entity="far"),
            _dp(ts=T0, value=10.0, entity="near"),
        ]
        summary = _summary(_anomaly(), points)
        entities = summary["sources"]["openaq"]["metrics"]["pm25"]["entities"]
        assert [e["entity_id"] for e in entities] == ["near", "far"]

    def test_value_range_min_max_mean(self) -> None:
        points = [
            _dp(ts=T0 - timedelta(hours=1), value=10.0),
            _dp(ts=T0, value=20.0),
            _dp(ts=T0 + timedelta(hours=1), value=60.0),
        ]
        rng = _summary(_anomaly(), points)["sources"]["openaq"]["metrics"]["pm25"][
            "value_range"
        ]
        assert rng["min"] == pytest.approx(10.0)
        assert rng["max"] == pytest.approx(60.0)
        assert rng["mean"] == pytest.approx(30.0)

    def test_nan_observations_are_dropped_before_aggregation(self) -> None:
        # GFS/S5P can carry a NaN value. Unfiltered it poisons mean (-> NaN)
        # and emits literal `NaN` into the JSON payload, which is invalid JSON.
        points = [
            _dp(ts=T0 - timedelta(hours=1), value=10.0),
            _dp(ts=T0, value=float("nan")),
            _dp(ts=T0 + timedelta(hours=1), value=20.0),
        ]
        pm25 = _summary(_anomaly(), points)["sources"]["openaq"]["metrics"]["pm25"]

        assert pm25["n_points"] == 2  # the NaN observation is gone
        rng = pm25["value_range"]
        assert rng["min"] == pytest.approx(10.0)
        assert rng["max"] == pytest.approx(20.0)
        assert rng["mean"] == pytest.approx(15.0)
        series_values = [v for e in pm25["entities"] for _, v in e["series"]]
        assert series_values and all(math.isfinite(v) for v in series_values)

    def test_metric_with_only_nonfinite_values_is_absent(self) -> None:
        points = [
            _dp(ts=T0, value=float("nan")),
            _dp(ts=T0 + timedelta(hours=1), value=float("inf")),
        ]
        summary = _summary(_anomaly(), points)
        assert summary["sources"] == {}
        assert summary["coverage"]["openaq"] is False

    def test_nearest_in_time_picks_closest_observation(self) -> None:
        points = [
            _dp(ts=T0 - timedelta(hours=2), value=10.0),
            _dp(ts=T0 - timedelta(hours=1), value=20.0),
            _dp(ts=T0 + timedelta(hours=3), value=30.0),
        ]
        nearest = _summary(_anomaly(), points)["sources"]["openaq"]["metrics"][
            "pm25"
        ]["nearest_in_time"]
        assert nearest["v"] == pytest.approx(20.0)
        assert nearest["dt_minutes"] == pytest.approx(60.0)

    def test_nearest_in_time_breaks_ties_by_distance(self) -> None:
        # Both observations are 1 h from the anomaly; the closer station wins.
        points = [
            _dp(ts=T0 - timedelta(hours=1), value=20.0, entity="near"),
            _dp(ts=T0 + timedelta(hours=1), value=99.0,
                lat=HOUSTON_LAT + 0.3, entity="far"),
        ]
        nearest = _summary(_anomaly(), points)["sources"]["openaq"]["metrics"][
            "pm25"
        ]["nearest_in_time"]
        assert nearest["v"] == pytest.approx(20.0)
        assert nearest["entity_id"] == "near"

    def test_coverage_flags_present_and_absent_sources(self) -> None:
        points = [
            _dp(ts=T0, value=180.0, source="openaq", metric="pm25"),
            _dp(ts=T0, value=1000.0, source="noaa_gfs", metric="pbl_height",
                entity="gfs:29.76,-95.37", unit="m"),
        ]
        coverage = _summary(_anomaly(), points)["coverage"]
        assert coverage["openaq"] is True
        assert coverage["noaa_gfs"] is True
        assert coverage["openweather"] is False
        assert coverage["sentinel5p"] is False

    def test_coverage_only_tracks_expected_sources(self) -> None:
        coverage = _summary(_anomaly(), [_dp(ts=T0, value=1.0)])["coverage"]
        assert set(coverage) == set(EXPECTED_SOURCES)

    def test_spatial_filter_excludes_points_beyond_radius(self) -> None:
        points = [
            _dp(ts=T0, value=20.0, entity="near"),
            # ~111 km north — outside the default 50 km radius.
            _dp(ts=T0, value=99.0, lat=HOUSTON_LAT + 1.0, entity="far"),
        ]
        pm25 = _summary(_anomaly(), points)["sources"]["openaq"]["metrics"]["pm25"]
        assert pm25["n_points"] == 1
        assert pm25["entities"][0]["entity_id"] == "near"

    def test_temporal_filter_excludes_points_outside_window(self) -> None:
        points = [
            _dp(ts=T0, value=20.0),
            _dp(ts=T0 - timedelta(hours=40), value=99.0),  # before window start
        ]
        summary = _summary(_anomaly(), points)
        assert summary["sources"]["openaq"]["metrics"]["pm25"]["n_points"] == 1

    def test_window_boundaries_are_inclusive(self) -> None:
        start, end = context_window(T0)
        points = [_dp(ts=start, value=1.0), _dp(ts=end, value=2.0)]
        summary = build_cross_source_summary(
            _anomaly(), points, window_start=start, window_end=end
        )
        assert summary["sources"]["openaq"]["metrics"]["pm25"]["n_points"] == 2

    def test_empty_points_yields_empty_sources(self) -> None:
        summary = _summary(_anomaly(), [])
        assert summary["sources"] == {}
        assert all(present is False for present in summary["coverage"].values())
        assert summary["anomaly"]["metric"] == "pm25"

    def test_unexpected_source_enriched_but_absent_from_coverage(self) -> None:
        # purpleair is a planned-but-not-yet-live source; it must still be
        # gathered if rows exist, just not scored in the coverage card.
        points = [_dp(ts=T0, value=12.0, source="purpleair", entity="pa-1")]
        summary = _summary(_anomaly(), points)
        assert "purpleair" in summary["sources"]
        assert "purpleair" not in summary["coverage"]

    def test_metric_unit_captured(self) -> None:
        points = [_dp(ts=T0, value=44.0, metric="no2", unit="ppb")]
        summary = _summary(_anomaly(), points)
        assert summary["sources"]["openaq"]["metrics"]["no2"]["unit"] == "ppb"

    def test_counts_points_and_entities(self) -> None:
        points = [
            _dp(ts=T0 - timedelta(hours=1), value=10.0, entity="openaq-1"),
            _dp(ts=T0, value=20.0, entity="openaq-1"),
            _dp(ts=T0, value=30.0, entity="openaq-2"),
        ]
        summary = _summary(_anomaly(), points)
        pm25 = summary["sources"]["openaq"]["metrics"]["pm25"]
        assert pm25["n_points"] == 3
        assert pm25["n_entities"] == 2
        assert summary["sources"]["openaq"]["n_points"] == 3
        assert summary["sources"]["openaq"]["n_entities"] == 2

    def test_gfs_u_and_v_metrics_share_entity_id(self) -> None:
        cell = "gfs:29.76,-95.37"
        points = [
            _dp(ts=T0, value=3.2, source="noaa_gfs", metric="u_10m",
                entity=cell, unit="m/s"),
            _dp(ts=T0, value=4.1, source="noaa_gfs", metric="v_10m",
                entity=cell, unit="m/s"),
        ]
        metrics = _summary(_anomaly(), points)["sources"]["noaa_gfs"]["metrics"]
        u_entity = metrics["u_10m"]["entities"][0]["entity_id"]
        v_entity = metrics["v_10m"]["entities"][0]["entity_id"]
        assert u_entity == v_entity == cell


def _load(db_session, start, end, radius_km=50.0):
    """load_context_points anchored on the anomaly used across these tests."""
    return load_context_points(
        db_session,
        lat=HOUSTON_LAT,
        lon=HOUSTON_LON,
        window_start=start,
        window_end=end,
        radius_km=radius_km,
    )


class TestLoadContextPoints:
    @pytest.mark.asyncio
    async def test_loads_points_within_window_and_radius(self, db_session) -> None:
        start, end = context_window(T0)
        await _seed(db_session, [_dp(ts=T0, value=42.0)])
        rows = await _load(db_session, start, end)
        assert len(rows) == 1
        assert rows[0].value == pytest.approx(42.0)

    @pytest.mark.asyncio
    async def test_excludes_points_before_window(self, db_session) -> None:
        start, end = context_window(T0)
        await _seed(db_session, [_dp(ts=T0 - timedelta(hours=40), value=9.0)])
        assert await _load(db_session, start, end) == []

    @pytest.mark.asyncio
    async def test_excludes_points_after_window(self, db_session) -> None:
        start, end = context_window(T0)
        await _seed(db_session, [_dp(ts=T0 + timedelta(hours=40), value=9.0)])
        assert await _load(db_session, start, end) == []

    @pytest.mark.asyncio
    async def test_excludes_points_beyond_radius(self, db_session) -> None:
        start, end = context_window(T0)
        # ~111 km north of the anomaly.
        await _seed(db_session, [_dp(ts=T0, value=9.0, lat=HOUSTON_LAT + 1.0)])
        assert await _load(db_session, start, end) == []

    @pytest.mark.asyncio
    async def test_precise_radius_excludes_bbox_corner(self, db_session) -> None:
        # (+0.4, +0.4) deg sits inside the lat/lon box but ~59 km away — the
        # great-circle filter must still drop it.
        start, end = context_window(T0)
        await _seed(
            db_session,
            [_dp(ts=T0, value=9.0, lat=HOUSTON_LAT + 0.4, lon=HOUSTON_LON + 0.4)],
        )
        assert await _load(db_session, start, end) == []

    @pytest.mark.asyncio
    async def test_orders_by_timestamp(self, db_session) -> None:
        start, end = context_window(T0)
        await _seed(
            db_session,
            [
                _dp(ts=T0 + timedelta(hours=2), value=3.0),
                _dp(ts=T0 - timedelta(hours=2), value=1.0),
                _dp(ts=T0, value=2.0),
            ],
        )
        rows = await _load(db_session, start, end)
        assert [r.value for r in rows] == [1.0, 2.0, 3.0]


class TestEnrichAnomaly:
    @pytest.mark.asyncio
    async def test_returns_record_with_window_and_anomaly_id(
        self, db_session
    ) -> None:
        anomaly = await _persist_anomaly(db_session)
        await _seed(db_session, [_dp(ts=T0, value=150.0)])
        record = await enrich_anomaly(db_session, anomaly)
        assert isinstance(record, EnrichmentRecord)
        assert record.anomaly_id == anomaly.id
        start, end = context_window(T0)
        assert record.context_window_start == start
        assert record.context_window_end == end

    @pytest.mark.asyncio
    async def test_summary_contains_cross_source_data(self, db_session) -> None:
        anomaly = await _persist_anomaly(db_session)
        await _seed(
            db_session,
            [
                _dp(ts=T0, value=150.0, source="openaq", metric="pm25"),
                _dp(ts=T0, value=3.0, source="noaa_gfs", metric="u_10m",
                    entity="gfs:29.76,-95.37", unit="m/s"),
            ],
        )
        record = await enrich_anomaly(db_session, anomaly)
        summary = record.cross_source_summary_json
        assert summary["coverage"]["openaq"] is True
        assert summary["coverage"]["noaa_gfs"] is True
        assert "pm25" in summary["sources"]["openaq"]["metrics"]

    @pytest.mark.asyncio
    async def test_does_not_persist(self, db_session) -> None:
        anomaly = await _persist_anomaly(db_session)
        await _seed(db_session, [_dp(ts=T0, value=150.0)])
        await enrich_anomaly(db_session, anomaly)
        rows = (await db_session.execute(select(EnrichmentRecord))).scalars().all()
        assert rows == []


class TestPersistEnrichment:
    @pytest.mark.asyncio
    async def test_inserts_record(self, db_session) -> None:
        anomaly = await _persist_anomaly(db_session)
        record = await enrich_anomaly(db_session, anomaly)
        inserted = await persist_enrichment(db_session, record)
        assert inserted is True
        rows = (await db_session.execute(select(EnrichmentRecord))).scalars().all()
        assert len(rows) == 1
        assert rows[0].anomaly_id == anomaly.id

    @pytest.mark.asyncio
    async def test_idempotent_skips_existing_anomaly(self, db_session) -> None:
        anomaly = await _persist_anomaly(db_session)
        first = await persist_enrichment(
            db_session, await enrich_anomaly(db_session, anomaly)
        )
        second = await persist_enrichment(
            db_session, await enrich_anomaly(db_session, anomaly)
        )
        assert first is True
        assert second is False
        rows = (await db_session.execute(select(EnrichmentRecord))).scalars().all()
        assert len(rows) == 1


class TestEnrichPendingAnomalies:
    @pytest.mark.asyncio
    async def test_enriches_anomalies_without_records(self, db_session) -> None:
        await _persist_anomaly(db_session, timestamp=T0)
        await _persist_anomaly(db_session, timestamp=T0 + timedelta(days=1))
        await _seed(db_session, [_dp(ts=T0, value=120.0)])
        summary = await enrich_pending_anomalies(db_session)
        assert summary.n_pending == 2
        assert summary.n_persisted == 2
        rows = (await db_session.execute(select(EnrichmentRecord))).scalars().all()
        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_records_persist_failure_and_continues(
        self, db_session, monkeypatch
    ) -> None:
        # One anomaly's persist failure must be recorded on its line and let
        # the pass finish the rest, not abort and discard the whole summary.
        await _persist_anomaly(db_session, timestamp=T0)
        await _persist_anomaly(db_session, timestamp=T0 + timedelta(days=1))
        await _seed(db_session, [_dp(ts=T0, value=120.0)])

        import app.detection.enrichment as enr

        real = enr.persist_enrichment
        calls = {"n": 0}

        async def flaky(session, record):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("commit failed")
            return await real(session, record)

        monkeypatch.setattr(enr, "persist_enrichment", flaky)

        summary = await enrich_pending_anomalies(db_session)

        assert summary.n_pending == 2
        assert summary.n_persisted == 1
        assert len(summary.lines) == 2
        errored = [ln for ln in summary.lines if ln.error is not None]
        assert len(errored) == 1
        assert "RuntimeError" in errored[0].error

    @pytest.mark.asyncio
    async def test_skips_already_enriched(self, db_session) -> None:
        anomaly = await _persist_anomaly(db_session)
        await persist_enrichment(
            db_session, await enrich_anomaly(db_session, anomaly)
        )
        summary = await enrich_pending_anomalies(db_session)
        assert summary.n_pending == 0
        assert summary.n_persisted == 0

    @pytest.mark.asyncio
    async def test_dry_run_does_not_persist(self, db_session) -> None:
        await _persist_anomaly(db_session)
        summary = await enrich_pending_anomalies(db_session, dry_run=True)
        assert summary.n_pending == 1
        assert summary.n_persisted == 0
        rows = (await db_session.execute(select(EnrichmentRecord))).scalars().all()
        assert rows == []

    @pytest.mark.asyncio
    async def test_severity_filter(self, db_session) -> None:
        await _persist_anomaly(db_session, severity="severe", timestamp=T0)
        await _persist_anomaly(
            db_session, severity="minor", timestamp=T0 + timedelta(hours=1)
        )
        summary = await enrich_pending_anomalies(db_session, severity="severe")
        assert summary.n_pending == 1
        assert summary.lines[0].severity == "severe"

    @pytest.mark.asyncio
    async def test_limit_caps_count(self, db_session) -> None:
        for i in range(3):
            await _persist_anomaly(db_session, timestamp=T0 + timedelta(hours=i))
        summary = await enrich_pending_anomalies(db_session, limit=2)
        assert summary.n_pending == 2
        assert summary.n_persisted == 2

    @pytest.mark.asyncio
    async def test_idempotent_rerun(self, db_session) -> None:
        await _persist_anomaly(db_session)
        first = await enrich_pending_anomalies(db_session)
        second = await enrich_pending_anomalies(db_session)
        assert first.n_persisted == 1
        assert second.n_pending == 0
        assert second.n_persisted == 0

    @pytest.mark.asyncio
    async def test_format_summary_accepts_real_result(self, db_session) -> None:
        await _persist_anomaly(db_session)
        summary = await enrich_pending_anomalies(db_session)
        rendered = _format_summary(summary)
        assert isinstance(rendered, str)
        assert "Pending anomalies: 1" in rendered


class TestEnrichBatchesContext:
    @pytest.mark.asyncio
    async def test_co_located_anomalies_share_one_context_query(
        self, db_session, monkeypatch
    ) -> None:
        # Three anomalies in one spatiotemporal bucket must collapse to a single
        # context query, not one per anomaly (the O(pending) bbox-scan N+1).
        import app.detection.enrichment as enr

        await _persist_anomaly(db_session, timestamp=T0)
        await _persist_anomaly(db_session, timestamp=T0 + timedelta(hours=1))
        await _persist_anomaly(db_session, timestamp=T0 + timedelta(hours=2))
        await _seed(db_session, [_dp(ts=T0, value=120.0)])

        calls = {"n": 0}
        real = enr._query_box_window

        async def _counting(session, box, window_start, window_end):
            calls["n"] += 1
            return await real(session, box, window_start, window_end)

        monkeypatch.setattr(enr, "_query_box_window", _counting)

        summary = await enrich_pending_anomalies(db_session)

        assert summary.n_pending == 3
        assert summary.n_persisted == 3
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_batched_summaries_match_per_anomaly_path(self, db_session) -> None:
        # The batched context must produce summaries byte-identical to the
        # per-anomaly DB path, with each anomaly filtered to its own window.
        a1 = await _persist_anomaly(db_session, timestamp=T0)
        a2 = await _persist_anomaly(db_session, timestamp=T0 + timedelta(hours=3))
        await _seed(
            db_session,
            [
                _dp(ts=T0, value=120.0),
                _dp(ts=T0 + timedelta(hours=3), value=80.0),
            ],
        )
        ref1 = (await enrich_anomaly(db_session, a1)).cross_source_summary_json
        ref2 = (await enrich_anomaly(db_session, a2)).cross_source_summary_json
        assert ref1["sources"]["openaq"]["n_points"] > 0

        await enrich_pending_anomalies(db_session)

        stored = {
            r.anomaly_id: r.cross_source_summary_json
            for r in (
                await db_session.execute(select(EnrichmentRecord))
            ).scalars().all()
        }
        assert stored[a1.id] == ref1
        assert stored[a2.id] == ref2


class TestCLI:
    def test_parse_args_defaults(self) -> None:
        args = _parse_args([])
        assert args.severity is None
        assert args.limit is None
        assert args.dry_run is False

    def test_parse_args_severity(self) -> None:
        assert _parse_args(["--severity", "severe"]).severity == "severe"

    def test_parse_args_limit_is_int(self) -> None:
        assert _parse_args(["--limit", "25"]).limit == 25

    def test_parse_args_dry_run_flag(self) -> None:
        assert _parse_args(["--dry-run"]).dry_run is True

    def test_parse_args_rejects_unknown_flag(self) -> None:
        with pytest.raises(SystemExit):
            _parse_args(["--bogus"])

    def test_format_summary_handles_empty_result(self) -> None:
        rendered = _format_summary(EnrichmentSummary())
        assert isinstance(rendered, str)
        assert "Pending anomalies: 0" in rendered
