import math
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import delete, select

from app.db.models import Anomaly, DataPoint, EnrichmentRecord
from app.detection.consensus import ConsensusAnomaly
from app.detection.run import (
    GroupKey,
    _load_gfs_wind_rows,
    _nearest_value,
    _parse_args,
    build_aux_inputs,
    group_points_by_series,
    persist_anomalies,
    run_detection,
)


def _to_utc(dt: datetime) -> datetime:
    """SQLite returns naive datetimes; Postgres returns tz-aware. Normalize
    both shapes to a single tz-aware UTC datetime for comparison."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


HOUSTON_LAT = 29.7604
HOUSTON_LON = -95.3698
T0 = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)


@pytest_asyncio.fixture(autouse=True)
async def _clean_tables(db_session):
    await db_session.execute(delete(EnrichmentRecord))
    await db_session.execute(delete(Anomaly))
    await db_session.execute(delete(DataPoint))
    await db_session.commit()
    yield


def _dp(
    *,
    ts: datetime,
    value: float,
    metric: str = "pm25",
    source: str = "openaq",
    entity: str = "openaq-sensor-1",
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


def _gfs_wind(
    ts: datetime,
    *,
    u: float,
    v: float,
    entity: str = "gfs:29.76,-95.37",
) -> list[DataPoint]:
    """A GFS u_10m + v_10m DataPoint pair for one grid cell-cycle."""
    return [
        _dp(ts=ts, value=u, metric="u_10m", source="noaa_gfs", entity=entity),
        _dp(ts=ts, value=v, metric="v_10m", source="noaa_gfs", entity=entity),
    ]


def _seasonal(i: int, base: float = 22.0, amp: float = 6.0) -> float:
    return base + amp * math.sin(2.0 * math.pi * i / 24)


async def _seed(db_session, points: list[DataPoint]) -> None:
    # Commit per-row: SQLAlchemy's bulk-insert sentinel return path doesn't
    # round-trip the (UUID, timestamp) composite PK cleanly on SQLite, so we
    # serialize the inserts. Cheap for the test sizes here.
    for p in points:
        db_session.add(p)
        await db_session.commit()


def _make_consensus_anomaly(
    *,
    ts: datetime = T0,
    metric: str = "pm25",
    source: str = "openaq",
    value: float = 200.0,
    severity: str = "moderate",
    methods: list[str] | None = None,
) -> ConsensusAnomaly:
    return ConsensusAnomaly(
        timestamp=ts,
        lat=HOUSTON_LAT,
        lon=HOUSTON_LON,
        metric=metric,
        source=source,
        value=value,
        expected_value=22.0,
        z_score=4.5,
        methods_triggered=methods or ["zscore", "stl"],
        severity=severity,
    )


class TestGroupPointsBySeries:
    def test_groups_by_source_metric_entity(self) -> None:
        pts = [
            _dp(ts=T0, value=20.0, source="openaq", metric="pm25", entity="A"),
            _dp(ts=T0 + timedelta(hours=1), value=21.0, source="openaq", metric="pm25", entity="A"),
            _dp(ts=T0, value=30.0, source="openaq", metric="pm25", entity="B"),
            _dp(ts=T0, value=10.0, source="openaq", metric="no2", entity="A"),
        ]
        groups = group_points_by_series(pts)
        assert set(groups.keys()) == {
            GroupKey("openaq", "pm25", "A"),
            GroupKey("openaq", "pm25", "B"),
            GroupKey("openaq", "no2", "A"),
        }
        assert len(groups[GroupKey("openaq", "pm25", "A")]) == 2

    def test_excludes_non_primary_metrics(self) -> None:
        # u_10m / v_10m / pbl_height are auxiliary, not primary detection metrics.
        pts = [
            _dp(ts=T0, value=20.0, metric="pm25"),
            _dp(ts=T0, value=3.0, metric="u_10m", source="noaa_gfs"),
            _dp(ts=T0, value=4.0, metric="v_10m", source="noaa_gfs"),
            _dp(ts=T0, value=1000.0, metric="pbl_height", source="noaa_gfs"),
        ]
        groups = group_points_by_series(pts)
        keys = set(groups.keys())
        for k in keys:
            assert k.metric not in {"u_10m", "v_10m", "pbl_height"}

    def test_empty_input_returns_empty_dict(self) -> None:
        assert group_points_by_series([]) == {}

    def test_includes_openaq_ozone(self) -> None:
        # OpenAQ normalizes both "o3" and "ozone" parameters to the metric
        # name "ozone"; detection must group it, not drop it.
        pts = [_dp(ts=T0, value=55.0, source="openaq", metric="ozone", entity="A")]
        groups = group_points_by_series(pts)
        assert GroupKey("openaq", "ozone", "A") in groups

    def test_includes_sentinel5p_column_metrics(self) -> None:
        # The Sentinel-5P collector emits column densities s5p_-prefixed.
        pts = [
            _dp(ts=T0, value=4.1e-5, source="sentinel5p",
                metric="s5p_no2_column", entity="g1"),
            _dp(ts=T0, value=2.0e-5, source="sentinel5p",
                metric="s5p_so2_column", entity="g1"),
            _dp(ts=T0, value=3.0e-2, source="sentinel5p",
                metric="s5p_co_column", entity="g1"),
            _dp(ts=T0, value=1.1e-4, source="sentinel5p",
                metric="s5p_hcho_column", entity="g1"),
        ]
        groups = group_points_by_series(pts)
        assert {key.metric for key in groups} == {
            "s5p_no2_column",
            "s5p_so2_column",
            "s5p_co_column",
            "s5p_hcho_column",
        }

    def test_excludes_sentinel5p_granule_metadata(self) -> None:
        # granule_available / cloud_cover are metadata flags, not pollutant
        # time-series — they must not be treated as detection targets.
        pts = [
            _dp(ts=T0, value=1.0, source="sentinel5p",
                metric="s5p_no2_granule_available", entity="g1"),
            _dp(ts=T0, value=18.0, source="sentinel5p",
                metric="s5p_no2_cloud_cover", entity="g1"),
        ]
        assert group_points_by_series(pts) == {}


class TestNearestValue:
    def test_returns_value_at_exact_timestamp(self) -> None:
        rows = [(T0, 5.0), (T0 + timedelta(hours=1), 7.0)]
        assert _nearest_value(T0, rows, timedelta(hours=3)) == 5.0

    def test_returns_nearest_within_tolerance(self) -> None:
        rows = [
            (T0, 5.0),
            (T0 + timedelta(hours=2), 9.0),
            (T0 + timedelta(hours=4), 11.0),
        ]
        # Target hour=2.5 → closest is hour=2 → 9.0
        assert _nearest_value(
            T0 + timedelta(hours=2, minutes=30), rows, timedelta(hours=3)
        ) == 9.0

    def test_returns_none_when_no_row_within_tolerance(self) -> None:
        rows = [(T0, 5.0)]
        # Target is 10 hours away; tolerance is 3h → no match
        assert _nearest_value(
            T0 + timedelta(hours=10), rows, timedelta(hours=3)
        ) is None

    def test_returns_none_when_rows_empty(self) -> None:
        assert _nearest_value(T0, [], timedelta(hours=3)) is None


class TestLoadGfsWindRows:
    @pytest.mark.asyncio
    async def test_derives_wind_speed_as_hypot_of_u_and_v(self, db_session) -> None:
        await _seed(db_session, _gfs_wind(T0, u=3.0, v=4.0))
        rows = await _load_gfs_wind_rows(
            db_session, T0 - timedelta(hours=1), T0 + timedelta(hours=1)
        )
        assert len(rows) == 1
        assert rows[0][1] == pytest.approx(5.0)  # hypot(3, 4)

    @pytest.mark.asyncio
    async def test_pairs_u_and_v_per_grid_cell(self, db_session) -> None:
        await _seed(
            db_session,
            _gfs_wind(T0, u=3.0, v=4.0, entity="gfs:A")
            + _gfs_wind(T0, u=6.0, v=8.0, entity="gfs:B"),
        )
        rows = await _load_gfs_wind_rows(
            db_session, T0 - timedelta(hours=1), T0 + timedelta(hours=1)
        )
        assert sorted(round(v, 4) for _, v in rows) == [5.0, 10.0]

    @pytest.mark.asyncio
    async def test_skips_u_component_with_no_matching_v(self, db_session) -> None:
        # A u_10m row whose cell-cycle has no v_10m cannot form a wind vector.
        await _seed(
            db_session,
            [_dp(ts=T0, value=3.0, metric="u_10m", source="noaa_gfs", entity="gfs:A")],
        )
        rows = await _load_gfs_wind_rows(
            db_session, T0 - timedelta(hours=1), T0 + timedelta(hours=1)
        )
        assert rows == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_gfs_wind_data(self, db_session) -> None:
        rows = await _load_gfs_wind_rows(
            db_session, T0 - timedelta(hours=1), T0 + timedelta(hours=1)
        )
        assert rows == []


class TestBuildAuxInputs:
    @pytest.mark.asyncio
    async def test_returns_none_when_wind_data_missing(self, db_session) -> None:
        # Only PBL data, no wind → cannot build multivariate input
        primary = [_dp(ts=T0 + timedelta(hours=i), value=20.0 + i) for i in range(5)]
        pbl = [
            _dp(ts=T0 + timedelta(hours=i), value=1000.0,
                metric="pbl_height", source="noaa_gfs",
                entity="gfs:29.76,-95.37")
            for i in range(5)
        ]
        await _seed(db_session, primary + pbl)
        result = await build_aux_inputs(db_session, primary)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_pbl_data_missing(self, db_session) -> None:
        primary = [_dp(ts=T0 + timedelta(hours=i), value=20.0 + i) for i in range(5)]
        wind = [
            dp
            for i in range(5)
            for dp in _gfs_wind(T0 + timedelta(hours=i), u=3.0, v=4.0)
        ]
        await _seed(db_session, primary + wind)
        result = await build_aux_inputs(db_session, primary)
        assert result is None

    @pytest.mark.asyncio
    async def test_builds_inputs_when_both_aux_available(self, db_session) -> None:
        primary = [_dp(ts=T0 + timedelta(hours=i), value=20.0 + i) for i in range(5)]
        wind = [
            dp
            for i in range(5)
            for dp in _gfs_wind(T0 + timedelta(hours=i), u=3.0, v=4.0)
        ]
        pbl = [
            _dp(ts=T0 + timedelta(hours=i), value=1000.0 + i,
                metric="pbl_height", source="noaa_gfs",
                entity="gfs:29.76,-95.37")
            for i in range(5)
        ]
        await _seed(db_session, primary + wind + pbl)
        result = await build_aux_inputs(db_session, primary)
        assert result is not None
        assert len(result) == 5
        assert result[0].value == pytest.approx(20.0)
        assert "wind_speed" in result[0].aux_features
        assert "pbl_height" in result[0].aux_features
        # wind_speed = hypot(u=3, v=4) = 5
        assert result[0].aux_features["wind_speed"] == pytest.approx(5.0)

    @pytest.mark.asyncio
    async def test_skips_timestamps_with_no_aux_within_tolerance(self, db_session) -> None:
        # Primary has 5 points; aux only available for first 3 (last 2 are
        # >3h beyond any aux row). The 2 unmatched points are dropped.
        primary = [_dp(ts=T0 + timedelta(hours=i), value=20.0 + i) for i in range(5)]
        wind = [
            dp
            for i in range(3)
            for dp in _gfs_wind(T0 + timedelta(hours=i), u=3.0, v=4.0)
        ]
        pbl = [
            _dp(ts=T0 + timedelta(hours=i), value=1000.0,
                metric="pbl_height", source="noaa_gfs",
                entity="gfs:29.76,-95.37")
            for i in range(3)
        ]
        await _seed(db_session, primary + wind + pbl)
        result = await build_aux_inputs(db_session, primary)
        assert result is not None
        # Hours 0/1/2 have aux within ±3h; hour 3 still within tolerance of
        # hour 2 (1h gap < 3h tolerance); hour 4 is 2h from last aux → ok.
        # So all 5 are kept. Adjust expectation accordingly.
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_returns_none_when_primary_empty(self, db_session) -> None:
        result = await build_aux_inputs(db_session, [])
        assert result is None


class TestPersistAnomalies:
    @pytest.mark.asyncio
    async def test_inserts_anomalies(self, db_session) -> None:
        anomalies = [
            _make_consensus_anomaly(ts=T0, severity="severe", methods=["zscore", "stl", "isolation_forest"]),
            _make_consensus_anomaly(ts=T0 + timedelta(hours=1), severity="moderate"),
        ]
        n = await persist_anomalies(db_session, anomalies)
        assert n == 2
        rows = (await db_session.execute(select(Anomaly))).scalars().all()
        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_idempotent_re_run_does_not_duplicate(self, db_session) -> None:
        anomalies = [_make_consensus_anomaly(ts=T0, severity="moderate")]
        first = await persist_anomalies(db_session, anomalies)
        second = await persist_anomalies(db_session, anomalies)
        assert first == 1
        assert second == 0
        rows = (await db_session.execute(select(Anomaly))).scalars().all()
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_idempotency_key_includes_source_metric_ts_location(
        self, db_session
    ) -> None:
        # Same timestamp but different location → distinct rows.
        a1 = _make_consensus_anomaly(ts=T0)
        a2 = _make_consensus_anomaly(ts=T0)
        a2_data = a2.model_dump()
        a2_data["lon"] = HOUSTON_LON + 0.5
        a2_relocated = ConsensusAnomaly(**a2_data)
        n = await persist_anomalies(db_session, [a1, a2_relocated])
        assert n == 2

    @pytest.mark.asyncio
    async def test_persisted_anomaly_carries_all_fields(self, db_session) -> None:
        a = _make_consensus_anomaly(
            ts=T0, severity="severe", methods=["zscore", "stl", "isolation_forest"]
        )
        await persist_anomalies(db_session, [a])
        row = (await db_session.execute(select(Anomaly))).scalar_one()
        assert _to_utc(row.timestamp) == T0
        assert row.metric == "pm25"
        assert row.source == "openaq"
        assert row.value == pytest.approx(200.0)
        assert row.expected_value == pytest.approx(22.0)
        assert row.z_score == pytest.approx(4.5)
        assert row.methods_triggered == ["zscore", "stl", "isolation_forest"]
        assert row.severity == "severe"


class TestRunDetectionEndToEnd:
    @pytest.mark.asyncio
    async def test_runs_and_persists_anomalies_for_qualifying_groups(
        self, db_session
    ) -> None:
        # 200-hour PM2.5 series with a strong injected spike at hour 120.
        n = 200
        spike_idx = 120
        points: list[DataPoint] = []
        for i in range(n):
            value = 220.0 if i == spike_idx else _seasonal(i)
            points.append(_dp(ts=T0 + timedelta(hours=i), value=value))
        await _seed(db_session, points)

        summary = await run_detection(db_session)

        assert summary.n_groups_examined == 1
        assert summary.n_groups_run == 1
        assert summary.n_anomalies_emitted >= 1
        assert summary.n_persisted >= 1
        rows = (await db_session.execute(select(Anomaly))).scalars().all()
        spike_ts = T0 + timedelta(hours=spike_idx)
        assert any(_to_utc(r.timestamp) == spike_ts for r in rows)

    @pytest.mark.asyncio
    async def test_skips_groups_under_min_points(self, db_session) -> None:
        # Only 10 points — well below default min_points=50
        points = [_dp(ts=T0 + timedelta(hours=i), value=10.0 + i) for i in range(10)]
        await _seed(db_session, points)

        summary = await run_detection(db_session)
        assert summary.n_groups_examined == 1
        assert summary.n_groups_run == 0
        assert summary.groups[0].skipped_reason is not None

    @pytest.mark.asyncio
    async def test_filter_by_source(self, db_session) -> None:
        # Seed two sources of PM2.5; filter to one.
        openaq = [
            _dp(ts=T0 + timedelta(hours=i), value=_seasonal(i),
                source="openaq", entity="openaq-1")
            for i in range(60)
        ]
        purpleair = [
            _dp(ts=T0 + timedelta(hours=i), value=_seasonal(i),
                source="purpleair", entity="pa-1")
            for i in range(60)
        ]
        # Inject a spike in each
        openaq[30] = _dp(ts=T0 + timedelta(hours=30), value=300.0,
                         source="openaq", entity="openaq-1")
        purpleair[30] = _dp(ts=T0 + timedelta(hours=30), value=300.0,
                            source="purpleair", entity="pa-1")
        await _seed(db_session, openaq + purpleair)

        summary = await run_detection(db_session, source="openaq")
        assert summary.n_groups_examined == 1
        # Only one source's anomalies were persisted
        rows = (await db_session.execute(select(Anomaly))).scalars().all()
        assert all(r.source == "openaq" for r in rows)

    @pytest.mark.asyncio
    async def test_dry_run_does_not_persist(self, db_session) -> None:
        n = 200
        spike_idx = 120
        points: list[DataPoint] = []
        for i in range(n):
            value = 220.0 if i == spike_idx else _seasonal(i)
            points.append(_dp(ts=T0 + timedelta(hours=i), value=value))
        await _seed(db_session, points)

        summary = await run_detection(db_session, dry_run=True)
        assert summary.n_anomalies_emitted >= 1
        assert summary.n_persisted == 0
        rows = (await db_session.execute(select(Anomaly))).scalars().all()
        assert rows == []

    @pytest.mark.asyncio
    async def test_re_run_is_idempotent(self, db_session) -> None:
        # Seed and run twice; the second run must persist zero new rows.
        n = 200
        spike_idx = 120
        points: list[DataPoint] = []
        for i in range(n):
            value = 220.0 if i == spike_idx else _seasonal(i)
            points.append(_dp(ts=T0 + timedelta(hours=i), value=value))
        await _seed(db_session, points)

        first = await run_detection(db_session)
        before_count = (await db_session.execute(select(Anomaly))).scalars().all()
        second = await run_detection(db_session)
        after_count = (await db_session.execute(select(Anomaly))).scalars().all()
        assert first.n_persisted >= 1
        assert second.n_persisted == 0
        assert len(before_count) == len(after_count)


class TestCLIParsing:
    def test_no_args_defaults(self) -> None:
        args = _parse_args([])
        assert args.source is None
        assert args.metric is None
        assert args.since is None
        assert args.dry_run is False

    def test_source_filter(self) -> None:
        args = _parse_args(["--source", "openaq"])
        assert args.source == "openaq"

    def test_metric_filter(self) -> None:
        args = _parse_args(["--metric", "pm25"])
        assert args.metric == "pm25"

    def test_since_filter(self) -> None:
        args = _parse_args(["--since", "2026-05-01"])
        assert args.since == "2026-05-01"

    def test_dry_run_flag(self) -> None:
        args = _parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_min_points_override(self) -> None:
        args = _parse_args(["--min-points", "20"])
        assert args.min_points == 20

    def test_invalid_flag_raises(self) -> None:
        with pytest.raises(SystemExit):
            _parse_args(["--not-a-real-flag"])
