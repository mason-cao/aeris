"""Eval-set freeze: event dedup, composite ranking, fixture output.

The freeze encodes the Month 2 inclusion rule ("top-50 summer anomalies by
composite severity") in reviewable code: one physical event contributes one
anomaly, ranking is (consensus count, |z|), and the fixture records the
criteria next to the ids.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.db.models import Anomaly, EnrichmentRecord
from app.eval.freeze import (
    FreezeResult,
    fixture_payload,
    freeze_eval_set,
    group_events,
)
from app.eval.harness import load_anomaly_set

T0 = datetime(2026, 7, 2, 12, 0, tzinfo=timezone.utc)
HOUSTON_LAT = 29.7604
HOUSTON_LON = -95.3698


def _anomaly(
    *,
    ts: datetime = T0,
    metric: str = "pm25",
    lat: float = HOUSTON_LAT,
    lon: float = HOUSTON_LON,
    methods: list[str] | None = None,
    z_score: float | None = 4.0,
    severity: str = "moderate",
) -> Anomaly:
    return Anomaly(
        id=uuid.uuid4(),
        timestamp=ts,
        lat=lat,
        lon=lon,
        metric=metric,
        source="openaq",
        value=100.0,
        expected_value=20.0,
        z_score=z_score,
        methods_triggered=methods or ["zscore", "stl"],
        severity=severity,
    )


class TestGroupEvents:
    def test_nearby_same_metric_anomalies_merge(self) -> None:
        events = group_events(
            [
                _anomaly(ts=T0),
                _anomaly(ts=T0 + timedelta(minutes=20), lat=HOUSTON_LAT + 0.02),
            ]
        )
        assert len(events) == 1
        assert len(events[0]) == 2

    def test_different_metrics_never_merge(self) -> None:
        events = group_events([_anomaly(metric="pm25"), _anomaly(metric="ozone")])
        assert len(events) == 2

    def test_distant_stations_do_not_merge(self) -> None:
        # ~28 km apart at the same instant: separate events.
        events = group_events(
            [_anomaly(), _anomaly(lat=HOUSTON_LAT + 0.25)]
        )
        assert len(events) == 2

    def test_far_apart_in_time_do_not_merge(self) -> None:
        events = group_events(
            [_anomaly(ts=T0), _anomaly(ts=T0 + timedelta(hours=2))]
        )
        assert len(events) == 2

    def test_single_linkage_chains_a_moving_event(self) -> None:
        # A->B and B->C are each within the merge window; A->C is not.
        # Chain linkage keeps the moving plume one event.
        events = group_events(
            [
                _anomaly(ts=T0),
                _anomaly(ts=T0 + timedelta(minutes=25)),
                _anomaly(ts=T0 + timedelta(minutes=50)),
            ]
        )
        assert len(events) == 1
        assert len(events[0]) == 3


class TestFreezeEvalSet:
    @pytest.mark.asyncio
    async def test_one_event_contributes_one_anomaly(self, db_session) -> None:
        # Five station-hours of one pm25 event must not fill the set.
        event_members = [
            _anomaly(ts=T0 + timedelta(minutes=10 * i)) for i in range(5)
        ]
        lone = _anomaly(metric="ozone", ts=T0 + timedelta(hours=8), z_score=3.2)
        for a in [*event_members, lone]:
            db_session.add(a)
        await db_session.commit()

        result = await freeze_eval_set(
            db_session,
            window_start=T0 - timedelta(days=1),
            window_end=T0 + timedelta(days=1),
            top_n=50,
        )
        assert result.n_anomalies == 6
        assert result.n_events == 2
        assert len(result.selected) == 2

    @pytest.mark.asyncio
    async def test_consensus_count_outranks_z_score(self, db_session) -> None:
        three_methods = _anomaly(
            metric="ozone",
            ts=T0 + timedelta(hours=8),
            methods=["zscore", "stl", "isolation_forest"],
            z_score=3.1,
            severity="severe",
        )
        big_z_two_methods = _anomaly(z_score=9.5)
        db_session.add(three_methods)
        db_session.add(big_z_two_methods)
        await db_session.commit()

        result = await freeze_eval_set(
            db_session,
            window_start=T0 - timedelta(days=1),
            window_end=T0 + timedelta(days=1),
            top_n=1,
        )
        assert result.selected[0].id == three_methods.id

    @pytest.mark.asyncio
    async def test_event_representative_is_its_strongest_member(
        self, db_session
    ) -> None:
        weak = _anomaly(ts=T0, z_score=3.0)
        strong = _anomaly(ts=T0 + timedelta(minutes=15), z_score=8.0)
        db_session.add(weak)
        db_session.add(strong)
        await db_session.commit()

        result = await freeze_eval_set(
            db_session,
            window_start=T0 - timedelta(days=1),
            window_end=T0 + timedelta(days=1),
            top_n=10,
        )
        assert [a.id for a in result.selected] == [strong.id]

    @pytest.mark.asyncio
    async def test_window_excludes_out_of_range_anomalies(self, db_session) -> None:
        may = _anomaly(ts=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
        july = _anomaly(ts=T0, metric="ozone")
        db_session.add(may)
        db_session.add(july)
        await db_session.commit()

        result = await freeze_eval_set(
            db_session,
            window_start=datetime(2026, 6, 1, tzinfo=timezone.utc),
            window_end=datetime(2026, 9, 1, tzinfo=timezone.utc),
            top_n=50,
        )
        assert [a.id for a in result.selected] == [july.id]

    @pytest.mark.asyncio
    async def test_reports_selected_anomalies_missing_enrichment(
        self, db_session
    ) -> None:
        enriched = _anomaly()
        bare = _anomaly(metric="ozone", ts=T0 + timedelta(hours=8))
        db_session.add(enriched)
        db_session.add(bare)
        await db_session.commit()
        db_session.add(
            EnrichmentRecord(
                anomaly_id=enriched.id,
                context_window_start=T0 - timedelta(hours=36),
                context_window_end=T0 + timedelta(hours=36),
                cross_source_summary_json={"schema_version": 1},
            )
        )
        await db_session.commit()

        result = await freeze_eval_set(
            db_session,
            window_start=T0 - timedelta(days=1),
            window_end=T0 + timedelta(days=1),
            top_n=50,
        )
        assert result.missing_enrichment == [bare.id]


class TestFixturePayload:
    def test_payload_loads_through_the_harness(self, tmp_path) -> None:
        selected = [_anomaly(), _anomaly(metric="ozone")]
        result = FreezeResult(
            window_start=datetime(2026, 6, 1, tzinfo=timezone.utc),
            window_end=datetime(2026, 9, 1, tzinfo=timezone.utc),
            top_n=50,
            n_anomalies=12,
            n_events=2,
            selected=selected,
            event_sizes={selected[0].id: 5, selected[1].id: 1},
            missing_enrichment=[],
        )
        payload = fixture_payload(result)
        path = tmp_path / "eval50.json"
        path.write_text(json.dumps(payload))

        assert load_anomaly_set(path) == [a.id for a in selected]
        assert payload["criteria"]["top"] == 50
        assert payload["n_events"] == 2
