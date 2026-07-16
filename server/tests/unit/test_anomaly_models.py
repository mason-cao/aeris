import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.db.models import Anomaly, EnrichmentRecord


HOUSTON_LAT = 29.7604
HOUSTON_LON = -95.3698


def _make_anomaly(**overrides) -> Anomaly:
    defaults = dict(
        timestamp=datetime(2026, 5, 16, 14, 30, tzinfo=timezone.utc),
        lat=HOUSTON_LAT,
        lon=HOUSTON_LON,
        metric="pm25",
        source="openaq",
        value=87.4,
        expected_value=22.1,
        z_score=4.8,
        methods_triggered=["zscore", "stl"],
        severity="moderate",
    )
    defaults.update(overrides)
    return Anomaly(**defaults)


class TestAnomalyModel:
    @pytest.mark.asyncio
    async def test_persists_with_all_fields(self, db_session) -> None:
        anomaly = _make_anomaly()
        db_session.add(anomaly)
        await db_session.commit()

        loaded = (await db_session.execute(select(Anomaly))).scalar_one()

        assert isinstance(loaded.id, uuid.UUID)
        assert loaded.metric == "pm25"
        assert loaded.source == "openaq"
        assert loaded.value == pytest.approx(87.4)
        assert loaded.expected_value == pytest.approx(22.1)
        assert loaded.z_score == pytest.approx(4.8)
        assert loaded.severity == "moderate"
        assert loaded.detected_at is not None

    @pytest.mark.asyncio
    async def test_methods_triggered_roundtrips_as_list(self, db_session) -> None:
        anomaly = _make_anomaly(methods_triggered=["zscore", "stl", "isolation_forest"])
        db_session.add(anomaly)
        await db_session.commit()

        loaded = (await db_session.execute(select(Anomaly))).scalar_one()
        assert loaded.methods_triggered == ["zscore", "stl", "isolation_forest"]

    @pytest.mark.asyncio
    async def test_expected_value_and_z_score_nullable(self, db_session) -> None:
        anomaly = _make_anomaly(expected_value=None, z_score=None)
        db_session.add(anomaly)
        await db_session.commit()

        loaded = (await db_session.execute(select(Anomaly))).scalar_one()
        assert loaded.expected_value is None
        assert loaded.z_score is None

    @pytest.mark.asyncio
    async def test_b18_trigger_and_detector_provenance_roundtrips(
        self, db_session
    ) -> None:
        availability = {
            "zscore": {"ran": True, "skip_code": None, "detail": None},
            "stl": {"ran": True, "skip_code": None, "detail": None},
            "isolation_forest": {
                "ran": False,
                "skip_code": "missing_complete_gfs_aux",
                "detail": None,
            },
        }
        anomaly = _make_anomaly(
            source_entity_id="openaq-sensor-7",
            detector_availability_json=availability,
        )
        db_session.add(anomaly)
        await db_session.commit()

        loaded = (await db_session.execute(select(Anomaly))).scalar_one()

        assert loaded.source_entity_id == "openaq-sensor-7"
        assert loaded.detector_availability_json == availability


class TestEnrichmentRecordModel:
    @pytest.mark.asyncio
    async def test_persists_linked_to_anomaly(self, db_session) -> None:
        anomaly = _make_anomaly()
        db_session.add(anomaly)
        await db_session.flush()

        window_start = anomaly.timestamp - timedelta(hours=36)
        window_end = anomaly.timestamp + timedelta(hours=36)
        enrichment = EnrichmentRecord(
            anomaly_id=anomaly.id,
            context_window_start=window_start,
            context_window_end=window_end,
            cross_source_summary_json={
                "openaq": {"n_points": 12, "max_value": 92.0},
                "sentinel5p": {"n_points": 1, "column_density": 4.1e15},
            },
        )
        db_session.add(enrichment)
        await db_session.commit()

        loaded = (await db_session.execute(select(EnrichmentRecord))).scalar_one()
        assert loaded.anomaly_id == anomaly.id
        assert loaded.context_window_start == window_start
        assert loaded.context_window_end == window_end
        assert loaded.cross_source_summary_json["openaq"]["n_points"] == 12
        assert "sentinel5p" in loaded.cross_source_summary_json

    @pytest.mark.asyncio
    async def test_cascades_when_anomaly_deleted(self, db_session) -> None:
        anomaly = _make_anomaly()
        db_session.add(anomaly)
        await db_session.flush()

        enrichment = EnrichmentRecord(
            anomaly_id=anomaly.id,
            context_window_start=anomaly.timestamp - timedelta(hours=36),
            context_window_end=anomaly.timestamp + timedelta(hours=36),
            cross_source_summary_json={"openaq": {"n_points": 0}},
        )
        db_session.add(enrichment)
        await db_session.commit()

        await db_session.delete(anomaly)
        await db_session.commit()

        remaining = (await db_session.execute(select(EnrichmentRecord))).scalars().all()
        assert remaining == []
