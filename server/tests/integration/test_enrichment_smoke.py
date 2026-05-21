"""Cross-source enrichment smoke tests.

Exercises the enrichment pipeline end-to-end against a realistic four-source
Houston scene — an afternoon Ship Channel PM2.5 event under stagnant,
low-PBL conditions. Where ``test_enrichment.py`` unit-tests each function in
isolation, this file feeds enrichment data shaped the way the real OpenAQ,
OpenWeather, NOAA GFS, and Sentinel-5P collectors emit it, and checks that
the EnrichmentRecord that comes out is coherent, complete, and queryable.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import delete, select

from app.db.models import Anomaly, DataPoint, EnrichmentRecord
from app.detection.enrichment import enrich_anomaly, enrich_pending_anomalies


# The anomaly: a severe PM2.5 spike near the Houston Ship Channel.
ANOMALY_TS = datetime(2026, 5, 18, 20, 0, tzinfo=timezone.utc)
ANOMALY_LAT = 29.7361
ANOMALY_LON = -95.2580

# Source entity locations. The "near"/"mid" stations sit inside the 50 km
# context radius; the "far" station and the stale reading are seeded so the
# enrichment can be checked for correctly dropping them.
NEAR_STATION = (29.7460, -95.2660)   # OpenAQ, ~1.3 km from the anomaly
MID_STATION = (29.8100, -95.1700)    # OpenAQ, ~12 km out
FAR_STATION = (30.4400, -95.2600)    # OpenAQ, ~78 km out — beyond the radius
OW_CENTER = (29.7604, -95.3698)      # OpenWeather grid + Sentinel-5P anchor
OW_NORTH = (29.9800, -95.3698)
OW_SOUTH = (29.5400, -95.3698)
GFS_CELL_A = (29.75, -95.25)         # nearest 0.25-deg GFS cell
GFS_CELL_B = (29.75, -95.50)


@pytest_asyncio.fixture(autouse=True)
async def _clean_tables(db_session):
    await db_session.execute(delete(EnrichmentRecord))
    await db_session.execute(delete(Anomaly))
    await db_session.execute(delete(DataPoint))
    await db_session.commit()
    yield


def _series(source, metric, entity, loc, unit, samples):
    """DataPoints for one (source, metric, entity) stream.

    ``samples`` is a list of ``(hour_offset_from_anomaly, value)`` pairs.
    """
    lat, lon = loc
    return [
        DataPoint(
            timestamp=ANOMALY_TS + timedelta(hours=hours),
            lat=lat,
            lon=lon,
            metric=metric,
            value=float(value),
            unit=unit,
            source=source,
            source_entity_id=entity,
            raw_json=None,
        )
        for hours, value in samples
    ]


async def _seed(db_session, points: list[DataPoint]) -> None:
    # Commit per row: SQLAlchemy's bulk-insert sentinel path does not
    # round-trip the (UUID, timestamp) composite PK cleanly on SQLite.
    for point in points:
        db_session.add(point)
        await db_session.commit()


async def _seed_houston_scene(db_session) -> Anomaly:
    """Seed a realistic four-source PM2.5 event and return the anomaly row."""
    points = (
        # OpenAQ near station: a stagnation buildup into an afternoon spike.
        _series("openaq", "pm25", "openaq-2117", NEAR_STATION, "ug/m3",
                [(-30, 13), (-18, 18), (-6, 42), (-2, 78), (0, 96),
                 (8, 55), (18, 21)])
        + _series("openaq", "no2", "openaq-2117", NEAR_STATION, "ppb",
                  [(-18, 16), (-6, 27), (0, 33), (12, 18)])
        # A stale reading on the same station, four days before the window.
        + _series("openaq", "pm25", "openaq-2117", NEAR_STATION, "ug/m3",
                  [(-96, 11)])
        # OpenAQ mid station.
        + _series("openaq", "pm25", "openaq-3043", MID_STATION, "ug/m3",
                  [(-24, 15), (-6, 40), (0, 58), (12, 24)])
        # OpenAQ far station — outside the 50 km radius.
        + _series("openaq", "pm25", "openaq-9981", FAR_STATION, "ug/m3",
                  [(-6, 20), (0, 22), (6, 19)])
        # OpenWeather 5-point grid (centre / north / south sampled here).
        + _series("openweather", "wind_speed", "grid:center", OW_CENTER, "m/s",
                  [(-24, 4.1), (-12, 2.3), (0, 1.1), (12, 3.0)])
        + _series("openweather", "wind_direction", "grid:center", OW_CENTER,
                  "degree", [(-12, 175), (0, 160), (12, 200)])
        + _series("openweather", "temperature", "grid:center", OW_CENTER,
                  "degC", [(-12, 30), (0, 33), (12, 29)])
        + _series("openweather", "wind_speed", "grid:north", OW_NORTH, "m/s",
                  [(-12, 2.6), (0, 1.5), (12, 3.2)])
        + _series("openweather", "wind_speed", "grid:south", OW_SOUTH, "m/s",
                  [(-12, 2.4), (0, 1.3), (12, 2.9)])
        # NOAA GFS 0.25-deg cells, 6-hourly analysis cycles.
        + _series("noaa_gfs", "u_10m", "gfs:29.75,-95.25", GFS_CELL_A, "m/s",
                  [(-14, -1.2), (-2, -2.1), (10, -0.8)])
        + _series("noaa_gfs", "v_10m", "gfs:29.75,-95.25", GFS_CELL_A, "m/s",
                  [(-14, 1.6), (-2, 2.6), (10, 1.1)])
        + _series("noaa_gfs", "pbl_height", "gfs:29.75,-95.25", GFS_CELL_A, "m",
                  [(-14, 700), (-2, 430), (10, 820)])
        + _series("noaa_gfs", "t_850", "gfs:29.75,-95.25", GFS_CELL_A, "degC",
                  [(-14, 17.2), (10, 16.9)])
        + _series("noaa_gfs", "u_10m", "gfs:29.75,-95.50", GFS_CELL_B, "m/s",
                  [(-2, -1.9), (10, -1.0)])
        + _series("noaa_gfs", "v_10m", "gfs:29.75,-95.50", GFS_CELL_B, "m/s",
                  [(-2, 2.4), (10, 1.4)])
        + _series("noaa_gfs", "pbl_height", "gfs:29.75,-95.50", GFS_CELL_B, "m",
                  [(-2, 460), (10, 760)])
        # Sentinel-5P granule means, anchored at the target centre.
        + _series("sentinel5p", "s5p_no2_column", "7e1c0d52-0517-a0b1",
                  OW_CENTER, "mol/m^2", [(-30, 6.2e-5)])
        + _series("sentinel5p", "s5p_no2_granule_available", "7e1c0d52-0517-a0b1",
                  OW_CENTER, "count", [(-30, 1.0)])
        + _series("sentinel5p", "s5p_no2_cloud_cover", "7e1c0d52-0517-a0b1",
                  OW_CENTER, "percent", [(-30, 22.0)])
        + _series("sentinel5p", "s5p_no2_column", "a1112233-0518-c2d3",
                  OW_CENTER, "mol/m^2", [(-4, 1.05e-4)])
        + _series("sentinel5p", "s5p_no2_granule_available", "a1112233-0518-c2d3",
                  OW_CENTER, "count", [(-4, 1.0)])
        + _series("sentinel5p", "s5p_no2_cloud_cover", "a1112233-0518-c2d3",
                  OW_CENTER, "percent", [(-4, 14.0)])
        + _series("sentinel5p", "s5p_no2_column", "b9988776-0519-e4f5",
                  OW_CENTER, "mol/m^2", [(20, 7.1e-5)])
        + _series("sentinel5p", "s5p_no2_granule_available", "b9988776-0519-e4f5",
                  OW_CENTER, "count", [(20, 1.0)])
        + _series("sentinel5p", "s5p_no2_cloud_cover", "b9988776-0519-e4f5",
                  OW_CENTER, "percent", [(20, 31.0)])
    )
    await _seed(db_session, points)

    anomaly = Anomaly(
        timestamp=ANOMALY_TS,
        lat=ANOMALY_LAT,
        lon=ANOMALY_LON,
        metric="pm25",
        source="openaq",
        value=96.0,
        expected_value=15.0,
        z_score=6.4,
        methods_triggered=["zscore", "stl", "isolation_forest"],
        severity="severe",
    )
    db_session.add(anomaly)
    await db_session.commit()
    return anomaly


class TestCrossSourceEnrichmentSmoke:
    @pytest.mark.asyncio
    async def test_enriches_realistic_four_source_scene(self, db_session) -> None:
        await _seed_houston_scene(db_session)
        summary = await enrich_pending_anomalies(db_session)
        assert summary.n_pending == 1
        assert summary.n_persisted == 1

        record = (
            await db_session.execute(select(EnrichmentRecord))
        ).scalar_one()
        payload = record.cross_source_summary_json

        assert payload["coverage"] == {
            "openaq": True,
            "openweather": True,
            "noaa_gfs": True,
            "sentinel5p": True,
        }
        assert payload["window"]["hours_before"] == 36.0
        assert payload["window"]["hours_after"] == 36.0
        assert "pm25" in payload["sources"]["openaq"]["metrics"]
        assert "wind_speed" in payload["sources"]["openweather"]["metrics"]
        assert "u_10m" in payload["sources"]["noaa_gfs"]["metrics"]
        assert "s5p_no2_column" in payload["sources"]["sentinel5p"]["metrics"]

        # Distances are computed from the anomaly: the near OpenAQ station
        # resolves to ~1-2 km, which a lat/lon mix-up would not.
        pm25_entities = payload["sources"]["openaq"]["metrics"]["pm25"]["entities"]
        near = next(e for e in pm25_entities if e["entity_id"] == "openaq-2117")
        assert near["distance_km"] < 5.0

    @pytest.mark.asyncio
    async def test_excludes_far_station_and_stale_points(self, db_session) -> None:
        anomaly = await _seed_houston_scene(db_session)
        record = await enrich_anomaly(db_session, anomaly)
        payload = record.cross_source_summary_json

        pm25 = payload["sources"]["openaq"]["metrics"]["pm25"]
        entity_ids = {e["entity_id"] for e in pm25["entities"]}
        assert "openaq-2117" in entity_ids   # near — kept
        assert "openaq-3043" in entity_ids   # mid — kept
        assert "openaq-9981" not in entity_ids  # ~78 km — dropped

        # The near station's four-days-stale reading is dropped from its
        # series; everything kept falls inside the context window.
        window_start = payload["window"]["start"]
        near = next(e for e in pm25["entities"] if e["entity_id"] == "openaq-2117")
        assert len(near["series"]) == 7
        assert all(row[0] >= window_start for row in near["series"])

    @pytest.mark.asyncio
    async def test_gfs_wind_components_share_cell_entities(
        self, db_session
    ) -> None:
        anomaly = await _seed_houston_scene(db_session)
        record = await enrich_anomaly(db_session, anomaly)
        gfs = record.cross_source_summary_json["sources"]["noaa_gfs"]["metrics"]

        # The corroboration scorer pairs u_10m with v_10m by cell to derive
        # wind vectors, so both metrics must key off identical entity ids.
        u_cells = {e["entity_id"] for e in gfs["u_10m"]["entities"]}
        v_cells = {e["entity_id"] for e in gfs["v_10m"]["entities"]}
        assert u_cells == v_cells
        assert u_cells == {"gfs:29.75,-95.25", "gfs:29.75,-95.50"}

    @pytest.mark.asyncio
    async def test_sentinel5p_granule_mean_present_with_column(
        self, db_session
    ) -> None:
        anomaly = await _seed_houston_scene(db_session)
        record = await enrich_anomaly(db_session, anomaly)
        s5p = record.cross_source_summary_json["sources"]["sentinel5p"]

        column = s5p["metrics"]["s5p_no2_column"]
        assert column["unit"] == "mol/m^2"
        assert column["n_entities"] == 3  # three granules in the window
        # Granule means are all reported at the Houston target anchor.
        for entity in column["entities"]:
            assert entity["lat"] == pytest.approx(OW_CENTER[0])
            assert entity["lon"] == pytest.approx(OW_CENTER[1])

    @pytest.mark.asyncio
    async def test_anomaly_metric_history_captured(self, db_session) -> None:
        anomaly = await _seed_houston_scene(db_session)
        record = await enrich_anomaly(db_session, anomaly)
        pm25 = record.cross_source_summary_json["sources"]["openaq"]["metrics"][
            "pm25"
        ]
        # The 72 h history of the anomaly's own metric is captured, and the
        # observation nearest the anomaly in time is the spike itself.
        assert pm25["value_range"]["max"] == pytest.approx(96.0)
        assert pm25["nearest_in_time"]["dt_minutes"] == 0.0
        assert pm25["nearest_in_time"]["v"] == pytest.approx(96.0)
        assert pm25["nearest_in_time"]["entity_id"] == "openaq-2117"

    @pytest.mark.asyncio
    async def test_summary_json_round_trips_through_database(
        self, db_session
    ) -> None:
        anomaly = await _seed_houston_scene(db_session)
        await enrich_pending_anomalies(db_session)

        record = (
            await db_session.execute(select(EnrichmentRecord))
        ).scalar_one()
        # refresh() re-reads the row so the JSON comes back out of the DB
        # column rather than the in-memory object the write left behind.
        await db_session.refresh(record)
        payload = record.cross_source_summary_json

        assert json.dumps(payload)  # the nested payload is serializable
        assert payload["schema_version"] == 1
        assert payload["anomaly"]["id"] == str(anomaly.id)
        assert payload["anomaly"]["severity"] == "severe"

    @pytest.mark.asyncio
    async def test_enrich_pending_is_idempotent(self, db_session) -> None:
        await _seed_houston_scene(db_session)
        first = await enrich_pending_anomalies(db_session)
        second = await enrich_pending_anomalies(db_session)

        assert first.n_persisted == 1
        assert second.n_pending == 0
        assert second.n_persisted == 0
        records = (
            await db_session.execute(select(EnrichmentRecord))
        ).scalars().all()
        assert len(records) == 1
