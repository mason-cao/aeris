# OpenAQ Collector + Schema Hardening — Design

**Date**: 2026-04-13
**Phase**: Month 1 Week 2 Step 2.1 (first collector of Week 2)
**Status**: Implemented in initial form on 2026-04-14

## Context

AERIS Week 1 delivered the foundation (DB, BaseCollector, FastAPI) and one collector (EPA AirNow). The Week 1 end-to-end smoke test revealed two latent bugs:

1. **Hypertable constraint**: fixed in Week 1 smoke test by making `DataPoint`'s primary key composite `(id, timestamp)`.
2. **Semantic mismatch**: the EPA AirNow collector stores EPA's derived **AQI** value (0–500 integer), but the Month 1 plan's intent (Step 2.2 explicitly) is to store **raw physical units**. OpenAQ only returns raw concentration (µg/m³ for particulates, ppm/ppb for gases), which is incompatible with the current `value` column semantics.

This spec describes the OpenAQ collector AND the bundled schema + EPA fix that must ship together to make cross-source air-quality comparison work correctly. Bundling is deliberate: shipping OpenAQ without fixing EPA would leave the `data_points` table mixing AQI and µg/m³ in the same `value` column, which would poison anomaly detection in Month 2.

## Goals

1. Add an `OpenAQCollector` that pulls current air-quality readings from OpenAQ v3 for stations within 50km of Suwanee, GA.
2. Harmonize `value` semantics across all collectors: **raw concentration in native physical units** (never AQI).
3. Make the schema self-describing via a `unit` column so downstream consumers (detection, visualization, API) never have to hardcode assumptions.
4. Establish a uniform dedup key `(source, metric, source_entity_id, timestamp)` that all future Week 2/3 collectors can reuse.
5. Preserve the existing BaseCollector abstraction and the hour-cadence collection pattern established in Week 1.

## Non-Goals

- Historical backfill of OpenAQ data (time series built by hourly runs over time)
- Concurrent/parallel station fetches (sequential is fast enough for ~20 stations)
- Ingesting OpenAQ temperature / relative humidity (OpenWeather owns those metrics)
- Ingesting ultrafine / specialty params (PM1, PM4, CO2, NO, NOx, CH4, UFP) — scope drift
- AQI pipeline (frontend can compute AQI at render time from raw concentrations if needed)
- Parameter-ID lookup via `/v3/parameters` endpoint (station-walk approach doesn't need it)

## Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Store **raw concentration**, not AQI | OpenAQ returns concentration; EPA can give concentration via `Value`/`Unit` fields; AQI is a derived metric that can always be recomputed from concentration but not vice versa. Scientific integrity preserved. |
| D2 | Add `unit` column to `DataPoint` | Makes unit semantics explicit. Different metrics use different units (`ug/m3`, `ppm`, `ppb`, `degC`, `hPa`, `m/s`). Self-describing schema. |
| D3 | Add `source_entity_id` column to `DataPoint` (NOT NULL) | Gives every observation a stable per-source identifier for dedup. EPA: `ReportingArea`. OpenAQ: `sensor_id`. PurpleAir: `sensor_index`. NOAA grid cell: grid ID. Uniform dedup across all 9 collectors. |
| D4 | Unique constraint `(source, metric, source_entity_id, timestamp)` | Idempotent collector runs — re-ingesting the same observation is a no-op. Includes `timestamp` per TimescaleDB hypertable requirements. |
| D5 | Station-walk endpoint strategy | `/v3/locations?bbox=...` + in-code radius filter + `/v3/locations/{id}/sensors` per station. OpenAQ caps coordinate-radius queries at 25km, so bbox preserves AERIS's configured 50km target while keeping station metadata for Month 3 map layers. |
| D6 | Current snapshot only | Latest reading is embedded in the sensor response — no separate `/measurements` call needed. Hourly runs accumulate the time series naturally. Backfill deferred to Month 2 if detection needs more history. |
| D7 | Per-station error isolation | One station's 4xx/5xx logs a warning and skips that station. Remaining stations still collected. A bad station cannot fail the run. |
| D8 | Sequential fetches | No concurrency. Simplicity wins at 20 calls/hour; OpenAQ rate limits are generous. Bounded concurrency can be added in Week 4 hardening if needed. |
| D9 | Metrics: `pm25, pm10, ozone, no2, so2, co, bc` | Exact match to Month 1 phase plan Step 2.1 and the existing EPA collector metric names. Covers criteria pollutants + black carbon. Temperature/humidity delegated to OpenWeather. |
| D10 | Parameter name map as module-level dict | Same pattern as `epa_airnow.PARAMETER_MAP`. Explicit, greppable, testable. |
| D11 | Fix EPA AirNow to use `Value`/`Unit` in same PR | Without this, EPA rows remain AQI-based and pollute the new concentration-centric schema. Bundling is correct. |
| D12 | Drop + recreate `data_points` in dev on next run | The existing 2 smoke-test rows have AQI semantics and no `unit`/`source_entity_id` values. Fresh start; no real data lost. |

## Architecture

```
server/app/collectors/openaq.py
├─ OpenAQCollector(BaseCollector)
│    source_name = "openaq"
│    collect_interval_minutes = 60
│    API_BASE = "https://api.openaq.org/v3"
│
│  async def fetch() -> dict[str, Any]:
│      1. GET /v3/locations?bbox={min_lon},{min_lat},{max_lon},{max_lat}&limit=1000
│         Header: X-API-Key: <settings.openaq_api_key>
│         → list of station dicts (meta.found, results)
│      2. Filter locations back to the configured target radius.
│      3. For each location in results:
│           GET /v3/locations/{id}/sensors
│           Header: X-API-Key
│           → list of sensor dicts (latest reading embedded)
│         Handle per-station exceptions: log warning, continue
│      4. Return {"locations": [...], "sensors_by_location_id": {id: [sensors]}}
│
│  def normalize(raw) -> list[DataPointCreate]:
│      Walk every (location, sensor) pair.
│      For each sensor:
│        - Look up PARAMETER_MAP[sensor.parameter.name] → our metric name
│        - Skip unknown params
│        - Pull latest.value, latest.datetime.utc, latest.coordinates
│        - Skip if value is None or datetime unparseable
│        - source_entity_id = str(sensor.id)
│        - unit = sensor.parameter.units  (e.g. "µg/m³" → "ug/m3" after normalization)
│        - Emit DataPointCreate
│      Return list.

server/app/collectors/epa_airnow.py  (MODIFIED)
├─ EPAAirNowCollector.normalize():
│    - Read obs["Value"] instead of obs["AQI"]
│    - Read obs["Unit"] (e.g. "UG/M3", "PPB") → normalize to lowercase/canonical
│    - source_entity_id = obs["ReportingArea"] (stable per-station string)
│    - PARAMETER_MAP unchanged (names → canonical metric)
│    - Skip obs with Value is None (same guard, different field)

server/app/db/models.py  (MODIFIED)
├─ DataPoint:
│    id: UUID (PK part 1)
│    timestamp: DateTime (PK part 2 — TimescaleDB partition)
│    lat: float
│    lon: float
│    metric: str(64)
│    value: float                        ← raw concentration in native units
│    unit: str(32)                       ← NEW, e.g. "ug/m3", "ppm"
│    source: str(64)
│    source_entity_id: str(128)          ← NEW, NOT NULL
│    raw_json: JSON
│    collected_at: DateTime
│    __table_args__ = (
│        Index("ix_data_points_timestamp", "timestamp"),
│        Index("ix_data_points_source", "source"),
│        Index("ix_data_points_metric", "metric"),
│        Index("ix_data_points_source_metric_ts", "source", "metric", "timestamp"),
│        Index("ix_data_points_location", "lat", "lon"),
│        UniqueConstraint(
│            "source", "metric", "source_entity_id", "timestamp",
│            name="uq_data_points_dedup",
│        ),
│    )

server/app/collectors/base.py  (MODIFIED)
├─ _store() uses ON CONFLICT DO NOTHING on the new unique constraint
│  so re-ingest of the same observation is a silent no-op.
```

## Parameter Mapping

OpenAQ parameter names (as they appear in `/v3/locations/{id}/sensors` responses under `parameter.name`):

```python
# server/app/collectors/openaq.py
PARAMETER_MAP: dict[str, str] = {
    "pm25": "pm25",
    "pm10": "pm10",
    "o3":   "ozone",
    "no2":  "no2",
    "so2":  "so2",
    "co":   "co",
    "bc":   "bc",
}
```

OpenAQ uses lowercase names. We keep this as a map (not just an allowlist) to document accepted params and map OpenAQ `o3` to AERIS's canonical `ozone` metric.

Unit normalization (OpenAQ → canonical):

```python
UNIT_MAP: dict[str, str] = {
    "µg/m³": "ug/m3",
    "ug/m3": "ug/m3",
    "ppm":   "ppm",
    "ppb":   "ppb",
}
```

EPA AirNow unit normalization (EPA → canonical):

```python
# server/app/collectors/epa_airnow.py
EPA_UNIT_MAP: dict[str, str] = {
    "UG/M3": "ug/m3",
    "PPM":   "ppm",
    "PPB":   "ppb",
}
```

## Error Handling

- **Top-level API errors** (auth, network outage, total 5xx) propagate to `BaseCollector.collect()`'s existing retry loop (30s/60s/120s backoff). Unchanged.
- **Per-station errors** inside `fetch()` are caught locally: log a structured warning with the station ID, continue to the next station. The run succeeds as long as ≥1 station returned data.
- **Per-sensor errors** inside `normalize()` are caught: skip the sensor, log debug, continue. Matches existing EPA pattern for skipping bad observations.
- **Pagination sentinel**: if `meta.found > 1000` on the locations query, log WARNING that we're clipping at max limit. For Suwanee's 50km radius this should never fire; it's a safety net.

## Config Changes

```python
# server/app/config.py
class Settings(BaseSettings):
    ...
    openaq_api_key: str = ""  # NEW — required at implementation time
    ...
```

```bash
# server/.env.example  (and Mason's local .env)
OPENAQ_API_KEY=your_key_here
```

## Tests

**New unit tests** (`server/tests/unit/test_openaq.py`):

- `test_normalize_maps_all_parameters` — given fixture with all 7 params, verify output
- `test_normalize_skips_unknown_parameter` — e.g. `"pm4"` is not in map, skipped
- `test_normalize_skips_null_latest_value` — sensor with `latest: null`
- `test_normalize_skips_bad_timestamp`
- `test_normalize_preserves_raw_json`
- `test_normalize_sets_source_entity_id_to_sensor_id`
- `test_normalize_normalizes_units` — `µg/m³` in → `ug/m3` out
- `test_normalize_empty_response`
- `test_unit_map_covers_common_openaq_units`
- `test_parameter_map_covers_plan_metrics`

**Updated unit tests** (`server/tests/unit/test_epa_airnow.py`):

- Fixtures gain `Value` and `Unit` fields
- Assertions updated: `value` now a concentration (e.g. 12.3), not AQI (42)
- New test: `test_normalize_uses_reporting_area_as_entity_id`
- New test: `test_normalize_skips_missing_value` (replaces `test_normalize_skips_missing_aqi`)

**Integration concern (deferred to Week 3 Step 3.6)**:
These unit tests mock HTTP. A proper real-DB integration test for OpenAQ goes into the Week 3 hardening suite per the Month 1 phase plan Step 3.6.

## Files Touched

| File | Action |
|------|--------|
| `server/app/collectors/openaq.py` | **new** |
| `server/app/collectors/epa_airnow.py` | modify (use Value/Unit, add entity_id) |
| `server/app/collectors/base.py` | modify (ON CONFLICT DO NOTHING on _store) |
| `server/app/db/models.py` | modify (unit + source_entity_id cols + unique constraint) |
| `server/app/config.py` | modify (add openaq_api_key) |
| `server/.env.example` | modify (add OPENAQ_API_KEY) |
| `server/app/api/routes/data.py` | modify (`DataPointResponse` gains `unit` + `source_entity_id`) |
| `server/tests/unit/test_openaq.py` | **new** |
| `server/tests/unit/test_epa_airnow.py` | modify (fixture + assertions) |

No changes to `db/schema.py`, `db/session.py`, or `main.py`. Within `api/routes/`, only `data.py` changes — purely to expose the two new columns through `DataPointResponse`. All other route modules, the WebSocket layer, and the collector orchestrator (`run_all.py`) are untouched.

## Resolved / Remaining Implementation Checks

1. **Resolved: OpenAQ `/v3/locations` geospatial query** — OpenAQ documents radius in meters and caps it at 25,000m, so the implementation uses `bbox` and filters by configured radius in code.
2. **Resolved from docs, still needs live smoke test: OpenAQ sensor response shape** — `/v3/locations/{locations_id}/sensors` documents `latest.datetime.utc`, `latest.value`, and `latest.coordinates`.
3. **Remaining: EPA `Value` field coverage** — spec assumes EPA AirNow's `Value` field is populated alongside `AQI` for all current observations. If it's sometimes null, add a fallback that computes concentration from AQI via EPA breakpoint formulas (Month 2 concern).

## Implementation Notes (2026-04-14)

- OpenAQ v3 documents `coordinates` + `radius` in meters, but caps radius at 25,000m. AERIS targets 50km, so `OpenAQCollector.fetch()` uses the documented `bbox` query shape and then filters returned locations back to the configured target radius with a Haversine distance check.
- OpenAQ parameter `o3` is normalized to the existing AERIS canonical metric `ozone`, matching `EPAAirNowCollector`. This keeps cross-source ozone comparisons aligned.
- The implementation follows the documented `/v3/locations/{locations_id}/sensors` payload, where each sensor includes `parameter` metadata and an optional `latest` measurement with `datetime.utc`, `value`, and `coordinates`.

## Commit Points

This implementation should be committed as one coherent slice because the schema, EPA semantics, and OpenAQ collector must stay in sync:

1. `feat(collectors): add OpenAQ collector and raw-unit data schema`

## Out of Scope / Future Work

- Historical backfill via `/v3/sensors/{id}/measurements?datetimeFrom=...` — deferred; add if Month 2 detection needs more baseline.
- Bounded-concurrency station fetches — Week 4 orchestration hardening if needed.
- Parameter-IDs-based `/v3/parameters/{id}/latest` endpoint — alternative strategy, not needed unless station-walk hits rate/performance issues.
- Caching the station list between runs — premature optimization.
- OpenAQ temperature / relative humidity — owned by OpenWeather collector (Week 2 Step 2.3).
