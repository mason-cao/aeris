# NASA FIRMS Collector Design

**Date**: 2026-04-20
**Phase**: Month 1 Week 2 Step 2.4
**Status**: Implemented in initial form on 2026-04-20

## Context

NASA FIRMS provides near-real-time active fire detections from satellite sensors. For AERIS, fire detections are a cross-reference source: if PM2.5 rises locally and FIRMS reports recent fire activity upwind or within transport range, the LLM explanation pipeline can cite that evidence instead of guessing. Month 1 only needs current fire detections stored in the common schema; smoke transport modeling is a Month 2 enrichment concern.

The original Month 1 plan marked FIRMS as "no auth." Current FIRMS API documentation requires a free `MAP_KEY` for API access, so this collector adds a dedicated `FIRMS_MAP_KEY` setting.

## Goals

1. Add `NASAFIRMSCollector` for active fire detections around the target area.
2. Query a 100km bounding box around Suwanee to capture nearby and potentially upwind fires.
3. Pull multiple near-real-time sources: `VIIRS_NOAA20_NRT`, `VIIRS_SNPP_NRT`, and `MODIS_NRT`.
4. Normalize fire radiative power, confidence, and brightness into the shared `DataPoint` schema.
5. Preserve enough raw row context for Month 2 enrichment and fire-distance analysis.
6. Add unit tests for CSV parsing, timestamp handling, categorical confidence conversion, and fetch URL construction.

## API Shape

Endpoint shape documented by NASA FIRMS:

```text
GET https://firms.modaps.eosdis.nasa.gov/api/area/csv/[MAP_KEY]/[SOURCE]/[AREA_COORDINATES]/[DAY_RANGE]
```

Parameters:

| Param | Value |
|-------|-------|
| `MAP_KEY` | `settings.firms_map_key` |
| `SOURCE` | one of `VIIRS_NOAA20_NRT`, `VIIRS_SNPP_NRT`, `MODIS_NRT` |
| `AREA_COORDINATES` | `west,south,east,north` |
| `DAY_RANGE` | `1` |

The collector uses the global FIRMS endpoint rather than the US/Canada endpoint to keep behavior portable when AERIS later expands regions.

## Metrics

| FIRMS field | AERIS metric | Unit | Notes |
|-------------|--------------|------|-------|
| `frp` | `fire_radiative_power` | `MW` | Fire Radiative Power |
| `confidence` | `fire_confidence` | `percent` | MODIS numeric confidence is stored directly; VIIRS `l/n/h` maps to `33/66/100` |
| `bright_ti4` or `brightness` | `fire_brightness` | `K` | VIIRS uses `bright_ti4`; MODIS uses `brightness` |

Each FIRMS detection emits up to three rows, one per metric. `source_entity_id` is a deterministic detection ID derived from source, satellite, timestamp, and coordinates.

## Timestamp Handling

FIRMS rows provide:

- `acq_date`: `YYYY-MM-DD`
- `acq_time`: `HHMM`, often without leading zero

The collector zero-pads `acq_time` to four digits and stores the timestamp as UTC.

## Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Require `FIRMS_MAP_KEY` | Current FIRMS API docs require a free map key for area CSV requests. |
| D2 | Query 100km radius | Fires outside the 50km data area can still influence air quality through smoke transport. |
| D3 | Query three NRT sources | VIIRS catches smaller/hotter fires, MODIS provides continuity with a longer-running product. |
| D4 | Map VIIRS confidence categories to numeric percent-like values | The normalized schema stores numeric values. Raw category remains in `raw_json`. |
| D5 | Treat empty CSV as a successful zero-row response | No active fires near Suwanee is a normal state and should not mark the collector unhealthy. |
| D6 | Raise only if every source request fails | One satellite source can be unavailable without invalidating the entire run. |

## Files Touched

| File | Action |
|------|--------|
| `server/app/config.py` | add `firms_map_key` |
| `server/.env.example` | add `FIRMS_MAP_KEY` |
| `README.md` | document FIRMS map key |
| `docs/specs/2026-04-05-month1-phase-plan.md` | correct FIRMS auth/API key details |
| `server/app/collectors/nasa_firms.py` | new collector |
| `server/tests/unit/test_nasa_firms.py` | new tests |
| `session_summary.md` | handoff update |

## Verification

- `venv/bin/pytest`

## Commit Point

`feat(collectors): add NASA FIRMS fire collector`
