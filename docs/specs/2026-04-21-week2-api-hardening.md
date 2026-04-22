# Week 2 Data API Hardening

**Date**: 2026-04-21
**Phase**: Month 1 Week 2 Step 2.5
**Status**: Implemented in initial form on 2026-04-21

## Context

By the end of Week 2, AERIS has five live collector modules: EPA AirNow, OpenAQ, PurpleAir, OpenWeather, and NASA FIRMS. The raw data API already supported source queries and filters, but the documented source-status endpoint was `GET /api/data/sources` while the implementation exposed the list at `GET /api/data`.

This hardening slice makes the documented route real, preserves the legacy route, and adds route-level tests so future collector/schema changes do not silently break the API contract.

## Goals

1. Add `GET /api/data/sources` for source status metadata.
2. Preserve `GET /api/data` as a compatibility alias for the same response.
3. Verify `GET /api/data/{source}` returns the new `unit` and `source_entity_id` fields.
4. Verify metric, time, location-radius, limit, and offset behavior at the FastAPI route layer.
5. Keep full Postgres/Timescale integration tests deferred to Week 3 Step 3.6.

## API Contract

### `GET /api/data/sources`

Returns data source status rows sorted by source name:

```json
[
  {
    "name": "openweather",
    "source_type": "openweather",
    "status": "active",
    "last_collected_at": "2026-04-21T12:00:00Z",
    "error_count": 0
  }
]
```

### `GET /api/data`

Compatibility alias for `GET /api/data/sources`.

### `GET /api/data/{source}`

Returns paginated raw observations:

```json
{
  "items": [
    {
      "metric": "pm25",
      "value": 12.5,
      "unit": "ug/m3",
      "source": "openaq",
      "source_entity_id": "12345"
    }
  ],
  "total": 1,
  "limit": 100,
  "offset": 0
}
```

Supported query parameters:

| Parameter | Behavior |
|-----------|----------|
| `metric` | exact metric filter |
| `start` / `end` | inclusive timestamp range |
| `lat` / `lon` / `radius_km` | approximate bounding-box radius filter |
| `limit` | max rows, capped at 1000 |
| `offset` | pagination offset |

## Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Define `/data/sources` before `/data/{source}` | FastAPI/Starlette route order matters; the static route must be registered first. |
| D2 | Keep `/data` as an alias | Existing clients or smoke scripts may already use the old route. Removing it would create avoidable churn. |
| D3 | Test through ASGI transport | Route tests should exercise dependency injection, serialization, and URL parsing, not only helper functions. |
| D4 | Use SQLite route tests for Week 2 | Full TimescaleDB verification remains a separate integration hardening task in Week 3. |

## Files Touched

| File | Action |
|------|--------|
| `server/app/api/routes/data.py` | add `/data/sources`, keep legacy alias |
| `server/tests/unit/test_data_routes.py` | new route tests |
| `docs/specs/2026-04-05-month1-phase-plan.md` | update Step 2.5 implementation detail |
| `session_summary.md` | handoff update |

## Verification

- `venv/bin/pytest tests/unit/test_data_routes.py`
- `venv/bin/pytest`

## Commit Point

`test(api): add data route coverage and source listing endpoint`
