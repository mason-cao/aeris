# PurpleAir Collector Design

**Date**: 2026-04-22
**Phase**: Month 1 Week 2 Step 2.2
**Status**: Implemented in initial form on 2026-04-22

## Context

PurpleAir is AERIS's hyperlocal particulate source. EPA AirNow and OpenAQ provide official/government-grade monitoring, while PurpleAir adds dense neighborhood-level PM measurements from low-cost sensors. These readings are valuable for localized anomaly detection, but they are not equivalent to regulatory monitors: PM values are raw sensor estimates, and temperature/humidity are measured inside the sensor housing.

## Goals

1. Add `PurpleAirCollector` for current outdoor sensors in the configured target area.
2. Store raw PurpleAir values with explicit units using the normalized `DataPoint` schema.
3. Preserve each sensor's `sensor_index` as `source_entity_id` for deduplication.
4. Keep corrections/calibration out of Month 1. Store raw values; correction factors belong in Month 2 detection/enrichment.
5. Add unit tests for fetch parameter construction, response normalization, timestamp handling, and bad row skipping.

## API Shape

Endpoint:

```text
GET https://api.purpleair.com/v1/sensors
Header: X-API-Key: <settings.purpleair_api_key>
```

Query parameters:

| Param | Value |
|-------|-------|
| `fields` | `name,last_seen,latitude,longitude,location_type,private,pm2.5_atm,pm10.0_atm,humidity,temperature,confidence,channel_flags` |
| `location_type` | `0` (outside sensors only) |
| `max_age` | `7200` seconds |
| `nwlng` / `nwlat` | northwest corner of target bounding box |
| `selng` / `selat` | southeast corner of target bounding box |

PurpleAir's response is columnar:

```json
{
  "fields": ["sensor_index", "name", "last_seen", "..."],
  "data": [
    [12345, "sensor name", 1776801600, "..."]
  ]
}
```

Normalize by zipping each `data` row against `fields`. `sensor_index` is included by default and should not be requested in the `fields` query parameter.

## Metrics

| PurpleAir field | AERIS metric | Unit | Notes |
|-----------------|--------------|------|-------|
| `pm2.5_atm` | `pm25` | `ug/m3` | raw atmospheric PM2.5 estimate |
| `pm10.0_atm` | `pm10` | `ug/m3` | raw atmospheric PM10 estimate |
| `humidity` | `humidity` | `percent` | inside sensor housing, not ambient |
| `temperature` | `temperature` | `degF` | inside sensor housing, not ambient |

`last_seen` is the measurement timestamp for the current sensor values. It is a Unix timestamp and is stored as UTC.

## Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Use `pm2.5_atm` and `pm10.0_atm` | Matches PurpleAir's documented atmospheric mass concentration fields and keeps Month 1 raw. |
| D2 | Query outdoor sensors only (`location_type=0`) | AERIS focuses on environmental exposure around the target area; indoor sensors would distort neighborhood maps. |
| D3 | Keep `temperature` in Fahrenheit | PurpleAir reports Fahrenheit; the `unit` column makes this explicit. Conversion can happen later if needed. |
| D4 | Use `max_age=7200` | PurpleAir sensors report frequently; a two-hour freshness window prevents stale sensors from being normalized as current data. |
| D5 | Preserve full row JSON | Quality fields such as `confidence` and `channel_flags` may be useful for Month 2 filtering. |
| D6 | Use a shared geospatial helper | OpenAQ, PurpleAir, NASA FIRMS, and future collectors all need target bounding boxes and radius math. |

## Files Touched

| File | Action |
|------|--------|
| `server/app/collectors/geo.py` | new shared bounding box / distance helpers |
| `server/app/collectors/purpleair.py` | new collector |
| `server/app/collectors/openaq.py` | use shared geospatial helper |
| `server/tests/unit/test_purpleair.py` | new tests |
| `server/tests/unit/test_openaq.py` | verify OpenAQ still uses bbox behavior |
| `session_summary.md` | handoff update |

## Verification

- `venv/bin/pytest`

## Commit Point

`feat(collectors): add PurpleAir collector`
