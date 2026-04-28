# OpenWeather Collector Design

**Date**: 2026-04-18
**Phase**: Month 1 Week 2 Step 2.3
**Status**: Implemented in initial form on 2026-04-18

## Context

Weather is a core cross-reference source for AERIS anomaly explanations. Wind direction and speed help explain pollutant transport, pressure and humidity affect particulate behavior, and precipitation/cloud cover provide context for air-quality changes. Month 1 needs current weather collection; Month 2 can enrich this with historical windows for detection features.

The Month 1 plan labels this collector "NOAA / OpenWeather." The first implementation uses OpenWeather Current Weather because the project already has `OPENWEATHER_API_KEY` in config and the endpoint gives the required current metrics by latitude/longitude. NOAA Weather API can remain a later no-key fallback.

## Goals

1. Add a weather collector for Suwanee and nearby grid points inside the configured radius.
2. Normalize temperature, humidity, wind, pressure, precipitation, and cloud cover into `DataPointCreate`.
3. Store values in explicit units, using OpenWeather metric units.
4. Preserve each requested grid point as `source_entity_id` so nearby samples do not deduplicate into one city-level row.
5. Add unit tests for fetch fan-out, metric normalization, optional precipitation handling, and bad row skipping.

## API Shape

Endpoint:

```text
GET https://api.openweathermap.org/data/2.5/weather
```

Query parameters:

| Param | Value |
|-------|-------|
| `lat` / `lon` | requested grid point |
| `appid` | `settings.openweather_api_key` |
| `units` | `metric` |

OpenWeather documents the relevant current weather fields as:

- `main.temp`: Celsius when `units=metric`
- `main.pressure`: hPa
- `main.humidity`: percent
- `wind.speed`: meter/sec when `units=metric`
- `wind.deg`: meteorological degrees
- `rain.1h` / `snow.1h`: precipitation in mm/h where available
- `clouds.all`: cloudiness percent
- `dt`: Unix UTC timestamp for data calculation

## Spatial Strategy

Query five points per hourly run:

| Grid ID | Location |
|---------|----------|
| `center` | configured target center |
| `north` | 25km north, capped by configured target radius |
| `east` | 25km east, capped by configured target radius |
| `south` | 25km south, capped by configured target radius |
| `west` | 25km west, capped by configured target radius |

This gives coarse weather coverage over the 50km target area without burning API calls. Each point is stored with `source_entity_id = "grid:<id>"`, not OpenWeather's city ID, because multiple nearby coordinates may resolve to the same city and would otherwise deduplicate incorrectly.

## Metrics

| OpenWeather field | AERIS metric | Unit | Notes |
|-------------------|--------------|------|-------|
| `main.temp` | `temperature` | `degC` | `units=metric` |
| `main.humidity` | `humidity` | `percent` | relative humidity |
| `wind.speed` | `wind_speed` | `m/s` | metric units |
| `wind.deg` | `wind_direction` | `degree` | meteorological degrees |
| `main.pressure` | `pressure` | `hPa` | sea-level pressure where provided |
| `rain.1h` + `snow.1h` | `precipitation` | `mm/h` | absent precipitation is stored as `0.0` |
| `clouds.all` | `cloud_cover` | `percent` | cloudiness |

## Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Use Current Weather API, not One Call 3.0 | Current Weather covers Month 1 metrics with lower payload complexity. One Call can be added for forecasts/history later. |
| D2 | Use `units=metric` | Keeps values aligned with scientific units and avoids later Fahrenheit conversion for OpenWeather. |
| D3 | Store `0.0 mm/h` for absent precipitation | The absence of `rain`/`snow` means no current precipitation in the response. Explicit zeros are easier for detection features than missing rows. |
| D4 | Use requested grid IDs for dedup | OpenWeather city IDs can collapse multiple nearby query points into the same entity. |
| D5 | Query sequentially | Five calls/hour is small; concurrency can be added only if a real bottleneck appears. |
| D6 | Raise if all grid-point requests fail | A completely failed run should trigger BaseCollector retry/status handling instead of being marked as an active zero-row run. |

## Files Touched

| File | Action |
|------|--------|
| `server/app/collectors/geo.py` | add short-distance coordinate offset helper |
| `server/app/collectors/noaa_weather.py` | new OpenWeather-backed collector |
| `server/tests/unit/test_noaa_weather.py` | new tests |
| `docs/specs/2026-04-05-month1-phase-plan.md` | note implementation details |
| `session_summary.md` | handoff update |

## Verification

- `venv/bin/pytest`

## Commit Point

`feat(collectors): add OpenWeather weather collector`
