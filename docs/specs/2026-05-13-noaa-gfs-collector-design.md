# NOAA GFS Collector Design

**Date**: 2026-05-13
**Phase**: Month 1 — fourth macro API
**Status**: Spec drafted, implementation pending

## Context

AERIS attributes climate anomalies by cross-referencing four macro data sources. OpenWeather covers surface-level fields at hourly cadence, but it does not expose upper-air physics — the 500 hPa flow patterns, 850 hPa temperature, planetary boundary layer height, and atmospheric water content that drive most of the synoptic-scale phenomena over Houston (sea-breeze fronts, inversions trapping petrochemical emissions, frontal passages). NOAA's Global Forecast System fills that gap.

This collector finalizes the four-API roster (OpenAQ, OpenWeather, Sentinel-5P, NOAA GFS) and replaces the placeholder NOAA filename that previously housed the OpenWeather collector.

## Goals

1. Add a `NOAAGFSCollector` that pulls 0.25° GFS analysis fields covering the configured target bounding box.
2. Normalize seven upper-air and surface variables into the standard `DataPointCreate` schema.
3. Run at the GFS native cadence (six-hourly cycles: 00z / 06z / 12z / 18z) with cycle-back fallback when the most recent run is not yet published.
4. Add unit tests for cycle selection, variable normalization, bounding-box subsetting, and Kelvin→Celsius / Pa→hPa unit conversion.
5. Keep dependencies pip-only — no `eccodes` or other system installs.

## Data Source & Transport

| Attribute | Value |
|-----------|-------|
| Source | NOAA GFS 0.25° global forecast |
| Endpoint | `https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs{YYYYMMDD}/gfs_0p25_{HH}z` |
| Protocol | OPeNDAP (DAP2) |
| Native cadence | Four model runs/day at 00z, 06z, 12z, 18z |
| Forecast hours | 0..384h. v1 reads forecast hour 0 (analysis only). |
| Library | `xarray` with `engine="pydap"` |

OPeNDAP lets the client request only the lat/lon/variable subset it needs, so each cycle transfers a few hundred KB instead of the multi-GB GRIB2 file. `pydap` is pure Python; this avoids the `cfgrib` + system `eccodes` dependency chain.

GFS variable names (the OPeNDAP server publishes these directly):

| GFS variable | Description |
|--------------|-------------|
| `hgtprs` | Geopotential height on pressure levels (`lev` dim) |
| `tmpprs` | Air temperature on pressure levels |
| `ugrd10m` | 10-meter eastward wind |
| `vgrd10m` | 10-meter northward wind |
| `pressfc` | Surface pressure |
| `pwatclm` | Precipitable water, entire atmospheric column |
| `hpblsfc` | Planetary boundary layer height |

## Spatial Strategy

Subset by the configured AERIS bounding box: `target_bounding_box(buffer_km=20)` from `app/collectors/geo.py`. For Houston (29.7604, -95.3698) with a 50km radius and 20km buffer, that yields a ~70km × 70km box → roughly **5 × 5 = 25 grid cells** at 0.25° resolution.

Each grid cell is stored as a separate `DataPoint`, with `source_entity_id = f"gfs:{lat:.2f},{lon:.2f}"`. This keeps cells independently addressable for spatial detection logic downstream.

## Metrics

| GFS variable | AERIS metric | Unit | Notes |
|--------------|--------------|------|-------|
| `hgtprs` @ lev=500 | `gh_500` | `m` | 500 hPa geopotential height — synoptic flow pattern |
| `tmpprs` @ lev=850 | `t_850` | `degC` | 850 hPa temperature — frontal / inversion detection. Convert from Kelvin. |
| `ugrd10m` | `u_10m` | `m/s` | 10-m eastward wind — pollutant transport |
| `vgrd10m` | `v_10m` | `m/s` | 10-m northward wind — pollutant transport |
| `pressfc` | `surface_pressure` | `hPa` | Surface pressure. Convert from Pa. |
| `pwatclm` | `precipitable_water` | `mm` | Column water vapor (kg/m² ≡ mm liquid equivalent) |
| `hpblsfc` | `pbl_height` | `m` | Mixed-layer depth — pollutant trapping |

Unit conventions match the rest of the AERIS schema: temperatures in degC (consistent with OpenWeather), pressures in hPa (consistent with OpenWeather), distances in meters.

## Cycle Selection

GFS cycles are published with a ~3-5 hour lag after their nominal time. The collector picks the freshest cycle that is actually available:

1. Compute the most recent nominal cycle: `latest = floor(utcnow().hour / 6) * 6`.
2. Walk back up to four cycles (24h), trying each `(date, hour)` pair in descending order.
3. Use the first cycle whose OPeNDAP dataset opens successfully.
4. If all four fail, raise `RuntimeError` — `BaseCollector` retry/status handling takes over.

`raw_json` records the cycle that was actually used so downstream code can correlate.

## Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | OPeNDAP via `xarray` + `pydap`, not GRIB2 + `cfgrib` | Pure-Python dep chain. Server-side subsetting keeps transfer small. No `eccodes` system install. |
| D2 | Pin to the 0.25° GFS dataset (`gfs_0p25`) | Finest publicly available global resolution. ~25 grid cells over Houston is enough detail for synoptic features without overwhelming storage. |
| D3 | Analysis only (forecast hour 0) for v1 | Attribution needs *observed* conditions, not forecasts. Forecast horizons can be added in v2 once the LLM pipeline starts asking for them. |
| D4 | One `DataPoint` per (grid cell, variable) | Matches the existing collector schema. Lets the spatial filters in `data` routes work uniformly across all four sources. |
| D5 | Houston-tuned 7-variable starter set | Covers the four attribution lenses: synoptic flow (`gh_500`), thermal structure (`t_850`), transport (`u_10m`/`v_10m`), boundary state (`surface_pressure`, `pbl_height`, `precipitable_water`). |
| D6 | Convert units at ingest (K→degC, Pa→hPa) | Consistent units across collectors simplify the detection and LLM stages. The original units stay in `raw_json` for audit. |
| D7 | Cycle-back fallback up to four cycles | GFS publication lag is typically 3-5h; walking back guarantees we always have *something* recent without hardcoding the lag. |
| D8 | Bounding-box subset with 20km buffer | The buffer lets `within_target_radius` filtering be done downstream without losing edge-of-radius cells to rounding. |
| D9 | No on-disk caching of OPeNDAP responses | Cycles change every six hours and each pull is small. Cache complexity isn't worth it for v1. |
| D10 | Raise on missing variables | If GFS renames or temporarily drops a variable, we want a loud failure, not silent partial ingestion. |

## Files Touched

| File | Action |
|------|--------|
| `server/app/collectors/noaa_gfs.py` | new collector |
| `server/app/collectors/registry.py` | register `NOAAGFSCollector` |
| `server/tests/unit/test_noaa_gfs.py` | new tests |
| `server/requirements.txt` | add `xarray`, `pydap` |
| `docs/specs/2026-05-13-noaa-gfs-collector-design.md` | this doc |
| `README.md` | flip NOAA GFS status from Pending to Live |

## Verification

- `venv/bin/pytest`
- Manual smoke: `python -m app.collectors.run_all --source=noaa_gfs` against a known-good cycle, confirm `DataPoint` rows with the seven expected metrics and ~25 distinct `source_entity_id` values.

## Open Questions

- **OPeNDAP reliability.** NOMADS occasionally throttles or returns intermittent errors. If we hit this in practice, D1 may need a follow-up: add the S3 GRIB2 path as a fallback (and the `cfgrib` dependency along with it).
- **PBL height accuracy over coasts.** GFS's diagnostic PBL height is known to be noisy over land-sea boundaries. Houston-area values may need a sanity floor/ceiling once we see real data — handle in v2.

## Commit Point

`feat(collectors): add NOAA GFS analysis collector`
