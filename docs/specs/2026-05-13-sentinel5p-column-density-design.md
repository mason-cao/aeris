# Sentinel-5P Column Density Extraction Design

**Date**: 2026-05-13
**Phase**: Month 1 — close out
**Status**: Implemented 2026-05-13

## Context

The current `Sentinel5PCollector` is catalog-only. It queries the Copernicus OData catalog for L2 granules covering the Houston bbox in the last 48 hours and stores two metrics per granule: `s5p_<product>_granule_available` (always 1.0) and `s5p_<product>_cloud_cover` (from catalog metadata). It never opens the actual granule netCDF files, so the LLM pipeline has no column-density values to reason over — only "a granule existed".

For petrochemical attribution this is the wrong abstraction. A NO2 plume drifting north from the Ship Channel shows up as elevated tropospheric NO2 column density in TROPOMI's swath. We need the column-density numbers, spatially subset to the target area and quality-filtered, alongside the existing catalog metrics.

This spec extends the existing collector rather than replacing it. Catalog metrics stay; column-density metrics are added.

## Goals

1. After each catalog query, download the granule payload for any L2 product mapped to a Houston-relevant pollutant.
2. Open each granule with xarray + netCDF4, subset to the target bounding box, and emit a single mean column-density `DataPoint` per (granule, product) that survives the QA filter.
3. Authenticate to Copernicus via CDSE password OAuth using new `CDSE_USERNAME` / `CDSE_PASSWORD` env vars.
4. Keep existing `_granule_available` and `_cloud_cover` metrics so we don't regress catalog observability.
5. Add unit tests for token exchange, granule download, QA filtering, bbox subsetting, and the new `_column` metric emission.

## Products

| L2 product | netCDF variable | AERIS metric | Unit |
|------------|----------------|--------------|------|
| NO2 | `PRODUCT/nitrogendioxide_tropospheric_column` | `s5p_no2_column` | `mol/m^2` |
| SO2 | `PRODUCT/sulfurdioxide_total_vertical_column` | `s5p_so2_column` | `mol/m^2` |
| CO | `PRODUCT/carbonmonoxide_total_column` | `s5p_co_column` | `mol/m^2` |
| HCHO | `PRODUCT/formaldehyde_tropospheric_vertical_column` | `s5p_hcho_column` | `mol/m^2` |

`PRODUCT_TYPE_MAP` already lists O3, CH4, AER_AI but those are out of scope for v1 — they pass through catalog-only and emit no `_column` metric.

## Authentication

CDSE exposes a Keycloak-backed OAuth2 endpoint at:

```
POST https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token
Content-Type: application/x-www-form-urlencoded

grant_type=password
client_id=cdse-public
username=<CDSE_USERNAME>
password=<CDSE_PASSWORD>
```

Response carries `access_token` (~10 min TTL) and `refresh_token` (~1 hour TTL). For v1 the collector fetches a fresh access token at the start of every collection run and reuses it for all granule downloads in that run. Token caching across runs is deferred to v2.

Granule download URL pattern:

```
GET https://catalogue.dataspace.copernicus.eu/odata/v1/Products({Id})/$value
Authorization: Bearer <access_token>
```

Streaming download to a temp file (granules are 50-500 MB), then `xr.open_dataset(path, engine="netcdf4")` for extraction, then delete.

## Extraction

For each granule we keep from the catalog query:

1. Skip the download entirely if the granule's product code is not in `COLUMN_PRODUCTS` (NO2, SO2, CO, HCHO). Catalog-only metrics still emit.
2. Stream the granule payload to a temp file.
3. Open with xarray + netCDF4. The relevant group is `PRODUCT`; coordinate variables `latitude` and `longitude` are 2D arrays (along-track × across-track) — TROPOMI is swath data, not gridded.
4. Build a boolean mask from:
   - `PRODUCT/qa_value >= QA_THRESHOLD` for the product
   - `latitude` in `[bbox.min_lat, bbox.max_lat]`
   - `longitude` in `[bbox.min_lon, bbox.max_lon]`
5. If fewer than `MIN_VALID_PIXELS` survive (default 10), emit no `_column` point — the granule didn't actually cover Houston well.
6. Compute the arithmetic mean of the column-density variable over the surviving pixels.
7. Emit one `DataPoint` with `metric=s5p_<product>_column`, `value=mean`, `unit=mol/m^2`, anchored at the target center (`settings.aeris_target_lat / aeris_target_lon`) and `source_entity_id=<product_id>`.

QA thresholds per ESA recommendations:

| Product | Threshold |
|---------|-----------|
| NO2 | 0.75 |
| SO2 | 0.50 |
| CO | 0.50 |
| HCHO | 0.50 |

## Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Extend the existing collector, don't fork a new one | Catalog query, filter, and metadata parsing are already correct. The gap is only the granule processing step. |
| D2 | Keep `_granule_available` and `_cloud_cover` metrics | Useful observability for collection health and a fallback signal when QA filtering drops every pixel. |
| D3 | CDSE password OAuth, fresh token per run | Simplest dev setup. Tokens are short-lived so per-run is the right cadence. Refresh-token reuse is a v2 optimization. |
| D4 | netCDF via `xarray + netcdf4` | We already added xarray for NOAA GFS. Adding `netCDF4` as a binary dep is unavoidable for L2 files. No GRIB libs needed. |
| D5 | Houston-tuned 4-product set | NO2/SO2/CO/HCHO directly track Ship Channel petrochem signatures. O3/CH4/AER_AI add noise without attribution value in v1. |
| D6 | Spatial subset by bounding box, not by haversine | TROPOMI pixels are ~5.5×3.5 km — finer than our bbox-vs-radius rounding error. Bbox is cheaper and good enough. |
| D7 | QA threshold per product, ESA-recommended | NO2's stricter cut (0.75) is needed to drop cloud-contaminated retrievals. Other products tolerate 0.5 per ESA guidance. |
| D8 | Single mean column-density point per granule, anchored at target center | Matches the simplest interpretation of "a Sentinel-5P observation over Houston". Per-pixel or sub-grid storage is a v2 question. |
| D9 | Stream downloads to temp file, delete after parse | Granules are 50-500 MB; no point persisting them. Reprocessing on demand is fine since CDSE retains the archive. |
| D10 | `MIN_VALID_PIXELS = 10` floor | Below 10 surviving pixels the mean is statistically meaningless — likely a swath that barely clipped Houston. Skip rather than emit a noisy point. |
| D11 | Skip download for unmapped products | The catalog query may return O3 or CH4 granules. They keep their catalog metrics but never get downloaded — saves bandwidth and the heavy netCDF parse. |

## Files Touched

| File | Action |
|------|--------|
| `server/app/collectors/sentinel5p.py` | extend — auth, download, parse, new metric emission |
| `server/app/config.py` | add `cdse_username`, `cdse_password` |
| `server/.env.example` | document CDSE credentials |
| `server/requirements.txt` | add `netCDF4` |
| `server/tests/unit/test_sentinel5p.py` | add tests for auth, download, QA, column emission |
| `docs/specs/2026-05-13-sentinel5p-column-density-design.md` | this doc |
| `README.md` | flip Sentinel-5P "pending" callout in the roadmap |

## Verification

- `venv/bin/pytest`
- Manual smoke: with valid `CDSE_USERNAME` / `CDSE_PASSWORD` in `.env`, run `python -m app.collectors.run_all --source=sentinel5p`. Expect new `s5p_<product>_column` rows in `data_points` for any granule whose qa_value mask retains ≥10 pixels over the Houston bbox.

## Open Questions

- **Granule overlap & double-counting.** A single TROPOMI orbit can drop two adjacent granules with slight overlap, both intersecting Houston. v1 emits one point per granule; the LLM might see two "observations" for one physical pass. If this matters, v2 should dedup by orbit number.
- **Off-orbit days.** TROPOMI revisits any spot ~daily, but cloud cover and orbit geometry mean Houston may go 24-48h with zero retained pixels. The catalog metrics still emit, so the LLM can tell the difference between "no observation" and "low signal". Worth confirming once we see real data.
- **Token refresh inside long runs.** If a collection cycle takes >10 minutes (slow CDSE day, many granules), the access token expires mid-run. v1 fetches once per run, so a long run could fail late. If we see this in practice, plug in refresh-token reuse before retrying the token-fetch.

## Commit Point

`feat(collectors): extract Sentinel-5P column densities from granule netCDF`
