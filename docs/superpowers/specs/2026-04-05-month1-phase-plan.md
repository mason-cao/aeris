# AERIS Month 1 Phase Plan: Infrastructure & Data Pipeline

## Context

AERIS is at design-complete, pre-implementation stage. The full design spec exists at `docs/superpowers/specs/2026-04-05-aeris-design.md`. No code has been written. This plan covers Month 1 (April 2026): standing up the server infrastructure and building the data pipeline that collects from all 9 environmental data sources continuously.

**Target area**: Suwanee, GA (north Atlanta metro). Center: 34.05°N, 84.07°W, 50km radius.
**Development strategy**: Vertical slices — build one collector end-to-end first (EPA AirNow), validate the architecture, then replicate across all 9 sources. Develop locally first (Postgres in Docker), deploy to home server in Week 4.

## Files to Create/Modify

### Server infrastructure
- `server/requirements.txt` — Python dependencies
- `server/docker-compose.yml` — Local dev PostgreSQL + TimescaleDB
- `server/app/__init__.py`
- `server/app/config.py` — Pydantic Settings, env var loading
- `server/app/main.py` — FastAPI entry, CORS, lifespan
- `server/app/db/__init__.py`
- `server/app/db/models.py` — SQLAlchemy ORM: DataPoint, DataSource
- `server/app/db/schema.py` — DDL + TimescaleDB hypertable creation
- `server/app/db/session.py` — Async session factory

### Collectors (one file per source + base)
- `server/app/collectors/__init__.py`
- `server/app/collectors/base.py` — Abstract BaseCollector
- `server/app/collectors/epa_airnow.py`
- `server/app/collectors/openaq.py`
- `server/app/collectors/purpleair.py`
- `server/app/collectors/noaa_weather.py`
- `server/app/collectors/nasa_firms.py`
- `server/app/collectors/sentinel5p.py`
- `server/app/collectors/traffic.py`
- `server/app/collectors/usgs_water.py`
- `server/app/collectors/eia_energy.py`
- `server/app/collectors/run_all.py` — Orchestrator + APScheduler
- `server/app/collectors/validation.py` — Data quality checks

### API routes
- `server/app/api/__init__.py`
- `server/app/api/routes/__init__.py`
- `server/app/api/routes/data.py` — Raw data endpoints
- `server/app/api/routes/system.py` — Health/status

### Tests
- `server/tests/conftest.py` — Fixtures, test DB setup (SQLite for CI)
- `server/tests/unit/test_base_collector.py`
- `server/tests/unit/test_epa_airnow.py`
- `server/tests/unit/test_openaq.py`
- `server/tests/unit/test_purpleair.py`
- `server/tests/unit/test_noaa_weather.py`
- `server/tests/unit/test_nasa_firms.py`
- `server/tests/unit/test_sentinel5p.py`
- `server/tests/unit/test_traffic.py`
- `server/tests/unit/test_usgs_water.py`
- `server/tests/unit/test_eia_energy.py`
- `server/tests/unit/test_validation.py`
- `server/tests/integration/test_data_pipeline.py`
- `server/tests/integration/test_api_endpoints.py`

---

## Week 1: Foundation + First Vertical Slice (EPA AirNow)

**Goal**: Dev environment running, database live, one collector working end-to-end.

### Step 1.1: Dev environment setup
- Create `server/requirements.txt`:
  - fastapi, uvicorn[standard], sqlalchemy[asyncio], asyncpg, psycopg2-binary
  - httpx (async HTTP client for collectors)
  - pydantic, pydantic-settings, python-dotenv
  - apscheduler (for cron orchestration in Week 4)
  - pytest, pytest-cov, pytest-asyncio, httpx (test client)
- Create `server/docker-compose.yml`: PostgreSQL 16 + TimescaleDB extension
- Create `server/.env` from `.env.example` with local dev values
- Verify: `docker compose up -d` starts Postgres, `psql` connects

### Step 1.2: Database schema + models
- `app/config.py`: Pydantic Settings class reading from `.env`
  - DATABASE_URL, all API keys, AERIS_TARGET_LAT/LON/RADIUS, log level, env
- `app/db/models.py`: SQLAlchemy ORM models
  - **DataPoint**: id (UUID), timestamp (DateTime, indexed), lat (Float), lon (Float), metric (String, indexed), value (Float), source (String, indexed), raw_json (JSON), collected_at (DateTime)
  - **DataSource**: id (UUID), name (String, unique), source_type (String), config (JSON), last_collected_at (DateTime), status (String), error_count (Integer)
- `app/db/schema.py`: Create tables + TimescaleDB hypertable on data_points partitioned by timestamp
- `app/db/session.py`: async sessionmaker with asyncpg
- Verify: Run schema creation, check tables exist in psql

### Step 1.3: BaseCollector abstract class
- `app/collectors/base.py`:
  - `fetch()` → raw API response
  - `normalize()` → list of DataPointCreate dicts
  - `collect()` → orchestrates fetch → normalize → validate → store
  - Built-in: retry with exponential backoff, request timeout, structured logging
  - `DataPointCreate`: Pydantic model for the normalized schema
  - `CollectionResult`: Pydantic model (success, record_count, duration_ms, errors)

### Step 1.4: EPA AirNow collector
- `app/collectors/epa_airnow.py`:
  - API: `https://www.airnowapi.org/aq/observation/latLong/current/`
  - Params: lat=34.05, lon=-84.07, distance=50 (miles), API_KEY from config
  - Returns: current AQI observations for reporting areas in range
  - Metrics normalized: pm25, ozone, no2, so2, co
  - Each observation → one DataPoint row
- Unit test: mock API response, verify normalize() output
- Integration test: call collect() against test DB, verify rows inserted

### Step 1.5: FastAPI skeleton + data endpoint
- `app/main.py`: FastAPI app, CORS (allow localhost:5173), lifespan (init DB)
- `app/api/routes/system.py`: `GET /api/health` → {"status": "ok", "db": "connected"}
- `app/api/routes/data.py`: `GET /api/data/{source}` → paginated DataPoints
  - Query params: metric, start, end, lat, lon, radius_km, limit, offset
- Verify: Run EPA collector, then `curl localhost:8000/api/data/epa_airnow` returns data

### Week 1 Deliverable
Run `python -m app.collectors.epa_airnow` → real Atlanta air quality data in Postgres → queryable via `GET /api/data/epa_airnow`. First passing test suite.

**Commit point**: `feat(core): add database schema, BaseCollector, and EPA AirNow collector`

---

## Week 2: Air Quality + Weather + Fire Collectors

**Goal**: 5 of 9 collectors working. Air quality from 3 independent sources.

**API key registrations (parallel)**: PurpleAir, OpenWeather. (OpenAQ and NASA FIRMS need no key.)

### Step 2.1: OpenAQ collector
- API: `https://api.openaq.org/v3/` — government air quality monitors
- No auth required. Query by coordinates + radius.
- Metrics: pm25, pm10, ozone, no2, so2, co, bc

### Step 2.2: PurpleAir collector
- API: `https://api.purpleair.com/v1/sensors`
- Auth: API key in header (`X-API-Key`)
- Query: bounding box around target area
- Metrics: pm25 (primary), pm10, humidity, temperature (sensor-reported)
- Note: PurpleAir PM2.5 readings run ~30-40% higher than EPA reference monitors. Store raw values; correction factor is a Month 2 concern.

### Step 2.3: NOAA / OpenWeather collector
- API: OpenWeather Current Weather + One Call 3.0 (free tier: 1000 calls/day)
- Metrics: temperature, humidity, wind_speed, wind_direction, pressure, precipitation, cloud_cover
- Query: lat/lon for Suwanee + nearby grid points (cover the 50km radius)
- Also consider NOAA Weather API (api.weather.gov) as free backup — no key needed

### Step 2.4: NASA FIRMS collector
- API: `https://firms.modaps.eosdis.nasa.gov/api/` — active fire data
- No auth. CSV download for bounding box + time range.
- Metrics: fire_radiative_power, confidence, brightness
- Query: NRT (Near Real-Time) MODIS and VIIRS data for larger bounding box (100km)

### Step 2.5: Enhance data API + full test suite
- Add query params to data endpoint: `?metric=pm25&start=...&end=...&radius_km=25`
- Add `GET /api/data/sources` — list all data sources with their last_collected_at and status
- Unit tests for all 4 new collectors
- Integration test: run all 5 collectors, verify data from each source

**Commit point**: `feat(collectors): add OpenAQ, PurpleAir, NOAA weather, and NASA FIRMS collectors`

---

## Week 3: Satellite + Traffic + Water + Energy + Static Data

**Goal**: All 9 collectors working. Static reference data loaded.

**API key registrations**: NASA Earthdata (Sentinel-5P), TomTom, EIA.

### Step 3.1: Sentinel-5P collector (most complex)
- API: NASA Earthdata / Copernicus Open Access Hub
- Auth: NASA Earthdata bearer token
- Data: NO2, SO2 tropospheric column density (daily satellite passes)
- Complexity: Large GeoTIFF/NetCDF files → extract values for bounding box
- Add `netCDF4` to requirements.txt (lighter than rasterio, sufficient for extracting grid values)
- Normalize: grid cell averages → DataPoint rows (one per grid cell in range)

### Step 3.2: TomTom Traffic collector
- API: `https://api.tomtom.com/traffic/services/` — Traffic Flow API
- Auth: API key as query param
- Metrics: traffic_speed, traffic_freeflow_speed, traffic_congestion (ratio)
- Query: road segments within bounding box. Focus on major roads: I-85, I-285, GA-316, US-23

### Step 3.3: USGS Water Services collector
- API: `https://waterservices.usgs.gov/nwis/iv/` — Instantaneous Values
- No auth. Query by bounding box or site codes.
- Metrics: streamflow, water_temperature, conductivity, dissolved_oxygen, turbidity, ph
- Key sites: Chattahoochee River gauges near Suwanee/Buford Dam

### Step 3.4: EIA Open Data collector
- API: `https://api.eia.gov/v2/` — Open Data API
- Auth: API key as query param
- Metrics: generation_mwh, co2_emissions, fuel_type, grid_carbon_intensity
- Scope: Georgia power plants + SERC region grid data

### Step 3.5: Static reference data + comprehensive tests
- **EPA TRI**: Load facility locations + annual emissions for Gwinnett County and surrounding counties
- **EPA ECHO**: Compliance violations for facilities in range
- **NWS Alerts**: RSS feed integration for severe weather + air quality alerts
- Full unit test suite for all 9 collectors
- Integration test: run all 9, verify data from each

**Commit point**: `feat(collectors): add Sentinel-5P, TomTom traffic, USGS water, and EIA energy collectors`

---

## Week 4: Orchestration + Server Deployment + Validation

**Goal**: Everything running autonomously on the home server. 48-hour validation pass.

### Step 4.1: Cron orchestration (run_all.py)
- `app/collectors/run_all.py` using APScheduler:

| Interval | Sources | Rationale |
|----------|---------|-----------|
| 15 min | PurpleAir, TomTom, USGS | High-frequency sources |
| 1 hour | EPA AirNow, OpenAQ, NOAA | Standard reporting intervals |
| 3 hours | NASA FIRMS | NRT data latency |
| Daily (6 AM ET) | Sentinel-5P, EIA | Daily products |

- Per-collector failure isolation (one crash doesn't stop others)
- Retry: 3 attempts with 30s / 60s / 120s exponential backoff
- Stale data alert: log WARNING if source hasn't updated in 2× expected interval
- CLI: `python -m app.collectors.run_all` (all sources) or `--source=epa` (single)
- Structured logging: JSON format, one log line per collection run

### Step 4.2: Data validation layer
- `app/collectors/validation.py`:
  - **Schema check**: all required fields present, correct types
  - **Range check**: values within plausible bounds per metric
  - **Duplicate detection**: reject if (timestamp, lat, lon, metric, source) exists
  - **Freshness check**: warn if data timestamp > 2× expected interval old
- Called automatically in BaseCollector.collect() between normalize() and store

### Step 4.3: Home server setup
- Ubuntu Server 22.04 LTS on the old PC
- Networking: static local IP, ufw (allow 22, 8000), DDNS for remote access
- Install: PostgreSQL 16 + TimescaleDB, Python 3.11, git
- Install Ollama + pull Llama 3 8B (for Month 2 — but install now to verify hardware)

### Step 4.4: Deploy AERIS to server
- Clone repo, create venv, pip install requirements
- Create production `.env` with real API keys + production DB URL
- Run schema creation against production Postgres
- Systemd services: `aeris-api.service` + `aeris-collector.service`
  - Both: auto-restart on failure, start on boot

### Step 4.5: 48-hour validation run
**Pass criteria**:
1. All 9 sources collected data within 2× their expected interval (no gaps)
2. Per-source failure rate <5%
3. No schema validation errors
4. API endpoints return valid data for all 9 sources
5. Server resources stable: CPU <50% avg, memory <80%
6. No unhandled exceptions in logs
7. Ollama inference test passes

**Commit point**: `feat(infra): add collector orchestration, validation, and server deployment config`

---

## Month 1 Acceptance Criteria

- [ ] All 9 collectors pulling real data from Atlanta metro area APIs
- [ ] Normalized schema `(timestamp, lat, lon, metric, value, source, raw_json)` for all data
- [ ] Scheduled collection at correct intervals with per-collector failure isolation
- [ ] 48-hour validation pass on home server (no gaps, no crashes)
- [ ] FastAPI endpoints returning filtered data per source
- [ ] Data validation layer catching schema/range/duplicate/freshness issues
- [ ] Unit tests for BaseCollector + all 9 collectors (80%+ coverage on collectors/)
- [ ] Integration tests for end-to-end data flow and API endpoints
- [ ] Home server running autonomously with systemd auto-restart
- [ ] Static reference data loaded (EPA TRI facilities, EPA ECHO violations)

## API Key Registration Checklist

| Key | Needed by | Registration URL |
|-----|-----------|-----------------|
| EPA AirNow | Week 1 | https://docs.airnowapi.org/account/request/ |
| PurpleAir | Week 2 | https://develop.purpleair.com/ |
| OpenWeather | Week 2 | https://openweathermap.org/api |
| NASA Earthdata | Week 3 | https://urs.earthdata.nasa.gov/users/new |
| TomTom | Week 3 | https://developer.tomtom.com/ |
| EIA | Week 3 | https://www.eia.gov/opendata/register.php |
| Mapbox | Month 3 | https://account.mapbox.com/auth/signup/ |

## Verification Plan

1. **Per-collector**: Each collector has unit tests (mock API) + can be run manually against real API
2. **Integration**: `pytest tests/integration/` hits test DB, verifies full collect → store → query flow
3. **End-to-end**: 48-hour continuous run on server with monitoring
4. **Manual spot-check**: Query each source via API, verify data looks reasonable for Atlanta area
