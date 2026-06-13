# Graph Report - aeris  (2026-06-13)

## Corpus Check
- 118 files · ~95,088 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2146 nodes · 5393 edges · 118 communities (106 shown, 12 thin omitted)
- Extraction: 66% EXTRACTED · 34% INFERRED · 0% AMBIGUOUS · INFERRED: 1830 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d12b07b8`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_L1|L1]]
- [[_COMMUNITY_L1|L1]]
- [[_COMMUNITY_Enrichment|Enrichment]]
- [[_COMMUNITY_L106|L106]]
- [[_COMMUNITY_L66|L66]]
- [[_COMMUNITY_L1|L1]]
- [[_COMMUNITY_Rate limiter|Rate limiter]]
- [[_COMMUNITY_L92|L92]]
- [[_COMMUNITY_L1|L1]]
- [[_COMMUNITY_L1|L1]]
- [[_COMMUNITY_L1|L1]]
- [[_COMMUNITY_L178|L178]]
- [[_COMMUNITY_L35|L35]]
- [[_COMMUNITY_L1|L1]]
- [[_COMMUNITY_L1|L1]]
- [[_COMMUNITY_L1|L1]]
- [[_COMMUNITY_L1|L1]]
- [[_COMMUNITY_L554|L554]]
- [[_COMMUNITY_L46|L46]]
- [[_COMMUNITY_L1|L1]]
- [[_COMMUNITY_L24|L24]]
- [[_COMMUNITY_L70|L70]]
- [[_COMMUNITY_L57|L57]]
- [[_COMMUNITY_Historical backfill|Historical backfill]]
- [[_COMMUNITY_L1|L1]]
- [[_COMMUNITY_L49|L49]]
- [[_COMMUNITY_L289|L289]]
- [[_COMMUNITY_L1|L1]]
- [[_COMMUNITY_L22|L22]]
- [[_COMMUNITY_L30|L30]]
- [[_COMMUNITY_L296|L296]]
- [[_COMMUNITY_L608|L608]]
- [[_COMMUNITY_L22|L22]]
- [[_COMMUNITY_L1|L1]]
- [[_COMMUNITY_L17|L17]]
- [[_COMMUNITY_L13|L13]]
- [[_COMMUNITY_L451|L451]]
- [[_COMMUNITY_L26|L26]]
- [[_COMMUNITY_L41|L41]]
- [[_COMMUNITY_L158|L158]]
- [[_COMMUNITY_Detection|Detection]]
- [[_COMMUNITY_L90|L90]]
- [[_COMMUNITY_L1|L1]]
- [[_COMMUNITY_L1|L1]]
- [[_COMMUNITY_L134|L134]]
- [[_COMMUNITY_L1|L1]]
- [[_COMMUNITY_L22|L22]]
- [[_COMMUNITY_L1|L1]]
- [[_COMMUNITY_L17|L17]]
- [[_COMMUNITY_L45|L45]]
- [[_COMMUNITY_L46|L46]]
- [[_COMMUNITY_L1|L1]]
- [[_COMMUNITY_L20|L20]]
- [[_COMMUNITY_L363|L363]]
- [[_COMMUNITY_L1|L1]]
- [[_COMMUNITY_L17|L17]]
- [[_COMMUNITY_L360|L360]]
- [[_COMMUNITY_L15|L15]]
- [[_COMMUNITY_Alembic env|Alembic env]]
- [[_COMMUNITY_L14|L14]]
- [[_COMMUNITY_L49|L49]]
- [[_COMMUNITY_L121|L121]]
- [[_COMMUNITY_L99|L99]]
- [[_COMMUNITY_Table|Table]]
- [[_COMMUNITY_L12|L12]]
- [[_COMMUNITY_Detection CLI|Detection CLI]]
- [[_COMMUNITY_L138|L138]]
- [[_COMMUNITY_L102|L102]]
- [[_COMMUNITY_L23|L23]]
- [[_COMMUNITY_Data routes|Data routes]]
- [[_COMMUNITY_L94|L94]]
- [[_COMMUNITY_L102|L102]]
- [[_COMMUNITY_L191|L191]]
- [[_COMMUNITY_L126|L126]]
- [[_COMMUNITY_L31|L31]]
- [[_COMMUNITY_Db|Db]]
- [[_COMMUNITY_Detection CLI|Detection CLI]]
- [[_COMMUNITY_DB schema|DB schema]]
- [[_COMMUNITY_Corroboration scorer|Corroboration scorer]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 83|Community 83]]
- [[_COMMUNITY_Community 84|Community 84]]
- [[_COMMUNITY_L18|L18]]
- [[_COMMUNITY_Test|Test]]
- [[_COMMUNITY_Community 88|Community 88]]
- [[_COMMUNITY_Collectors|Collectors]]
- [[_COMMUNITY_Community 92|Community 92]]
- [[_COMMUNITY_Community 93|Community 93]]
- [[_COMMUNITY_Community 95|Community 95]]
- [[_COMMUNITY_Community 99|Community 99]]
- [[_COMMUNITY_Community 103|Community 103]]
- [[_COMMUNITY_Community 104|Community 104]]
- [[_COMMUNITY_Test|Test]]
- [[_COMMUNITY_Community 106|Community 106]]
- [[_COMMUNITY_Community 107|Community 107]]
- [[_COMMUNITY_Community 108|Community 108]]
- [[_COMMUNITY_Community 109|Community 109]]
- [[_COMMUNITY_Community 110|Community 110]]
- [[_COMMUNITY_Community 111|Community 111]]
- [[_COMMUNITY_Community 112|Community 112]]
- [[_COMMUNITY_Community 113|Community 113]]
- [[_COMMUNITY_Community 126|Community 126]]

## God Nodes (most connected - your core abstractions)
1. `DataPoint` - 143 edges
2. `Anomaly` - 101 edges
3. `EnrichmentRecord` - 76 edges
4. `ConsensusAnomaly` - 71 edges
5. `LLMClient` - 69 edges
6. `Sentinel5PCollector` - 66 edges
7. `DataPointCreate` - 62 edges
8. `AsyncRateLimiter` - 61 edges
9. `BaseCollector` - 58 edges
10. `NOAAGFSCollector` - 54 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `create_tables()`  [INFERRED]
  deploy/windows-collector/init_db.py → server/app/db/schema.py
- `AsyncConnection` --uses--> `Base`  [INFERRED]
  server/app/db/schema.py → server/app/db/models.py
- `AsyncSession` --uses--> `Base`  [INFERRED]
  server/tests/conftest.py → server/app/db/models.py
- `Namespace` --uses--> `Anomaly`  [INFERRED]
  server/app/eval/freeze.py → server/app/db/models.py
- `ClaimDraft` --uses--> `ClaimDraft`  [INFERRED]
  server/app/llm/validate.py → server/app/llm/parser.py

## Import Cycles
- 1-file cycle: `server/app/api/routes/data.py -> server/app/api/routes/data.py`
- 1-file cycle: `server/app/main.py -> server/app/main.py`
- 1-file cycle: `server/app/collectors/backfill.py -> server/app/collectors/backfill.py`
- 1-file cycle: `server/app/collectors/noaa_gfs.py -> server/app/collectors/noaa_gfs.py`
- 1-file cycle: `server/app/collectors/openaq.py -> server/app/collectors/openaq.py`
- 1-file cycle: `server/app/collectors/openweather.py -> server/app/collectors/openweather.py`
- 1-file cycle: `server/app/collectors/sentinel5p.py -> server/app/collectors/sentinel5p.py`
- 1-file cycle: `server/tests/unit/test_backfill.py -> server/tests/unit/test_backfill.py`
- 1-file cycle: `server/app/detection/consensus.py -> server/app/detection/consensus.py`
- 1-file cycle: `server/app/detection/stl.py -> server/app/detection/stl.py`
- 1-file cycle: `server/app/detection/zscore.py -> server/app/detection/zscore.py`
- 1-file cycle: `server/app/detection/engine.py -> server/app/detection/engine.py`
- 1-file cycle: `server/app/detection/run.py -> server/app/detection/run.py`
- 1-file cycle: `server/app/detection/enrichment.py -> server/app/detection/enrichment.py`
- 1-file cycle: `server/app/eval/freeze.py -> server/app/eval/freeze.py`
- 1-file cycle: `server/app/llm/corroboration.py -> server/app/llm/corroboration.py`
- 1-file cycle: `server/app/llm/explain.py -> server/app/llm/explain.py`
- 1-file cycle: `server/tests/unit/test_data_routes.py -> server/tests/unit/test_data_routes.py`
- 1-file cycle: `server/tests/unit/detection/test_consensus.py -> server/tests/unit/detection/test_consensus.py`
- 1-file cycle: `server/tests/unit/detection/test_engine.py -> server/tests/unit/detection/test_engine.py`

## Communities (118 total, 12 thin omitted)

### Community 0 - "L1"
Cohesion: 0.06
Nodes (52): _Bucket, _bucket_for(), _build_consensus_anomaly(), merge(), Consensus across the three detectors.  Each detector (Z-score, STL, IsolationFor, Internal mutable accumulator: detector outputs at a single timestamp., Pick the observed value and verify detectors agree.      All detectors share the, Z-score's expected_value wins over STL's; IF doesn't provide one. (+44 more)

### Community 1 - "L1"
Cohesion: 0.14
Nodes (38): ConsensusAnomaly, A merged anomaly record across detectors at a single (timestamp, location)., DetectionEngine, Detection orchestrator.  Runs the configured detectors (Z-score, STL, IsolationF, Orchestrates one or more detectors over a single time-series.      Detectors are, Standard 3-detector engine with each detector's library defaults., Run all configured detectors on ``series`` and merge via consensus.          ``s, IsolationForestDetector (+30 more)

### Community 2 - "Enrichment"
Cohesion: 0.28
Nodes (6): _make_anomaly(), _make_claim(), _make_explanation(), TestClaimModel, TestExpertLabelModel, TestExplanationModel

### Community 3 - "L106"
Cohesion: 0.10
Nodes (30): ABC, _amain(), archive_key(), available_strategies(), BackfillStrategy, _days_between(), db_location_ids(), _format_result() (+22 more)

### Community 4 - "L66"
Cohesion: 0.04
Nodes (46): AERIS Month 2 — AI Attribution Phase Plan, Architecture additions, Bracco prep window: June 1 → June 5 (5 days), Bracco-readiness checkpoint (June 2 meeting), Bracco-readiness gate (Mon Jun 1, 22:00 CT), Claim taxonomy, Connection to broader research narrative, Context (+38 more)

### Community 5 - "L1"
Cohesion: 0.32
Nodes (6): Any, Sentinel5PCollector, make_payload(), make_record(), TestSentinel5PColumnNormalize, TestSentinel5PNormalize

### Community 6 - "Rate limiter"
Cohesion: 0.11
Nodes (38): BaseCollector, CollectorClass, BackfillResult, NOAAGFSBackfill, OpenWeatherBackfill, Walk the Sentinel-5P catalog backward in 48h windows, extracting columns.      C, Walk past GFS cycles within the NOMADS retention window., Documented no-op: the free OpenWeather tier has no historical API. (+30 more)

### Community 7 - "L92"
Cohesion: 0.14
Nodes (8): area_polygon(), extract_product_code(), granule_download_url(), _load_granule_arrays(), odata_filter(), parse_iso_datetime(), Open a downloaded S5P granule and return flat (value, qa, lat, lon) lists., TestSentinel5PHelpers

### Community 8 - "L1"
Cohesion: 0.09
Nodes (33): _amain(), _ensure_utc(), fixture_payload(), _format_result(), freeze_eval_set(), FreezeResult, group_events(), load_window_anomalies() (+25 more)

### Community 9 - "L1"
Cohesion: 0.08
Nodes (33): Enum, GenerationResult, A structured generation: the parsed schema instance plus call metadata., ClaimType, The 10-type claim taxonomy, plus a routing fallback for claims the     rules don, extract_claim_drafts(), Flatten the per-step claims into ordered ClaimDrafts.      step_index is 1-based, build_step_prompt() (+25 more)

### Community 10 - "L1"
Cohesion: 0.09
Nodes (38): _amain(), _format_explanation(), generate_explanation(), main(), make_client(), _metric_points(), _parse_args(), persist_explanation() (+30 more)

### Community 11 - "L178"
Cohesion: 0.08
Nodes (21): _iso_z(), _measurement_timestamp(), Render UTC datetime as an OpenAQ-friendly ISO-8601 string., Pull the measurement timestamp from an OpenAQ /measurements record.      The v3, target_bounding_box(), within_target_radius(), location_within_target_radius(), normalize_openaq_unit() (+13 more)

### Community 12 - "L35"
Cohesion: 0.11
Nodes (20): offset_coordinate(), Approximate coordinate offset for short distances., extract_precipitation(), OpenWeatherCollector, parse_observation_time(), Transform OpenWeather current weather responses into DataPoints., Fetch current weather for target-area grid points., weather_query_points() (+12 more)

### Community 13 - "L1"
Cohesion: 0.06
Nodes (49): _anomaly_ts(), BackgroundTolerance, _blank_match(), _claim_trend_window(), _claimed_temperature(), _combine(), ConcentrationTolerance, _gfs_wind_components() (+41 more)

### Community 14 - "L1"
Cohesion: 0.11
Nodes (27): classify_claim(), Rule-based routing of a claim into taxonomy types, most verifiable first.      A, One claim routed and scored: everything Phase 2 contributes to a Claim row., Classify a claim and run its primary type's scorer.      Unclassified claims agg, score_claim(), ScoredClaim, Claim-type classification + Phase 2 dispatch.  Step 1 of the corroboration flow:, _summary_with() (+19 more)

### Community 15 - "L1"
Cohesion: 0.12
Nodes (11): check_grounding(), ground_claim_drafts(), Verify a claim against the context the model was shown., Run the Phase 1 grounding check over every extracted claim., _salient_terms(), ClaimDraft, Threshold-worded quantities match directionally, not by point tolerance.      "e, TestCheckGrounding (+3 more)

### Community 16 - "L1"
Cohesion: 0.10
Nodes (35): _amain(), _cell_exists(), _format_summaries(), load_anomaly_set(), main(), ModelSweepSummary, _parse_args(), Sweep the eval anomaly set across all comparison models.  One cell = one (anomal (+27 more)

### Community 17 - "L554"
Cohesion: 0.16
Nodes (38): Score a regional-vs-local claim by spatial CV across OpenAQ stations., Score a 'pollutant was elevated' claim against OpenAQ + Sentinel-5P.      Three, score_background_vs_event(), score_concentration_elevation(), _entity(), _hourly_series(), _metric_from_entities(), Phase 2 corroboration scorer — shared aggregator.  The aggregator turns per-sour (+30 more)

### Community 18 - "L46"
Cohesion: 0.16
Nodes (14): build_aux_inputs(), Assemble IsolationForest inputs by joining wind speed + pbl_height aux.      Win, _dp(), _gfs_wind(), SQLite returns naive datetimes; Postgres returns tz-aware. Normalize     both sh, A GFS u_10m + v_10m DataPoint pair for one grid cell-cycle., _seasonal(), _seed() (+6 more)

### Community 19 - "L1"
Cohesion: 0.13
Nodes (27): _amain(), _ask_verdict(), collect_claim_groups(), main(), _parse_args(), Interactive labeling tool for the expert evaluation set.  Loads one anomaly's co, Unique claim texts for an anomaly, in (model, step) first-seen order., One verdict letter, re-prompting until valid; None means quit. (+19 more)

### Community 20 - "L24"
Cohesion: 0.12
Nodes (15): Clock, _defer_if_budget_spent(), _header_seconds(), rate_limited_get(), Push the next acquire at least ``seconds`` into the future.          Used when a, GET through the limiter, honoring Retry-After on 429 responses., _retry_wait_seconds(), AsyncClient (+7 more)

### Community 21 - "L70"
Cohesion: 0.09
Nodes (46): GroundingResult, LLMClient, Raw model output plus usage metadata from a single model call., Abstract base class for all LLM clients (local and cloud).      Subclasses imple, RawCompletion, _claim_row(), One Claim row: Phase 1 verdict always, Phase 2 only for grounded claims.      Un, GeminiClient (+38 more)

### Community 22 - "L57"
Cohesion: 0.18
Nodes (11): aggregate_verdicts(), CorroborationResult, Aggregated cross-source verdict for one claim.      ``corroboration_score`` is `, Collapse per-source verdicts into a scalar score and evidence count.      ``scor, test_all_contradicting_scores_minus_one(), test_all_silent_returns_null_and_unverified(), test_all_supporting_scores_plus_one(), test_empty_verdicts_is_unverified() (+3 more)

### Community 23 - "Historical backfill"
Cohesion: 0.16
Nodes (11): OpenAQArchiveBackfill, parse_archive_csv(), Map archive CSV rows to normalized DataPoints.      Columns: location_id, sensor, Bulk history from the OpenAQ S3 data archive instead of the API., archive_csv(), TestApiBackfillRateLimit, TestArchiveKey, TestDbLocationIds (+3 more)

### Community 24 - "L1"
Cohesion: 0.19
Nodes (8): attribute_value(), Column-product records minus those already extracted to the DB., Stream a granule to ``fd``, following redirects by hand.          The OData ``$v, safe_float(), Any, AsyncClient, BoundingBox, DataPointCreate

### Community 25 - "L49"
Cohesion: 0.12
Nodes (30): distance_km(), Great-circle distance between two WGS84 coordinates., _amain(), build_cross_source_summary(), EnrichmentLine, _ensure_utc(), _Entry, _format_summary() (+22 more)

### Community 26 - "L289"
Cohesion: 0.18
Nodes (8): _match_variable(), _message_matches(), Download the freshest reachable GFS analysis cycle from the NOMADS GRIB filter., Fetch and parse one GFS cycle; empty list means the cycle is unavailable., GET one cycle's GRIB subset; None if the cycle is not published yet., Extract one (lat, lon, values) cell per grid point from a GRIB2 subset., Any, datetime

### Community 27 - "L1"
Cohesion: 0.31
Nodes (5): RawCompletion, _completion(), MockLLMClient, Concrete LLMClient returning scripted raw completions, for testing., TestGenerate

### Community 28 - "L22"
Cohesion: 0.04
Nodes (44): AERIS (Autonomous Environmental RAG & Inference System) - Design Specification, Anomaly Detection Engine, Anomaly Enrichment, Architecture, Consensus & Severity, Context, Core Environmental, Cross-Reference Sources (+36 more)

### Community 29 - "L30"
Cohesion: 0.22
Nodes (11): clear_locations_cache(), OpenAQCollector, Any, collector(), fast_limiter(), _fresh_locations_cache(), make_location(), make_raw() (+3 more)

### Community 30 - "L296"
Cohesion: 0.18
Nodes (10): _nearest_cell_value(), _nearest_value(), The time-nearest value from the spatially nearest cell that has one., Return the value at the row whose timestamp is closest to ``target``,     provid, Samples per diurnal cycle from the observed cadence, or None to skip STL., _stl_period_for(), TestNearestCellValue, TestNearestValue (+2 more)

### Community 31 - "L608"
Cohesion: 0.18
Nodes (11): ChemistryTolerance, _direction_verdict(), Draft tolerance for chemistry (pending Dr. Bracco).      TROPOMI HCHO is noisy a, {'hcho': 'up', 'ozone': 'down', ...} for adjective-species mentions., Score a chemical-signature claim. Qualitative-only per Bracco 2026-06-10:     sc, score_chemistry(), _species_directions(), test_chemistry_clear_contradiction() (+3 more)

### Community 32 - "L22"
Cohesion: 0.05
Nodes (36): AERIS Month 1 Phase Plan: Infrastructure & Data Pipeline, API Key Registration Checklist, API routes, Collectors (one file per source + base), Context, Files to Create/Modify, Month 1 Acceptance Criteria, Server infrastructure (+28 more)

### Community 33 - "L1"
Cohesion: 0.19
Nodes (20): _amain(), _ensure_utc(), _format_summary(), _load_aux_cells(), _load_gfs_component(), _load_gfs_wind_cells(), load_points(), main() (+12 more)

### Community 34 - "L17"
Cohesion: 0.17
Nodes (7): _hourly_inputs(), TestIFDetectorBasics, TestIFDetectorConfiguration, TestIFDetectorContract, TestIFDetectorMultivariate, TestIFDetectorUnivariate, IsolationForestInput

### Community 35 - "L13"
Cohesion: 0.13
Nodes (7): Pure sinusoidal series, evenly-spaced hourly, no noise., _seasonal_series(), TestSTLDetectorBasics, TestSTLDetectorConfiguration, TestSTLDetectorContract, TestSTLDetectorOutliers, datetime

### Community 36 - "L451"
Cohesion: 0.17
Nodes (16): _point_value(), The numeric threshold in an 'exceeded N' style claim, else None.      Clock time, The numeric value in a 'was N' style point claim, else None.      Threshold-word, _threshold_value(), _match_numbers(), _quantities(), (value, unit, character offset) for every quantity in the text., How the quantity at ``position`` relates to the measurement it cites.      A thr (+8 more)

### Community 37 - "L26"
Cohesion: 0.32
Nodes (4): Any, make_grid_cell(), make_raw(), TestNOAAGFSNormalize

### Community 38 - "L41"
Cohesion: 0.15
Nodes (10): GUID, UUID column stored in a form SQLite cannot misread as a number.      Postgres ke, Dialect, Any, UUID, TypeEngine, Document the corruption mode the dashed format defends against.      The product, TestGUIDBindAndResult (+2 more)

### Community 40 - "Detection"
Cohesion: 0.14
Nodes (14): _earliest_keyword(), _metric_block(), (OpenAQ metric, Sentinel column metric) named in the claim, if any., Per-station mean of in-window values, for stations with enough coverage., Coefficient of variation across station means; None when undefined., The group whose keyword appears first in the text; None if none match.      Firs, Draft thresholds for emissions_source_type (pending Dr. Bracco)., Score a mobile / point / area source-type claim against OpenAQ patterns.      mo (+6 more)

### Community 41 - "L90"
Cohesion: 0.17
Nodes (6): collector(), Minimal stand-in for an httpx streaming response.      httpx rewrites Content-Le, _StubStreamClient, _StubStreamResponse, TestDownloadGranule, TestExtractColumnsTokenRefresh

### Community 42 - "L1"
Cohesion: 0.32
Nodes (5): lifespan(), FastAPI, health_check(), HealthResponse, AsyncSession

### Community 43 - "L1"
Cohesion: 0.16
Nodes (13): AsyncConnection, AsyncEngine, configure_postgres_compatibility(), configure_timescale(), create_tables(), drop_tables(), Create all tables and enable TimescaleDB hypertable on data_points., Keep existing development databases compatible with the current ORM. (+5 more)

### Community 44 - "L134"
Cohesion: 0.25
Nodes (6): _anomaly(), _dp(), build_cross_source_summary with the window derived from the anomaly., _summary(), TestBuildCrossSourceSummary, DataPoint

### Community 45 - "L1"
Cohesion: 0.12
Nodes (15): A.E.R.I.S. - Autonomous Environmental RAG & Inference System, Acknowledgements, Architecture, Data Sources, Environment Variables, Getting Started, License, Prerequisites (+7 more)

### Community 46 - "L22"
Cohesion: 0.23
Nodes (7): _hourly_series(), _iso_inputs(), _run_kwargs(), _seasonal(), TestEngineEmptyAndStable, TestEngineReturnContract, timedelta

### Community 47 - "L1"
Cohesion: 0.19
Nodes (10): _build_series_with_events(), _houston_pm25_baseline(), Multi-event integration tests for the detection engine.  Builds synthetic Housto, Synthetic Houston PM2.5 baseline.      Diurnal cycle (peak around morning rush h, Typical wind / PBL for a Houston spring day., Build (univariate series, IF inputs) with injected events.      ``events`` maps, TestKnownEventsReproducibility, TestKnownHoustonEvents (+2 more)

### Community 48 - "L17"
Cohesion: 0.23
Nodes (4): GeminiClient, _client_with(), _no_sleep(), TestGeminiClient

### Community 49 - "L45"
Cohesion: 0.12
Nodes (15): Architecture, Commit Points, Config Changes, Context, Design Decisions, Error Handling, Files Touched, Goals (+7 more)

### Community 50 - "L46"
Cohesion: 0.44
Nodes (9): DataPointResponse, DataSourceResponse, fetch_data_sources(), get_data_by_source(), list_data_sources(), list_data_sources_legacy(), PaginatedDataPoints, AsyncSession (+1 more)

### Community 51 - "L1"
Cohesion: 0.30
Nodes (11): OpenAQBackfill, Paginated historical fetch of OpenAQ sensor measurements., Any, _build_client(), _fast_limiter(), _location(), _measurement(), Mirror OpenAQ v3's actual /measurements response shape.      The timestamp lives (+3 more)

### Community 52 - "L20"
Cohesion: 0.06
Nodes (29): configure_logging(), _ExtraFormatter, Process-level logging for the collector entry points.  The scheduled runs redire, Append ``extra={...}`` fields as key=value pairs.      The collectors put their, collector_names(), create_collector(), create_collectors(), get_collector_class() (+21 more)

### Community 53 - "L363"
Cohesion: 0.10
Nodes (23): _angular_diff(), _bearing_deg(), _claimed_coordinates(), _claimed_from_bearing(), Meteorological 'from' bearing (deg, 0=N, 90=E) for wind components.      ``u`` i, Smallest absolute difference between two bearings, in [0, 180]., The wind 'from' bearing a transport claim implies, or None.      Handles wind-so, Score a wind-transport claim against GFS 10 m wind + OpenWeather direction. (+15 more)

### Community 54 - "L1"
Cohesion: 0.12
Nodes (15): AERIS Windows Collector — Setup, Step 0 — Get the code onto the Acer (~10 min), Step 10 — Monitoring from vacation (optional, ~5 min), Step 1 — Install Miniforge (~10 min), Step 2 — Build the Python environment (~20 min), Step 3 — Point the config at SQLite (~5 min), Step 4 — Create the database (~3 min), Step 5 — Test a real collection run (~5 min) (+7 more)

### Community 55 - "L17"
Cohesion: 0.26
Nodes (3): GPTClient, _client_with(), TestGPTClient

### Community 56 - "L360"
Cohesion: 0.15
Nodes (12): Commit Point, Context, Cycle Selection, Data Source & Transport, Design Decisions, Files Touched, Goals, Metrics (+4 more)

### Community 57 - "L15"
Cohesion: 0.11
Nodes (33): Anomaly, Base, Claim, EnrichmentRecord, ExpertLabel, Explanation, DeclarativeBase, TestEnrichmentConfig (+25 more)

### Community 58 - "Alembic env"
Cohesion: 0.18
Nodes (6): Settings, app.db.models.Base, BaseSettings, alembic.env.run_migrations_offline, alembic.env.run_migrations_online, test_settings_normalizes_plain_postgres_url_for_async_engine()

### Community 59 - "L14"
Cohesion: 0.40
Nodes (4): datetime, api_client(), seed_data_point(), TestDataPointRoutes

### Community 60 - "L49"
Cohesion: 0.17
Nodes (11): AERIS — Claude Code Instructions, Architecture Map, graphify, Non-Obvious Coding Rules, Output Format, Project Purpose, Repo-Specifics, Run / Build / Test Commands (+3 more)

### Community 61 - "L121"
Cohesion: 0.17
Nodes (11): API Contract, Commit Point, Context, Design Decisions, Files Touched, `GET /api/data`, `GET /api/data/{source}`, `GET /api/data/sources` (+3 more)

### Community 62 - "L99"
Cohesion: 0.42
Nodes (5): persist_anomalies(), Persist consensus anomalies as Anomaly rows; skip duplicates.      Idempotent na, _make_consensus_anomaly(), TestPersistAnomalies, ConsensusAnomaly

### Community 63 - "Table"
Cohesion: 0.31
Nodes (9): b7c11efca53b.upgrade, d2b9f4a7c1e8.upgrade, anomalies, claims, data_points, data_sources, enrichment_records, expert_labels (+1 more)

### Community 64 - "L12"
Cohesion: 0.39
Nodes (6): Config, _alembic_config(), test_downgrade_base_leaves_no_app_tables(), test_upgrade_head_creates_claims_fk_to_explanations(), test_upgrade_head_creates_enrichment_fk_to_anomalies(), test_upgrade_head_creates_expected_table()

### Community 65 - "Detection CLI"
Cohesion: 0.17
Nodes (11): Authentication, Commit Point, Context, Design Decisions, Extraction, Files Touched, Goals, Open Questions (+3 more)

### Community 66 - "L138"
Cohesion: 0.18
Nodes (10): API Shape, Commit Point, Context, Design Decisions, Files Touched, Goals, Metrics, OpenWeather Collector Design (+2 more)

### Community 67 - "L102"
Cohesion: 0.18
Nodes (10): Addendum — Bracco feedback, 2026-06-10, Addendum — implementation fixes, 2026-06-12, AERIS Corroboration Scorer — Design Memo, Build structure, Claim taxonomy, Open questions for the June 2 Bracco meeting, Phase 1 / Phase 2 sequencing (added 2026-05-26), Purpose (+2 more)

### Community 68 - "L23"
Cohesion: 0.06
Nodes (33): AERIS Codebase Review — Prioritized Findings, Scope Assessment & Remediation Roadmap, Context, Cross-cutting themes (fixing the pattern beats fixing instances), Eval-store provenance is undefined and the store is empty for the eval window (2026-06-13, verified), Execution plan (2026-06-13) — phased, surgical, P0.1 — `validate.py` grounding gate: unitless claim numbers ground against any context number **[verified, high]**, P0.2 — Corroboration scorer ignores "under" thresholds → inverted semantics on a headline type **[verified, high]**, P0.3 — Eval freeze under-merges events (first-match, not transitive single-linkage) **[verified, high]** (+25 more)

### Community 69 - "Data routes"
Cohesion: 0.07
Nodes (38): ArgumentParser, BaseCollector, CollectionResult, DataPointCreate, Bulk insert normalized data points, ignoring duplicate observations., Update the DataSource record with collection status., Normalized data point schema. All collectors output this format., Result of a single collection run. (+30 more)

### Community 70 - "L94"
Cohesion: 0.18
Nodes (10): Addendum 2026-06-12, AERIS Month 2 — Rebaseline, Bracco re-anchor (next email), Revised gates, Revised June deliverable, Revised timeline, Risks added by the extension, Scope changes (+2 more)

### Community 71 - "L102"
Cohesion: 0.20
Nodes (9): Collector Registry + Manual Runner, Commit Point, Context, Design Decisions, Files Touched, Goals, Registry Contract, Runner Contract (+1 more)

### Community 72 - "L191"
Cohesion: 0.40
Nodes (4): Consequences for the eval design (already applied in code), Data pipeline repair — findings and checklist, Repair checklist (ordered), What the audit found

### Community 75 - "Db"
Cohesion: 0.67
Nodes (3): app.db.schema.create_tables, app.db.session.engine, init_db.main

### Community 82 - "Community 82"
Cohesion: 0.42
Nodes (3): filter_and_mean(), Mean of pixel values that survive QA + bbox + finite-value masking., TestFilterAndMean

### Community 83 - "Community 83"
Cohesion: 0.27
Nodes (4): _make_anomaly, _make_anomaly(), TestAnomalyModel, TestEnrichmentRecordModel

### Community 88 - "Community 88"
Cohesion: 0.20
Nodes (10): _claimed_pbl_height(), _nearest(), Draft tolerances for atmospheric_trap (pending Dr. Bracco)., Score a PBL / inversion trapping claim.      Inversion = air aloft (GFS 850 hPa), score_atmospheric_trap(), TrapTolerance, test_trap_inversion_contradicted_when_surface_warmer(), test_trap_inversion_supported_when_aloft_warmer_than_surface() (+2 more)

### Community 89 - "Collectors"
Cohesion: 0.17
Nodes (8): BaseModel, Call the model once and return raw text plus token counts.          ``schema`` i, Generate structured output validated against schema, retrying on parse failure., _Attribution, _Attribution, _Attribution, _client_with(), TestOllamaClient

### Community 92 - "Community 92"
Cohesion: 0.18
Nodes (6): NOAAGFSCollector, collector(), Parses a real NOMADS GRIB-filter subset captured for the Houston bbox., TestLoadCycle, TestNOAAGFSFetch, TestParseGrib

### Community 93 - "Community 93"
Cohesion: 0.43
Nodes (3): filter_params(), Build the NOMADS GRIB-filter query for a cycle's f000 analysis over the target b, TestFilterParams

### Community 95 - "Community 95"
Cohesion: 0.47
Nodes (3): cycle_candidates(), GFS cycle times to try, newest first, walking back up to CYCLE_FALLBACK cycles., TestCycleCandidates

### Community 99 - "Community 99"
Cohesion: 0.47
Nodes (3): fetch_access_token(), Exchange CDSE password credentials for a short-lived bearer token., TestFetchAccessToken

### Community 104 - "Community 104"
Cohesion: 0.50
Nodes (3): from_gfs_longitude(), Convert GFS 0..360 longitude back to WGS84 (-180..180)., TestFromGfsLongitude

### Community 106 - "Community 106"
Cohesion: 0.18
Nodes (12): enrich_anomaly(), enrich_pending_anomalies(), EnrichmentConfig, persist_enrichment(), Build the EnrichmentRecord for one anomaly. Does not persist it., Insert an EnrichmentRecord unless the anomaly already has one.      Insert-only, Enrich every anomaly that has no EnrichmentRecord yet.      Each record is commi, Window geometry for one enrichment pass.      ``hours_before``/``hours_after`` a (+4 more)

### Community 107 - "Community 107"
Cohesion: 0.23
Nodes (8): context_window(), ``(start, end)`` of the context window centred on ``timestamp``., _load(), load_context_points anchored on the anomaly used across these tests., _seed(), TestContextWindow, TestLoadContextPoints, datetime

### Community 108 - "Community 108"
Cohesion: 0.20
Nodes (7): Cross-source enrichment smoke tests.  Exercises the enrichment pipeline end-to-e, DataPoints for one (source, metric, entity) stream.      ``samples`` is a list o, Seed a realistic four-source PM2.5 event and return the anomaly row., _seed(), _seed_houston_scene(), _series(), TestCrossSourceEnrichmentSmoke

### Community 109 - "Community 109"
Cohesion: 0.33
Nodes (5): group_points_by_series(), GroupKey, Identifies one time-series: a single station / granule / grid cell., Group DataPoints into (source, metric, source_entity_id) time-series.      Drops, TestGroupPointsBySeries

### Community 110 - "Community 110"
Cohesion: 0.24
Nodes (3): MonkeyPatch, TestCollectSkipsStoredColumns, TestSentinel5PFetch

### Community 111 - "Community 111"
Cohesion: 0.25
Nodes (3): EnrichmentSummary, Outcome of an :func:`enrich_pending_anomalies` pass., TestCLI

### Community 112 - "Community 112"
Cohesion: 0.47
Nodes (3): anomaly_bounding_box(), Lat/lon box enclosing ``radius_km`` around a point.      A cheap pre-filter for, TestAnomalyBoundingBox

### Community 126 - "Community 126"
Cohesion: 0.28
Nodes (5): _is_sqlite(), run_migrations_offline(), run_migrations_online(), get_session(), AsyncSession

## Knowledge Gaps
- **274 isolated node(s):** `AsyncSession`, `LogRecord`, `Clock`, `AsyncClient`, `TypeEngine` (+269 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DataPoint` connect `Rate limiter` to `L1`, `L106`, `L1`, `L92`, `L1`, `L178`, `L46`, `Historical backfill`, `L1`, `L49`, `L296`, `L1`, `L41`, `L158`, `L90`, `L134`, `L46`, `L1`, `L15`, `L14`, `L99`, `Data routes`, `Community 82`, `Community 99`, `Community 106`, `Community 107`, `Community 108`, `Community 109`, `Community 110`, `Community 111`, `Community 112`?**
  _High betweenness centrality (0.230) - this node is a cross-community bridge._
- **Why does `Anomaly` connect `L15` to `L1`, `Enrichment`, `L1`, `L1`, `L1`, `L46`, `L1`, `L70`, `L49`, `L296`, `L1`, `L158`, `L134`, `L99`, `Community 83`, `Community 106`, `Community 107`, `Community 108`, `Community 109`, `Community 111`, `Community 112`?**
  _High betweenness centrality (0.148) - this node is a cross-community bridge._
- **Why does `_claim_row()` connect `L70` to `L1`, `L1`, `L1`?**
  _High betweenness centrality (0.097) - this node is a cross-community bridge._
- **Are the 140 inferred relationships involving `DataPoint` (e.g. with `BackfillResult` and `BackfillStrategy`) actually correct?**
  _`DataPoint` has 140 INFERRED edges - model-reasoned connections that need verification._
- **Are the 97 inferred relationships involving `Anomaly` (e.g. with `EnrichmentConfig` and `EnrichmentLine`) actually correct?**
  _`Anomaly` has 97 INFERRED edges - model-reasoned connections that need verification._
- **Are the 73 inferred relationships involving `EnrichmentRecord` (e.g. with `EnrichmentConfig` and `EnrichmentLine`) actually correct?**
  _`EnrichmentRecord` has 73 INFERRED edges - model-reasoned connections that need verification._
- **Are the 64 inferred relationships involving `ConsensusAnomaly` (e.g. with `IsolationForestAnomaly` and `STLAnomaly`) actually correct?**
  _`ConsensusAnomaly` has 64 INFERRED edges - model-reasoned connections that need verification._