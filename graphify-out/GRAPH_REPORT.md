# Graph Report - aeris  (2026-06-13)

## Corpus Check
- 118 files · ~93,636 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1945 nodes · 5211 edges · 127 communities (97 shown, 30 thin omitted)
- Extraction: 65% EXTRACTED · 35% INFERRED · 0% AMBIGUOUS · INFERRED: 1813 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `2e683d4c`
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
- [[_COMMUNITY_Corroboration scorer|Corroboration scorer]]
- [[_COMMUNITY_Corroboration scorer|Corroboration scorer]]
- [[_COMMUNITY_Corroboration scorer|Corroboration scorer]]
- [[_COMMUNITY_L18|L18]]
- [[_COMMUNITY_Test|Test]]
- [[_COMMUNITY_Collectors|Collectors]]
- [[_COMMUNITY_Collectors|Collectors]]
- [[_COMMUNITY_Consensus tiers|Consensus tiers]]
- [[_COMMUNITY_Data routes|Data routes]]
- [[_COMMUNITY_Enrichment|Enrichment]]
- [[_COMMUNITY_Collector registry|Collector registry]]
- [[_COMMUNITY_Test|Test]]
- [[_COMMUNITY_Test|Test]]
- [[_COMMUNITY_Test|Test]]
- [[_COMMUNITY_Test|Test]]
- [[_COMMUNITY_Test|Test]]
- [[_COMMUNITY_Test|Test]]
- [[_COMMUNITY_Test|Test]]
- [[_COMMUNITY_Test|Test]]
- [[_COMMUNITY_Test|Test]]
- [[_COMMUNITY_Test|Test]]
- [[_COMMUNITY_Test|Test]]
- [[_COMMUNITY_Test|Test]]
- [[_COMMUNITY_Test|Test]]
- [[_COMMUNITY_Test|Test]]
- [[_COMMUNITY_Test|Test]]
- [[_COMMUNITY_Test|Test]]
- [[_COMMUNITY_Test|Test]]
- [[_COMMUNITY_Community 123|Community 123]]
- [[_COMMUNITY_Community 124|Community 124]]
- [[_COMMUNITY_Community 125|Community 125]]
- [[_COMMUNITY_Community 126|Community 126]]

## God Nodes (most connected - your core abstractions)
1. `DataPoint` - 146 edges
2. `Anomaly` - 112 edges
3. `EnrichmentRecord` - 80 edges
4. `ConsensusAnomaly` - 73 edges
5. `LLMClient` - 70 edges
6. `Sentinel5PCollector` - 65 edges
7. `AsyncRateLimiter` - 63 edges
8. `DataPointCreate` - 62 edges
9. `BaseCollector` - 59 edges
10. `NOAAGFSCollector` - 55 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `create_tables()`  [INFERRED]
  deploy/windows-collector/init_db.py → server/app/db/schema.py
- `OpenAQBackfill` --calls--> `target_bounding_box`  [EXTRACTED]
  server/app/collectors/backfill.py → app/collectors/geo.py
- `run_collectors` --calls--> `BaseCollector`  [EXTRACTED]
  app/collectors/run_all.py → server/app/collectors/base.py
- `load_context_points` --references--> `DataPoint`  [EXTRACTED]
  app/detection/enrichment.py → server/app/db/models.py
- `_series` --calls--> `DataPoint`  [EXTRACTED]
  tests/integration/test_enrichment_smoke.py → server/app/db/models.py

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

## Communities (127 total, 30 thin omitted)

### Community 0 - "L1"
Cohesion: 0.23
Nodes (6): merge(), Group detector outputs by timestamp and emit unified anomaly records.      All t, _if(), _merge_kwargs(), _stl(), _z()

### Community 1 - "L1"
Cohesion: 0.12
Nodes (40): merge, ConsensusAnomaly, A merged anomaly record across detectors at a single (timestamp, location)., DetectionEngine, Detection orchestrator.  Runs the configured detectors (Z-score, STL, IsolationF, Orchestrates one or more detectors over a single time-series.      Detectors are, Standard 3-detector engine with each detector's library defaults., Run all configured detectors on ``series`` and merge via consensus.          ``s (+32 more)

### Community 2 - "Enrichment"
Cohesion: 0.07
Nodes (56): db_session, test_engine, Anomaly, Base, Claim, EnrichmentRecord, ExpertLabel, Explanation (+48 more)

### Community 3 - "L106"
Cohesion: 0.10
Nodes (32): _amain(), available_strategies(), BackfillResult, _days_between(), db_location_ids(), _format_result(), _gfs_cycles_in_range(), _iso_z() (+24 more)

### Community 4 - "L66"
Cohesion: 0.09
Nodes (25): BaseCollector, CollectorClass, DataPointCreate, Normalized data point schema. All collectors output this format., NOAAGFSCollector, location_within_target_radius(), normalize_openaq_unit(), OpenAQCollector (+17 more)

### Community 5 - "L1"
Cohesion: 0.10
Nodes (16): BoundingBox, Return min lon, min lat, max lon, max lat., filter_and_mean(), Mean of pixel values that survive QA + bbox + finite-value masking., MonkeyPatch, Any, Sentinel5PCollector, collector() (+8 more)

### Community 6 - "Rate limiter"
Cohesion: 0.12
Nodes (34): BackfillStrategy, NOAAGFSBackfill, OpenAQBackfill, OpenWeatherBackfill, Per-source historical loader interface., Paginated historical fetch of OpenAQ sensor measurements., Walk the Sentinel-5P catalog backward in 48h windows, extracting columns.      C, Walk past GFS cycles within the NOMADS retention window. (+26 more)

### Community 7 - "L92"
Cohesion: 0.08
Nodes (24): area_polygon(), attribute_value(), extract_product_code(), fetch_access_token(), granule_download_url(), ids_with_stored_columns(), _load_granule_arrays(), odata_filter() (+16 more)

### Community 8 - "L1"
Cohesion: 0.09
Nodes (34): _amain(), _ensure_utc(), fixture_payload(), _format_result(), freeze_eval_set(), FreezeResult, group_events(), load_window_anomalies() (+26 more)

### Community 9 - "L1"
Cohesion: 0.11
Nodes (23): Enum, ClaimType, The 10-type claim taxonomy, plus a routing fallback for claims the     rules don, extract_claim_drafts(), Flatten the per-step claims into ordered ClaimDrafts.      step_index is 1-based, build_step_prompt(), ReasoningStep, ReasoningStepResponse (+15 more)

### Community 10 - "L1"
Cohesion: 0.16
Nodes (21): The ANOMALY block of the prompt: what was detected, where, how unusual., The DATA CONTEXT block of the prompt, from the enrichment summary.      Also the, station means: AAA (2.1 km) 18.2, ...' or None for single entities., render_anomaly_text(), render_enrichment_text(), _render_station_means(), _anomaly(), Explanation orchestrator: chain -> claims -> Phase 1 gate -> Phase 2 -> rows.  U (+13 more)

### Community 11 - "L178"
Cohesion: 0.09
Nodes (48): parse_archive_csv, settings, target_bounding_box, within_target_radius, GroundingResult, _amain(), _claim_row(), _format_explanation() (+40 more)

### Community 12 - "L35"
Cohesion: 0.12
Nodes (18): extract_precipitation(), OpenWeatherCollector, parse_observation_time(), Transform OpenWeather current weather responses into DataPoints., Fetch current weather for target-area grid points., weather_query_points(), WeatherQueryPoint, OpenWeatherCollector (+10 more)

### Community 13 - "L1"
Cohesion: 0.09
Nodes (34): _anomaly_ts(), _blank_match(), _claim_trend_window(), ConcentrationTolerance, _local_day_slice(), _metric_block(), _pooled_series(), _pre_anomaly_baseline() (+26 more)

### Community 14 - "L1"
Cohesion: 0.09
Nodes (31): _claimed_from_bearing(), classify_claim(), Rule-based routing of a claim into taxonomy types, most verifiable first.      A, One claim routed and scored: everything Phase 2 contributes to a Claim row., Classify a claim and run its primary type's scorer.      Unclassified claims agg, The wind 'from' bearing a transport claim implies, or None.      Handles wind-so, {'hcho': 'up', 'ozone': 'down', ...} for adjective-species mentions., score_claim() (+23 more)

### Community 15 - "L1"
Cohesion: 0.08
Nodes (24): _point_value(), The numeric threshold in an 'exceeded N' style claim, else None.      Clock time, The numeric value in a 'was N' style point claim, else None.      Threshold-word, _threshold_value(), check_grounding(), _match_numbers(), _quantities(), (value, unit, character offset) for every quantity in the text. (+16 more)

### Community 16 - "L1"
Cohesion: 0.11
Nodes (33): _amain(), _cell_exists(), _format_summaries(), load_anomaly_set(), main(), ModelSweepSummary, _parse_args(), Sweep the eval anomaly set across all comparison models.  One cell = one (anomal (+25 more)

### Community 17 - "L554"
Cohesion: 0.12
Nodes (33): BackgroundTolerance, _earliest_keyword(), Draft thresholds for background_vs_event (pending Dr. Bracco).      min_stations, Score a regional-vs-local claim by spatial CV across OpenAQ stations., Per-station mean of in-window values, for stations with enough coverage., Coefficient of variation across station means; None when undefined., The group whose keyword appears first in the text; None if none match.      Firs, Draft thresholds for emissions_source_type (pending Dr. Bracco). (+25 more)

### Community 18 - "L46"
Cohesion: 0.11
Nodes (24): build_aux_inputs(), group_points_by_series(), GroupKey, _load_gfs_wind_cells(), Identifies one time-series: a single station / granule / grid cell., Group DataPoints into (source, metric, source_entity_id) time-series.      Drops, Assemble IsolationForest inputs by joining wind speed + pbl_height aux.      Win, Derive 10-m wind speed per grid cell, nearest cell to the station first.      GF (+16 more)

### Community 19 - "L1"
Cohesion: 0.13
Nodes (27): _amain(), _ask_verdict(), collect_claim_groups(), main(), _parse_args(), Interactive labeling tool for the expert evaluation set.  Loads one anomaly's co, Unique claim texts for an anomaly, in (model, step) first-seen order., One verdict letter, re-prompting until valid; None means quit. (+19 more)

### Community 20 - "L24"
Cohesion: 0.12
Nodes (15): Clock, _defer_if_budget_spent(), _header_seconds(), rate_limited_get(), Push the next acquire at least ``seconds`` into the future.          Used when a, GET through the limiter, honoring Retry-After on 429 responses., _retry_wait_seconds(), AsyncClient (+7 more)

### Community 21 - "L70"
Cohesion: 0.05
Nodes (40): BaseModel, make_client, GeminiClient._complete, GeminiClient._retry_delay, GPTClient._complete, _strict_json_schema, LLMClient, Raw model output plus usage metadata from a single model call. (+32 more)

### Community 22 - "L57"
Cohesion: 0.13
Nodes (22): aggregate_verdicts(), CorroborationResult, low_corroboration_flag(), Aggregated cross-source verdict for one claim.      ``corroboration_score`` is `, Score a wind-transport claim against GFS 10 m wind + OpenWeather direction., Collapse per-source verdicts into a scalar score and evidence count.      ``scor, Phase 2 metadata flag: strongly contradicted across >= 2 sources.      Not a gat, score_transport_direction() (+14 more)

### Community 23 - "Historical backfill"
Cohesion: 0.11
Nodes (23): _store_points, archive_key(), OpenAQArchiveBackfill, parse_archive_csv(), Map archive CSV rows to normalized DataPoints.      Columns: location_id, sensor, Bulk history from the OpenAQ S3 data archive instead of the API., AsyncRateLimiter, Spaces requests evenly so a shared API key stays under its rate limit.      One (+15 more)

### Community 24 - "L1"
Cohesion: 0.20
Nodes (20): _amain(), build_cross_source_summary(), _ensure_utc(), _Entry, _format_summary(), _iso(), main(), _nearest_in_time() (+12 more)

### Community 25 - "L49"
Cohesion: 0.25
Nodes (6): _anomaly(), _dp(), build_cross_source_summary with the window derived from the anomaly., _summary(), TestBuildCrossSourceSummary, DataPoint

### Community 26 - "L289"
Cohesion: 0.18
Nodes (12): enrich_anomaly(), enrich_pending_anomalies(), EnrichmentConfig, persist_enrichment(), Build the EnrichmentRecord for one anomaly. Does not persist it., Insert an EnrichmentRecord unless the anomaly already has one.      Insert-only, Enrich every anomaly that has no EnrichmentRecord yet.      Each record is commi, Window geometry for one enrichment pass.      ``hours_before``/``hours_after`` a (+4 more)

### Community 27 - "L1"
Cohesion: 0.31
Nodes (5): RawCompletion, _completion(), MockLLMClient, Concrete LLMClient returning scripted raw completions, for testing., TestGenerate

### Community 28 - "L22"
Cohesion: 0.18
Nodes (15): GenerationResult, A structured generation: the parsed schema instance plus call metadata., Run the 4-step reasoning chain, threading each step's summary forward., ReasoningChainResult, run_reasoning_chain(), StepResult, _sum_optional(), run_reasoning_chain (+7 more)

### Community 29 - "L30"
Cohesion: 0.22
Nodes (11): clear_locations_cache(), OpenAQCollector, Any, collector(), fast_limiter(), _fresh_locations_cache(), make_location(), make_raw() (+3 more)

### Community 30 - "L296"
Cohesion: 0.18
Nodes (10): _nearest_cell_value(), _nearest_value(), The time-nearest value from the spatially nearest cell that has one., Return the value at the row whose timestamp is closest to ``target``,     provid, Samples per diurnal cycle from the observed cadence, or None to skip STL., _stl_period_for(), TestNearestCellValue, TestNearestValue (+2 more)

### Community 31 - "L608"
Cohesion: 0.09
Nodes (35): ChemistryTolerance, _claimed_pbl_height(), _combine(), _direction_verdict(), _nearest(), Score a 'pollutant was elevated' claim against OpenAQ + Sentinel-5P.      Three, Per-source roll-up across aspects: contradiction dominates, else support., Score a PBL / inversion trapping claim.      Inversion = air aloft (GFS 850 hPa) (+27 more)

### Community 32 - "L22"
Cohesion: 0.23
Nodes (6): Any, MockCollector, Concrete implementation for testing the abstract BaseCollector., TestMockCollectorCollect, TestMockCollectorFetch, TestMockCollectorNormalize

### Community 33 - "L1"
Cohesion: 0.17
Nodes (22): _amain(), _engine_for(), _ensure_utc(), _format_summary(), _load_aux_cells(), _load_gfs_component(), load_points(), main() (+14 more)

### Community 34 - "L17"
Cohesion: 0.17
Nodes (7): _hourly_inputs(), TestIFDetectorBasics, TestIFDetectorConfiguration, TestIFDetectorContract, TestIFDetectorMultivariate, TestIFDetectorUnivariate, IsolationForestInput

### Community 35 - "L13"
Cohesion: 0.13
Nodes (7): Pure sinusoidal series, evenly-spaced hourly, no noise., _seasonal_series(), TestSTLDetectorBasics, TestSTLDetectorConfiguration, TestSTLDetectorContract, TestSTLDetectorOutliers, datetime

### Community 36 - "L451"
Cohesion: 0.14
Nodes (14): _claimed_temperature(), _gfs_wind_components(), Draft tolerances for the wind / met headline types (pending Dr. Bracco)., ('value', x) for a numeric speed, ('low', None) for stagnation, else None., A surface-temperature value in C named in the claim, else None., Score a 'stagnant / hot' state claim against GFS wind + OpenWeather.      Wind s, score_meteorological_state(), _wind_intent() (+6 more)

### Community 37 - "L26"
Cohesion: 0.26
Nodes (6): NOAAGFSCollector, Any, make_grid_cell(), make_raw(), TestNOAAGFSFetch, TestNOAAGFSNormalize

### Community 38 - "L41"
Cohesion: 0.15
Nodes (10): GUID, UUID column stored in a form SQLite cannot misread as a number.      Postgres ke, Dialect, Any, UUID, TypeEngine, Document the corruption mode the dashed format defends against.      The product, TestGUIDBindAndResult (+2 more)

### Community 40 - "Detection"
Cohesion: 0.14
Nodes (12): enrich_anomaly, enrich_pending_anomalies, Cross-source enrichment smoke tests.  Exercises the enrichment pipeline end-to-e, DataPoints for one (source, metric, entity) stream.      ``samples`` is a list o, Seed a realistic four-source PM2.5 event and return the anomaly row., _seed(), _seed_houston_scene(), _series() (+4 more)

### Community 41 - "L90"
Cohesion: 0.23
Nodes (8): context_window(), ``(start, end)`` of the context window centred on ``timestamp``., _load(), load_context_points anchored on the anomaly used across these tests., _seed(), TestContextWindow, TestLoadContextPoints, datetime

### Community 42 - "L1"
Cohesion: 0.19
Nodes (8): lifespan(), get_session(), FastAPI, health_check(), HealthResponse, AsyncSession, AsyncSession, health_check

### Community 43 - "L1"
Cohesion: 0.15
Nodes (14): AsyncConnection, AsyncEngine, configure_postgres_compatibility(), configure_timescale(), create_tables(), drop_tables(), Create all tables and enable TimescaleDB hypertable on data_points., Keep existing development databases compatible with the current ORM. (+6 more)

### Community 44 - "L134"
Cohesion: 0.17
Nodes (9): GFSVariable, _match_variable(), _message_matches(), Download the freshest reachable GFS analysis cycle from the NOMADS GRIB filter., Fetch and parse one GFS cycle; empty list means the cycle is unavailable., GET one cycle's GRIB subset; None if the cycle is not published yet., Extract one (lat, lon, values) cell per grid point from a GRIB2 subset., Any (+1 more)

### Community 45 - "L1"
Cohesion: 0.20
Nodes (8): collector_names(), create_collector(), create_collectors(), get_collector_class(), source_choices(), create_collector, BaseCollector, TestCollectorRegistry

### Community 46 - "L22"
Cohesion: 0.23
Nodes (7): _hourly_series(), _iso_inputs(), _run_kwargs(), _seasonal(), TestEngineConsensusEndToEnd, TestEngineEmptyAndStable, timedelta

### Community 47 - "L1"
Cohesion: 0.19
Nodes (10): _build_series_with_events(), _houston_pm25_baseline(), Multi-event integration tests for the detection engine.  Builds synthetic Housto, Synthetic Houston PM2.5 baseline.      Diurnal cycle (peak around morning rush h, Typical wind / PBL for a Houston spring day., Build (univariate series, IF inputs) with injected events.      ``events`` maps, TestKnownEventsReproducibility, TestKnownHoustonEvents (+2 more)

### Community 48 - "L17"
Cohesion: 0.23
Nodes (4): GeminiClient, _client_with(), _no_sleep(), TestGeminiClient

### Community 49 - "L45"
Cohesion: 0.26
Nodes (24): IsolationForestAnomaly, STLAnomaly, TestMergeDistinctTimestamps, TestMergeEmptyInputs, TestMergeFieldPriority, TestMergeOrdering, TestMergePreservesDetails, TestMergeSingleMethod (+16 more)

### Community 50 - "L46"
Cohesion: 0.18
Nodes (18): fetch_data_sources, get_data_by_source, list_data_sources, list_data_sources_legacy, DataSource, DataPointResponse, DataSourceResponse, fetch_data_sources() (+10 more)

### Community 51 - "L1"
Cohesion: 0.28
Nodes (6): _make_anomaly(), _make_claim(), _make_explanation(), TestClaimModel, TestExpertLabelModel, TestExplanationModel

### Community 52 - "L20"
Cohesion: 0.12
Nodes (17): ArgumentParser, async_main(), build_parser(), exit_code(), format_result(), main(), missing_credentials(), Map source name -> unset settings fields it needs. (+9 more)

### Community 53 - "L363"
Cohesion: 0.18
Nodes (14): _angular_diff(), _bearing_deg(), _claimed_coordinates(), Meteorological 'from' bearing (deg, 0=N, 90=E) for wind components.      ``u`` i, Smallest absolute difference between two bearings, in [0, 180]., Initial bearing from one point to another (equirectangular, fine at 50 km)., Direction-only check: does the wind blow from the claimed source toward     the, score_point_source_attribution() (+6 more)

### Community 54 - "L1"
Cohesion: 0.21
Nodes (8): configure_logging(), _ExtraFormatter, Process-level logging for the collector entry points.  The scheduled runs redire, Append ``extra={...}`` fields as key=value pairs.      The collectors put their, LogRecord, LogRecord, _record(), TestExtraFormatter

### Community 55 - "L17"
Cohesion: 0.26
Nodes (3): GPTClient, _client_with(), TestGPTClient

### Community 56 - "L360"
Cohesion: 0.25
Nodes (3): EnrichmentSummary, Outcome of an :func:`enrich_pending_anomalies` pass., TestCLI

### Community 57 - "L15"
Cohesion: 0.27
Nodes (4): _make_anomaly, _make_anomaly(), TestAnomalyModel, TestEnrichmentRecordModel

### Community 58 - "Alembic env"
Cohesion: 0.18
Nodes (6): Settings, app.db.models.Base, BaseSettings, alembic.env.run_migrations_offline, alembic.env.run_migrations_online, test_settings_normalizes_plain_postgres_url_for_async_engine()

### Community 59 - "L14"
Cohesion: 0.35
Nodes (5): AsyncClient, datetime, api_client(), seed_data_point(), TestDataPointRoutes

### Community 60 - "L49"
Cohesion: 0.13
Nodes (9): Build an evenly-spaced (timestamp, value) series anchored at T0., _series(), TestZScoreArithmetic, TestZScoreDetectorBasics, TestZScoreDetectorConfiguration, TestZScoreDetectorLookbackOnly, TestZScoreDetectorOutliers, datetime (+1 more)

### Community 61 - "L121"
Cohesion: 0.50
Nodes (3): from_gfs_longitude(), Convert GFS 0..360 longitude back to WGS84 (-180..180)., TestFromGfsLongitude

### Community 62 - "L99"
Cohesion: 0.39
Nodes (4): _make_consensus_anomaly(), TestPersistAnomalies, persist_anomalies, ConsensusAnomaly

### Community 63 - "Table"
Cohesion: 0.31
Nodes (9): b7c11efca53b.upgrade, d2b9f4a7c1e8.upgrade, anomalies, claims, data_points, data_sources, enrichment_records, expert_labels (+1 more)

### Community 64 - "L12"
Cohesion: 0.39
Nodes (6): Config, _alembic_config(), test_downgrade_base_leaves_no_app_tables(), test_upgrade_head_creates_claims_fk_to_explanations(), test_upgrade_head_creates_enrichment_fk_to_anomalies(), test_upgrade_head_creates_expected_table()

### Community 65 - "Detection CLI"
Cohesion: 0.33
Nodes (6): configure_logging, create_collectors, async_main, missing_credentials, run_collectors, warn_missing_credentials

### Community 67 - "L102"
Cohesion: 0.43
Nodes (3): filter_params(), Build the NOMADS GRIB-filter query for a cycle's f000 analysis over the target b, TestFilterParams

### Community 68 - "L23"
Cohesion: 0.08
Nodes (24): AERIS Codebase Review — Prioritized Findings, Scope Assessment & Remediation Roadmap, Context, Cross-cutting themes (fixing the pattern beats fixing instances), Eval-store provenance is undefined and the store is empty for the eval window (2026-06-13, verified), P0.1 — `validate.py` grounding gate: unitless claim numbers ground against any context number **[verified, high]**, P0.2 — Corroboration scorer ignores "under" thresholds → inverted semantics on a headline type **[verified, high]**, P0.3 — Eval freeze under-merges events (first-match, not transitive single-linkage) **[verified, high]**, P0.4 — STL silently disabled for sub-hourly series; "severe" tier unreachable **[verified, downgraded high→medium]** (+16 more)

### Community 69 - "Data routes"
Cohesion: 0.15
Nodes (11): ABC, BaseCollector, Bulk insert normalized data points, ignoring duplicate observations., Update the DataSource record with collection status., Abstract base class for all data source collectors.      Subclasses must impleme, Fetch raw data from the external API., Transform raw API response into normalized DataPointCreate records., Orchestrate fetch → normalize → store with retry and logging. (+3 more)

### Community 70 - "L94"
Cohesion: 0.47
Nodes (3): cycle_candidates(), GFS cycle times to try, newest first, walking back up to CYCLE_FALLBACK cycles., TestCycleCandidates

### Community 71 - "L102"
Cohesion: 0.47
Nodes (3): anomaly_bounding_box(), Lat/lon box enclosing ``radius_km`` around a point.      A cheap pre-filter for, TestAnomalyBoundingBox

### Community 72 - "L191"
Cohesion: 0.25
Nodes (3): collector(), TestFetchCycleGrib, TestLoadCycle

### Community 73 - "L126"
Cohesion: 0.38
Nodes (3): parse_cycle_time(), DataPointCreate, TestParseCycleTime

### Community 75 - "Db"
Cohesion: 0.67
Nodes (3): app.db.schema.create_tables, app.db.session.engine, init_db.main

### Community 76 - "Detection CLI"
Cohesion: 0.67
Nodes (3): build_aux_inputs, persist_anomalies, run_detection

### Community 77 - "DB schema"
Cohesion: 0.67
Nodes (3): configure_postgres_compatibility, configure_timescale, create_tables

### Community 85 - "L18"
Cohesion: 0.18
Nodes (12): _Bucket, _bucket_for(), _build_consensus_anomaly(), Consensus across the three detectors.  Each detector (Z-score, STL, IsolationFor, Internal mutable accumulator: detector outputs at a single timestamp., Pick the observed value and verify detectors agree.      All detectors share the, Z-score's expected_value wins over STL's; IF doesn't provide one., _resolve_expected_value() (+4 more)

### Community 88 - "Collectors"
Cohesion: 0.19
Nodes (7): CollectionResult, Result of a single collection run., DataPointCreate, Any, DataPointCreate, TestCollectionResult, TestDataPointCreate

### Community 89 - "Collectors"
Cohesion: 0.29
Nodes (4): OllamaClient, _Attribution, _client_with(), TestOllamaClient

### Community 123 - "Community 123"
Cohesion: 0.36
Nodes (3): Map the count of triggering methods to a severity tier., severity_for(), TestSeverityFor

### Community 124 - "Community 124"
Cohesion: 0.29
Nodes (7): distance_km(), offset_coordinate(), Great-circle distance between two WGS84 coordinates., Approximate coordinate offset for short distances., target_bounding_box(), within_target_radius(), group_events

### Community 126 - "Community 126"
Cohesion: 0.83
Nodes (3): _is_sqlite(), run_migrations_offline(), run_migrations_online()

## Knowledge Gaps
- **100 isolated node(s):** `Context`, `Source-independence audit — headline types verified thin (2026-06-13, code + DB)`, `Eval-store provenance is undefined and the store is empty for the eval window (2026-06-13, verified)`, `Scope assessment (feasibility · novelty · stability · focus)`, `Cross-cutting themes (fixing the pattern beats fixing instances)` (+95 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **30 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DataPoint` connect `Historical backfill` to `L1`, `Enrichment`, `L106`, `L66`, `L1`, `Rate limiter`, `L92`, `L1`, `L46`, `L1`, `L49`, `L289`, `L296`, `L22`, `L1`, `L41`, `L158`, `Detection`, `L90`, `L46`, `L360`, `L14`, `L99`, `Data routes`, `L102`, `Collectors`, `Community 125`?**
  _High betweenness centrality (0.237) - this node is a cross-community bridge._
- **Why does `Anomaly` connect `Enrichment` to `L1`, `L1`, `L1`, `L178`, `L1`, `L46`, `L1`, `L70`, `L1`, `L49`, `L289`, `L296`, `L1`, `L158`, `Detection`, `L90`, `L1`, `L360`, `L15`, `L99`, `L102`, `Detection CLI`, `Community 124`?**
  _High betweenness centrality (0.176) - this node is a cross-community bridge._
- **Why does `_claim_row()` connect `L178` to `L1`, `L57`?**
  _High betweenness centrality (0.123) - this node is a cross-community bridge._
- **Are the 138 inferred relationships involving `DataPoint` (e.g. with `BackfillResult` and `BackfillStrategy`) actually correct?**
  _`DataPoint` has 138 INFERRED edges - model-reasoned connections that need verification._
- **Are the 96 inferred relationships involving `Anomaly` (e.g. with `EnrichmentConfig` and `EnrichmentLine`) actually correct?**
  _`Anomaly` has 96 INFERRED edges - model-reasoned connections that need verification._
- **Are the 73 inferred relationships involving `EnrichmentRecord` (e.g. with `EnrichmentConfig` and `EnrichmentLine`) actually correct?**
  _`EnrichmentRecord` has 73 INFERRED edges - model-reasoned connections that need verification._
- **Are the 64 inferred relationships involving `ConsensusAnomaly` (e.g. with `IsolationForestAnomaly` and `STLAnomaly`) actually correct?**
  _`ConsensusAnomaly` has 64 INFERRED edges - model-reasoned connections that need verification._