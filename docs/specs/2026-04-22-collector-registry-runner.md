# Collector Registry + Manual Runner

**Date**: 2026-04-22
**Phase**: Month 1 Week 2 closeout / Week 4 orchestration prep
**Status**: Implemented in initial form on 2026-04-22

## Context

By the end of Week 2, AERIS has five collector modules:

- EPA AirNow
- OpenAQ
- PurpleAir
- OpenWeather
- NASA FIRMS

Each collector can be imported and tested independently, but there is no single registry or manual command to run the current pipeline. Week 4 calls for scheduled orchestration, but the scheduler should not own collector discovery. A small registry and CLI runner gives developers a stable manual workflow now and gives APScheduler a clean integration point later.

## Goals

1. Add a central registry for implemented collectors.
2. Provide lookup helpers for one source or all sources.
3. Add a manual runner:
   - `python -m app.collectors.run_all`
   - `python -m app.collectors.run_all --source=purpleair`
   - `python -m app.collectors.run_all --max-retries=1`
4. Keep per-source failure isolation when running all collectors.
5. Return a non-zero process exit code if any collector fails.
6. Keep APScheduler out of this slice.

## Registry Contract

`server/app/collectors/registry.py` owns:

- `COLLECTOR_REGISTRY`
- `collector_names()`
- `get_collector_class(source)`
- `create_collector(source)`
- `create_collectors(source=None)`

The registry uses collector `source_name` values as keys:

| Key | Collector |
|-----|-----------|
| `epa_airnow` | `EPAAirNowCollector` |
| `openaq` | `OpenAQCollector` |
| `purpleair` | `PurpleAirCollector` |
| `openweather` | `OpenWeatherCollector` |
| `nasa_firms` | `NASAFIRMSCollector` |

## Runner Contract

`server/app/collectors/run_all.py` owns:

- argument parsing
- database session creation
- sequential collector execution
- collector cleanup via `close()`
- plain text summary output
- process exit code

Per-collector `BaseCollector.collect()` still owns fetch/normalize/store retry logic. The runner only isolates one collector's final failure from the rest of the run.

## Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Sequential execution | Five current collectors are cheap enough to run serially; easier logs and lower API burst risk. |
| D2 | Source names, not aliases | Explicit `source_name` keys avoid ambiguity. Short aliases can be added later if useful. |
| D3 | Non-zero exit when any collector fails | Shell/cron/systemd wrappers need a simple failure signal. |
| D4 | No scheduler yet | APScheduler belongs in Week 4 once all current collectors can be run manually and reliably. |
| D5 | Registry owns imports | Future orchestration should not duplicate imports or source names. |

## Files Touched

| File | Action |
|------|--------|
| `server/app/collectors/registry.py` | new collector registry |
| `server/app/collectors/run_all.py` | new manual CLI runner |
| `server/tests/unit/test_collector_registry.py` | registry tests |
| `server/tests/unit/test_run_all.py` | runner helper tests |
| `docs/specs/2026-04-05-month1-phase-plan.md` | note manual runner foundation |
| `session_summary.md` | handoff update |

## Verification

- `venv/bin/pytest`

## Commit Point

`feat(collectors): add registry and manual runner`
