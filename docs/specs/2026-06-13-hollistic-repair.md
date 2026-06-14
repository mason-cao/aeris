# AERIS Codebase Review — Prioritized Findings, Scope Assessment & Remediation Roadmap

## Context

Full read-only review of the AERIS backend (~19k LOC, 50 source + 45 test files) across correctness, stability, performance, edge cases, refactors, and project scope. Run as two multi-agent passes: pass 1 produced 103 findings across 13 subsystems with 5 verifiers completing before a spend-limit cutoff; pass 2 (after the limit reset) ran the 2 missing finders (`cross-cutting`, `tests-quality`), the 8 skipped verifiers, both scope evaluators, and a synthesis. I merged the two passes manually (the synthesis agent's loader for the 5 pass-1-verified subsystems silently failed, so its auto-ranked list omitted them — those findings are re-incorporated here) and personally read-verified the five most consequential highs.

**Net confidence:** the headline items below are agent-verified and/or read-verified. The review corrected itself in pass 2: 1 finding rejected (noted), several severities recalibrated. Items still tagged **[unconfirmed]** are credible with specific line refs but were not independently challenged — confirm as step one of implementing them.

**Read this first:** the most important conclusion is _not_ a code bug. It's a thesis-validity risk (next section). Fixing the P0 code bugs is necessary but not sufficient — the eval also needs data that can actually support the corroboration claim.

---

## ⚠️ Thesis-validity risks (existential — no code fix resolves these)

The scope evaluators independently converged on this, and it outranks every code finding:

> **Durable takeaway — persist to project memory (capture once out of plan mode):** AERIS's Month-2 corroboration thesis needs ≥2 _measurement-independent_ data channels per claim, but as of mid-June the only live sources are GFS + OpenWeather — both meteorological and largely NWP-derived, so _not_ independent (S5P columns never actually collected, zero ground NO₂/SO₂/CO). This is a standing constraint on **eval validity**, not a transient bug or footgun: until an independent chemistry channel (real S5P columns and/or ground sensors) is restored, the headline Phase-2 corroboration result is unsupportable as written and must be caveated. Relates to [[project_pipeline_repair]].

1. **Source-independence is broken right now.** The corroboration proxy assumes ≥2 _measurement-independent_ channels per claim. But Sentinel-5P column density was **never actually collected** (all ~283 S5P rows are catalog availability markers, not measurements), there is **zero ground NO₂/SO₂/CO** in Houston OpenAQ, and OpenAQ live collection has a multi-week hole. That leaves **GFS + OpenWeather as the only live pair — both meteorological, and OpenWeather largely derives from NWP**, so they are _not_ independent. As written, the headline Phase-2 result would rest on two correlated sources agreeing with each other. **This is the load-bearing assumption of the whole Month-2 thesis and it is currently unsupported.**
2. **`evidence_n` is likely 1 for a large fraction of claims** (S5P ≈1 granule-mean/species/day, GFS 6-hourly). A corroboration "score" computed from a single source is single-source agreement, not cross-source — weakening the framing regardless of independence.
3. **The "expert-labeled" framing depends on an unsecured commitment.** IRR (κ≥0.6) hinges on Dr. Bracco's July labeling availability, which isn't confirmed in writing (the re-anchor email still owes this).
4. **No schedule slack below the Jul 13 freeze**, and accumulation is gated on data that is broken as of mid-June.

**Highest-leverage actions (from the scope evaluators):**

- Treat the 2026-06-12 pipeline repair as a **hard blocker, not a parallel track**: restore CDSE (real S5P columns) + the OpenAQ key on the Acer and _verify non-zero satellite + chemistry rows before the Jul 1 accumulation window_. Re-scope every satellite/chemistry-dependent claim type to qualitative if columns stay near-zero (as already done for types 6/7/9).
- Add a **source-independence audit** to the analysis plan: report per-source-pair contribution to each verdict and collapse GFS+OpenWeather into a single meteorological channel where both derive the same NWP quantity.
- Define a **minimum-`evidence_n` bar for the headline correlation**: report the Spearman both unweighted and restricted to `evidence_n ≥ 2` as a named figure.
- **Lock the Bracco labeling commitment in writing now** (fold into the owed correction email): subset size, ~Jul 16 delivery / ~Jul 20 return, July availability; pre-agree the κ<0.6 "expert-audited" fallback.
- **Reconcile the README with reality** (S5P/OpenAQ "Live" status, drop the NO₂/SO₂/CO column-density and named-event-category promises) — a 15-minute edit that removes a credibility risk for a college/competition audience.
- Add a **liveness alarm on the Acer** (daily row-count delta → notification on zero growth) for the unattended Jul 1-12 window; the credential preflight only catches startup, not mid-window stalls.
- State the source-independence assumption as an explicit **limitation in the write-up** regardless of outcome — naming it pre-empts the strongest reviewer objection.

### Source-independence audit — headline types verified thin (2026-06-13, code + DB)

Risks 1–2 above were spot-checked against `corroboration.py` and the live Postgres store. They hold and are sharper than stated: **none of the three headline types currently has two measurement-independent channels.**

| #   | Headline type             | Sources (verified in code)                     | Live channels now        | Independent ≥2?                       |
| --- | ------------------------- | ---------------------------------------------- | ------------------------ | ------------------------------------- |
| 1   | `concentration_elevation` | OpenAQ + Sentinel-5P (`corroboration.py:258`)  | OpenAQ only — S5P absent | **No — single channel**               |
| 2   | `transport_direction`     | GFS + OpenWeather (`corroboration.py:424`)     | GFS + OpenWeather        | **No — both NWP-lineage, correlated** |
| 3   | `meteorological_state`    | GFS + OpenWeather (`corroboration.py:507-508`) | GFS + OpenWeather        | **No — same pair**                    |

Verified data reality (Postgres `aeris`, 2026-06-13):

- **Sentinel-5P: 21 rows, all `s5p_*_granule_available = 1.0` — zero column-density values.** The scorers query `s5p_hcho_column` / resolved sentinel metrics that do not exist in the store, so every S5P leg is silent. "Never collected" is empirically confirmed, not inferred. (The "~283 S5P rows" figure in risk #1 was from a different store — the Mac eval DB has 21 — see the provenance finding below.)
- **OpenAQ ends 2026-05-22.** pm25 (~31 stations) + ozone (~40) dominate; no2/so2/co have **16/7/4 rows total** (stragglers, not usable series) — confirms addendum #6's zero-ground-gas finding, and shows the outage runs into the eval window, not just before it.
- GFS ends May 21; OpenWeather is 35 rows on May 22. The eval store holds **only May shakeout data — zero rows in the June 1+ eval window for any source.**

Two structural consequences:

1. **The qualitative-chemistry demotion compounds the problem.** Demoting types 6/7 to qualitative (correct for a non-chemist labeler) removed exactly the claim types where S5P satellite columns are the independent channel, concentrating the headline onto the three met-adjacent types where independence is weakest. Type 1 is the only headline type S5P restoration rescues — the strongest single argument for treating the S5P backfill (P1.1) as the gate.
2. **The type-2 scorer cannot use the one independent met cross-check.** `score_transport_direction` (`corroboration.py:424`) checks claimed bearing only against modeled GFS/OpenWeather wind; it never tests whether an observed OpenAQ aerosol gradient actually shifted the claimed way. So restoring OpenAQ does nothing for types 2/3 unless the scorer is redesigned to corroborate transport against observed aerosol movement.

Added actions (extend the GFS+OpenWeather collapse bullet above):

- **Measure GFS↔OpenWeather agreement empirically** on collected data (wind-direction concordance, temp-delta distribution) and report it as a number. If they are ~redundant, types 2/3 are single-channel and must be relabeled "model-consistency," not "cross-source corroboration"; if they disagree materially, `evidence_n=2` is defensible. Do not assert independence either way without the number.
- **Decide types 2/3 before the freeze:** redesign type-2 to use observed aerosol transport, or relabel. The headline framing for 2/3 changes or is defended with the agreement number.

### Eval-store provenance is undefined and the store is empty for the eval window (2026-06-13, verified)

The configured eval store (`postgresql://…@localhost:5432/aeris`) holds **only May shakeout data — no eval-window rows for any source** — and **no SQLite→Postgres migration script exists in the repo** (`deploy/windows-collector/` has only `init_db.py` + `run_collectors.bat`). With production collection on the Acer (SQLite) and the eval intended to run on Mac Postgres+TimescaleDB, the path between them is undefined. **Resolve before July 13:** either (a) the Acer migrates to Mac Postgres via a written, dialect-safe load — verify UUID + tz-aware timestamp round-trip, where P1.5's `GUID`/`sa.UUID` divergence becomes load-bearing — or (b) the Mac runs collectors directly into Postgres for the eval window, or (c) the eval runs on the Acer SQLite and the TimescaleDB claim is dropped. This is a prerequisite on par with the OpenAQ/CDSE restoration, not a P-tier code fix.

**Runway note:** the rebaseline freezes the eval set **July 13** from June 1 → July 12 data (`2026-06-10-month2-rebaseline.md:24`), so the operative accumulation window is ~4 weeks, not the full June–Aug span. Every data prerequisite above (OpenAQ key, S5P columns, eval-store provenance) must be producing **before July 13**.

---

## Scope assessment (feasibility · novelty · stability · focus)

- **Novelty — strong (both evaluators).** Inter-source physical corroboration as a _label-free eval proxy_, explicitly distinct from retrieval-grounded factuality (FActScore-style), is a genuinely defensible contribution and the build reflects it. This is the project's intellectual core.
- **Feasibility — engineering adequate/ahead, research at-risk on data.** The build is _ahead_ of the 2026-06-10 rebaseline checkpoint (417 tests green; detection, enrichment, reasoning chain, Phase-1 grounding, and several scorers built). The binding constraint is data, not code — see thesis-validity risks. The S5P backfill bug below (P1.1) is part of _why_ the data isn't flowing.
- **Stability — thesis framing stable; pipeline + production paths at-risk.** The thesis has been stable since the 2026-05-26 Phase1/Phase2 split. But pipeline data health is the real exposure (much of it unrecoverable for this window, not just unbuilt), and production runs on **SQLite on a Windows Acer synced by ZIP, not git** — diverging from the documented Postgres+TimescaleDB stack, so the eval never exercises TimescaleDB and zip-sync risks code/DB drift. Resolve the split explicitly (accept SQLite as store-of-record and drop the TimescaleDB claim for Month 2, _or_ stand up Postgres on the Acer) and replace zip-sync with a git remote. (Clarified 2026-06-13: the Acer SQLite is **collection-only**; the eval is intended to run on Mac Postgres+TimescaleDB, so the TimescaleDB claim stays — the real gap is the undefined Acer→Mac data path, and the eval store is currently empty for the eval window. See the eval-store-provenance finding above.)
- **Focus — strong.** Scope discipline is real and visible (RAG/ChromaDB, API routes, frontend correctly deferred; a mandatory "Month 4 backlog" for temptations). Don't broaden into Month 3 until eval-integrity + data are fixed.

**One-line verdict:** Genuinely novel and feasible to _build_; the immediate threat is that the eval would be computed by code with correctness bugs (P0) _and_ fed by non-independent, data-starved inputs (thesis-validity). Both must be addressed before any Month-2 result is reportable.

---

## Cross-cutting themes (fixing the pattern beats fixing instances)

1. **Dual-dialect (Postgres/SQLite) correctness debt.** The `sa.UUID()`/`GUID` divergence (P1.5), the naive-vs-aware datetime comparisons (`data.py`, detection `--since`), and `postgresql.JSON` columns on a dual-dialect schema all silently misbehave **only on SQLite** — which is exactly the CI _and_ Acer production store, so a green suite does not prove these paths correct.
2. **Phase-1 (`validate.py`) ↔ Phase-2 (`corroboration.py`) semantic drift.** The scorer re-introduces inverted/asymmetric logic (under-cue handling) that Phase-1 handles correctly, directly skewing the Phase1→Phase2 delta the thesis correlates — and the regression test for that exact path is missing.
3. **Silent statistical degradation in detection.** STL disabled for sub-hourly series with no logged skip; residual scale inflated by the anomalies it scores; cadence guard catches too-coarse but not too-short — so the "severe" consensus tier silently becomes unreachable.
4. **Transaction / lifecycle hygiene across CLIs and collectors.** `collect()` double-commits with no outer transaction; `persist_anomalies`/`persist_enrichment` commit per-iteration with no rollback; **4 of 5 CLI entry points leak the asyncpg pool** (never call `engine.dispose()`).
5. **Relative tolerance ÷ reference magnitude → zero-width band at zero.** `validate._supports` (validate.py:146) and `corroboration` point branch (corroboration.py:280): a legitimate 0 reading makes nonzero claims mismatch. Fix once with a shared `within_tolerance(a,b,pct,floor=EPS)` helper.
6. **The test suite cannot prove correctness (so every bug above ships green).** Session-scoped engine + rollback-only teardown gives no real isolation (worked around three incompatible ways); known-event fixtures are tautological sine-wave spikes; the riskiest paths (STL masking, under-cue corroboration, mid-chain parse failure) are untested.
7. **Exceptions / parsing not classified.** `base.collect` retries everything incl. 401s; OpenAQ dead-key is opaque; cloud clients raise `KeyError`/`IndexError` on safety-blocked 200s instead of `LLMParseError` (escapes retry, crashes the sweep).
8. **Dependency/config hygiene.** Dead SDK pins (`openai`, `google-genai` never imported), `chromadb` pinned before RAG exists, hardcoded CORS, `Settings` with empty-string secret defaults and no app-level preflight.

---

## P0 — Eval-integrity bugs (fix + add regression tests before reporting any Month-2 number)

### P0.1 — `validate.py` grounding gate: unitless claim numbers ground against any context number **[verified, high]**

- **Where:** `server/app/llm/validate.py:170-187` (`_match_numbers`), guard at :174.
- **Why:** `if unit and ctx_unit and unit != ctx_unit: continue` only fires when _both_ sides carry a unit, so a unitless fabricated quantity ("levels hit 80") matches _any_ context number within 25% regardless of metric — a fabricated number passes the mandatory hallucination gate and `explain._claim_row` feeds it into Phase-2, inflating corroboration. (The synthesis auto-list dropped this due to a loader bug; it is real and agent-verified.)
- **How:** Require the matched context quantity to carry a unit when the claim number is unitless, or only allow a unitless match when the claim's nearest metric token co-occurs with the context number.

### P0.2 — Corroboration scorer ignores "under" thresholds → inverted semantics on a headline type **[verified, high]**

- **Where:** `corroboration.py:200-217` (`_threshold_value`), `220-233` (`_point_value`), `273-305` (`score_concentration_elevation`).
- **Why:** `_threshold_value` collects only `kind=="over"` cues → under-claims yield `threshold=None`; `_point_value` returns `None` because a threshold cue exists → the claim falls into the qualitative `else` branch scored as `nearest > baseline*ratio` (opposite of under-semantics). `validate._supports` grounds the same claim correctly, so Phase-1 and Phase-2 disagree on the exact headline type the delta is built on.
- **How:** Mirror `validate._threshold_relation`: have `_threshold_value` also return under-thresholds, add an under branch (`SUPPORTING if nearest <= threshold else CONTRADICTING`), and stop `_point_value` returning `None` for under-only cues.

### P0.3 — Eval freeze under-merges events (first-match, not transitive single-linkage) **[verified, high]**

- **Where:** `freeze.py:74-94` (`group_events`).
- **Why:** Joins each anomaly to the **first** matching event and `break`s, never unioning two events a later anomaly transitively bridges (A–B–C stays two events). Contradicts the documented moving-plume guarantee, lets one physical event seed multiple top-N representatives, and is order-dependent on equal-timestamp ties.
- **How:** Collect _all_ matching events in the inner loop and union them, or run union-find / connected-components over the per-metric proximity graph. **Re-freeze the eval set after fixing.**

### P0.4 — STL silently disabled for sub-hourly series; "severe" tier unreachable **[verified, downgraded high→medium]**

- **Where:** `run.py:84-107` (`_stl_period_for`/`_engine_for`) ↔ `stl.py:33-49`.
- **Why:** `_engine_for` builds `STLDetector(period=period)` so the floor is `2*period+1`; 15-min cadence → `period=96` → floor 193, so a 50–192-point group clears `MIN_GROUP_POINTS=50`, is reported "ok", but STL returns `[]` invisibly with `skipped_reason` never set. (The cadence-aware `_stl_period_for` (`MIN_STL_SAMPLES_PER_DAY=4`) added since pass 1 guards _too-coarse_, not _too-short_ — hence the downgrade to medium, but it remains a silent detection degradation.)
- **How:** In `_engine_for` gate STL on `len(series) >= 2*period+1`; when short, skip deliberately and set `GroupSummary.skipped_reason`. Fix the stale run.py:71-74 comment.

### P0-tests (and a prerequisite)

- **Prerequisite — fix test isolation first (see P1.6)**, or these regression tests can pass in isolation and rot in the full suite.
- `test_validate.py`: empty/whitespace context → UNVERIFIED; "so2 0 ppb" vs "so2 0.2 ppb"; "index hit 45" (no unit) must NOT ground against "humidity 45%"; `cited_sources=[""]` must not auto-pass.
- `test_corroboration.py`: "stayed below N" / "elevated but below N" assert correct SUPPORTING/CONTRADICTING.
- `test_run.py`: ~60-point 15-min-cadence series asserts STL runs or is explicitly reported skipped.

---

## P1 — Data-pipeline correctness, production stability & test integrity

### P1.1 — Sentinel-5P backfill never fetches deep history (no upper time bound) **[unconfirmed — confirm first; directly tied to "S5P never collected"]**

- **Where:** `backfill.py:600-616` (`_fetch_window`) + `sentinel5p.py:117-126` (`odata_filter`).
- **Why:** `odata_filter` emits only `ContentDate/Start gt (window_end - LOOKBACK)`, no upper bound, and CDSE returns newest 200 (`$orderby desc`, `RESULT_LIMIT=200`); each backward window re-returns the same most-recent granules, so older granules near `--since` are never fetched (dedup hides it as zero inserts).
- **How:** Add an upper bound — `odata_filter(window_end, window_hours)` emitting both `... gt (window_end - window_hours)` AND `... le window_end` — passing `self.window_hours` from `_fetch_window`.

### P1.2 — S5P accepts silently-truncated granule downloads **[unconfirmed]**

- **Where:** `sentinel5p.py:419-433` (`_download_granule`). Streams chunks, returns on first 200 with no bytes-written vs `Content-Length` check → partial `.nc` yields a column mean from incomplete swath data.
- **How:** Track bytes written vs `content-length`; raise on short/aborted streams so `collect()`'s retry re-attempts.

### P1.3 — Default httpx client doesn't follow redirects → silent GFS/S5P "outages" **[unconfirmed]**

- **Where:** `base.py:63` (`follow_redirects=False`), used by `sentinel5p.py:272`, `noaa_gfs.py:199-210`. A 301/302 from NOMADS/CDSE makes `_fetch_cycle_grib` skip the cycle and `_fetch_catalog`/`fetch_access_token` raise. Only the granule `$value` path handles redirects manually.
- **How:** Build the catalog/token calls with `follow_redirects=True` (or extend manual-redirect handling).

### P1.4 — OpenAQ `/latest` re-inserts stale readings every hour **[unconfirmed]**

- **Where:** `openaq.py:214-226` (`_normalize_sensor`). An offline station's weeks-old `latest` is emitted as a fresh DataPoint each run; insert-only persistence writes the same stale (timestamp, value) every cycle, bloating storage and feeding detection stale-but-current-looking values.
- **How:** Drop the point when `now - timestamp` exceeds a configurable `OPENAQ_MAX_READING_AGE_S`.

### P1.5 — Alembic migrations use `sa.UUID()` not the `GUID` type → SQLite corruption **[verified, high; synthesis ranked #1]**

- **Where:** both `server/alembic/versions/*.py` — every id/FK column (confirmed via grep); contrast `models.py:25-60` `GUID`.
- **Why:** On SQLite `sa.UUID()` → numeric-affinity token; an undashed-hex id can be coerced (the GUID docstring cites a real id that landed as `Inf`). Migration DDL diverges from `Base.metadata.create_all`, so any alembic-provisioned SQLite DB (CI + the SQLite Acer path) is corruption-vulnerable. _Confirm whether the Acer DB is built via alembic or `init_db.py`/`create_all`; if the latter, severity drops but the CI divergence remains._
- **How:** Import `GUID` into both migration modules and use it for every id/FK column so SQLite stores dashed `CHAR(36)`/TEXT, matching `create_all`. Add the round-trip test (P1.6/P0-tests).

### P1.6 — `db_session` fixture gives no real isolation; committed rows leak across the whole suite **[verified, high — NEW from tests-quality]**

- **Where:** `conftest.py:9-37` + ad-hoc workarounds in `test_run.py:37-43`, `test_enrichment_smoke.py:40-46`, `test_explain.py:275-288`, `test_eval_harness.py:156-163`, `test_data_routes.py:80-81`.
- **Why:** `test_engine` is session-scoped (one SQLite file) and `db_session` only rolls back at teardown — a no-op for committed rows — so every DB test defends itself by hand in three incompatible ways (per-module autouse delete, UUID-scoped counts, name namespacing) and some have none. **Any new test that writes an unscoped count passes alone and fails in the full suite by ordering** — and the failure looks like a real bug, not contamination. This is _why every P0/P1 correctness fix would "ship green."_
- **How:** Make `db_session` isolating — run each test inside `connection.begin()` + `begin_nested()` joined to the session, roll back the outer transaction at teardown — then delete the per-module `_clean_tables`/UUID-scoping workarounds. **Do this before the P0 regression tests** so they actually catch regressions.

---

## P2 — Medium robustness, correctness & performance (grouped)

**Transaction / lifecycle (theme 4)**

- `collect()` double-commits (`_store` + `_update_source_status` commit independently, no outer txn) → data can land while status is stale/error and retries inflate `error_count` after a partial commit. Wrap one `async with session.begin()` per successful attempt; make `_store`/`_update_source_status` execute-only. _(base.py:79-203)_ **[verified — NEW]**
- `persist_anomalies` commits per group with no try/except → a mid-run DB error leaves groups 0..N-1 committed and prints no summary. Single end transaction, or per-group try/except that records the failure in `GroupSummary`. _(run.py:417,483)_
- `persist_enrichment` commits per record → one failure aborts the whole pass and loses the partial summary. Wrap per-anomaly, record, continue. _(enrichment.py:340-342,409-416)_ **[verified]**
- **4/5 CLIs leak the asyncpg pool** — only `run_all` calls `engine.dispose()`; the others emit "Event loop is closed" on shutdown. Add an `@asynccontextmanager` lifecycle helper in `db/session.py` and use it in all `_amain` bodies. _(run.py:579, explain.py:390, harness.py:170, freeze.py:294, label_cli.py:208)_ **[verified — NEW]**

**Dual-dialect / timezone (theme 1)**

- detection `--since` is tz-fragile on SQLite: `_parse_since` returns aware UTC ("+00:00"), but `DateTime(timezone=True)` stores naive ISO on SQLite → boundary rows wrong-windowed; CI+Acer both SQLite. Strip tzinfo for non-postgres in `load_points`, add a SQLite boundary test. _(run.py:153-154,528-533)_ **[verified — NEW]**
- `data.py` compares naive `start`/`end` to tz-aware `timestamp` → silently wrong range filtering on asyncpg. Coerce to UTC-aware (or strip on SQLite). _(data.py:66-67,83-88)_
- `configure_postgres_compatibility` creates the dedup UNIQUE index inside the create_all txn with no duplicate handling → hard startup crash on a dirty dev DB. Wrap in try/except like `configure_timescale`. _(schema.py:40-45)_

**Error handling & resilience (theme 7)**

- `base.collect` retries non-transient errors (incl. dead-key 401) 3× / 90s — classify exceptions, break on 4xx/auth. _(base.py:84-128)_ **[verified]**
- Cloud clients raise `KeyError`/`IndexError` on safety-blocked/empty-content 200s, escaping retry and crashing the sweep — guard extraction, raise `LLMParseError`. _(gpt_client.py:90, gemini_client.py:100)_ **[verified]**
- OpenAQ dead-key surfaces as opaque `HTTPStatusError` — special-case 401/403 into a clear "key rejected" log. _(openaq.py:136-145)_
- Malformed location coordinates raise uncaught `float()` aborting the whole fetch — try/except to skip one bad station. _(openaq.py:87,240)_
- CDSE token can expire mid-download (300s timeout vs ~600s token) — refresh-and-retry once on 401 in `_download_granule`. _(sentinel5p.py:354-363,403-436)_

**Grounding / scoring robustness**

- `validate` cited-source check is raw substring; `cited_sources=[""]` auto-passes — filter empties, tokenize against known source names. _(validate.py:211-214)_ **[verified]**
- `validate` grounding has no relation/verb check, so fabricated causal claims ground on co-occurring nouns — route causal claim types to a stricter check. _(validate.py:209,226)_ **[verified]**
- `explain` `stated_confidence` is an unbounded float; 5/-1/NaN flows into the corroboration analysis — bound with `Field(ge=0, le=1)` in prompt.py. _(prompt.py:40)_ **[verified]**
- `corroboration.score_emissions_source_type` uses default `min_obs=1` (unlike type-10) — add `min_obs_per_station` to `SourceTypeTolerance`. _(corroboration.py:1008)_

**Performance**

- Z-score O(n²) full-series rescan per point — two-pointer window / `np.searchsorted`. _(zscore.py:38-43)_
- `build_aux_inputs` re-loads the full GFS u/v/PBL window per group — hoist out of the loop / memoize by (metric, window). _(run.py:181-303,459)_
- Enrichment N+1 bbox scan per pending anomaly — batch by overlapping window/cluster. _(enrichment.py:409-416)_ **[verified; offline CLI bounded by `limit`]**
- Backfill one INSERT over the whole points list — chunk ~500 rows (SQLite 999 / PG 65535 param caps). _(backfill.py:126-143)_
- Backfill archive N×D serial GETs — bounded `asyncio.Semaphore` + `gather`. _(backfill.py:498-518)_
- Backfill OpenAQ measurements `while True` no max-page guard. _(backfill.py:300-363)_
- S5P granule raveled to ~1.8M boxed floats then pure-Python masked — vectorize in numpy. _(sentinel5p.py:236-240)_
- Harness builds GPT/Gemini clients for every model even when all cells are resumed-complete — lazily create on first non-skipped cell. _(harness.py:87-116)_

**Detection statistics**

- STL residual scale = plain std over _all_ residuals incl. anomalies → masking — use MAD×1.4826. _(stl.py:64-71)_
- `sentinel5p` qa_value relies on implicit xarray scale-decoding — pass `mask_and_scale=True`, guard `if qa.max()>1: /100`. _(sentinel5p.py:209,237)_

**Eval / labeling integrity & API**

- Labeler-blinding cross-module invariant is untested — add a test asserting labeler-facing text contains no grounding/corroboration/model tokens. _(label_cli.py:130-135, explain.py:127-143)_ **[verified]**
- `/data/{source}` lat/lon unbounded + partial geo subset silently drops the filter — `Query(ge/le)` bounds + 422. _(data.py:68-69,89-100)_ **[downgraded medium→low by verifier: read-only endpoint, worst case is unfiltered results]**

**Test gaps (medium)** — under-cue corroboration (rank-2 ships green without it); detection known-events are synthetic-only (add `tests/fixtures/known_anomalies.json` with ~5-10 hand-labeled Houston anomalies + a few confirmed non-events, assert recall + bounded FP rate); no alembic-upgrade-head-on-SQLite UUID round-trip test; GRIB/netCDF parsing + masking + redirect/truncation download; collector orchestration (retry classification, `_store` dialect, failure status).

---

## P3 — Low (terse, grouped; verifier corrections applied)

**Collectors:** rate-limit reset header has no sanity cap (∞ sleep on absolute-epoch) `ratelimit.py:66-73`; 429 backoff doesn't `limiter.defer` the shared budget `ratelimit.py:105-112`; generic-dialect `_store`/`_store_points` fallback omits `on_conflict` (raise `NotImplementedError` for unsupported dialects — theme, 2 sites) `base.py:161, backfill.py:139`; OpenWeather sends `appid=''` wasting 5 requests `openweather.py:96-124`; OpenWeather precip reads only `1h`, drops `3h`-only as 0.0 `openweather.py:73-84`; OpenAQ sensors fetch unpaginated `openaq.py:154-169`; process-global `_locations_cache` leaks across instances `openaq.py:25-29`; GFS bbox no antimeridian/clamp `noaa_gfs.py:104-112`; GFS `_extract_grid` assumes all messages share one grid `noaa_gfs.py:243-272`.
**Backfill:** `db_location_ids` loads full `raw_json` blobs to read one id `backfill.py:427-442`; S5PBackfill closes a collector created once in `__init__` → 2nd `backfill()` uses a closed client `backfill.py:569`; 997-line module mixes 4 strategies + persistence + CLI (split into a package) `backfill.py:1-997`.
**Detection:** STL treats irregular spacing as even (**downgraded medium→low**: `run.py` cadence guard bounds it) `stl.py:52,59`; consensus merge silently overwrites on duplicate-timestamp `consensus.py:86-91`; IF↔Z/STL matched by exact-timestamp across two lists (unasserted invariant) `engine.py:73-82`; `_stl_period_for` uses upper-middle element not true median (`statistics.median`) `run.py:95`; engine uses only `group_points[0].lat/lon` `run.py:114-116`.
**Enrichment:** `_summarize_metric` min/max/mean unguarded against NaN (S5P/GFS can store NaN; poisons aggregates + emits invalid JSON `NaN`) — filter NaN, ideally reject at `_store` `enrichment.py:223-232`; `_nearest_in_time` mixes timedelta+float in sort key `enrichment.py:240-243`; `unit` taken from `entries[0]` with no consistency check `enrichment.py:225`.
**LLM:** Gemini all-thought/empty-parts returns "" and burns both retries `gemini_client.py:100-101`; `generate()` reports only the successful attempt's latency (understates eval cost summed in reasoning_chain) `client_base.py:83-110`; parse-retry resends identical prompt (deterministic Ollama → wasted) `client_base.py:82-99`; `parser` accesses `claim.statement/.cited_sources` directly despite getattr-guarding `.claims` `parser.py:28-37`; `validate._supports`/`corroboration` zero-division tolerance band (theme 5, 2 sites) `validate.py:146, corroboration.py:280`; `_render_station_means`/enrichment render emit literal `'None'` into grounding context `explain.py:107,160-165`; positional `step_index` attribution `explain.py:260-261`; corroboration threshold/point branch ignores nearest-sample recency `corroboration.py:273-284`; `_resolve_pollutant` returns first-in-tuple not first-mentioned `corroboration.py:191-197`; mobile verdict keyed on single global peak hour `corroboration.py:1000-1006`.
**Eval:** `--top<=0` produces a silently wrong-sized fixture `freeze.py:153,250`; persist returns False (raced/dup) but counted `completed` `harness.py:99-100`; `load_anomaly_set` doesn't de-dup/warn empty id list `harness.py:46-62`; claim dedup ordering non-deterministic (**downgraded medium→low**: labels are order-invariant; only presentation order) — add `Claim.claim_text, Claim.id` tiebreak `label_cli.py:52-69`; note fan-out has no dedup-group marker `label_cli.py:147-151`.
**DB/API/deps:** dead SDK pins `openai`/`google-genai` never imported (drop from requirements) `requirements.txt:26-27`; `postgresql.JSON` used on dual-dialect columns (use generic `JSON`) `models.py:20,85,121,149`; naive datetimes vs tz-aware columns `data.py:66-67`; `data_sources` status not in `collect()` txn (health-view divergence — folds into P2 double-commit fix) `base.py:183-187`.
**Test gaps (low):** high-cadence STL-disable path (50–192-point sub-hourly); STL masking + irregular cadence + Z-score duplicate timestamps; determinism test doesn't pin IF scores/membership `test_engine_known_events.py:212-238`; reasoning-chain mid-step parse-failure path (most likely real Llama failure) `test_reasoning_chain.py`; enrichment over-fetch radius/window drop; explanation with zero claims; /health degraded + /data invalid-params.

---

## P4 — Minor (one-liners; corrections applied)

`run_all.exit_code` returns 0 for an empty result set (vacuous `all()`) `run_all.py:103-104`; `collector.close()` in `finally` can mask the original exception `run_all.py:73-83`; lazy `import asyncio` inside the retry loop `base.py:126`; local `import time`/intra-package imports inside hot loops `backfill.py:187,479,580,642`; window/day enumeration materializes full lists `backfill.py:540-543,776-784`; GFS `raw_value` key name misleads vs converted `value` `noaa_gfs.py:307-317`; `client_base._get_client` hardcodes a dead 30s timeout shadowed by per-request 120s `client_base.py:55`; `generate(max_attempts<1)` raises without calling the model `client_base.py:82`; `_TIME_RE` blanks any `d:dd` token dropping colon-ratios `validate.py:53`; blank step summaries threaded forward as empty bullets `reasoning_chain.py:56-58`; `_resolve_value` exact float `!=` (**downgraded low→minor**: can't trigger today) → `math.isclose` `consensus.py:192`; `get_session` annotated `-> AsyncSession` but is an async generator (**downgraded low→minor**: cosmetic) `session.py:15`; `Settings()` empty-secret defaults, no app-level preflight (**downgraded low→minor**: CLIs already preflight) `config.py:49`; CORS origins hardcoded to localhost:5173 (**downgraded low→minor**: frontend deferred) `main.py:47-53`; `event_sizes` recomputes the representative independently of selection `freeze.py:150,155-157`; `_cell_exists` duplicates persist idempotency `harness.py:92`; `chromadb==1.5.9` pinned before RAG exists `requirements.txt:25`; `drop_tables` doesn't drop the Timescale hypertable/chunks first `schema.py:67-71`; `corroboration.py` 1379 lines — leave per design spec (optionally extract pure text parsers).

**Rejected in re-verify (do NOT act on):** "NaN z_score ranks as top-tier via `abs(nan)`" — refuted: `zscore.py:51-52` guards `if std == 0: continue` _before_ the division, so a zero-variance window never produces a NaN z_score.

---

## Recommended sequencing & verification

0. **Thesis-validity first (non-code):** confirm the S5P backfill (P1.1) is fixed and producing real column rows, restore OpenAQ, and decide the source-independence story — these gate whether a Month-2 result is even possible. Do the README reconciliation + Bracco commitment email now.
1. **Fix test isolation (P1.6)**, then **P0** + its regression tests, then **re-freeze the eval set and re-run the harness** — every Month-2 number depends on P0 being correct _and_ on tests that actually catch regressions.
2. **P1** data-pipeline + production (S5P backfill/truncation/redirects/stale latest, alembic UUID).
3. **P2 themes** — the 8 cross-cutting fixes (single-transaction collect, `engine.dispose` lifecycle helper, shared tolerance helper, dialect/timezone normalization, exception taxonomy) collapse many individual findings.
4. **P3/P4** opportunistically alongside the modules they touch.

**Verification per tier:** the P0/P0-tests cases; `pytest server/tests/unit/llm server/tests/unit/detection server/tests/unit/test_eval_*` after P0 (with the isolation fix in place); an `alembic upgrade head` + hex-`e` UUID round-trip test for P1.5/P1.6; a SQLite `--since` boundary test for the tz fix; for P1 data fixes a live `python -m app.collectors.backfill --source=sentinel5p --since <date>` asserting distinct bounded windows and non-zero column inserts. Re-run `graphify update .` after edits.

**Confirm `[unconfirmed]` P1 items against current source before implementing** (their pass-1 verifiers were the ones that hit the spend limit; pass-2 re-verified all _except_ these data-pipeline ones were already re-run — backfill/collectors-climate verifiers ran in pass 2, so P1.1–P1.4 now carry pass-2 verdicts; re-read the cited spans if acting on them).

---

## Execution plan (2026-06-13) — phased, surgical

Operationalizes the sequencing above into atomic steps: one coherent change per step, each independently verifiable, with pauses on phase boundaries. Tags: **[code]** = source/test change; **[Acer]** = production run on the Windows Acer; **[decide]** = a decision required before dependent steps.

**Out of scope for this track** (owned separately, still open risks): the README "Live"-status reconciliation, and the consolidated Bracco correction + labeling-commitment email.

Two long-lead items run in parallel with the code work without serializing it: the **Acer re-runs** (A5) and the **two decisions** (B1, F3).

### Phase A — S5P/GFS data-flow (unblock real data; gates the Acer re-run)

- **A1** [code] Confirm P1.1 / P1.2 / P1.3 against current source (read-only). Independent verdict each: confirmed-open / refuted / already-covered by the 2026-06-12 granule-download fixes.
- **A2** [code] Fix P1.1 (backfill upper time bound) + distinct-bounded-window test — only if A1 confirms open.
- **A3** [code] Fix P1.2 (short/aborted-stream guard, bytes vs `Content-Length`) + test — only if confirmed.
- **A4** [code] Fix P1.3 (follow redirects on catalog/token/GFS paths) + test — only if confirmed.
- **A5** [Acer] Pull code, re-run `backfill --source sentinel5p --since 2026-06-01` + OpenAQ top-up; verify non-zero `s5p_*_column` and chemistry counts. Long lead; runs while C/D proceed.

### Phase B — Eval-store data path (longest data-clock lead)

- **B1** [decide] **DONE 2026-06-13 — chose (a)** Acer→Mac loader. Decisive reason: the live sources (GFS retains ~10 days on NOMADS; OpenWeather has no history API) can't be backfilled at freeze, so collection is pinned to the always-on Acer — (b) is dominated, and (c) would force moving Ollama + the whole eval stack onto the Windows box and run the analysis code on the buggier SQLite dialect. (a) moves one table instead of all the compute and keeps the documented Postgres+TimescaleDB stack.
- **B2** [code] **DONE 2026-06-13.** `server/app/eval/load_store.py` + `server/tests/integration/test_load_store.py` (15 tests, TDD). Scope-cut to raw `data_points` only (downstream tables regenerate on the Mac). ORM-typed copy → `pg_insert(...).on_conflict_do_nothing(constraint="uq_data_points_dedup")`, batched/streamed, `--since`/`--dry-run`, engines disposed in `finally` (no asyncpg pool leak). The load-bearing risk was the **timestamp**, not the UUID: SQLite drops tzinfo, so `_aware_utc` re-anchors to UTC; the round-trip test runs against a Postgres DB whose session tz is forced to `America/Chicago` and is **sabotage-verified** (neutering `_aware_utc` shifts the stored instant and the test fails). Real-scale smoke: 127,923 rows loaded from the 2026-06-10 snapshot, per-`(source,metric)` reconciliation exact, idempotent re-run (0 inserted), detection read the loaded rows clean.
  - **Corrupt-id pre-flight — DONE 2026-06-13.** `_check_source_ids` runs before any write: on a SQLite source it counts `typeof(id) <> 'text'` (the numeric-affinity `Inf` class — the Acer's id column is still NUMERIC affinity post-fix) and raises `CorruptSourceError` naming the repair instead of crashing mid-stream with an opaque `ValueError`; the CLI exits 3. Sabotage-verified (removing the guard reverts to `ValueError: badly formed hexadecimal UUID string`).

### Phase C — Test isolation (gates every regression test in D)

- **C1** [code] P1.6 — transactional `db_session` (`connection.begin()` + `begin_nested()`, outer rollback at teardown); remove the per-module `_clean_tables` / UUID-scoping workarounds; confirm the full suite stays green.

### Phase D — P0 eval-integrity bugs (one bug + its regression test per step)

- **D1** [code] P0.1 — unitless-claim grounding leak (`validate._match_numbers`).
- **D2** [code] P0.2 — under-threshold corroboration inversion (`corroboration` `_threshold_value` / `_point_value` / `score_concentration_elevation`). Also resolves cross-cutting theme 2.
- **D3** [code] P0.3 — eval-freeze under-merge (`freeze.group_events`); flag re-freeze for E.
- **D4** [code] P0.4 — STL silently disabled for sub-hourly series (`run._engine_for` ↔ `stl`).

### Phase E — Re-baseline (needs D complete + data in store from A5/B)

- **E1** [code][Acer] Re-freeze the eval set, re-run the harness.

### Phase F — Source-independence framing (decide before the freeze; needs eval-window data)

- **F1** [code] Source-independence audit: report per-source-pair contribution to each verdict.
- **F2** [code] Measure GFS↔OpenWeather agreement empirically (wind-direction concordance, temp-delta distribution); report as a named number.
- **F3** [decide] Types 2/3: redesign type-2 to corroborate against observed aerosol transport, or relabel "model-consistency." F1/F2 feed this.

### Phase G — Remaining P1

- **G1** [code] P1.4 — drop stale `/latest` readings past `OPENAQ_MAX_READING_AGE_S` (`openaq._normalize_sensor`).
- **G2** [code] P1.5 — alembic migrations use `GUID` for every id/FK column (the 2026-06-12 fix only touched `models.py`).

### Phase H — P2 / P3 / P4 cleanup (coarse by design)

Re-expanded into atomic one-theme-per-step units when reached, not before — D and E collapse overlaps (theme 2 ≈ D2, theme 3 ≈ D4) and the exact decomposition depends on what re-baseline reveals. Covers the remaining cross-cutting themes (single-transaction `collect`, `engine.dispose` lifecycle helper, shared `within_tolerance`, dialect/timezone normalization, exception taxonomy) and the grouped P3/P4 one-liners.
