# Data pipeline repair — findings and checklist

> **Date:** 2026-06-12
> **Source of truth:** audit of the Acer's `aeris.db` (snapshot copied to the Mac 2026-06-10 13:56 ET) and `collector.log` (10 MB, through 2026-06-10 12:57 ET).
> **Status:** every item below is unfixed until checked off. Items 1 and 2 block the Month 2 eval; do them first.

---

## What the audit found

| Source | Rows | Reality vs. assumption |
| --- | --- | --- |
| OpenAQ | 105,492 | Live hourly collection worked **only May 22–23** (6 ok runs, 275 rows). The API key has returned **401 on 447 of 453 hourly runs since the first logged run (May 22 22:57)** — the suspension hit almost immediately, not "last week." Everything else was ingested by the June 10 archive backfill, which ends at **June 7 05:00 UTC** (archive publication lag). No live rows since; the restored account's new key is not on the Acer. |
| Sentinel-5P | 283 | **Zero column-density rows exist.** All 283 rows are `s5p_*_granule_available` catalog markers. `CDSE_USERNAME`/`CDSE_PASSWORD` were never added to `server/.env` (the keys aren't in the file), so the collector has run in catalog-only mode since deployment. The "283 satellite column rows" figure in the 6/11 email to Dr. Bracco was availability markers, not measurements. |
| NOAA GFS | 6,384 | Complete: 19 days × 4 cycles × 12 cells × 7 metrics, no gaps. |
| OpenWeather | 15,841 | Effectively complete (~17 missing point-hours over 19 days). |

Additional facts:

- **Ground chemistry is absent, and it is not an ingest bug.** Since May 1 the OpenAQ data is pm25 (88,592 rows / 42 stations), ozone (15,074 / 18), pm10 (1,749 / 2) — **zero NO2, SO2, or CO**. Verified against the public S3 archive directly: TCEQ Clinton C403 (location 1910, the flagship Ship Channel monitor) publishes only `o3/pm10/pm25` to OpenAQ in this period, and the DB ingested it 1:1.
- **Station count:** 62 stations reported in May (61 in June), not the 104 quoted in the 6/11 email — 104 was the bbox location count, not stations actually reporting. The network is a mix of TCEQ CAMS sites (the C-numbers) and community PM-only sensors (ACTS/COCO/park-district).
- **77 stale rows** with timestamps from 2016–2026-01 (dead stations' "latest" values swept up on May 22–23). Harmless to detection (min-points floor) but they pollute row counts and date ranges.
- **One corrupted row:** `data_points.id = Inf` (REAL, not a UUID string) on an OpenAQ row written during the June 10 archive backfill (timestamp 2026-05-28 12:39:32). It crashes **every** ORM read of `data_points` — detection cannot run against this file until it's fixed.
- **SQLAlchemy echo is on** (`AERIS_ENV` is not `production` on the Acer despite SETUP.md step 3), which is most of the 10 MB log.
- App-level INFO logs (e.g. "CDSE credentials not set; running in catalog-only mode") never reach `collector.log` — only WARNING+ and the engine echo do. That is why catalog-only mode was invisible for three weeks.

## Repair checklist (ordered)

1. [ ] **OpenAQ key.** Generate the new key in the restored account, update `OPENAQ_API_KEY` in the Acer's `C:\temp\aeris\aeris\server\.env` (and the Mac copy), run `python -m app.collectors.run_all --source=openaq`, confirm `ok`.
2. [ ] **CDSE credentials.** Register the free Copernicus Data Space account, add `CDSE_USERNAME` / `CDSE_PASSWORD` to both `.env` files, run `python -m app.collectors.run_all --source=sentinel5p`, confirm `s5p_*_column` rows appear (not just `_granule_available`). Granules are hundreds of MB each — leave the Acer on Ethernet and don't schedule anything else heavy in that hour.
3. [ ] **Backfill the gaps.** Run on the Acer (the production DB lives there); both are idempotent and safe to re-run.
   - OpenAQ: `python -m app.collectors.backfill --source openaq --since 2026-06-06` (archive lags ~1–2 days; re-run a few days later to top up).
   - Sentinel-5P: `python -m app.collectors.backfill --source sentinel5p --since 2026-06-01` **after item 2**. As of 2026-06-12 the backfill downloads granules and extracts column densities (it was catalog-only before — the missed weeks were unrecoverable by code). Granules are hundreds of MB each; an interrupted run resumes on re-run because already-extracted granules are skipped.
4. [ ] **Fix the corrupted id** on the Acer's file before the next detection run:
   `UPDATE data_points SET id = lower(hex(randomblob(16))) WHERE typeof(id) = 'real';`
   Root cause (found 2026-06-12, see `test_guid_type.py`): the ORM declared ids with the bare `UUID` DDL keyword, which SQLite gives NUMERIC affinity, and bound them as undashed 32-char hex. ~1 in 10⁶ uuid4 hex strings is all digits plus a single `e`; SQLite reads that as scientific notation and stores a REAL — here the exponent overflowed to `Inf`. Fixed in `models.py` (`GUID` type, binds the dashed 36-char form, which can never parse as a number — no migration needed; legacy undashed rows still read fine). The UPDATE above repairs the one already-corrupted row.
5. [ ] **Delete the 77 stale rows** (optional, hygiene): `DELETE FROM data_points WHERE source = 'openaq' AND timestamp < '2026-05-01';`
6. [ ] **Set `AERIS_ENV=production`** in the Acer's `.env` (kills the engine echo). The code half landed 2026-06-12: `run_all` and `backfill` now configure stderr logging (`logsetup.py`), so collector INFO/WARNING lines reach `collector.log`, and `run_all` prints a loud per-source warning when credentials are missing — the check that would have caught item 2 on day one.
7. [ ] **Weekly row-count check** during the accumulation window (already in the rebaseline): per-source rows + max timestamp, eyeballed against expectations. The numbers go in the weekly Bracco update emails.

## Consequences for the eval design (already applied in code)

- `emissions_source_type` defaults to pm25 instead of no2 (no ground NO2 exists to default to).
- `secondary_formation` (type 9, O₃-lags-NO₂) **cannot be scored from ground data in this window**. Once CDSE works, satellite NO2 is 1/day — no hourly lag test. Realistic options: re-scope type 9 to qualitative-only like types 6/7, or add a TCEQ/AirNow collector for gaseous pollutants (scope change — Month 4 backlog and a Bracco conversation, not a unilateral add).
- The corrected data table and the type 9 situation go to Dr. Bracco in one consolidated email **after** items 1–3 land, so it carries real numbers instead of a second correction.
