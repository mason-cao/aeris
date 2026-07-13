# Server-side analysis on Postgres + TimescaleDB (Option A)

The Acer collects on **SQLite** (its `.env` keeps `DATABASE_URL=sqlite+...`, untouched).
The planned official evaluation (`detection -> enrichment -> explain -> harness`)
runs against **Postgres + TimescaleDB** here. The pipeline uses the same ORM
models on SQLite and Postgres, with engine-specific schema setup.

**Do not run an official freeze yet.** This flow remains blocked by the data-QC,
methodology, and labeling-protocol decisions recorded in the July 15 call-prep
document. The July 12 snapshot is for audit and disposable rehearsal only.

Everything in this directory is disposable: the analysis DB is rebuilt from a snapshot each time.

## One-time

- Docker Desktop running.
- `cp .env.analysis.example .env.analysis` (defaults are fine for local use).

## Freeze-day flow

```bash
cd server/deploy/analysis

# 1. Start Postgres + TimescaleDB (host port 5433; extension preloaded).
docker compose up -d --wait

# 2. Load the frozen SQLite snapshot. --reset wipes the analysis DB first.
#    create_tables() builds the data_points hypertable (empty) before rows land.
cd ../..                       # back to server/
DATABASE_URL='postgresql+asyncpg://aeris:aeris@localhost:5433/aeris' \
  python -m app.db.migrate --source-sqlite /path/to/frozen/aeris.db --reset

# 3. Point the pipeline at Postgres for the rest of the session.
set -a && source deploy/analysis/.env.analysis && set +a

# 4. Run the eval on Timescale (unchanged commands).
python -m app.detection.run
python -m app.detection.enrichment
python -m app.eval.freeze --start 2026-06-01 --end 2026-07-12 --top 50 --out fixtures/eval50.json
python -m app.eval.harness --anomaly-set fixtures/eval50.json

# 5. Done — wipe the container + volume.
cd deploy/analysis && docker compose down -v
```

## Verify the hypertable (optional)

```bash
docker exec aeris-analysis-db psql -U aeris -d aeris \
  -c "SELECT hypertable_name, num_chunks FROM timescaledb_information.hypertables;"
```

## Notes

- **Migration uses typed ORM tables across engines.** The copy goes through the ORM `Table`
  objects, so the `GUID` decorator (CHAR(36) ↔ native `uuid`), `JSON`, and naive→UTC
  `timestamptz` conversions happen automatically. A June 18 smoke test loaded 71,262
  rows into a 23-chunk hypertable with native `uuid`/`timestamptz`/`json` columns
  and bit-identical values. Source tables absent from an older snapshot (`explanations`,
  `claims`, `expert_labels`) are skipped.
- **Re-runnable.** Without `--reset` the loader refuses a non-empty target so you can't
  double-load; with `--reset` it drops and rebuilds.
- **Image is pinned** (`timescale/timescaledb:2.17.2-pg16`) for reproducibility.
- This does **not** touch the Acer or its collection. Moving edge collection to
  Postgres would require a separate design and soak test.
