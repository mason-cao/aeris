# A.E.R.I.S. - Autonomous Environmental RAG & Inference System

**A preregistered study of how much of what a language model says about the
physical world can be checked against an instrument.**

Language models will produce a fluent causal explanation for any scientific
observation on demand, and a sound explanation is difficult to tell apart from
a merely plausible one. Air quality is a setting where no answer key exists.
Nobody recorded the true cause of the nitrogen dioxide spike southeast of
Houston at 23:00 UTC on 15 June 2026. What exists instead is redundancy: a
claim about wind direction can be put to three separate wind products, and a
claim about a concentration can be put to the monitor that recorded it and
sometimes to a satellite column overhead. This study builds a deterministic
verifier on that redundancy and measures how far it actually reaches.

| | |
| --- | --- |
| Study window | 1 June to 5 August 2026, Houston, Texas |
| Observations | 405,432 across eight instrument networks |
| Events | 50 detected anomalies, stratified by pollutant |
| Generators | Llama 3 8B (local), GPT-5.4, Gemini 3.6 Flash |
| Claims | 1,781 extracted, 514 scored |
| Preregistration | OSF `osf.io/hb92y`, registered 9 August 2026 |
| Status | Coverage measured; expert labels not yet collected |

Each of the three models explained the same 50 anomalies. Their explanations
decompose into 1,781 individual claims. A deterministic checker assigns every
claim to one of ten types and puts it only to the measurement channels that can
physically bear on that type. Where no channel can speak, the checker abstains.

No verdict is returned for 1,267 of the 1,781 claims. Verdicts exist for 514,
or 28.9 percent. An unverified claim is not a false one; it is one that no
instrument in the network can weigh in on either way. Coverage rather than
accuracy is the binding constraint on a verifier of this kind, and measuring it
is what this repository reports.

The project name describes the intended complete system. Collection, detection,
and verification are implemented. Retrieval, the frontend, and autonomous
operation are planned, and are described under Planned Product Architecture
below.

## Current Status

The repository holds a Python backend and a command-line research workflow that
can:

1. Collect and normalize seven live environmental feeds, plus historical EPA AQS
   samples.
2. Detect anomalies with z-score, STL, and isolation-forest methods.
3. Build a 72-hour, cross-source evidence summary for each anomaly.
4. Generate explanations with a local Ollama model or cloud comparison models.
5. Check claims for context grounding and score them with deterministic,
   channel-aware corroboration rules.
6. Freeze anomaly sets, run three-model sweeps, collect blinded expert labels,
   and run source and channel ablations.

The evaluation set is frozen at `server/fixtures/eval50.json` and the
three-model sweep has run: 150 of 150 cells, 1,781 claims. The coverage result
below comes from that sweep and required no expert labels to compute.

Official expert labels have not been collected. The `expert_labels` table holds
zero rows, so the preregistered agreement analysis in
`server/app/eval/phase_analysis.py` has not been run and no agreement result
exists. There is no vector-retrieval pipeline, web frontend, interactive map, or
WebSocket service in the current repository.

## Coverage Results

| Outcome for the 1,781 claims | Count | Share |
| --- | --- | --- |
| Rejected by the grounding gate before any instrument was consulted | 860 | 48.3% |
| Passed the gate, but no channel could weigh in | 407 | 22.9% |
| Received a verdict | 514 | 28.9% |

Of the 514 scored claims, 331 were weighed by a single measurement channel and
183 by two. No claim in the corpus reached three. Adding sensors does not raise
that ceiling, because the constraint is which channels can physically bear on a
given claim type rather than how many instruments are deployed.

A further 235 claims cannot be scored regardless of what the instruments
recorded: 209 fall outside the ten-type taxonomy entirely, and 26 belong to a
type designated qualitative-only in June 2026. The taxonomy therefore forecloses
13 percent of the corpus independently of the sensor network.

Coverage is also not a fixed property of the sensor network. Across identical
events, identical stored evidence, and identical rules, the unscored fraction
varies substantially between the three models, so generator behavior accounts
for much of it. The per-model breakdown is withheld until expert labels are
returned, because the labeling is blinded and publishing it now would reveal
which model is which.

## Limits of the Result

Coverage is descriptive and was computed with no labels of any kind. The
following are open.

- **Agreement is untested.** Whether the corroboration score tracks expert
  judgment of the same claim is the preregistered question and remains
  unanswered. If the 514 verdicts are unreliable, the coverage figure is
  unanchored as well, which is what makes the labeling stage load-bearing rather
  than confirmatory.
- **No model is ranked.** Three models were run. Their relative accuracy is not
  reported and cannot be, absent labels.
- **Two scorer defects are known.** On claims reporting several measurements at
  once, the numeric comparator can match a value against the wrong species. In
  one claim type, intent keywords are selected by earliest match and do not read
  negation, so a sentence arguing against a mechanism can score as though it
  asserted one. Both inject disagreement attributable to the implementation
  rather than to the method, and neither rate has been measured. Both remain in
  place: the scorer ran under a protocol fixed in advance, and amending it after
  the sweep would invalidate that claim.

## The Corroboration Checker

The checker contains no language model. Every rule is explicit and
deterministic.

1. A claim is assigned to one of ten types (concentration elevation, transport
   direction, atmospheric trap, secondary formation, and others) or to
   `unclassified`, which is never scored.
2. A grounding gate checks the claim's cited sources, terms, and quantities
   against the evidence the model was shown. Claims that fail stop here.
3. Surviving claims are put only to the measurement channels that can physically
   bear on the assigned type. A transport-direction claim is checked against
   wind products and not against satellite columns.
4. Each channel returns supporting, contradicting, or silent. When every
   eligible channel is silent, the checker abstains rather than scoring.

Sources that share a measurement process collapse into one channel, so two
monitors operated under the same program cannot vote twice. That grouping is a
design choice about instrument physics, not a demonstration of statistical
independence. GFS assimilates ASOS observations, so the two meteorological
channels share inputs, and the residual correlation has not been measured. Any
independence claim must be tested rather than inferred from source names.

## Data Sources

Eight sources, grouped into five measurement channels.

| Source | Role | Cadence | Channel |
| --- | --- | --- | --- |
| OpenAQ | PM2.5 and ozone, restricted to entities verified as regulatory monitors | Hourly | Ground in-situ |
| TCEQ CAMS | Preliminary ground NO2, SO2, CO from a scraped public report | Hourly | Ground in-situ |
| EPA AQS | Historical ground NO2, SO2, CO; backfill only, certification status not retained | Backfill | Ground in-situ |
| PurpleAir | Low-cost optical PM2.5, time-aware quality screen applied, values uncorrected | Hourly | Ground optical |
| Sentinel-5P | Satellite NO2, SO2, CO, and HCHO columns | Daily when available | Satellite column |
| NOAA GFS | NWP meteorology, winds, and boundary-layer fields | 6-hour cycles | NWP |
| OpenWeather | Blended surface weather at five query points | Hourly | NWP |
| ASOS / METAR | Direct airport weather observations | Hourly | Met in-situ |

## Scope

The configured target is centered on Houston, Texas, with a default 50 km
radius. Point-based collectors apply that radius. Sentinel-5P column extraction
averages quality-filtered pixels inside the corresponding bounding box, so its
spatial footprint is not an exact 50 km circle.

The evaluation covers summer air-quality anomalies. Collection is not inherently
limited to that window.

## Evaluation Set and Preregistration

Every parameter below was locked before any model was run.

- Study window: 2026-06-01 to 2026-08-05 UTC, end-exclusive.
- 405,432 observations fall inside that window. The snapshot holds 466,507 rows
  in total; the remainder are the 2025 EPA AQS baseline and rows past the
  cutoff. Seven of the eight networks contribute in-window rows, since EPA AQS
  is historical only.
- 6,395 in-window anomalies, deduplicated to 2,304 events, from which the top 50
  were drawn and stratified by pollutant: NO2 18, ozone 11, CO 6, PM10 5,
  PM2.5 5, SO2 5.
- Frozen fixture `server/fixtures/eval50.json`, snapshot sha256 `1e50e007...`,
  code commit `5072549`.
- Models: Llama 3 8B served locally through Ollama, GPT-5.4, and Gemini 3.6
  Flash.

Anomaly identifiers are random UUIDs, so re-running detection regenerates them
and invalidates the freeze. Rows must be copied with identifiers intact.

The analysis plan was preregistered on OSF on 2026-08-09, before any expert saw
a claim: `osf.io/hb92y`, sole contributor, embargoed to 2027-06-30 so that it
does not become public during a blind review. The registered primary statistic
is a cluster-bootstrap Spearman correlation between the corroboration score and
the expert label, restricted to three headline claim types. Cohen's kappa in
`phase_analysis.py` measures inter-rater reliability between human labelers and
is not a machine-versus-expert statistic.

## Implemented Architecture

```text
Windows collector box
  -> seven scheduled live collectors
  -> SQLite edge database

Analysis workflow
  -> PostgreSQL + TimescaleDB analysis database
  -> anomaly detection
  -> cross-source enrichment
  -> LLM explanation generation
  -> context grounding and deterministic corroboration
  -> freeze, labeling, and ablation CLIs

FastAPI
  -> health endpoint
  -> data-source and paginated raw-data endpoints
```

The pipeline is ORM-only, so the edge and analysis tiers share one code path and
differ only in connection URL. Local and cloud model clients call HTTP APIs
directly through `httpx`. Structured database evidence is rendered into the
prompt; it is not retrieved from a vector store.

## Planned Product Architecture

The project name describes the intended complete system, not only the current
evaluation milestone. The remaining stages are:

- **RAG with ChromaDB:** retrieve relevant historical anomalies, validated
  explanations, and supporting environmental context for generation and
  natural-language queries.
- **React application:** anomaly feed, anomaly detail, evidence, evaluation,
  query, and system-status views.
- **Interactive map:** Mapbox GL display of monitors, weather fields, satellite
  coverage, anomalies, and supporting evidence.
- **FastAPI WebSockets:** stream collector health, new observations, detections,
  and explanation status to the frontend.
- **Autonomous operation:** continuously collect, detect, enrich, explain,
  evaluate, and publish new events behind explicit quality and confidence gates.

Retrieval is the priority among these. Evidence currently reaches the prompt as
structured rows read straight from the database, so the retrieval stage in the
project name is planned rather than implemented.

## Getting Started

### Prerequisites

- Python 3.11+
- Ollama with `llama3:8b` for local generation
- SQLite for edge collection, or PostgreSQL/TimescaleDB for analysis
- API credentials for the sources and cloud baselines intended to run

### Backend Setup

```bash
git clone https://github.com/mason-cao/aeris.git
cd aeris/server
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
pytest -q
uvicorn app.main:app --reload --port 8000
```

The suite reports 1,572 passed and 8 skipped in roughly one minute on
Python 3.13.

### Common Commands

```bash
# Run all registered live collectors, or one source.
python -m app.collectors.run_all
python -m app.collectors.run_all --source openaq

# Backfill a supported historical source.
python -m app.collectors.backfill --source epa_aqs \
  --since 2025-06-01 --until 2025-08-31

# Run detection and enrichment.
python -m app.detection.run
python -m app.detection.enrichment

# Generate one explanation.
python -m app.llm.explain --anomaly-id=<UUID>
```

`server/demo_checker.py` runs the scorer over one real event in the terminal,
claim by claim, showing the measurements behind each verdict. It reads the
August analysis snapshot, a 5 GB SQLite file kept out of the repository, so it
does not run on a fresh clone.

The freeze and labeling CLIs are included for inspection. Re-running either
would break the frozen evaluation set and the registered protocol.

### Environment Variables

See `server/.env.example`. The active settings are:

- `DATABASE_URL` and `DATABASE_URL_SYNC`
- `OPENAQ_API_KEY`, `OPENAQ_MAX_READING_AGE_S`, and `PURPLEAIR_API_KEY`
- `OPENWEATHER_API_KEY`
- `AQS_EMAIL` and `AQS_API_KEY`
- `CDSE_USERNAME` and `CDSE_PASSWORD`
- `OPENAI_API_KEY` and `GOOGLE_API_KEY`
- `AERIS_ENV`, `AERIS_LOG_LEVEL`, and the three `AERIS_TARGET_*` settings

## Roadmap

- [x] Live collector registry and historical backfill strategies
- [x] Anomaly detection and cross-source enrichment
- [x] LLM generation, grounding, corroboration, labeling, and ablation CLIs
- [x] Resolve provenance, quality-control, and scorer methodology blockers
- [x] Freeze the evaluation set and run the three-model sweep
- [ ] Collect official expert labels and run the preregistered analysis
- [ ] Add the ChromaDB retrieval layer and RAG evaluation
- [ ] Build the React and Mapbox application
- [ ] Add WebSocket-driven live updates and autonomous product workflows

## Acknowledgements

Dr. Annalisa Bracco is the scientific mentor for the attribution evaluation.
Lester Mackey provided methodological pointers on weighting evidential strength
across sources.

## License

MIT
