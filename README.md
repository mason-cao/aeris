# A.E.R.I.S. - Autonomous Environmental RAG & Inference System

**Measuring how much of what a language model says about air quality can
actually be checked against real instruments.**

Ask a language model why air pollution spiked somewhere and it will give a
confident, detailed answer every time. The hard part is telling a correct
explanation apart from one that only sounds right, and air quality has no answer
key. Nobody recorded the true cause of the nitrogen dioxide spike southeast of
Houston at 23:00 UTC on 15 June 2026.

The instruments do overlap, though. A claim about wind direction can be checked
against three separate wind sources. A claim about a concentration can be
checked against the monitor that recorded it, and sometimes against a satellite
passing overhead. AERIS builds a checker on that overlap. The checker runs on
fixed rules and has no language model inside it. The study measures how much it
can actually check, and whether its verdicts line up with expert judgment.

| | |
| --- | --- |
| Study window | 1 June to 5 August 2026, Houston, Texas |
| Observations | 405,432 in the window from seven live networks, plus a 2025 EPA AQS baseline |
| Events | 50 detected anomalies, stratified by pollutant |
| Models | Llama 3 8B (run locally), GPT-5.4, Gemini 3.6 Flash |
| Claims | 1,781 extracted, 514 scored (401 once parsing errors are corrected) |
| Preregistration | OSF `osf.io/hb92y`, registered 9 August 2026, embargoed until 30 June 2027 |
| Status | First-author labels complete and analyzed. A second labeler's review is in progress. |

The three models explained the same 50 events, and their explanations break down
into 1,781 separate claims. The checker sorts each claim into one of ten types
and compares it only against the kinds of measurement that can physically say
something about that type. If none of them can, it doesn't guess. It gives no
verdict.

The checker reached a verdict on 514 of the 1,781 claims, or 28.9 percent. Of
the other 1,267, 860 were stopped at a first check because they didn't match the
evidence the model had been given, and 407 passed that check but no instrument
could confirm or deny them. No verdict doesn't mean a claim is false, only that
the checker couldn't test it.

That fraction matters. A checker's accuracy score only covers the claims it was
able to check, so if that's under a third of what the model said, the score
describes a small slice of the output.

Coverage needs no expert labels. The second question, whether the checker's
verdicts agree with expert judgment, does. The primary labels come from the
first author, who also wrote the checker's rules, so they can't show agreement
with anyone else. A second labeler, who has advised the project but didn't write
the rules, is labeling a subset without seeing the first author's labels, the
checker's scores, or which model wrote what. Those comparisons are still
pending.

## Where things stand

The repo has a Python backend and a set of command-line tools that can:

1. Collect and normalize seven live environmental data feeds, plus historical
   EPA AQS samples.
2. Detect anomalies with z-score, STL, and isolation-forest methods.
3. Build a 72-hour, cross-source evidence summary for each anomaly.
4. Generate explanations with a local Ollama model or the two cloud models.
5. Check each claim against the evidence the model was shown, then score it
   with fixed, channel-aware corroboration rules.
6. Freeze anomaly sets, run three-model sweeps, collect blinded expert labels,
   and run source and channel ablations.

The evaluation set is frozen at `server/fixtures/eval50.json`, and the
three-model sweep has run: all 150 model-event pairs, 1,781 claims. The coverage
numbers below come straight from that sweep.

First-author labeling of all 50 events finished in September, with every claim
marked Valid, Invalid, or Unsure (701, 102, and 978). The marks were read off
the returned PDFs twice, on separate days, by two separately written extraction
methods, and all 1,781 decisions matched. Before import, every claim was matched
against the released packets, the frozen packet database, and the live
database.

The preregistered statistics ran with the analysis code in
`server/app/eval/phase_analysis.py` unchanged. The full analysis command expects
two labelers and stops with `zero overlap pairs` when there's only one, so a
small local script called the same first-author functions directly, with
identical inputs, filters, seeds, and settings. The second labeler's marks
haven't been inspected. Inter-rater agreement and the same-claim comparisons
come after that return. On 24 September an OSF update applied the
registration's SO2 exclusion, which the first run had missed, and declared three
deviation checks before they were computed.

The name describes the full planned system. Collection, detection, and checking
are built. There's no vector retrieval yet (the "RAG" in the name), no web
frontend or map, and no WebSocket service. Those are under
[What's planned](#whats-planned).

## Coverage results

| What happened to the 1,781 claims | Count | Share |
| --- | --- | --- |
| Stopped by the grounding check before any instrument was consulted | 860 | 48.3% |
| Passed the check, but no channel could weigh in | 407 | 22.9% |
| Got a verdict | 514 | 28.9% |

Of the 514 scored claims, 331 were checked by one measurement channel and 183
by two. None reached three. More monitors of the same kind wouldn't raise that
ceiling, because instruments that measure the same way count as one channel.
The limit is which kinds of measurement can physically speak to a claim type.

Those counts come from the frozen scorer. With the parsing errors described
under the limits corrected, 401 claims (22.5%) get a verdict, so the frozen
figure overstates coverage rather than understating it.

Looked at another way, 235 of the 1,781 claims could never have been scored,
whatever the instruments recorded. 209 don't fit any of the ten types, and
unclassified claims are never scored. The other 26 are point-source attribution
claims, a type designated qualitative-only in June 2026. Its rule only checks
claims that give coordinates, and the two that did were stopped at the grounding
check. So the taxonomy and its rules close off about 13 percent of the corpus
no matter what sensors are deployed.

Coverage also isn't a fixed property of the sensor network. With the same
events, the same stored evidence, and the same rules, the unscored share varies
a lot between the three models, so how each model writes accounts for much of
it. The per-model breakdown stays private until the second labeler finishes, so
that labeling stays blind.

## First-author results

The primary pool is the grounded, scored claims from the three registered
headline types (concentration elevation, transport direction, and
meteorological state). The registration keeps SO2 out of headline inference, so
claims from the five SO2 events are left out. That leaves 156 claims, and
leaving out the 50 marked Unsure leaves 106 claims across 41 anomalies.
Intervals come from 10,000 bootstrap resamples of whole anomalies.

| Analysis | Registered | First run, SO2 left in by mistake | Parser-corrected check |
| --- | --- | --- | --- |
| Primary Spearman, Unsure excluded | 0.119 (-0.014 to 0.251) | 0.156 (0.030 to 0.281) | 0.342 (0.170 to 0.511) |
| Unsure counted as Invalid | 0.081 (-0.077 to 0.233) | 0.080 (-0.057 to 0.215) | 0.283 (0.108 to 0.449) |
| Sign-mapped kappa (exploratory) | -0.005 (-0.074 to 0.060) | 0.023 (-0.044 to 0.090) | 0.057 (-0.052 to 0.167) |
| Claim-length control Spearman | 0.562 (0.437 to 0.664) | 0.591 (0.474 to 0.685) | 0.572 (0.433 to 0.679) |
| Sign accuracy vs. always guessing Valid | 30.8% vs 57.1% | 31.8% vs 53.8% | 42.5% vs 55.2% |
| Claims / anomalies in the primary | 106 / 41 | 113 / 46 | 90 / 40 |

The registered primary interval includes zero, so the registration's prediction
isn't supported. The first run on 21 September left SO2 in because the analysis
code never got a pollutant filter. An OSF update on 24 September applied the
exclusion as registered before the SO2-excluded numbers were computed. In the
registered results, a screen that flags negative scores has 17.0% precision and
catches 10.8% of the headline claims labeled Invalid.

The same update declared three deviation checks before they were computed.
Removing SO2 claim by claim instead of event by event gives 0.162 (0.033 to
0.289). Dropping the 42 claims scored on the wrong number gives 0.361 (0.177 to
0.531). Re-scoring every claim with the corrected parser gives 0.342, and its
Unsure-as-Invalid version also stays above zero. That suggests the scorer bugs,
more than the method itself, account for much of the weak registered result.

It still isn't a strong result. The checks were declared after the first results
were known, so they can't replace the registered numbers. Even with the parser
fixed, claim length correlates with the labels more strongly than the score
does, the score's sign matches the label less often than always guessing Valid,
and a screen that flags negative scores catches 8.1% of the headline claims
labeled Invalid. All of these numbers compare the checker with the judgments of
the person who wrote its rules.

## What these results don't show

Coverage is descriptive and didn't use any labels. The label results have real
limits.

- The primary labeler wrote the scoring rules, which is the study's main
  limitation. Neither labeler is independent of the project, since the second
  has advised it, and the second labeler's subset has only 20 scored headline
  claims to compare on.
- Unsure matters. The primary leaves out uncertain judgments, and those might
  not be missing at random. Several exploratory breakdowns can't be computed at
  all, because the scores or labels in them are constant or there are too few
  claims.
- The model comparisons are narrow. The unchanged code compares per-anomaly
  averages and leaves out qualitative-only types. It doesn't compare the models
  on identical claims, and it doesn't account for them writing different kinds
  of claims. Model-specific results stay private while the second labeler's
  review is in progress.
- The first run included SO2 by mistake. The registration keeps SO2 out of
  headline inference because almost all SO2 baseline readings sit below
  detection limits, but the frozen analysis code never got a pollutant filter.
  The registered results above apply the exclusion, as filed in an OSF
  correction on 24 September 2026, and the first-run values are reported beside
  them.
- The frozen scorer misreads a real share of claims. A label-free audit in
  September 2026 read every scored claim and found these problems:
  - numbers compared against the wrong pollutant (19 concentration claims), and
    dates, durations, percentages or distances read as concentrations (23 more)
  - wind speeds read as boundary-layer heights (20 claims), and heights written
    in meters not read at all, which left 14 checkable claims unscored
  - 99 claims whose meaning gets reversed, mostly sentences arguing against a
    mechanism that are scored as if they asserted it

  A corrected scorer re-scored every claim without looking at any label. It was
  generated from the frozen one by small exact patches, with claim routing,
  thresholds and aggregation unchanged, and it abstains on the 99 misread
  claims. It returns a verdict for 401 claims (22.5%) instead of 514, and among
  headline claims it contradicts 45 of 150 instead of 83 of 173. The fixes cut
  both ways: 48 supporting verdicts are withdrawn along with 86 contradicting
  ones. The registered results use the frozen scorer. The corrected scores are
  the parser-corrected check in the table above, declared in the OSF update
  before they were compared with any label.

## How the checker works

The checker has no language model in it. Every rule is written out and gives the
same answer every time.

1. Each claim gets one of ten types (concentration elevation, transport
   direction, atmospheric trap, secondary formation, and others) or
   `unclassified`, which is never scored.
2. A grounding check compares the claim's cited sources, terms, and numbers with
   the evidence the model was shown. Claims that fail stop here.
3. Claims that pass are compared only with the measurement channels that can
   physically speak to their type. A transport-direction claim is checked
   against wind data, not satellite columns.
4. Each channel returns supports, contradicts, or silent. If every eligible
   channel is silent, the checker gives no verdict instead of a score.

Sources that measure the same way are grouped into one channel, so two monitors
run under the same program can't vote twice. That grouping is a design choice
based on how the instruments work. It doesn't prove the channels are
statistically independent. GFS assimilates ASOS observations, for example, so
the two weather channels share inputs, and the leftover correlation between them
hasn't been measured. Independence has to be measured before it can be claimed.

## Data sources

Eight sources, grouped into five measurement channels.

| Source | Role | Cadence | Channel |
| --- | --- | --- | --- |
| OpenAQ | PM2.5, PM10, and ozone, restricted to entities verified as regulatory monitors | Hourly | Ground in-situ |
| TCEQ CAMS | Preliminary ground NO2, SO2, CO from a scraped public report | Hourly | Ground in-situ |
| EPA AQS | Historical ground NO2, SO2, CO; backfill only, certification status not retained | Backfill | Ground in-situ |
| PurpleAir | Low-cost optical PM2.5, time-aware quality screen applied, values uncorrected | Hourly | Ground optical |
| Sentinel-5P | Satellite NO2, SO2, CO, and HCHO columns | Daily when available | Satellite column |
| NOAA GFS | NWP meteorology, winds, and boundary-layer fields | 6-hour cycles | NWP |
| OpenWeather | Blended surface weather at five query points | Hourly | NWP |
| ASOS / METAR | Direct airport weather observations | Hourly | Met in-situ |

## Scope

The study area is centered on Houston, Texas, with a default 50 km radius.
Collectors that report by point apply that radius. Sentinel-5P columns are
averaged over quality-filtered pixels inside the matching bounding box, so the
satellite footprint isn't an exact 50 km circle.

The evaluation covers summer air-quality anomalies, but collection isn't limited
to that window.

## Evaluation set and preregistration

Every parameter below was locked before any model was run.

- Study window: 2026-06-01 to 2026-08-05 UTC, end-exclusive.
- 405,432 observations fall inside that window. The snapshot holds 466,507 rows
  in total. The rest are 44,168 EPA AQS rows from summer 2025, 11,554 rows
  collected before 1 June, and 5,353 rows after the cutoff. Seven of the eight
  networks have rows in the window, since EPA AQS is historical only.
- 6,395 anomalies in the window, merged into 2,304 events. The top 50 were drawn
  with stratification by pollutant: NO2 18, ozone 11, CO 6, PM10 5, PM2.5 5,
  SO2 5.
- Frozen fixture `server/fixtures/eval50.json`, snapshot sha256 `1e50e007...`,
  code commit `5072549`.
- Models: Llama 3 8B served locally through Ollama, GPT-5.4, and Gemini 3.6
  Flash.

Anomaly IDs are random UUIDs, so re-running detection generates new ones and
breaks the freeze. Rows have to be copied with their IDs intact.

The analysis plan was preregistered on OSF on 2026-08-09, before any expert saw
a claim: `osf.io/hb92y`, with a single contributor. It's embargoed until
2027-06-30 so it doesn't go public in the middle of blind review. The registered
primary statistic is a cluster-bootstrap Spearman correlation between the
corroboration score and the expert label, restricted to the three headline claim
types. Cohen's kappa in `phase_analysis.py` measures agreement between human
labelers. It isn't a machine-versus-expert statistic.

## What's implemented

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

The pipeline only talks to the database through the ORM, so the edge and
analysis tiers share one code path and differ only in the connection URL. The
local and cloud model clients call their HTTP APIs directly through `httpx`.
Evidence from the database goes into the prompt as structured rows. It isn't
retrieved from a vector store.

## What's planned

These parts of the system aren't built yet:

- Retrieval with ChromaDB, to pull relevant past anomalies, validated
  explanations, and supporting environmental context into generation and
  natural-language questions.
- A React app with an anomaly feed, anomaly detail, evidence, evaluation, query,
  and system-status views.
- An interactive Mapbox GL map of monitors, weather fields, satellite coverage,
  anomalies, and supporting evidence.
- FastAPI WebSockets that stream collector health, new observations, detections,
  and explanation status to the frontend.
- Autonomous operation that keeps collecting, detecting, enriching, explaining,
  evaluating, and publishing new events, behind explicit quality and confidence
  gates.

Retrieval comes first. Right now evidence reaches the prompt as rows read
straight from the database, so the retrieval stage in the name doesn't exist
yet.

## Getting started

### Prerequisites

- Python 3.11+
- Ollama with `llama3:8b` for local generation
- SQLite for edge collection, or PostgreSQL/TimescaleDB for analysis
- API credentials for whichever sources and cloud models you want to run

### Backend setup

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

With a local Postgres running, all 1,580 tests pass in under a minute on
Python 3.13. Without one, the 8 Postgres integration tests skip (1,572 passed,
8 skipped).

### Common commands

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

`server/demo_checker.py` runs the checker over one real event in the terminal,
claim by claim, and shows the measurements behind each verdict. It reads the
August analysis snapshot, a 5 GB SQLite file that isn't in the repo, so it won't
run on a fresh clone.

The freeze and labeling CLIs are included so they can be read. Re-running either
one would break the frozen evaluation set and the registered protocol.

### Environment variables

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
- [x] Verify and import first-author labels, and run the preregistered first-author statistics
- [x] Audit every scored claim for parsing errors and re-score with a corrected parser
- [x] File the OSF SO2 correction and compute the registered results and deviation checks
- [ ] Take in the second labeler's return and run the registered overlap analyses
- [ ] Add the ChromaDB retrieval layer and RAG evaluation
- [ ] Build the React and Mapbox application
- [ ] Add WebSocket-driven live updates and autonomous product workflows

## Acknowledgements

Dr. Annalisa Bracco is the scientific mentor for the attribution evaluation.
Lester Mackey provided methodological pointers on weighting evidential strength
across sources.

## License

MIT
