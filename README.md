# A.E.R.I.S. - Autonomous Environmental RAG & Inference System

AERIS is a planned end-to-end environmental intelligence system for detecting
Houston-area anomalies, retrieving relevant evidence, generating and evaluating
LLM explanations, and presenting the results through a real-time map interface.
The repository currently contains the research backend and evaluation workflow.

## Current Status

The repository currently contains a Python backend and command-line research
workflow. It can:

1. Collect and normalize seven live environmental feeds, plus historical EPA
   AQS samples.
2. Detect anomalies with z-score, STL, and isolation-forest methods.
3. Build a 72-hour, cross-source evidence summary for each anomaly.
4. Generate explanations with local Ollama or cloud comparison models.
5. Check claims for context grounding and score them with deterministic,
   channel-aware corroboration rules.
6. Prepare frozen anomaly sets, run three-model sweeps, collect blinded expert
   labels, and run source/channel ablations.

The official evaluation has not been frozen, official expert labels have not
been collected, and the final statistical analysis is not implemented. There
is no vector-retrieval pipeline, web frontend, interactive map, or WebSocket
service in the current repository.

## Planned Product Architecture

The project name describes the intended complete system, not only the current
evaluation milestone. The planned product stages remain:

- **RAG with ChromaDB:** retrieve relevant historical anomalies, validated
  explanations, and supporting environmental context for model generation and
  natural-language queries.
- **React application:** provide the anomaly feed, anomaly detail, evidence,
  evaluation, query, and system-status views.
- **Interactive map:** use Mapbox GL to display monitors, weather fields,
  satellite coverage, anomalies, and supporting evidence spatially.
- **FastAPI WebSockets:** stream collector health, new observations, anomaly
  detections, and explanation status to the frontend.
- **Autonomous operation:** continuously collect, detect, enrich, explain,
  evaluate, and publish new events with explicit quality and confidence gates.

These are planned, crucial parts of AERIS. They are not removed from the
roadmap merely because the July research evaluation is being completed first.

## Scope

The configured target is centered on Houston, Texas, with a default 50 km
radius. Point-based collectors apply the configured radius. Sentinel-5P column
extraction currently averages quality-filtered pixels inside the corresponding
Houston bounding box, so its spatial footprint is not an exact 50 km circle.

The planned evaluation focuses on summer air-quality anomalies. Collection is
not inherently limited to that evaluation window.

## Implemented Architecture

```text
Windows collector box
  -> seven scheduled live collectors
  -> SQLite edge database

Analysis workflow
  -> optional PostgreSQL + TimescaleDB analysis database
  -> anomaly detection
  -> cross-source enrichment
  -> LLM explanation generation
  -> context-grounding and deterministic corroboration
  -> frozen-set, labeling, and ablation CLIs

FastAPI
  -> health endpoint
  -> data-source and paginated raw-data endpoints
```

The local and cloud model clients call HTTP APIs directly through `httpx`.
Structured database evidence is rendered into the prompt; it is not retrieved
from a vector store.

## Data Sources

| Source | Role | Cadence | Implementation status |
| --- | --- | --- | --- |
| OpenAQ | PM2.5 and ozone from mixed ground networks | Hourly | Live; PM2.5 provider/instrument classification is unresolved |
| TCEQ CAMS | Preliminary ground NO2, SO2, and CO | Hourly | Live; scraped public report |
| EPA AQS | Historical ground NO2, SO2, and CO | Backfill only | Stored rows are one-hour AQS samples; certification status is not retained |
| PurpleAir | Low-cost optical PM2.5 | Hourly | Live; official use still needs correction and outlier policy |
| Sentinel-5P | Satellite NO2, SO2, CO, and HCHO columns | Daily when available | Live catalog and column extraction |
| NOAA GFS | NWP meteorology, winds, and boundary-layer fields | 6-hour cycles | Live |
| OpenWeather | Blended surface weather at five query points | Hourly | Live; no free historical backfill |
| ASOS / METAR | Direct airport weather observations | Hourly | Live |

The scorer groups sources into measurement-process channels. Those groups are
a research design choice, not proof of statistical independence. In particular,
the current OpenAQ PM2.5 block mixes provider classes, and NWP products can
share model inputs or assimilated observations with direct weather feeds. Any
independence claim must be tested rather than inferred from source names.

## Research Question

The planned study asks whether agreement across process-distinct measurement
channels can act as an automated signal for the quality of LLM explanations of
environmental anomalies, and how a local Llama 3 8B model compares with GPT-5.4
and Gemini 3.5 Flash baselines.

Implemented evaluation infrastructure includes:

- A context-grounding gate with source, term, and numeric checks.
- A ten-type deterministic claim scorer.
- A channel-aware aggregation rule and leave-one-source/channel-out ablation.
- A resumable three-model harness.
- A blinded per-claim expert-labeling CLI.

Planned but not yet implemented or completed:

- The official anomaly freeze and model-output set.
- Official expert labels and inter-rater reliability.
- Confidence intervals, clustered inference, calibration analysis, and final
  model comparisons.
- Empirical validation of the proposed channel grouping.

## Getting Started

### Prerequisites

- Python 3.11+
- Ollama with `llama3:8b` for local generation
- SQLite for edge collection or PostgreSQL/TimescaleDB for analysis
- API credentials for the sources and cloud baselines you intend to run

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

### Common Commands

```bash
# Run all registered live collectors or one source.
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

Do not run the freeze or official-label commands until the evaluation protocol,
data-quality rules, and official data snapshot have been locked.

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
- [ ] Resolve provenance, quality-control, and scorer methodology blockers
- [ ] Freeze and run the official evaluation
- [ ] Collect official labels and implement the statistical analysis
- [ ] Add the ChromaDB retrieval layer and RAG evaluation
- [ ] Build the React and Mapbox application
- [ ] Add WebSocket-driven live updates and autonomous product workflows

## Acknowledgements

Dr. Annalisa Bracco is the scientific mentor for the attribution evaluation.

## License

MIT
