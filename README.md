# A.E.R.I.S. - Autonomous Environmental RAG & Inference System

A self-hosted environmental intelligence platform that detects anomalies in real-time environmental data and generates causal explanations using a locally-hosted LLM with RAG and multi-source cross-referencing.

---

## The Problem

Attributing complex climate events and anomalies to underlying atmospheric physics is a highly specialized task. The necessary data (vertical temperature profiles, wind vectors, and satellite imaging) exists, but it is scattered across various meteorological agencies in different formats. 

Big cloud models could probably reason through this if you piped enough data in, but running GPT-class inference on a 24/7 stream isn't realistic for independent deployment. Small local models are cheap to run, but they hallucinate badly when asked to reason about climate physics and AI attribution. AERIS is an attempt to bridge that gap: an 8B model with a specialized RAG pipeline and structured cross-referencing to perform accurate climate attribution locally.

## What AERIS Does

AERIS runs on a home server and:

1. **Aggregates** real-time atmospheric and climate data sources (including OpenWeather and Sentinel-5P)
2. **Detects anomalies** using a three-method engine (statistical, seasonal decomposition, isolation forest)
3. **Explains causes** via a locally-hosted LLM that cross-references all data sources through a RAG pipeline
4. **Visualizes** everything on an interactive map, translating complex atmospheric anomalies into actionable regional health advisories and natural language summaries.

All inference runs locally, ensuring complete data privacy and independent operation.

## Scope

Geographic target: a 50km radius around downtown Houston, Texas. Houston was chosen for three high-contrast inputs that stress-test a 4-API attribution model: massive petrochemical emissions from the Ship Channel refinery complex, a dense government sensor network (EPA + TCEQ + harbor monitors), and dynamic Gulf-coast weather (sea-breeze fronts, hurricane corridor, frequent inversions). All collectors filter to this bounding box; the center coordinate is configurable via `AERIS_TARGET_LAT` / `AERIS_TARGET_LON` / `AERIS_TARGET_RADIUS_KM`.

## Architecture

```
Home Server (Always-On)
├── Data Collectors ──── 4 Macro APIs (NOAA GFS, OpenWeather, Sentinel-5P, OpenAQ)
├── PostgreSQL + TimescaleDB ──── Time-series storage
├── Anomaly Detection ──── Z-score | STL decomposition | Isolation Forest
├── Ollama (Llama 3 8B) ──── Local LLM inference
├── ChromaDB ──── RAG vector store
└── FastAPI ──── REST API + WebSocket

Web Application (React)
├── Interactive Map ──── Mapbox GL JS with meteorological/anomaly/satellite layers
├── Anomaly Feed ──── Real-time detected anomalies with LLM attribution summaries
├── Anomaly Detail ──── Full physics explanation + downstream regional health advisories
├── NL Query ──── "What atmospheric conditions caused the temperature inversion yesterday?"
└── System Dashboard ──── Collection status, model metrics, server health
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11+, FastAPI, SQLAlchemy |
| Database | PostgreSQL + TimescaleDB |
| Vector Store | ChromaDB |
| Local LLM | Ollama (Llama 3 8B) |
| ML | scikit-learn, statsmodels |
| Frontend | React 18, TypeScript, Vite |
| Mapping | Mapbox GL JS |
| Charts | Recharts |
| Styling | Tailwind CSS |

## Data Sources

| Source | Data | Frequency | Status |
|--------|------|-----------|--------|
| NOAA Global Forecast System (GFS) | Upper-air temperature & geopotential height, 10 m winds, boundary layer height, surface pressure, precipitable water | 6 hours | Live |
| OpenWeather | Surface temperature, humidity, pressure, wind speed/direction, cloud cover, precipitation | Hourly | Live |
| Sentinel-5P | Satellite atmospheric chemistry (NO2, SO2, CO, HCHO column densities) | Daily | Live |
| OpenAQ | Global baseline atmospheric sensor data | Hourly | Live |

## Research

**Question**: Can a locally-hosted 8B model match cloud GPT-class accuracy on climate-anomaly attribution when paired with a structured cross-source RAG pipeline and hallucination detection? Linking weather patterns to environmental events is well-established science; the open question is whether a small, locally-hosted model can do it reliably enough for independent deployment.

**Contributions**:
1. A four-API cross-referencing architecture that gives an 8B model the structured context it needs to reason about heterogeneous atmospheric data.
2. An automated hallucination-detection layer specialized for climate-physics claims, evaluated against intentionally-injected false statements.
3. An empirical local-vs-cloud comparison on a domain where small models are widely assumed to fail.

**Evaluation**:
- Expert-labeled anomaly ground truth (50-100 events, kept broad across categories — petrochemical upsets, ozone exceedances, wildfire-smoke transport, hurricanes, hard freezes — to test cross-category generalization, not category-specific tuning)
- Local (Llama 3 8B) vs. cloud (GPT 5.4, Gemini 3 Thinking) attribution accuracy
- Hallucination-detection precision/recall on injected false claims
- User comprehension and actionability study with non-expert participants

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ with TimescaleDB extension
- Ollama with Llama 3 8B pulled
- Node.js 18+ (Month 3, frontend)

### Setup

```bash
# Clone
git clone https://github.com/mason-cao/aeris.git
cd aeris

# Backend
cd server
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in API keys
uvicorn app.main:app --reload

# Frontend — Month 3 (client/ is not scaffolded yet)
# cd client
# npm install
# npm run dev
```

### Environment Variables

Copy `server/.env.example` and fill in:
- `DATABASE_URL` - PostgreSQL connection string
- `OPENAQ_API_KEY` - OpenAQ
- `OPENWEATHER_API_KEY` - OpenWeather
- `CDSE_USERNAME` / `CDSE_PASSWORD` - Copernicus Data Space (Sentinel-5P granule downloads)
- `NASA_EARTHDATA_TOKEN` - NASA Earthdata fallback (optional)
- `OPENAI_API_KEY` / `GOOGLE_API_KEY` - cloud LLM comparison (GPT-5.4, Gemini 3 Thinking)
- `MAPBOX_TOKEN` - Mapbox GL JS (Month 3, frontend)

## Roadmap

- [x] Design specification
- [x] **Month 1**: Server infrastructure + data pipeline (all four macro APIs live — OpenAQ, OpenWeather, Sentinel-5P column extraction, NOAA GFS analysis)
- [ ] **Month 2**: Anomaly detection engine + LLM explanation pipeline *(in progress — detection engine complete)*
- [ ] **Month 3**: Web application (map, feed, detail, query, dashboard)
- [ ] **Month 4**: Research evaluation + polish
- [ ] **Month 5**: Paper, competition submissions, stretch goals

## Acknowledgements

Dr. Annalisa Bracco, Senior Scientist @ CMCC & Professor, Georgia Institute of Technology - Formal mentor for the AI attribution phase

## License

MIT
