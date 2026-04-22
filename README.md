# A.E.R.I.S. &mdash; Autonomous Environmental RAG & Inference System

A self-hosted environmental intelligence platform that detects anomalies in real-time environmental data and generates causal explanations using a locally-hosted LLM with RAG and multi-source cross-referencing.

---

## The Problem

Environmental hazards vary block-by-block and change hour-by-hour. Public data exists across dozens of agencies (EPA, NOAA, NASA, USGS) in incompatible formats. When a pollution spike hits your neighborhood, no system automatically detects it, determines the cause, and tells you what to do.

## What AERIS Does

AERIS runs on a home server and:

1. **Aggregates** 9 real-time environmental data sources (air quality, weather, wildfires, satellite, traffic, water, energy)
2. **Detects anomalies** using a three-method engine (statistical, seasonal decomposition, isolation forest)
3. **Explains causes** via a locally-hosted LLM that cross-references all data sources through a RAG pipeline
4. **Visualizes** everything on an interactive map with real-time feeds, evidence panels, and natural language querying

All inference runs locally. No student data, health queries, or location data leaves the server.

## Architecture

```
Home Server (Always-On)
├── Data Collectors ──── 9 public APIs (EPA, OpenAQ, PurpleAir, NOAA, NASA FIRMS,
│                        Sentinel-5P, TomTom, USGS, EIA)
├── PostgreSQL + TimescaleDB ──── Time-series storage
├── Anomaly Detection ──── Z-score | STL decomposition | Isolation Forest
├── Ollama (Llama 3 8B) ──── Local LLM inference
├── ChromaDB ──── RAG vector store
└── FastAPI ──── REST API + WebSocket

Web Application (React)
├── Interactive Map ──── Mapbox GL JS with sensor/anomaly/satellite layers
├── Anomaly Feed ──── Real-time detected anomalies with LLM summaries
├── Anomaly Detail ──── Full explanation + evidence panels + health advisory
├── NL Query ──── "Why was air quality bad yesterday?"
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

| Source | Data | Frequency |
|--------|------|-----------|
| EPA AirNow | AQI, PM2.5, ozone, NO2, SO2, CO | Hourly |
| OpenAQ | Global government air quality monitors | Hourly |
| PurpleAir | Crowdsourced hyperlocal PM2.5 | ~2 min |
| NOAA / OpenWeather | Weather (temp, wind, humidity, pressure) | Hourly |
| NASA FIRMS | Active wildfire locations | ~3 hours |
| Sentinel-5P | Satellite NO2, SO2, CO columns | Daily |
| TomTom | Real-time traffic density | ~15 min |
| USGS Water Services | Stream flow, water quality | 15 min - hourly |
| EIA Open Data | Power plant emissions, grid carbon | Hourly - daily |

## Research

**Question**: Can locally-hosted LLMs generate accurate causal explanations for environmental anomalies by cross-referencing heterogeneous data sources?

**Evaluation**:
- Expert-labeled anomaly ground truth (50-100 events)
- Local (Llama 3 8B) vs. cloud (GPT 5.4, Gemini 3 Thinking) comparison
- Automated hallucination detection accuracy
- User comprehension and actionability study

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 15+ with TimescaleDB extension
- Ollama with Llama 3 8B pulled

### Setup

```bash
# Clone
git clone https://github.com/<your-username>/aeris.git
cd aeris

# Backend
cd server
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in API keys
uvicorn app.main:app --reload

# Frontend (new terminal)
cd client
npm install
npm run dev
```

### Environment Variables

Copy `server/.env.example` and fill in:
- `DATABASE_URL` &mdash; PostgreSQL connection string
- `AIRNOW_API_KEY` &mdash; EPA AirNow
- `OPENAQ_API_KEY` &mdash; OpenAQ
- `PURPLEAIR_API_KEY` &mdash; PurpleAir
- `OPENWEATHER_API_KEY` &mdash; OpenWeather
- `TOMTOM_API_KEY` &mdash; TomTom traffic
- `EIA_API_KEY` &mdash; EIA energy data
- `FIRMS_MAP_KEY` &mdash; NASA FIRMS active fire detections
- `MAPBOX_TOKEN` &mdash; Mapbox GL JS
- `NASA_EARTHDATA_TOKEN` &mdash; Sentinel-5P satellite data

## Roadmap

- [x] Design specification
- [ ] **Month 1**: Server infrastructure + data pipeline (9 collectors)
- [ ] **Month 2**: Anomaly detection engine + LLM explanation pipeline
- [ ] **Month 3**: Web application (map, feed, detail, query, dashboard)
- [ ] **Month 4**: Research evaluation + polish
- [ ] **Month 5**: Paper, competition submissions, stretch goals

## License

MIT
