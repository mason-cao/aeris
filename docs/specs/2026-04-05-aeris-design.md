# AERIS (Autonomous Environmental RAG & Inference System) — Design Specification

## Context

Mason is a high school student with 4-5 months to build a flagship capstone project for college applications (CS/CE major), competition submissions (ISEF, Regeneron STS, Congressional App Challenge), and potential academic publication. He has guidance from CS professors and undergraduates, strong Python + React skills, and access to LLM APIs (OpenAI, Anthropic) plus an old PC to use as a home server.

**The problem**: Environmental hazards (air pollution spikes, water quality changes, extreme heat events) vary block-by-block and change hour-by-hour. Public environmental data exists but is scattered across dozens of agencies in incompatible formats that no normal person can interpret. When an anomaly occurs — a sudden PM2.5 spike, an unusual ozone reading — there is no system that automatically detects it, cross-references multiple data sources to determine the cause, and explains it in plain language.

**The opportunity**: Self-hosted LLMs running on commodity hardware can now perform sophisticated reasoning tasks. By combining real-time environmental data aggregation with LLM-powered causal explanation, we can build a system that democratizes environmental intelligence — and study whether locally-hosted models can do this accurately.

---

## Project Overview

**AERIS (Autonomous Environmental RAG & Inference System)** is a self-hosted environmental intelligence platform that:
1. Aggregates 9 real-time environmental data sources on a home server
2. Detects anomalies using a three-method statistical + ML detection engine
3. Generates causal explanations via a locally-hosted LLM with RAG, multi-source cross-referencing, and hallucination detection
4. Presents everything through a React web app with an interactive map, real-time anomaly feeds, and natural language querying

**Research question**: *Can locally-hosted LLMs generate accurate causal explanations for environmental anomalies by cross-referencing heterogeneous data sources?*

---

## Deployment Scope

**Geographic focus**: Start with a single metropolitan area (Mason's local area or a data-rich city like Los Angeles, Houston, or New York). All data sources have strongest coverage in US metro areas. Expanding to additional regions is a month 5 stretch goal.

**Hardware requirements for home server**:
- CPU: 4+ cores (Intel i5/i7 or AMD Ryzen 5+ recommended)
- RAM: 16 GB minimum (Llama 3 8B needs ~8 GB for inference + overhead for DB and collectors)
- Storage: 256 GB+ SSD (time-series data grows ~1-2 GB/month at full collection rate)
- GPU: Optional but helpful. NVIDIA GTX 1060+ with 6GB VRAM significantly speeds Ollama inference. CPU-only works but explanations take ~30-60s instead of ~5-10s.
- Network: Stable internet connection (data collection runs 24/7). Static IP or DDNS for remote access.

If the old PC doesn't meet RAM or CPU requirements, a used workstation (Dell Optiplex, HP ProDesk) can be found for $100-200 and would be more than sufficient.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  LAYER 1: Home Server (Always-on Data Engine)       │
│  ───────────────────────────────────────────────    │
│  • 9 data collectors (cron-scheduled)               │
│  • PostgreSQL + TimescaleDB (time-series storage)   │
│  • Three-method anomaly detection engine            │
│  • Ollama (Llama 3 8B / Mistral 7B)                │
│  • ChromaDB (RAG vector store)                      │
│  • FastAPI (all endpoints)                          │
└──────────────────────┬──────────────────────────────┘
                       │ REST API / WebSocket
┌──────────────────────▼──────────────────────────────┐
│  LAYER 2: Web Application                           │
│  ───────────────────────────────────────────────    │
│  • React 18 + TypeScript + Vite                     │
│  • Mapbox GL JS (interactive map)                   │
│  • Recharts (data visualization)                    │
│  • Tailwind CSS                                     │
│  • 5 core pages (map, feed, detail, query, dash)    │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  LAYER 3: LLM Intelligence (Research Core)          │
│  ───────────────────────────────────────────────    │
│  • RAG over environmental knowledge base            │
│  • Multi-source context assembly                    │
│  • Structured explanation generation                │
│  • Post-processing hallucination detection          │
│  • Evaluation framework                             │
└─────────────────────────────────────────────────────┘
```

---

## Data Sources (9 total)

All free/public APIs. Common normalized schema: `(timestamp, lat, lon, metric, value, source, raw_json)`

### Core Environmental

| # | Source | Data | Update Freq | API |
|---|--------|------|-------------|-----|
| 1 | EPA AirNow | Official AQI, PM2.5, ozone, NO2, SO2, CO | Hourly | AirNow API (free key) |
| 2 | OpenAQ | Global government-grade air quality monitors | Hourly | OpenAQ v3 API (open) |
| 3 | PurpleAir | Crowdsourced hyperlocal PM2.5 sensors | ~2 min | PurpleAir API (free key) |
| 4 | NOAA / OpenWeather | Temp, humidity, wind speed/dir, pressure, precip | Hourly | OpenWeather API (free tier) |

### Cross-Reference Sources

| # | Source | Data | Update Freq | API |
|---|--------|------|-------------|-----|
| 5 | NASA FIRMS | Active wildfire locations + intensity | ~3 hours | FIRMS API (open) |
| 6 | Sentinel-5P (NASA Earthdata) | Satellite NO2, SO2, CO column density | Daily | Earthdata API (free account) |
| 7 | TomTom / HERE | Real-time traffic density and congestion | ~15 min | TomTom API (free tier, 2500 req/day) |

### Supplementary Sources

| # | Source | Data | Update Freq | API |
|---|--------|------|-------------|-----|
| 8 | USGS Water Services | Stream flow, water temperature, conductivity | 15 min - hourly | USGS Water Services (open) |
| 9 | EIA Open Data | Power plant emissions, grid carbon intensity | Hourly - daily | EIA API (free key) |

### Static Reference Data (loaded once, updated periodically)

- **EPA TRI**: Industrial facility locations + annual emissions (toxic release inventory)
- **EPA ECHO**: Facility compliance violations
- **NWS Alerts**: Severe weather and air quality alerts (real-time via RSS/API)

---

## Anomaly Detection Engine

Three complementary methods running on each new data batch:

### Method 1: Statistical (Z-Score on Rolling Windows)
- Compute rolling 7-day mean and standard deviation per (location, metric)
- Flag data points exceeding ±3σ as anomalies
- **Catches**: Sudden spikes and drops
- **Misses**: Gradual drift, seasonal patterns

### Method 2: Seasonal Decomposition (STL)
- Apply STL decomposition to separate trend, seasonal, and residual components
- Flag anomalies in the residual component (beyond ±2.5σ of residual distribution)
- **Catches**: Anomalies hidden by time-of-day or seasonal patterns
- **Misses**: Novel patterns not in historical data

### Method 3: Isolation Forest (Multivariate ML)
- scikit-learn `IsolationForest` trained on multivariate features: (metric_value, temperature, humidity, wind_speed, traffic_density, hour_of_day, day_of_week)
- Retrained weekly on rolling 30-day data
- **Catches**: Multivariate anomalies (e.g., PM2.5 high + wind high is more anomalous than PM2.5 high + wind calm)
- **Misses**: Depends on feature engineering

### Anomaly Enrichment
When any method flags an anomaly:
1. Gather concurrent data from ALL other sources for that time/location
2. Pull 72-hour historical context for the anomaly metric
3. Identify nearby industrial facilities (TRI data) within 25 km
4. Check for active wildfires (FIRMS) within 100 km
5. Retrieve satellite observations if available
6. Check NWS active alerts
7. Package all context into a structured anomaly record → pass to LLM pipeline

### Consensus & Severity
- **Minor**: Flagged by 1 method
- **Moderate**: Flagged by 2 methods
- **Severe**: Flagged by all 3 methods, OR value exceeds EPA "Hazardous" threshold
- De-duplication: Anomalies within 30 min and 10 km of each other for the same metric are merged

---

## LLM Explanation Pipeline

### Step 1: Context Assembly
Input: Enriched anomaly record
Output: Structured context document (~1500-3000 tokens) containing:
- Anomaly summary (metric, value, z-score, location, time)
- 72-hour time-series summary for the metric
- Weather conditions at anomaly time
- Traffic density at anomaly time
- Active fires within 100 km (with distance, direction, intensity)
- Satellite observations (NO2/SO2 column density if available)
- Nearby industrial facilities (name, type, distance, recent violations)
- Power plant data for the region
- Water quality data if relevant
- Historical baseline comparison

### Step 2: RAG Retrieval
ChromaDB vector store containing:
- EPA AQI breakpoints and health guidelines
- Environmental science reference material (atmospheric chemistry, meteorology patterns)
- Known pollution patterns (temperature inversions, wildfire smoke transport, industrial incident profiles)
- Local geography context (terrain, water bodies, urban heat islands)

Query: Anomaly type + top concurrent conditions → retrieve top 5 most relevant chunks

### Step 3: Structured LLM Prompt

```
System: You are an environmental scientist analyzing real-time monitoring
data. Your task is to explain an environmental anomaly using ONLY the
data provided below. Never speculate beyond what the data supports.
Cite specific data points when making causal claims.

Context: {assembled context from Step 1}
Reference: {RAG chunks from Step 2}

Generate an explanation with these sections:
A) SUMMARY: Plain-language description of what happened (2-3 sentences)
B) LIKELY CAUSES: Ranked list of probable causes with evidence from the
   cross-referenced data. Each cause must cite specific data points.
C) HEALTH IMPLICATIONS: Impact by risk group (general public, sensitive
   groups, children, elderly, outdoor workers)
D) CONFIDENCE: Overall confidence level (HIGH/MEDIUM/LOW) with reasoning
   about what data supports or limits the explanation
E) RECOMMENDED ACTIONS: Practical steps for affected populations
```

Model: Llama 3 8B (primary, local via Ollama) — with GPT 5.4 Standard Thinking / Gemini 3 Thinking as cloud baselines for research comparison.

### Step 4: Post-Processing & Hallucination Detection
- **Factual grounding check**: Extract all factual claims from the explanation. For each claim, verify against the actual data provided in Step 1:
  - "Wildfire 40 miles NW" → check FIRMS data for fires in that direction/distance
  - "Temperature inversion" → check if temperature profile supports inversion
  - "Industrial facility X" → check if facility exists in TRI data at stated location
- **Claim validation score**: % of claims that can be verified (target: >90%)
- **Flagging**: Unverifiable claims are flagged with "[unverified]" annotation
- **Metadata extraction**: Parse explanation for structured fields (primary_cause_category, confidence_level, affected_radius_km, risk_level_by_group)
- Store explanation + metadata + validation results in database

---

## Web Application

### Tech Stack
- React 18 + TypeScript + Vite
- Mapbox GL JS (interactive map with custom layers)
- Recharts (charts and time-series visualization)
- Tailwind CSS (styling)
- TanStack Query (data fetching and caching)
- WebSocket connection for real-time anomaly notifications

### Page 1: Live Map (Primary View)
- Full-screen Mapbox map
- Sensor location markers color-coded by current AQI (green→yellow→orange→red→purple gradient)
- Active anomaly markers with pulsing animation and severity-colored rings
- Toggle layers: satellite NO2/SO2 overlay, wildfire markers, traffic density heatmap, industrial facilities
- Click anomaly marker → slide-in side panel with LLM explanation summary
- Click "View Full Analysis" → navigate to Anomaly Detail page

### Page 2: Anomaly Feed
- Reverse-chronological list of detected anomalies
- Each card: severity badge, location name, time (relative), metric, one-line LLM summary
- Filter by: severity, metric type, date range, location radius
- WebSocket-driven: new anomalies appear in real-time at top of feed

### Page 3: Anomaly Detail
- Hero section: anomaly metric + value + severity + time/location
- 72-hour time-series chart with anomaly point highlighted
- Evidence panel grid (each as a card):
  - Weather conditions (temp, humidity, wind rose diagram)
  - Traffic density map snippet
  - Nearby fires (mini map + distance/direction)
  - Satellite observation (NO2/SO2 visualization if available)
  - Industrial facilities (list with distance, type, compliance status)
  - Power plant data
  - Water quality (if relevant)
- Full LLM explanation with inline citations linking to evidence cards
- Confidence indicator with reasoning
- Health advisory cards by risk group
- Hallucination check results (% claims verified, any flagged claims)

### Page 4: Natural Language Query
- Chat-style interface
- User types questions like:
  - "Why was air quality bad downtown yesterday?"
  - "What are the most common causes of PM2.5 spikes in this area?"
  - "Which neighborhoods had the most anomalies this month?"
  - "Is it safe to exercise outdoors tomorrow?"
- System queries database + RAG pipeline, generates grounded response
- Responses include inline charts/maps where relevant

### Page 5: System Dashboard
- Data collection status (per source: last update, success rate, data points/day)
- Anomaly statistics (total detected, by severity, by metric, trend over time)
- Model performance metrics (explanation generation latency, claim verification rate)
- Server resource utilization (CPU, memory, disk, Ollama inference stats)

### Mobile Responsiveness
- Responsive web design (not native app — native is month 5 stretch goal)
- Mobile priority: map view and anomaly feed
- Detail pages scroll vertically with stacked evidence cards
- Query interface works as full-screen chat on mobile

---

## Research Evaluation Framework

### Research Question
*Can locally-hosted LLMs generate accurate causal explanations for environmental anomalies by cross-referencing heterogeneous data sources?*

### Sub-questions
1. How accurately do LLM-generated explanations identify the true cause of environmental anomalies?
2. Does multi-source cross-referencing improve explanation quality compared to single-source context?
3. How does explanation quality from locally-hosted 8B-parameter models compare to cloud-scale models (GPT 5.4 Standard Thinking, Gemini 3 Thinking)?
4. Can automated hallucination detection reliably identify factual errors in generated explanations?

### Evaluation Metrics

| Metric | Measurement Method | Target |
|--------|-------------------|--------|
| **Causal accuracy** | Expert panel (professors) labels 50-100 anomalies with ground-truth causes (target — actual count depends on anomaly frequency in the chosen region over the study period; supplement with synthetic anomalies if needed). Score LLM on cause identification (correct/partial/incorrect). | >70% correct |
| **Factual grounding** | Automated: % of LLM claims verifiable against input data | >90% |
| **Cross-reference utilization** | Count distinct data sources cited per explanation | >3 sources avg |
| **Comprehensibility** | Flesch-Kincaid readability score + user study (10-15 people, Likert scale) | Grade 8-10 reading level |
| **Actionability** | User study: "Would you change your behavior based on this advisory?" (Likert) | >3.5/5 avg |
| **Local vs. cloud** | Same 50 anomalies through Llama 3 8B, GPT 5.4 Standard Thinking, Gemini 3 Thinking. Compare all metrics. | Document gap |
| **Ablation: multi-source vs. single** | Same anomalies explained with full context vs. only air quality data. Compare causal accuracy. | Multi-source > single |
| **Latency** | End-to-end time from anomaly detection to explanation generation | <60s local, <15s cloud |

### Study Design
- **Expert labeling**: 2-3 professors/grad students independently label anomaly causes. Inter-rater reliability (Cohen's kappa) reported.
- **User study**: 10-15 participants (mix of general public and environmental science students). Read 10 anomaly explanations, rate comprehensibility and actionability. IRB likely not needed for this scale but confirm with professors.
- **Ablation study**: Automated — run same anomalies through pipeline with different context levels.

---

## Timeline

**Note**: Each month below is a high-level overview. Before beginning each month, a detailed actionable phase plan will be created with specific tasks, file-level implementation steps, dependencies, and acceptance criteria.

### Month 1: Infrastructure & Data Pipeline (April 2026)
- Server setup: Ubuntu Server, networking, firewall, SSH, DDNS
- PostgreSQL + TimescaleDB installation and schema design
- Build data collectors for all 9 sources
- Common normalization layer
- FastAPI skeleton serving raw data endpoints
- Cron scheduling for all collectors
- **Milestone**: Server live, collecting data from all 9 sources

### Month 2: Anomaly Detection & LLM Pipeline (May 2026)
- Implement three anomaly detection methods (Z-score, STL, Isolation Forest)
- Anomaly enrichment pipeline (context assembly)
- Set up Ollama + Llama 3 8B
- Build ChromaDB knowledge base
- Implement RAG retrieval pipeline
- Build explanation generation with structured prompts
- Post-processing and hallucination detection
- API endpoints for anomalies and explanations
- **Milestone**: End-to-end anomaly detection → LLM explanation working

### Month 3: Web Application (June 2026)
- Live map with Mapbox GL JS (all layers and overlays)
- Anomaly feed with real-time WebSocket updates
- Anomaly detail page with evidence panels
- Natural language query interface
- System dashboard
- Responsive design for mobile/tablet
- **Milestone**: Fully functional web app, demo-ready

### Month 4: Research Evaluation & Polish (July 2026)
- Build evaluation harness (automated grounding checks, metrics)
- Expert labeling sessions (50-100 anomalies)
- Local vs. cloud model comparison
- Multi-source vs. single-source ablation
- User comprehension/actionability study
- Statistical analysis
- UI/UX polish, performance optimization
- **Milestone**: Evaluation complete, results analyzed

### Month 5: Paper, Competition & Stretch Goals (August 2026)
- Write research paper (target: AAAI undergraduate consortium or similar)
- Competition submission materials (ISEF, Regeneron STS, Congressional App Challenge)
- **Stretch**: React Native mobile app with push notifications
- **Stretch**: Expand to multiple geographic regions
- Demo preparation and rehearsal
- **Milestone**: Paper draft, competition materials, polished demo

---

## Verification Plan

### End-to-End Testing
1. **Data pipeline**: Verify each of the 9 collectors returns valid data. Check normalization to common schema. Confirm cron schedule runs reliably over 48 hours.
2. **Anomaly detection**: Inject known anomalies into test data. Verify all three methods detect them. Verify consensus scoring and de-duplication.
3. **LLM explanations**: Feed 10 manually-created anomaly records through the pipeline. Verify explanations are structured correctly, cite actual data, and hallucination detection catches injected false claims.
4. **Web app**: Navigate all 5 pages. Verify map renders with correct layers, anomaly feed updates in real-time, detail page shows evidence panels, NL query returns grounded responses, dashboard shows system metrics.
5. **Research evaluation**: Run the full evaluation pipeline on a test set of 10 anomalies. Verify metrics computation produces valid numbers.

### Demo Script
For competitions/presentations, prepare a live demo path:
1. Open live map → show real-time sensor data
2. Point to an active or recent anomaly → click for explanation
3. Show the evidence panels and cross-referenced data
4. Ask a natural language question about the anomaly
5. Show the system dashboard (engineering credibility)
6. Show local vs. cloud comparison results (research credibility)

---

## Tech Stack Summary

| Layer | Technology |
|-------|-----------|
| Server OS | Ubuntu Server 22.04+ |
| Backend | Python 3.11+, FastAPI, SQLAlchemy |
| Database | PostgreSQL + TimescaleDB |
| Vector Store | ChromaDB |
| Local LLM | Ollama (Llama 3 8B primary, Mistral 7B secondary) |
| Cloud LLMs (research comparison) | GPT 5.4 Standard Thinking, Gemini 3 Thinking |
| ML | scikit-learn (Isolation Forest), statsmodels (STL) |
| Frontend | React 18 + TypeScript + Vite |
| Mapping | Mapbox GL JS |
| Charts | Recharts |
| Styling | Tailwind CSS |
| Data Fetching | TanStack Query + WebSocket |
| Stretch: Mobile | React Native |
