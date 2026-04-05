# AERIS — Claude Code Instructions

## Project Purpose

AERIS (Autonomous Environmental RAG & Inference System) is a self-hosted environmental intelligence platform. It aggregates 9 real-time environmental data sources on a home server, detects anomalies using statistical + ML methods, and generates causal explanations via a locally-hosted LLM with RAG and multi-source cross-referencing. A React web app visualizes everything on an interactive map. The research question: can locally-hosted LLMs accurately explain environmental anomalies by cross-referencing heterogeneous data sources?

## Run / Build / Test Commands

```bash
# Backend
cd server
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000        # dev server
pytest                                             # all tests
pytest tests/unit                                  # unit only
pytest tests/integration                           # integration only
pytest --cov=app                                   # with coverage

# Frontend
cd client
npm install
npm run dev                                        # dev server (Vite, port 5173)
npm run build                                      # production build
npm run lint                                       # ESLint
npm run test                                       # Vitest

# Data collectors
python -m app.collectors.run_all                   # manual run of all collectors
python -m app.collectors.run_all --source=epa      # single source

# Anomaly detection
python -m app.detection.run                        # run detection on latest data

# LLM pipeline
python -m app.llm.explain --anomaly-id=<id>        # generate explanation for one anomaly
```

## Architecture Map

```
server/                         # Python backend (FastAPI)
  app/
    main.py                     # FastAPI app entry, CORS, lifespan
    config.py                   # env vars, API keys, DB URL
    db/
      models.py                 # SQLAlchemy ORM models
      schema.py                 # DB schema + TimescaleDB hypertables
      session.py                # DB session factory
    collectors/                 # One module per data source
      base.py                   # Abstract collector class
      epa_airnow.py
      openaq.py
      purpleair.py
      noaa_weather.py
      nasa_firms.py
      sentinel5p.py
      traffic.py
      usgs_water.py
      eia_energy.py
      run_all.py                # Orchestrator for all collectors
    detection/
      engine.py                 # Runs all 3 methods, consensus scoring
      zscore.py                 # Statistical Z-score detector
      stl.py                    # Seasonal decomposition detector
      isolation_forest.py       # ML multivariate detector
      enrichment.py             # Gathers cross-source context for anomaly
    llm/
      context.py                # Assembles structured context from enrichment
      rag.py                    # ChromaDB retrieval
      prompt.py                 # Structured prompt templates
      explain.py                # Orchestrates explanation generation
      validate.py               # Post-processing hallucination detection
    api/
      routes/                   # FastAPI routers grouped by domain
        data.py                 # Raw data endpoints
        anomalies.py            # Anomaly CRUD + explanation endpoints
        query.py                # Natural language query endpoint
        system.py               # Health, metrics, status
    ws/
      manager.py                # WebSocket connection manager for real-time

client/                         # React frontend (Vite + TypeScript)
  src/
    components/
      map/                      # Mapbox GL map + layers
      anomaly/                  # Feed, detail, evidence panels
      query/                    # NL query chat interface
      dashboard/                # System dashboard components
      shared/                   # Reusable UI components
    hooks/                      # Custom React hooks (data fetching, WS)
    pages/                      # Top-level route pages
    lib/                        # API client, utils, types
    styles/                     # Tailwind config, global styles

docs/
  superpowers/specs/            # Design specs
  research/                     # Paper drafts, evaluation scripts
```

## Non-Obvious Coding Rules

- **Collectors inherit from `base.py`**: Every data source collector extends `BaseCollector` and implements `fetch()` and `normalize()`. Do not create standalone scripts.
- **All data normalizes to common schema**: `(timestamp, lat, lon, metric, value, source, raw_json)`. Never store source-specific schemas in the main tables.
- **LLM prompts live in `prompt.py`**: Never inline prompt strings in other modules. All prompt templates are centralized.
- **Hallucination checks are not optional**: Every LLM explanation passes through `validate.py` before storage. Never skip this step.
- **API keys in `.env` only**: Never hardcode keys. Access via `config.py` which reads from environment.
- **No LangChain**: We use a custom RAG pipeline. Do not introduce LangChain or LlamaIndex dependencies.
- **Type hints everywhere in Python**: All function signatures have type annotations. Use Pydantic models for API request/response schemas.
- **Strict TypeScript**: `strict: true` in tsconfig. No `any` types except when interfacing with untyped third-party libraries.

## Testing and Verification

- **Backend**: pytest. Unit tests for each collector, detector, and LLM pipeline stage. Integration tests hit a test database (SQLite for CI, Postgres locally). Target 80%+ coverage on core modules (`detection/`, `llm/`).
- **Frontend**: Vitest + React Testing Library. Test components that contain logic. Don't test pure layout.
- **Anomaly detection validation**: Maintain a fixture set of known anomalies (injected into test data). All three detectors must catch them. Run with `pytest tests/detection/test_known_anomalies.py`.
- **LLM explanation validation**: Maintain 10 manually-crafted anomaly records with expected explanation elements. Verify structural correctness, source citation presence, and hallucination detection catches injected false claims.
- **Before claiming a feature is done**: Run the relevant test suite AND manually verify in the running app.

## Workflow Rules

- **Never auto-commit.** Instead, tell Mason when it's a good time to commit and suggest the commit message. Mason will run the git commands himself.
- **Commits follow conventional format**: `type(scope): description` (e.g., `feat(collectors): add EPA AirNow data collector`, `fix(detection): correct z-score window alignment`).
- **Each month has a detailed phase plan** created before work begins, with file-level tasks, dependencies, and acceptance criteria.
- **Branch strategy**: `main` is stable. Feature branches: `feat/<name>`, bug fixes: `fix/<name>`. No direct pushes to main.
- **Session handoff**: Update `session_summary.md` at the end of every session with accomplishments, current state, and next steps.

## Repo-Specifics

- **Monorepo**: `server/` and `client/` in the same repo. No separate repos.
- **Python 3.11+**, managed with venv. Requirements in `server/requirements.txt`.
- **Node 18+**, managed with npm. Package manifest in `client/package.json`.
- **Database**: PostgreSQL + TimescaleDB for production/local dev. SQLite for CI tests.
- **Vector store**: ChromaDB, persisted to `server/data/chromadb/`.
- **Local LLM**: Ollama running Llama 3 8B. Assumed to be running on the same machine at `http://localhost:11434`.
- **Environment variables**: `.env` file in `server/` (gitignored). See `.env.example` for required keys.

## Output Format

- Respond concisely. Lead with action or answer, not reasoning.
- When referencing code, use `file_path:line_number` format.
- For multi-step work, use task lists to track progress.
- When suggesting commits, format as: `type(scope): description` with a brief explanation of what changed.
- Keep PR descriptions short: 1-3 bullet summary + test plan.
- When explaining architectural decisions, reference the design spec at `docs/superpowers/specs/`.
- Do not output large file contents unless explicitly asked. Summarize or reference by path instead.
- State uncertainty clearly instead of guessing. Ask Mason clarifying questions when unsure.
