# AERIS Session Summary

> Updated at the end of each session. Read this first in every new session to restore context.

---

## Last Session: 2026-04-05

### Phase
Pre-development (project definition and setup)

### Accomplishments
- Brainstormed project ideas, evaluated 3 directions (AI tutor, environmental intelligence, Socratic companion)
- Selected: Hyperlocal Environmental Intelligence Platform with anomaly detection + LLM explanation
- Named the project: AERIS (Autonomous Environmental RAG & Inference System)
- Completed full design specification covering:
  - 3-layer architecture (home server, web app, LLM intelligence)
  - 9 data sources (EPA, OpenAQ, PurpleAir, NOAA, NASA FIRMS, Sentinel-5P, TomTom, USGS, EIA)
  - 3-method anomaly detection (Z-score, STL, Isolation Forest)
  - LLM explanation pipeline (context assembly, RAG, structured prompts, hallucination detection)
  - 5-page React web app (map, feed, detail, NL query, dashboard)
  - Research evaluation framework (causal accuracy, grounding, local vs. cloud comparison)
  - 5-month timeline with monthly milestones
- Created CLAUDE.md, README.md, session_summary.md
- Design spec saved to: `docs/superpowers/specs/` (pending — currently in plan file)

### Key Decisions
- **Research comparison models**: Llama 3 8B (local) vs. GPT 5.4 Standard Thinking + Gemini 3 Thinking (cloud)
- **No LangChain**: Custom RAG pipeline
- **Monorepo**: `server/` + `client/` in same repo
- **Home server**: Old PC running Ubuntu Server
- **Geographic scope**: Single US metro area to start

### Current State
- Git initialized locally, GitHub repo not yet created
- Full directory structure in place (`server/`, `client/`, `docs/`)
- Design spec copied to `docs/superpowers/specs/2026-04-05-aeris-design.md`
- `.gitignore` and `server/.env.example` created
- No code written yet — ready to begin Month 1

### Open Questions / Blockers
- Mason needs to choose target metro area (his local area recommended)
- Need to verify home server hardware specs (16 GB RAM minimum)
- API key registration needed for: EPA AirNow, PurpleAir, OpenWeather, TomTom, EIA, NASA Earthdata, Mapbox

### Next Steps
1. Create GitHub repo and push initial commit
2. Begin Month 1 phase plan: detailed actionable task breakdown for server infrastructure + data pipeline
3. Start registering for API keys (can be done in parallel)
