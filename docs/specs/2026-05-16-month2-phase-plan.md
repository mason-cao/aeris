# AERIS Month 2 — AI Attribution Phase Plan

> **Window:** 2026-05-16 → 2026-06-16 (4 weeks, ~17 working days)
> **Travel gap:** 2026-05-22 → 2026-05-31 (back June 1)
> **Mentor checkpoint:** 2026-06-02 with Dr. Annalisa Bracco
> **Convention:** AERIS phase plans live in `docs/specs/YYYY-MM-DD-monthN-phase-plan.md`. After approval, copy this file to `docs/specs/2026-05-16-month2-phase-plan.md` and commit it.

---

## Context

Month 1 delivered a working multi-source data pipeline: OpenAQ (ground sensors), Sentinel-5P (satellite column densities), NOAA GFS (upper-air reanalysis), and OpenWeather (surface meteorology) all live for a 50km Houston radius, with a normalized DataPoint schema and async retry-backed collection. Month 2 is the **core research phase**: turn that data into detected anomalies and into LLM-generated causal attributions — and produce a defensible novelty claim publishable as research.

Bracco's stated expertise is **"weather fields and AI physics-to-logic"** (her phrasing). The novelty angle has to land where she can evaluate it. She also sent three literature anchors on May 14: Carloni et al. 2025 (WIREs Data Mining Knowl., AI for climate physics), a 2024 ScienceDirect review on AI for air quality, and an industrial-city air quality AI case study. These define the prior-art frame the work must differentiate against.

**The core problem with the README's stated contributions** (4-API architecture, hallucination detection, local-vs-cloud comparison): individually they read as solid engineering, not novel research. Reviewers see RAG + multi-source + comparison and shrug. This phase reframes them around one tight thesis with a discovery question, so the three contributions collapse into one coherent system rather than three parallel engineering deliverables.

---

## Research thesis

> **Inter-source physical corroboration over causally-coupled heterogeneous sensors is a label-free proxy for ground-truth verification of LLM scientific attributions, empirically distinct from retrieval-grounded factuality checks because it leverages constraints from the underlying physical system rather than textual overlap.**

In a 4-API heterogeneous-sensor architecture, every claim an LLM makes about an atmospheric anomaly can be scored against the agreement (or disagreement) of independent physical sources: satellite chemistry, ground-level sensors, reanalysis dynamics, surface meteorology. These sources are _causally coupled_ by atmospheric physics — they sense different facets of the same underlying physical state — which means their agreement carries information that retrieval against a static knowledge base (FActScore-style) cannot provide. The corroboration score is computed mechanically, requires no human label, and the framework generalizes to any multi-source scientific domain with causally-coupled sensors (medical multi-modal, geophysical inversions, financial multi-feed).

### Relation to prior art

The closest neighbors are: **FActScore** (Min et al. 2023) for atomic-claim decomposition and label-free factuality scoring against retrieved evidence; **SelfCheckGPT** (Manakul et al. 2023) for using internal consistency across model samples as a hallucination signal; **RAGAs** (Es et al. 2023) for retrieval-grounded LLM evaluation; and the long-standing sensor-fusion / inter-comparison literature in atmospheric science (e.g., TROPOMI vs. ground-monitor validation, Verhoelst et al. 2021). The contribution here is _neither_ a new factuality-via-retrieval method _nor_ a sensor-fusion product — it is a methodology that imports the sensor-fusion lineage's "agreement-across-physical-channels" intuition into the LLM-evaluation regime that FActScore opened, with the explicit recognition that physical-source agreement is a different signal class than textual-retrieval agreement. The Carloni et al. 2025 WIREs review and Sayeed et al. 2024 ScienceDirect review (both sent by Bracco) frame the air-quality AI methodological context this work differentiates against.

### Discovery question

**Does inter-source corroboration correlate strongly enough with expert-labeled correctness that it can stand in for ground truth at scale — especially for small local models?**

Either outcome is publishable:

| Outcome                    | Publishable claim                                                                                                                                                           |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Strong correlation         | "Inter-source corroboration is a viable scalable eval proxy for LLM scientific reasoning, demonstrated on N=50 industrial-city air quality anomalies."                      |
| Weak / partial correlation | "Inter-source corroboration is necessary but not sufficient — characterizing the limits of cross-source eval and the claim types where it breaks."                          |
| Localized correlation      | "Corroboration tracks expert truth for {X} claim types but not {Y}; small models specifically produce low-corroboration claims at higher rate, with higher overconfidence." |

### Sub-contributions delivered in Month 2

1. **Mechanism** — Decomposed 4-step reasoning chain (physical signature → candidate causes → evidence evaluation → synthesis) with **per-step corroboration scoring** against the 4 APIs.
2. **Validation** — Correlation analysis between corroboration scores and expert labels on ~50 anomalies (10–20 Bracco-labeled, rest Mason-labeled with Bracco audit subset for IRR).
3. **Local-vs-cloud finding** — Characterize _where_ in the reasoning chain Llama 3 8B produces low-corroboration claims that GPT-5.4 / Gemini 3 Thinking do not, and whether the local model is _overconfident_ on exactly those claims.
4. **Calibration measurement** — Calibration curves of stated confidence vs. corroboration score, per model.

---

## Scope guardrails

### In scope for Month 2

- 3-method detection engine (Z-score, STL, Isolation Forest) + consensus scoring
- Anomaly enrichment (cross-source context gathering for the 72-hour window around each anomaly)
- ChromaDB knowledge base ingestion (EPA breakpoints, atmospheric reference, Houston-specific context)
- Ollama + Llama 3 8B local inference
- Decomposed reasoning chain prompt structure (4 steps)
- Cross-source corroboration scorer (the core novelty)
- Cloud comparison wrapper (GPT-5.4, Gemini 3 Thinking)
- Eval harness on ~50 anomalies (10–20 expert-labeled)
- Corroboration↔truth correlation analysis, calibration curves, disagreement analysis
- Bracco-ready artifact for June 2 meeting

### Explicitly deferred

- **Web app** (Month 3 — not yet scaffolded per CLAUDE.md)
- **User comprehensibility study** (Month 4)
- **Scale to 100+ anomalies** (Month 4)
- **Counterfactual attribution** ("what would NOT explain this") — flagged as a Month 4 ablation that could become a second paper if Month 2 results justify it
- **Physics-grounded prompt ablation** (Month 4) — does adding explicit atmospheric priors change calibration?
- **Cross-domain generalization** (post-AERIS, if results are strong)
- **Tree coverage / urban heat island linking** that Bracco suggested for Atlanta — note in the June 2 meeting that the Houston pivot redirected this; reconsider as a Month 4 secondary signal

### Hard scope discipline

If anything not on the in-scope list is tempting during Month 2, write it on a "Month 4 backlog" instead. Stretch ideas are how a 4-week research sprint becomes 8 weeks.

---

## Timeline

| Window              | Dates                    | Theme                                                                           |
| ------------------- | ------------------------ | ------------------------------------------------------------------------------- |
| Pre-vacation sprint | May 16 → May 21 (6 days) | Infrastructure + detection engine, parallelized with dad on server setup        |
| Travel gap          | May 22 → May 31 (10 days)| Offline; vacation deliverables in Google Doc (demo selection + hand-curated chain) |
| Bracco prep window  | Jun 1 → Jun 5 (5 days)   | Catch-up + meeting + reasoning chain / corroboration scorer build sprint        |
| Eval week           | Jun 6 → Jun 12 (7 days)  | Eval harness, cloud comparison, expert-label coordination, correlation analysis |
| Polish & write-up   | Jun 13 → Jun 16 (4 days) | Final analysis, write-up draft, buffer                                          |

The June 2 meeting falls 1 day into the post-vacation window. With only June 1 as a working day before the meeting, the plan commits up front to a **hand-curated demo** and treats a working LLM pipeline as upside, not the default. The design work that would otherwise have happened in the May 30–31 slots is pushed onto the vacation deliverables list (Google Doc work, no code).

---

## Power & statistics

N=50 anomalies × 4 reasoning steps × ~3 claims per step ≈ 600 claims. Split across 10 claim types, the per-type N is highly uneven and the long-tail types will have N too small for inferential statistics. The plan commits to:

**Three "headline" claim types** with N≥20 targeted: `concentration_elevation`, `transport_direction`, `meteorological_state`. These get Spearman correlation analysis with confidence intervals. For Spearman ρ to be distinguishable from zero at α=0.05, power=0.80, true ρ=0.4, n ≈ 47 is needed; at ρ=0.6, n ≈ 19 is sufficient — so the headline claim types target the latter regime.

**The other 7 claim types are descriptive only.** Report N, per-claim corroboration scores, qualitative examples — _no_ inferential claims (no p-values, no headline correlations). If a rare claim type's N reaches 20+ organically, promote it to inferential reporting post-hoc with a Bonferroni-adjusted threshold.

**Calibration analysis (ECE / reliability diagrams) uses 5 bins for N=50.** Anything finer is noise.

**Inter-rater reliability (IRR) target.** Mason labels and Bracco labels overlap on a 10–15 anomaly audit subset. Cohen's κ target:

- **κ ≥ 0.6 minimum** — acceptable; labels can be treated as exchangeable in the headline analysis
- **κ ≥ 0.75 target** — strong agreement
- **κ < 0.6** — downgrade the validation framing from "expert-labeled" to "labeled by Mason, audited by Bracco." The analysis still works but is less load-bearing as a publishable claim.

**Cap claims per anomaly to ~3 per reasoning step.** Without a cap, a chatty model produces 10+ claims per step and the per-anomaly evidence is dominated by one verbose run rather than 50 distinct events. The prompt template enforces this.

**Eval execution is idempotent.** Partial runs persist Explanation rows individually; re-running the harness only fills missing (anomaly, model) cells. This matters because the wall-clock for 600 LLM calls is closer to 2 days than the "overnight" estimate (see Eval week schedule).

---

## Architecture additions

All paths relative to `/Users/mason/project/Untitled/aeris/`.

### New: `server/app/detection/`

| File                  | Responsibility                                                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `engine.py`           | Orchestrator: runs all 3 methods on a metric/source/window, emits Anomaly records via consensus                                 |
| `zscore.py`           | Rolling 7-day Z-score detector, ±3σ default threshold (configurable per metric)                                                 |
| `stl.py`              | STL decomposition (statsmodels), residual >±2.5σ                                                                                |
| `isolation_forest.py` | Multivariate IsolationForest (scikit-learn) over (value, hour_of_day, day_of_week, plus contemporaneous wind_speed, PBL height) |
| `consensus.py`        | Severity scoring: 1 method = minor, 2 = moderate, 3 = severe                                                                    |
| `enrichment.py`       | Pulls 72h cross-source context around an anomaly's (lat, lon, timestamp); produces structured EnrichmentRecord                  |

### New: `server/app/llm/`

| File                 | Responsibility                                                                                                                                                                                                                                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `client_base.py`     | `LLMClient` ABC defining the async `generate(prompt, schema) → structured_output` interface, plus latency/token-count capture. Mirrors `collectors/base.py:BaseCollector` discipline.                                                                                                                               |
| `ollama_client.py`   | Async wrapper around Ollama HTTP API at `http://localhost:11434`; uses `format: "json"` + pydantic parse with one retry-on-failure                                                                                                                                                                                  |
| `gpt_client.py`      | OpenAI client for GPT-5.4 using `response_format=json_schema` for structured output                                                                                                                                                                                                                                 |
| `gemini_client.py`   | Google client for Gemini 3 Thinking using `response_schema`; handles RPM throttling separately from OpenAI                                                                                                                                                                                                          |
| `rag.py`             | ChromaDB retrieval; query by (metric, signature, region)                                                                                                                                                                                                                                                            |
| `prompt.py`          | All prompt templates centralized per CLAUDE.md rule; one template per reasoning step                                                                                                                                                                                                                                |
| `reasoning_chain.py` | Orchestrator: physical_signature → candidate_causes → evidence_evaluation → synthesis. Calls the LLM 4 times via the `LLMClient` interface.                                                                                                                                                                         |
| `parser.py`          | Structural parsing + claim extraction from each reasoning step's output; schema check; emits a `list[ClaimDraft]`                                                                                                                                                                                                   |
| `corroboration.py`   | **Core novelty.** Scores each `ClaimDraft` against cross-source evidence. See "Corroboration scorer design." Pure scoring — no pass/fail decisions.                                                                                                                                                                 |
| `validate.py`        | The CLAUDE.md-mandated hallucination gate. Consumes scored claims, applies decision rules (e.g., `corroboration_score ≤ -0.5` or `unverified` with low corroboration confidence → flag), produces the persisted `Explanation` record's hallucination flags. This file is what `explain.py` calls before persisting. |
| `explain.py`         | Top-level orchestrator: Anomaly → enrichment → `reasoning_chain` → `parser` → `corroboration` → `validate` → persist Explanation. CLI entrypoint preserved: `python -m app.llm.explain --anomaly-id=<id>`.                                                                                                          |

### New: `server/app/eval/`

| File                           | Responsibility                                                                                                  |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `harness.py`                   | Runs all 3 models (local + 2 cloud) on the labeled anomaly set; persists ExplanationResult per (anomaly, model) |
| `corroboration_correlation.py` | Computes Spearman/Pearson correlation between corroboration scores and expert labels, per claim type            |
| `calibration.py`               | Reliability diagrams, ECE (expected calibration error), per model                                               |
| `disagreement.py`              | Pairwise model disagreement matrix; structure of disagreement by claim type / reasoning step                    |
| `report.py`                    | Generates a Markdown report from the four analyses for Mason and Bracco                                         |

### Database schema additions in `server/app/db/models.py`

| Table              | Key fields                                                                                                                                                                                                                                                                                                                                                                                |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Anomaly`          | `id`, `timestamp`, `lat`, `lon`, `metric`, `source`, `value`, `expected_value`, `z_score`, `methods_triggered` (list), `severity`, `detected_at`                                                                                                                                                                                                                                          |
| `EnrichmentRecord` | `id`, `anomaly_id`, `context_window_start`, `context_window_end`, `cross_source_summary_json`                                                                                                                                                                                                                                                                                             |
| `Explanation`      | `id`, `anomaly_id`, `model_name`, `model_version`, `reasoning_steps_json`, `final_narrative`, `stated_confidence`, `latency_ms`, `prompt_tokens`, `completion_tokens`, `created_at`                                                                                                                                                                                                       |
| `Claim`            | `id`, `explanation_id`, `step_index` (1–4), `claim_type` (one of 10 taxonomy types), `claim_text`, `cited_sources`, `corroboration_score` (scalar, [-1,+1] or null), `evidence_n` (int, supporting + contradicting count), `per_source_verdicts` (JSON: `{source_name: +1/-1/0}`), `partial_verifiability` (bool, true for claim types where the data resolution constrains verification) |
| `ExpertLabel`      | `id`, `anomaly_id`, `labeler`, `true_cause`, `claim_validations_json` (list of {claim_id, verdict, note}), `created_at`                                                                                                                                                                                                                                                                   |

TimescaleDB hypertable conversion applies to `Anomaly` (partition by `timestamp`); the others stay as regular tables.

### New API routes in `server/app/api/routes/`

| File              | Endpoints                                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------------------ |
| `anomalies.py`    | `GET /api/anomalies`, `GET /api/anomalies/{id}`, `POST /api/detection/run`                             |
| `explanations.py` | `POST /api/explain/{anomaly_id}?model=local\|gpt-5.4\|gemini-3-thinking`, `GET /api/explanations/{id}` |
| `evaluation.py`   | `POST /api/eval/run`, `GET /api/eval/report`                                                           |

### Existing patterns reused

- **`server/app/collectors/base.py:BaseCollector`** — same retry/backoff/structured-logging pattern carries over for the `LLMClient` ABC in `client_base.py` and its three concrete implementations (`ollama_client.py`, `gpt_client.py`, `gemini_client.py`). Wrap with the same async + 30s timeout discipline.
- **`server/app/db/models.py:DataPoint`** — Anomaly references `DataPoint(source, metric, source_entity_id, timestamp)` via the existing unique constraint; no need to invent a new key scheme.
- **`server/app/db/session.py`** — async session factory reused for all new tables.
- **`server/tests/unit/`** — pytest + mock pattern from `test_openaq.py` / `test_noaa_gfs.py` extends directly to detector and LLM unit tests.

### New dependencies for `server/requirements.txt`

```
chromadb==0.5.x
scikit-learn==1.5.x
statsmodels==0.14.x
ollama==0.4.x        # or use httpx directly against the Ollama HTTP API
openai==1.x          # GPT-5.4
google-genai==0.x    # Gemini 3 Thinking
```

---

## Corroboration scorer design (core novelty — get this right)

The corroboration scorer is the load-bearing piece. Spend Wednesday/Thursday of the pre-vacation sprint on its design, not its implementation.

### Claim taxonomy

Each LLM-generated claim falls into one or more **claim types**, each with a defined verifiability profile against the 4 APIs. Three of these are designated **headline claim types** for inferential analysis (sufficient N expected); the rest are reported descriptively. **Industrial-city air quality** has three patterns a small LLM is most likely to get wrong; those three (`emissions_source_type`, `secondary_formation`, `background_vs_event`) were added on the Plan-agent critique recommendation and Bracco will validate them on Jun 2.

| Claim type                        | Verifiable against                                                         | Headline | Notes / verifiability constraint                                                                                                                                                                                                      | Example                                                                                                     |
| --------------------------------- | -------------------------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `concentration_elevation`         | OpenAQ, Sentinel-5P                                                        | **Yes**  | Direct comparison; well-resolved                                                                                                                                                                                                      | "Ground-level NO₂ exceeded 80 ppb between 14:00–18:00 CT"                                                   |
| `transport_direction`             | NOAA GFS (10m wind), OpenWeather                                           | **Yes**  | Direct comparison; well-resolved                                                                                                                                                                                                      | "Southerly winds advected pollutants from the Ship Channel northward"                                       |
| `meteorological_state`            | OpenWeather, NOAA GFS                                                      | **Yes**  | Direct comparison; well-resolved                                                                                                                                                                                                      | "Stagnant conditions: wind speed <2 m/s"                                                                    |
| `atmospheric_trap`                | NOAA GFS (PBL height, T@850), OpenWeather (surface T)                      | No       | Descriptive only; well-resolved but rarer                                                                                                                                                                                             | "A low PBL combined with thermal inversion trapped emissions near the surface"                              |
| `temporal_pattern`                | Time-series of any source                                                  | No       | Descriptive; verification is monotonicity/trend test                                                                                                                                                                                  | "Concentrations rose monotonically over a 4-hour window"                                                    |
| `chemistry`                       | Sentinel-5P (NO₂/HCHO ratio), OpenAQ (O₃/NO₂ relationship)                 | No       | **Partial verifiability**: TROPOMI HCHO retrievals are noisy at single-orbit resolution; tolerance is wider (±50%) and HCHO-silent granules are reported as silent not contradicting                                                  | "Elevated HCHO with depressed O₃ suggests fresh VOC emissions"                                              |
| `point_source_attribution`        | Sentinel-5P (granule-mean only in v1), OpenAQ (station proximity)          | No       | **Partial verifiability**: current Sentinel-5P collector emits one mean per granule at target center, not per-pixel gradients. Verification at granule-mean only; per-pixel deferred to Month 4. Marked `partial_verifiability=true`. | "Plume signature consistent with a refinery upset near 29.73°N, -95.22°W"                                   |
| `emissions_source_type` _(added)_ | OpenAQ time-of-day patterns, Sentinel-5P spatial pattern, OpenWeather wind | No       | Distinguishes mobile / point / area sources. Heuristics: morning-peak + I-610 corridor = mobile; persistent NO₂ near Ship Channel = point; broad regional = area.                                                                     | "Pattern consistent with a Ship Channel point source rather than mobile-source rush-hour"                   |
| `secondary_formation` _(added)_   | OpenAQ O₃/NO₂ time-lag, OpenWeather solar radiation/cloud cover            | No       | Dominant Houston summer pattern. Verification: O₃ peak lags NO₂ peak by hours with sufficient insolation.                                                                                                                             | "Afternoon O₃ peak consistent with photochemical formation downwind of morning NOx emissions"               |
| `background_vs_event` _(added)_   | Spatial uniformity across OpenAQ stations vs. localized gradient           | No       | Distinguishes regional regime (Saharan dust, regional ozone day) from local event. **Most common attribution error for small LLMs.**                                                                                                  | "Elevated PM2.5 across all monitors with similar magnitude suggests regional transport, not a local source" |

### Scoring procedure (per claim)

For each claim, the scorer:

1. **Classifies** the claim into one or more types (lightweight rule-based; LLM-based classification deferred unless rules underperform)
2. **Identifies** the relevant sources for each type
3. **Queries** the EnrichmentRecord for those sources in the relevant spatiotemporal window
4. **Compares** the claim's quantitative or directional content against the data:
   - Match within tolerance → +1 supporting (per source)
   - Contradiction beyond tolerance → -1 contradicting (per source)
   - No data available → 0 silent (per source)
5. **Aggregates** into a per-claim record:
   - `per_source_verdicts: {openaq: +1, sentinel5p: 0, gfs: -1, openweather: +1}` — the load-bearing detail for downstream disagreement analysis
   - `evidence_n` = count of non-silent verdicts (supporting + contradicting)
   - `corroboration_score`:
     - If `evidence_n ≥ 1`: `(supporting - contradicting) / evidence_n`, scalar in [-1, +1]
     - If `evidence_n == 0` (all silent): `null` with `unverified=true` flag — tracked separately in analysis, never treated as 0
   - `partial_verifiability=true` for `chemistry` and `point_source_attribution`; these are reported separately in the headline correlation analysis

**`evidence_n` is reported alongside the score.** A claim with `(score=+1, evidence_n=1)` and a claim with `(score=+1, evidence_n=3)` are _not_ the same evidential weight. Downstream analyses weight by `evidence_n`.

### Why this is the hardest engineering of the month

- Tolerance thresholds per claim type need defaults (these are research artifacts in themselves — bring to Bracco)
- Spatial windowing per claim (point source = tight, but limited by collector resolution; meteorological state = broad)
- Temporal windowing per claim (instantaneous concentrations vs. multi-hour trends)
- Handling claims that _partially_ match (LLM says "elevated NO₂" but data shows elevated SO₂)
- Per-source verdict logic differs by source (Sentinel-5P quality flags, GFS interpolation, OpenAQ station-to-claim-location distance weighting)

**Mitigation:** define a Claim → Verification mapping spec on paper before writing code (the `2026-05-21-corroboration-scorer-design.md` design memo). Build the scorer as 10 small functions (one per claim type) rather than one big one. Each returns `(per_source_verdicts: dict, evidence_summary: str)`; the scalar score and `evidence_n` are derived in a single shared aggregator.

---

## Week-by-week breakdown

### Pre-vacation sprint: May 16 → May 21 (6 days)

**Goal:** Detection engine + enrichment + DB schema + home server up. Both Mason and dad working in parallel on different tracks.

**Day 1 (Fri May 16):**

- Mason: scaffold `server/app/detection/` module, `models.py` Anomaly + EnrichmentRecord tables, Alembic migration
- Dad: home server inventory; pick OS, Ollama install plan
- Both: confirm GPU/RAM specs sufficient for Llama 3 8B (16GB GPU min, or M-series Mac with 24GB+ unified memory)

**Day 2 (Sat May 17):**

- Mason: implement `zscore.py` + unit test against synthetic data
- Dad: server OS install + Ollama installation
- Read: Carloni 2025 WIREs review (skim sections on AI hallucination + evaluation methodology)

**Day 3 (Sun May 18):**

- Mason: implement `stl.py` + `isolation_forest.py` + unit tests
- Dad/Mason: pull Llama 3 8B model; smoke test with `ollama run llama3:8b "test prompt"`
- Read: Sayeed 2024 ScienceDirect review (air quality AI eval methods)

**Day 4 (Mon May 19):**

- Mason: implement `consensus.py` + `engine.py`; run detection on existing 4 weeks of collected data
- Mason: write design memo for the corroboration scorer — claim taxonomy, tolerance defaults, edge cases (this is the artifact to bring to Bracco)
- Read: industrial-city case study Bracco sent

**Day 5 (Tue May 20):**

- Mason: implement `enrichment.py` — gather 72h cross-source context around each detected anomaly
- Mason: triage detected anomalies into a candidate eval set (~80 candidates → trim to 50 strong ones)
- Mason: draft email to Bracco confirming June 2 meeting and previewing the agenda

**Day 6 (Wed May 21):**

- Mason: cross-source enrichment smoke tests
- Mason: finalize the corroboration scorer design memo (claim taxonomy with 10 types, tolerance defaults, edge cases, partial-verifiability flags); commit to `docs/specs/2026-05-21-corroboration-scorer-design.md`. The memo mirrors Section 4 of the Bracco meeting notes Doc, but lives in the repo as the durable design record. **This is the load-bearing artifact for the June 2 meeting — Mason walks Bracco through it live, no pre-read.**
- Mason: buffer for slipped tasks from Days 1–5
- Mason: commit + push everything; tag the repo `month2-prevacation`
- Dad: verify home server is accessible from Mason's dev laptop (SSH, tunneled Ollama port)

_Note: `ollama_client.py` + `prompt.py` skeleton deferred to Jun 1 (stretch goal post-vacation) or Jun 3+ (post-meeting build sprint); this day stays a buffer for the design + reading._

**Acceptance criteria for the pre-vacation sprint:**

- Detection engine catches anomalies in real collected data (verified by Mason eyeballing 5 detected anomalies for plausibility)
- ~50 candidate anomalies enriched and in the database
- Llama 3 8B responding to test prompts via local Ollama
- Corroboration scorer design memo committed to `docs/specs/`
- Repo tagged `month2-prevacation`

### Travel gap: May 22 → May 31 (10 days)

Home server keeps collecting data in the background; detection runs nightly via APScheduler (carry-over from Month 1 Week 4 if not yet integrated; if not integrated, manual catch-up first day back).

**Vacation deliverables (lightweight, Google Doc only — no code):**

- Pick the one demo anomaly from the existing ~50 candidates (ideally a known refinery upset or ozone exceedance)
- Hand-draft the 4-step LLM output for that anomaly (what the model would say at each reasoning step)
- Hand-score each claim against the 4 sources using the design's tolerances; produce a printable claim-by-claim table
- Polish the Bracco meeting notes doc; print the 1-page version + the claim taxonomy table
- (Optional) draft prompt templates and the parser JSON schema on paper, so June 1 coding is transcription not design

### Bracco prep window: June 1 → June 5 (5 days)

**Goal:** Bracco meeting June 2 morning. Land the mechanism + taxonomy demonstration via a hand-curated demo; build the working LLM stack in the 3 days after the meeting.

**Mon Jun 1 (return + Bracco eve):**

The only working day before the meeting. Treat it as catch-up + meeting prep, not as a coding sprint.

- Catch up on accumulated vacation data; verify the collectors didn't silently fail during the trip (row-count check per source)
- Clear `Anomaly` + `EnrichmentRecord` tables and re-run detection on the now-extended dataset (insert-only persistence, per project guardrail)
- Confirm the demo anomaly chosen on vacation is still in the top-50 candidate set after the re-run; swap demo if not
- Final pass on the meeting notes doc; print the 1-page version + the claim taxonomy table
- (Stretch, only if time) skeleton `client_base.py` + `ollama_client.py` + one prompt template — does NOT block the meeting

**Tue Jun 2 — Bracco meeting (7:30 am):**

- Present the mechanism + the hand-curated demo (a working pipeline is upside if the June 1 stretch landed; otherwise the hand-curated path is the planned demo)
- Get her sign-off on the corroboration scorer's claim taxonomy and tolerance thresholds (especially `emissions_source_type`, `secondary_formation`, `background_vs_event`)
- Get her commitment on how many anomalies she'll label (target 10–15) and by when (target Jun 12)
- Get co-authorship / CMCC data-sharing preferences in writing
- Get her literature pointers refined (any further reviews to cite/differentiate against?)
- Post-meeting: write up notes within 24h

**Wed Jun 3 — Fri Jun 5 (compressed build sprint):**

The LLM stack that was originally May 30 / 31 / Jun 1 work lands here, post-meeting. Tight. If any item slips to Jun 6, eval week absorbs it.

- Implement `client_base.py` + `ollama_client.py` + `prompt.py` (deferred from pre-meeting)
- Implement `reasoning_chain.py` all 4 steps + `parser.py`
- Implement `corroboration.py` — all 10 claim types (apply any Bracco taxonomy adjustments from Jun 2 first)
- Implement `validate.py` (hallucination decision gate consuming corroboration scores)
- Implement `gpt_client.py` (GPT-5.4) and `gemini_client.py` (Gemini 3 Thinking) against the `LLMClient` ABC; verify quotas/budget
- Run reasoning chain on 10 anomalies with all 3 models; eyeball outputs for sanity
- Iterate on prompts based on what's clearly broken

**Acceptance criteria for Bracco prep window:**

- Bracco meeting happened; her labeling commitment + tolerance feedback + co-authorship + CMCC constraints captured in writing
- Reasoning chain runs end-to-end on 10 anomalies with all 3 models (target Jun 5; slip to Jun 6 acceptable)
- Corroboration scorer covers all 10 claim types

### Eval week: June 6 → June 12 (7 days)

**Goal:** Full eval harness + first round of analysis.

**Sat Jun 6 — Mon Jun 8 (eval execution, 3-day wall-clock):**

- Build `eval/harness.py` — runs all 3 models on the 50-anomaly set; persists Explanation + Claim records incrementally so re-runs are idempotent (resumable from any failure point)
- Run full eval. Realistic budget: 50 anomalies × 4 reasoning steps × 3 models = **600 calls**. Llama 3 8B at ~60s effective latency (cold-load + retries on parse failures) × 200 calls = **~3.3h local**. Cloud: GPT-5.4 200 calls × ~5s parallel ≈ 17 min; Gemini 3 Thinking 200 calls throttled at default RPM ≈ **~30 min** (apply async with rate-limit backoff). Plus inevitable 1–2 re-runs after prompt iteration → **plan ~2 full days of eval wall-clock**, not "overnight."
- Build `eval/label_cli.py` — minimal CLI labeling tool: `python -m app.eval.label --anomaly-id=<id>` loads context, presents claims one at a time, captures verdict + freeform note. Without this the labeling load becomes a 6-hour ordeal.
- Mason labels: 10 anomalies × 3 days = **30 anomalies cap, not in one push**. Avoid burnout.
- Ship the 10–20 selected for Bracco to her with the labeling instructions doc (sent in batch, not a CLI link — she works in Outlook)

**Tue Jun 9 — Thu Jun 11:**

- Build `corroboration_correlation.py` — Spearman/Pearson between corroboration scores and expert labels, overall and per claim type
- Build `calibration.py` — reliability diagrams + ECE per model
- Build `disagreement.py` — pairwise disagreement structure (which claim types do local + cloud disagree on?)
- First-pass results; iterate on scorer thresholds if a claim type is obviously miscalibrated

**Fri Jun 12:**

- Receive Bracco's labels (target); merge with Mason's labels
- Re-run correlation analysis on combined label set
- Generate report.py output

**Acceptance criteria for eval week:**

- 50 anomalies × 3 models fully evaluated
- 10–20 Bracco labels + 30+ Mason labels
- Three analyses (correlation, calibration, disagreement) complete with first-pass figures

### Polish & write-up: June 13 → June 16 (4 days)

- Refine figures + tables
- Write **1-page Month 2 results summary** (headline correlation, calibration findings, disagreement findings) — short artifact suitable for sending Bracco
- The 5-page methodology + results draft is **pushed to Month 3 Week 1** — it depends on the visualization choices Month 3 makes anyway, and 4 days at end of month is too tight for both
- Identify findings strong enough to send Bracco for review
- Tag repo `month2-complete`
- Write Month 3 plan (web app) starting with the eval results as the source of truth for what to visualize

---

## Bracco-readiness checkpoint (June 2 meeting)

This is the meeting that justifies the whole Month 2 framing. Materials to have ready:

1. **1-page memo (printed)** brought to the meeting — Section 1 of the Bracco notes doc: thesis (with FActScore differentiation), mechanism, the demo plan, the four questions. No pre-read — the meeting is for her to hear the ideas live and check feasibility / novelty.
2. **The demo** — hand-curated by default (one anomaly → 4-step reasoning chain output → corroboration scores per claim, side-by-side). If the working pipeline landed by Jun 1 it's a bonus; the meeting framing is the same either way.
3. **Claim taxonomy spec** — the corroboration scorer's 10 claim types with tolerance defaults and partial-verifiability flags; she should be able to challenge any of them. Printed as a small table for in-meeting pointing.
4. **Labeling instructions doc** — for the 10–15 anomalies she'll label: what counts as a correct/incorrect claim, how to flag uncertainty, time budget (~5 min per anomaly target). Format is plain-text/PDF, not a CLI link — she works in Outlook.
5. **Four questions for her:**
   - "Are these tolerance thresholds reasonable for Houston-area meteorology?"
   - "Are there claim types I'm missing that would be obvious to a climate scientist? Specifically, am I handling `emissions_source_type`, `secondary_formation`, and `background_vs_event` correctly for the Houston regime?"
   - "What would convince you, in 5 minutes of looking at one example, that an LLM's atmospheric attribution is wrong?"
   - "Will you commit to labeling 10 anomalies by Jun 12, and is this format usable? Are there 2–3 more papers on cross-source eval or LLM scientific reasoning you'd recommend I cite?"
6. **Post-meeting (within 24h):** written confirmation from Bracco of (a) labeling commitment, (b) attribution/co-authorship preferences for any eventual preprint, (c) any CMCC data-sharing constraints on the labeled dataset. This is a 30-second conversation that becomes a 6-month IRB problem if skipped.

---

## Risks + mitigations

| Risk                                                                                                                                     | Likelihood  | Mitigation                                                                                                                                                                                                                                                                                                                           |
| ---------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Server / Ollama setup eats more than 2 days                                                                                              | Medium      | Prototype on Mason's dev machine in parallel (Apple Silicon Ollama works); migrate to server later. Don't block the detection engine on it.                                                                                                                                                                                          |
| Corroboration scorer design takes >1 week                                                                                                | Medium-High | The Wed May 21 design memo is the gate. If the memo isn't done by then, postpone first reasoning-chain implementation and finish the design before Bracco meeting; better to walk in with a clean mechanism design than a half-built one.                                                                                            |
| Bracco can't label 10 anomalies by Jun 12                                                                                                | Medium      | Mason labels everything by Jun 10; Bracco labels become an _audit set_ for IRR (inter-rater reliability), not the ground truth. The validation analysis still works; the title shifts from "expert-labeled" to "expert-audited."                                                                                                     |
| Cloud API budget runs out mid-eval                                                                                                       | Low-Medium  | 50 anomalies × 4 steps × 2 cloud models = ~400 cloud calls. At <$0.50/call worst case = $200. Budget for $300. If GPT-5.4 access blocked, fall back to GPT-4-class equivalent + note in methodology.                                                                                                                                 |
| Llama 3 8B latency exceeds estimate                                                                                                      | Medium      | See Power & Statistics — realistic wall-clock is ~2 full days for the 600-call sweep, not "overnight." Plan accordingly; idempotent eval means partial runs accumulate.                                                                                                                                                              |
| Corroboration↔truth correlation is weak                                                                                                  | Medium      | This is a _finding_, not a failure. Reframe to "limits of cross-source eval" and characterize which claim types are reliably corroborated vs. not. Still publishable.                                                                                                                                                                |
| Scope creep (counterfactuals, physics priors, additional metrics, etc.) tempts in Eval week                                              | High        | Hard "Month 4 backlog" doc. Anything not in the in-scope list goes there, not into Month 2.                                                                                                                                                                                                                                          |
| Detection engine catches too few or too many anomalies                                                                                   | Medium      | Tunable thresholds per metric; eyeball-tune on Day 4 before committing to the 50-anomaly eval set. If detection is over-noisy, prefer fewer high-confidence anomalies (severity = severe) for the eval.                                                                                                                              |
| Per-claim-type N too small for inferential statistics                                                                                    | High        | Restrict inferential claims to the three headline types (`concentration_elevation`, `transport_direction`, `meteorological_state`) where N≥20 is targeted. Other 7 types are descriptive only with N reporting.                                                                                                                      |
| Detection produces zero "events" — Houston has limited anomalous days in a 4-week window                                                 | Medium      | Define eval set inclusion as **"top-50 by composite severity score"** rather than "above threshold." Confirm Day 4 the top-50 ranking is meaningful (severity gradient, not flat list of borderline events).                                                                                                                         |
| Reasoning-chain output not reliably parseable across 3 models                                                                            | Medium-High | (a) Use JSON-schema-constrained generation: OpenAI `response_format=json_schema`, Gemini `response_schema`, Ollama `format: "json"` + pydantic with one retry-on-error-feedback. (b) **Measure parse-failure rate as a first-class result — "local model fails structured output 15% of the time" is itself a publishable finding.** |
| Sentinel-5P spatial resolution (5.5×3.5 km, granule-mean only in v1) too coarse for `point_source_attribution` and noisy for `chemistry` | Confirmed   | These two claim types marked `partial_verifiability=true`. Wider tolerance for `chemistry` (±50% rather than ±20%); HCHO-silent treated as silent not contradicting. Excluded from headline correlation analysis. Per-pixel gradient extraction deferred to Month 4.                                                                 |
| Mason burnout from labeling 30+ anomalies in one push                                                                                    | Medium      | (a) Build the `label_cli.py` tool early so labeling is not a JSON-editing chore. (b) Cap labels at 10/day across 3 days, not 30 in one push. (c) Skip a few intentionally if a particular anomaly has insufficient enrichment context.                                                                                               |
| Cloud models refuse or heavily hedge causal attributions (e.g., "I cannot attribute pollution to a specific facility")                   | Medium      | Add explicit prompt framing: **"scientific hypothesis generation, not legal attribution; this is a research evaluation context."** Measure hedging rate per model as a result.                                                                                                                                                       |
| Bracco's labels become a publishable dataset → co-authorship / data-sharing complications                                                | Low-Medium  | In the Jun 2 meeting, ask explicitly: "Are you comfortable with these labels being released alongside an eventual preprint, with you credited?" Document the answer in writing. Ask about CMCC data-sharing constraints. Get this in writing within 24h of the meeting.                                                              |

---

## Verification

### Detection engine

- Unit tests per detector against synthetic data with injected anomalies (`tests/detection/test_zscore.py`, etc.)
- Fixture set of 5–10 hand-labeled real anomalies in `tests/detection/fixtures/known_anomalies.json`; all three detectors must catch them (`tests/detection/test_known_anomalies.py`)
- Smoke test: `python -m app.detection.run` produces nonzero anomalies on real data

### Reasoning chain

- Unit test that each step produces structured (parsable) output across all 3 models (`tests/llm/test_reasoning_chain.py`)
- Integration test: full chain runs end-to-end against a fixture EnrichmentRecord, produces an Explanation with non-empty claims at each step
- **Parse-failure rate is measured per model and reported as a result** — not treated as a bug
- Manual review: 5 anomalies' chain outputs eyeballed by Mason for narrative coherence and source citation presence

### Corroboration scorer

- Unit test per claim type (10 functions) with hand-crafted (claim, evidence) pairs verifying per-source verdict classification (+1 / -1 / 0)
- Sanity test: a claim with explicit data support across 3+ sources gets score ≈ +1 with `evidence_n ≥ 3`; an explicitly contradicted claim gets ≈ -1; a claim with no relevant data gets `score=null, unverified=true`
- `evidence_n` always reported alongside score; `per_source_verdicts` JSON is queryable for downstream slicing
- Cross-check: corroboration scores for an entire reasoning chain should be roughly monotonic in plausibility on a small Mason-eyeballed set

### Eval harness

- End-to-end: `python -m app.eval.harness --anomaly-set fixtures/eval50.json` produces full results for all 3 models
- **Idempotent + resumable** — partial runs persist Explanation rows individually; re-running only fills missing (anomaly, model) cells
- Report regenerable from persisted Explanation records (no rerun needed for figure tweaks)
- Per-claim-type N reported in every figure; headline correlation analysis only on the 3 designated headline types

### Bracco-readiness gate (Mon Jun 1, 22:00 CT)

- Demo anomaly chosen + hand-curated 4-step reasoning chain + per-claim corroboration scores ready (drafted on vacation, finalized June 1)
- 1-page printed memo (Section 1 of the Bracco notes doc) ready to bring to the meeting
- Labeling template doc shareable as plain PDF (not CLI link)
- (Bonus, not required) `python -m app.llm.explain --anomaly-id=<one-demo>` runs end-to-end with local Llama 3 8B; if it works, it becomes the live demo, otherwise the hand-curated path is the planned demo

### Post-meeting gate (Wed Jun 3, 18:00 CT)

- `docs/bracco/2026-06-02-postmeeting-notes.md` committed with: labeling commitment confirmed in writing, attribution/co-authorship preferences captured, CMCC data-sharing constraints documented, any additional literature pointers added to the project reading list

### IRR gate (Jun 12)

- Cohen's κ computed on the Mason-vs-Bracco overlap subset (10–15 anomalies labeled by both)
- κ ≥ 0.6: proceed with headline "expert-labeled" framing
- κ < 0.6: downgrade to "expert-audited" framing in all results

### Month 2 completion gate (June 16)

- `eval/report.py` generates correlation, calibration, and disagreement figures from real data
- 50 anomalies in DB with ≥1 expert label each (Mason or Bracco)
- 1-page Month 2 results summary committed to `docs/research/2026-06-16-month2-results-summary.md`
- Repo tagged `month2-complete`
- 5-page methodology + results draft scheduled into Month 3 Week 1, not Month 2

---

## Critical files (paths to be created/modified)

```
server/app/detection/{engine,zscore,stl,isolation_forest,consensus,enrichment}.py    [new]
server/app/llm/{client_base,ollama_client,gpt_client,gemini_client,rag,prompt,reasoning_chain,parser,corroboration,validate,explain}.py    [new]
server/app/eval/{harness,corroboration_correlation,calibration,disagreement,report,label_cli}.py    [new]
server/app/db/models.py    [add Anomaly, EnrichmentRecord, Explanation, Claim (with per_source_verdicts JSON + evidence_n + partial_verifiability), ExpertLabel]
server/app/api/routes/{anomalies,explanations,evaluation}.py    [new]
server/requirements.txt    [add chromadb, scikit-learn, statsmodels, ollama, openai, google-genai]
server/data/chromadb/    [populate with EPA breakpoints + atmospheric reference + Houston context]
server/tests/{detection,llm,eval}/    [new test trees]
docs/specs/2026-05-16-month2-phase-plan.md    [copy of this plan after approval]
docs/specs/2026-05-21-corroboration-scorer-design.md    [design memo committed; mirrors Section 4 of the Bracco meeting notes Doc, kept in repo as the durable design record]
docs/bracco/2026-06-02-memo.pdf    [printed 1-page meeting handout — Section 1 of the Bracco notes doc]
docs/bracco/2026-06-02-postmeeting-notes.md    [labeling commitment + co-authorship + CMCC constraints, captured in writing within 24h]
docs/research/2026-06-16-month2-results-summary.md    [1-page end-of-month summary; 5-page methodology+results draft pushed to Month 3 W1]
```

### Load-bearing files (the ones that most need to be right)

- `server/app/llm/corroboration.py` — where the paper's core claim lives
- `server/app/llm/validate.py` — the CLAUDE.md-guaranteed hallucination gate; do not let claim-extraction logic squat here (that's `parser.py`)
- `server/app/db/models.py` — `per_source_verdicts: JSON` is the load-bearing schema detail enabling downstream disagreement analysis
- `server/app/detection/enrichment.py` — spatiotemporal cross-source joining; underestimated complexity (4 different spatial conventions across collectors: gridded GFS at 0.25°, point OpenAQ stations, single-anchor Sentinel-5P granule mean, 5-point OpenWeather grid)
- `docs/specs/2026-05-21-corroboration-scorer-design.md` — if this memo is sharp, the Bracco meeting succeeds

---

## Connection to broader research narrative

This phase produces the _mechanism_ and the _core empirical result_. Month 3 builds the visualization (web app) that exposes the corroboration scores per claim to a human user. Month 4 scales the eval (100+ anomalies, ablations including physics-grounded prompts and counterfactual attribution), runs the user comprehensibility study, and writes the preprint. Month 5 is competition submissions + stretch goals.

The corroboration-as-eval-proxy framing established here is what makes the preprint defensible. Without it, the project is "yet another local-RAG-vs-cloud comparison." With it, the project contributes a generalizable methodology for evaluating LLM scientific reasoning at scale.
