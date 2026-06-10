# AERIS Corroboration Scorer — Design Memo

> **Date:** 2026-05-21 (initial design); committed 2026-05-24 from in-trip notes; revised 2026-05-26 to add Phase 1 / Phase 2 split per Dr. Bracco's 2026-05-25 email feedback; revised 2026-06-10 per her post-meeting email on claim types 6, 7, and 10 (see addendum at end)
> **Status:** Draft for Dr. Bracco review at the June 2 meeting
> **Related:** [Month 2 phase plan](2026-05-16-month2-phase-plan.md); Section 4 of the June 2 Bracco meeting notes (Google Doc, where this same content lives in Mason's conversational voice)

---

## Purpose

The corroboration scorer is the load-bearing piece of Month 2's **Phase 2 (novelty contribution)**. For each LLM-generated claim about an atmospheric anomaly that survives Phase 1 grounding, it computes a `corroboration_score ∈ [-1, +1]` against the agreement of the four data sources — Sentinel-5P, OpenAQ, NOAA GFS, OpenWeather — which sense different facets of one shared physical state through measurement processes that are largely independent. Their agreement is informative as a correctness signal to the degree those measurement errors stay independent even where the physical signal is shared. The score is a label-free proxy for ground-truth verification; the Month 2 research question is whether the score correlates with expert labels strongly enough to stand in as a scalable evaluation signal.

Three of the ten claim types are designated **headline** (N≥20 targeted, sufficient for inferential statistics). The other seven are descriptive only.

This memo specifies the taxonomy, the per-source scoring rules, the tolerance defaults, and the build structure. The tolerance defaults are draft values intended for Bracco to challenge.

### Phase 1 / Phase 2 sequencing (added 2026-05-26)

Per Bracco's 2026-05-25 email reply: do the more basic analysis first, then layer corroboration on top, to ensure the cross-source signal isn't dominated by fabricated claims. Month 2 therefore sequences as:

- **Phase 1 (`validate.py`)** — retrieval-grounded factuality check (FActScore-style). For each `ClaimDraft`, verify the claim's content is present in the retrieved enrichment context the model was given. Emits `grounding_verdict ∈ {grounded, unverified}` and `grounding_evidence_ref` (which slice of the context grounded the claim, null if unverified). Runs before this scorer. This is the CLAUDE.md-mandated hallucination gate and is independent of the corroboration logic below.
- **Phase 2 (this scorer, `corroboration.py`)** — per-claim agreement scoring across the 4 APIs, run only on Phase 1 survivors. Fabricated/unverified claims skip Phase 2 entirely with `corroboration_score=null, skipped_phase2=true`.

The thesis (corroboration as a label-free eval proxy *distinct from retrieval-grounded factuality*) is preserved and arguably strengthened: the Phase 1 → Phase 2 delta — claims that pass grounding but fail corroboration — is what empirically demonstrates the distinctness.

---

## Claim taxonomy

| #   | Type                       | Verifiable against                                                                | Headline | Partial | Default tolerance                                                                                                                                                                                                                                                                                |
| --- | -------------------------- | --------------------------------------------------------------------------------- | -------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | `concentration_elevation`  | OpenAQ, Sentinel-5P                                                               | **Yes**  | No      | ±25% of measured value; qualitative ("elevated") requires data to exceed local rolling baseline                                                                                                                                                                                                  |
| 2   | `transport_direction`      | NOAA GFS 10m wind, OpenWeather                                                    | **Yes**  | No      | claimed bearing within ±45° of measured wind                                                                                                                                                                                                                                                     |
| 3   | `meteorological_state`     | OpenWeather, NOAA GFS                                                             | **Yes**  | No      | wind speed ±1.5 m/s (or qualitative match, e.g., "stagnant" = <2 m/s); temperature ±2 °C                                                                                                                                                                                                         |
| 4   | `atmospheric_trap`         | NOAA GFS (PBL height, T@850), OpenWeather (surface T)                             | No       | No      | PBL height ±200 m; inversion = upper-air T > surface T                                                                                                                                                                                                                                           |
| 5   | `temporal_pattern`         | Time series of any source                                                         | No       | No      | monotonicity / trend-direction test; no quantitative tolerance                                                                                                                                                                                                                                   |
| 6   | `chemistry`                | Sentinel-5P (NO₂/HCHO ratio), OpenAQ (O₃/NO₂)                                     | No       | **Yes** | ±50% (HCHO retrievals are noisy at single-orbit resolution); HCHO-silent granules treated as silent, not contradicting                                                                                                                                                                           |
| 7   | `point_source_attribution` | Sentinel-5P (granule-mean only in v1), OpenAQ                                     | No       | **Yes** | direction-only check at granule-mean resolution; per-pixel gradient extraction deferred to Month 4                                                                                                                                                                                               |
| 8   | `emissions_source_type`    | OpenAQ time-of-day pattern, Sentinel-5P spatial pattern, OpenWeather wind         | No       | No      | rule-based: morning peak + I-610 corridor = mobile; persistent NO₂ near Ship Channel = point; broad anthropogenic signal across many neighborhoods = area. "Spread out and uniform across all stations" is the regional regime — that's type 10, not type 8                                      |
| 9   | `secondary_formation`      | OpenAQ O₃/NO₂ time-lag, OpenWeather solar radiation / cloud cover                 | No       | No      | O₃ peak lags NO₂ peak by ≥2 h on a high-insolation day                                                                                                                                                                                                                                           |
| 10  | `background_vs_event`      | Spatial coefficient of variation across OpenAQ stations; gradient vs. uniformity  | No       | No      | low CV across stations = regional / transport regime; localized spike = local event. **Most common attribution error for small LLMs.**                                                                                                                                                           |

Types 8–10 were added specifically to stress the patterns a small LLM is most likely to confuse for an industrial-city air-quality regime. They are the claim types most in scope for Bracco's review.

---

## Scoring procedure (per claim)

**Pre-condition:** Phase 1 grounding (`validate.py`) has already marked the claim `grounded`. Phase-1-unverified claims skip this scorer entirely (`corroboration_score=null, skipped_phase2=true`).

For each Phase-1-grounded claim:

1. **Classify** the claim into one or more types (lightweight rule-based; LLM-based classification deferred unless rules underperform).
2. **Identify** the relevant sources for each type from the table above.
3. **Query** the EnrichmentRecord for those sources in the relevant spatiotemporal window.
4. **Compare** the claim's quantitative or directional content against the data, one source at a time:
   - Match within tolerance → **+1 supporting**
   - Contradiction beyond tolerance → **−1 contradicting**
   - No data available → **0 silent**
5. **Aggregate** into a per-claim record:
   - `per_source_verdicts`: `{openaq: +1, sentinel5p: 0, gfs: -1, openweather: +1}` — the load-bearing detail for downstream disagreement analysis
   - `evidence_n` = count of non-silent verdicts (supporting + contradicting)
   - `corroboration_score`:
     - If `evidence_n ≥ 1`: `(supporting − contradicting) / evidence_n`, scalar in [−1, +1]
     - If `evidence_n == 0` (all silent): `null` with `unverified=true` — tracked separately in analysis, never treated as 0
   - `partial_verifiability=true` for `chemistry` and `point_source_attribution`; these are reported separately in the headline correlation analysis

**`evidence_n` is reported alongside the score.** A claim with `(score=+1, evidence_n=1)` and one with `(score=+1, evidence_n=3)` are not the same evidential weight. Downstream analyses weight by `evidence_n`.

---

## Build structure

Implemented as **10 small functions, one per claim type**, in `server/app/llm/corroboration.py`. Each returns `(per_source_verdicts: dict, evidence_summary: str)`. A shared aggregator derives the scalar score and `evidence_n`.

The scorer **only scores** — flagging is tracked as metadata alongside the raw score but is kept analytically separate so the raw scores remain available for the correlation analysis.

- **Phase 1 (`validate.py`) runs first.** Phase 1 is the retrieval-grounded factuality check (not the corroboration decision gate). It emits `grounding_verdict` + `grounding_evidence_ref` independently of any corroboration logic. Phase-1-unverified claims skip Phase 2 with `corroboration_score=null, skipped_phase2=true`.
- **Phase 2 (this scorer) emits a `low_corroboration_flag` as metadata**, computed at scoring time as `corroboration_score ≤ -0.5 AND evidence_n ≥ 2`. The flag is metadata, not a gate — downstream product code can consume it, but the raw `corroboration_score` is what the research analysis correlates against expert labels.

This separation matters: scoring is a research artifact for the correlation analysis; Phase 1 grounding and Phase 2 flagging are independent signals on the same Claim record.

---

## Why this is the hardest engineering of the month

- Tolerance thresholds per claim type — the defaults above are research artifacts in themselves, to be defended or revised with Bracco.
- Spatial windowing per claim type (point-source = tight; meteorological state = broad).
- Temporal windowing per claim type (instantaneous concentrations vs. multi-hour trends).
- Half-matching claims (LLM says "elevated NO₂" but the data shows elevated SO₂ instead).
- Per-source verdict logic differs by source: Sentinel-5P needs quality-flag handling, GFS needs interpolation, OpenAQ needs station-to-claim-location distance weighting.

---

## Open questions for the June 2 Bracco meeting

1. **Does the Phase 1 (retrieval-grounded factuality) → Phase 2 (cross-source corroboration) sequencing address the hallucination-dominance concern from your May 25 email? Is there a Phase 1 check I should add (e.g., entity-level grounding vs. statement-level grounding)?**
2. Are the tolerance defaults above reasonable for Houston-area meteorology?
3. Are claim types 8 (`emissions_source_type`), 9 (`secondary_formation`), and 10 (`background_vs_event`) correctly framed for the Houston regime?
4. Is there a claim type missing that an atmospheric scientist would consider obvious?
5. For the `chemistry` and `point_source_attribution` partial-verifiability flags, is ±50% tolerance and silent-granule treatment defensible, or should they be excluded from any quantitative reporting?

**Meeting talking point (not a question):** the Phase 2 corroboration mechanism is structurally distinct from SHAP. SHAP attributes model output to input features via Shapley values; this scorer attributes per-claim correctness to physical evidence agreement across independent sensor channels. Carloni's physics-as-attribution framing is the inspiration; the mechanism itself is not Shapley-based. Bracco's May 25 reply used "evolution of SHAP" loosely — worth clarifying so Phase 2 isn't expected to literally compute Shapley values.

---

## Addendum — Bracco feedback, 2026-06-10

Her post-meeting email flagged types 6 and 7 as "difficult to handle" if they rely on satellite data, and type 10 ("regional/local") as dependent on data quality. This effectively answers open question 5 above. Changes:

1. **Types 6 (`chemistry`) and 7 (`point_source_attribution`) are demoted to qualitative-only.** They were already `partial_verifiability=true` and excluded from headline correlations; they are now excluded from *all* quantitative reporting (no per-type correlation numbers at any N). They stay in the taxonomy and still get scored — how often the LLM leans on weakly-verifiable claim types is itself a result — but they are reported as case examples only. The `partial_verifiability` flag on the Claim record is unchanged.

2. **Type 10 (`background_vs_event`) gains a verifiability precondition.** The spatial-CV check is only meaningful when enough stations report. The scorer returns all-silent (`evidence_n=0`, `unverified=true` — not a verdict) unless, within the claim's spatiotemporal window:
   - **≥ 5 distinct OpenAQ stations** report the relevant pollutant, and
   - each counted station has **≥ 6 in-window observations** (enough to call it coverage rather than a stray reading).

   Both thresholds are draft values to confirm with Bracco, like every other tolerance in this memo. The station count and coverage actually available in the Houston network will be measured from collected data and reported alongside any type-10 result, so "data quality" is an explicit, inspectable precondition rather than an assumption.
