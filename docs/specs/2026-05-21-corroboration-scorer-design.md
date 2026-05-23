# AERIS Corroboration Scorer — Design Memo

> **Date:** 2026-05-21 (design); committed 2026-05-24 from in-trip notes
> **Status:** Draft for Dr. Bracco review at the June 2 meeting
> **Related:** [Month 2 phase plan](2026-05-16-month2-phase-plan.md); Section 4 of the June 2 Bracco meeting notes (Google Doc, where this same content lives in Mason's conversational voice)

---

## Purpose

The corroboration scorer is the load-bearing piece of Month 2. For each LLM-generated claim about an atmospheric anomaly, it computes a `corroboration_score ∈ [-1, +1]` against the agreement of the four causally-coupled data sources: Sentinel-5P, OpenAQ, NOAA GFS, OpenWeather. The score is a label-free proxy for ground-truth verification; the Month 2 research question is whether the score correlates with expert labels strongly enough to stand in as a scalable evaluation signal.

Three of the ten claim types are designated **headline** (N≥20 targeted, sufficient for inferential statistics). The other seven are descriptive only.

This memo specifies the taxonomy, the per-source scoring rules, the tolerance defaults, and the build structure. The tolerance defaults are draft values intended for Bracco to challenge.

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

The scorer **only scores** — it does not make pass/fail decisions. Decisions (e.g., "corroboration_score ≤ −0.5 with `evidence_n ≥ 2` triggers a hallucination flag") live in `validate.py`. The separation matters: scoring is a research artifact for the correlation analysis; flagging is a product decision that depends on downstream tuning.

---

## Why this is the hardest engineering of the month

- Tolerance thresholds per claim type — the defaults above are research artifacts in themselves, to be defended or revised with Bracco.
- Spatial windowing per claim type (point-source = tight; meteorological state = broad).
- Temporal windowing per claim type (instantaneous concentrations vs. multi-hour trends).
- Half-matching claims (LLM says "elevated NO₂" but the data shows elevated SO₂ instead).
- Per-source verdict logic differs by source: Sentinel-5P needs quality-flag handling, GFS needs interpolation, OpenAQ needs station-to-claim-location distance weighting.

---

## Open questions for the June 2 Bracco meeting

1. Are the tolerance defaults above reasonable for Houston-area meteorology?
2. Are claim types 8 (`emissions_source_type`), 9 (`secondary_formation`), and 10 (`background_vs_event`) correctly framed for the Houston regime?
3. Is there a claim type missing that an atmospheric scientist would consider obvious?
4. For the `chemistry` and `point_source_attribution` partial-verifiability flags, is ±50% tolerance and silent-granule treatment defensible, or should they be excluded from any quantitative reporting?
