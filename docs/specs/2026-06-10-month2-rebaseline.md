# AERIS Month 2 — Rebaseline

> **Date:** 2026-06-10
> **Status:** Supersedes the timeline, eval-window, and completion gates of the [Month 2 phase plan](2026-05-16-month2-phase-plan.md). The thesis, Phase 1 / Phase 2 design, claim taxonomy, statistics plan, and scope guardrails of that plan are unchanged and still authoritative.
> **Related:** [Corroboration scorer memo](2026-05-21-corroboration-scorer-design.md)

---

## Why rebaseline

Two facts force this, and they compound:

1. **The summer-only eval scope is incompatible with the original calendar.** The eval restriction to summer-window anomalies (adopted 2026-06-02 per Bracco, to remove seasonal confounds) means the 50-anomaly eval set cannot exist by the original June 12–16 eval dates — barely ten days of summer have happened. Houston's photochemical season also peaks July–September, so a later eval draws from richer, more anomalous data, not just more of it.
2. **The build is ~5 days behind the original plan.** As of June 10: detection, enrichment, the reasoning chain, the parser, Phase 1 grounding, and 3 of 10 corroboration scorers are built and tested (417 tests green). Still missing: 7 claim-type scorers, `explain.py`, the cloud clients, the whole `eval/` tree, and the labeling CLI — roughly 1.5–2 weeks of work.

The rebaseline also answers the June 2 meeting concerns directly: the data-volume concern is resolved by letting the eval window grow with the season (and by reporting per-claim `evidence_n` as a first-class figure), and the black-box concern is already answered by the claim taxonomy + open prompts — nothing in the eval path is an LLM judge.

Month 2 was always the research anchor; Months 3–4 (web app, scaling/user study) are cheaper and shift back without consequence.

## Summer window, operationally defined

"Summer" was never given a date range in any doc. Fixing that here:

- **Eval window: June 1 – August 31, 2026.** The Month 2 eval set is drawn from June 1 onward and **frozen on July 13** as the top-50 summer anomalies by composite severity (per the original inclusion rule).
- **May data is shakeout only.** It validates the pipeline end-to-end and tunes prompts/thresholds. No May anomaly enters the eval set, no May result enters the analysis.
- Anomaly categories stay broad within the window — the cross-category breadth rule is unchanged. Categories are whatever the summer delivers; no specific event type (hurricane, dust intrusion) is promised.

## Revised timeline

| Window         | Dates           | Theme                                                                                                                                          |
| -------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Build finish   | Jun 10 → Jun 24 | Phase 1 numeric grounding fix; remaining 7 claim-type scorers; `explain.py`; `gpt_client.py` + `gemini_client.py`; `eval/harness.py` + `label_cli.py` |
| May shakeout   | Jun 24 → Jun 30 | End-to-end runs on ≥10 May anomalies × 3 models; prompt iteration; scorer threshold sanity pass; labeling-tool dry run                          |
| Accumulation   | Jul 1 → Jul 12  | Summer data accrues; nightly detection; rolling Mason labels (≤10/day); weekly Bracco update emails with data-volume numbers                    |
| Eval execution | Jul 13 → Jul 16 | Freeze top-50 summer set; full 3-model eval sweep (idempotent, ~2 days wall-clock); ship Bracco her 10–15 labeling subset                       |
| Analysis       | Jul 17 → Jul 20 | Merge labels; Phase 1 baseline, Phase 2 correlation, Phase 1 → Phase 2 delta, calibration, disagreement; IRR (κ)                                |
| Write-up       | Jul 21 → Jul 24 | Figures, 1-page results summary, tag `month2-complete`                                                                                          |

Downstream: Month 3 (web app) starts ~Jul 25; Month 4 shifts back the same ~5–6 weeks; Month 5 compresses if needed. Update CLAUDE.md's "frontend work begins Month 3 (June 2026)" line when Month 3 planning starts.

### Revised June deliverable

The Month 2 deliverable for June is no longer "eval complete." It is: **pipeline validated end-to-end on May shakeout data** — every stage from detected anomaly to persisted Claim rows with both Phase 1 and Phase 2 signals, across all three models, on ≥10 May anomalies. That is exactly the validation framing already given to Bracco in the post-meeting email ("backfilling data back to the start of May to make sure the whole pipeline works before the real summer data comes in").

## Scope changes

1. **ChromaDB knowledge-base ingestion: deferred to Month 3.** Phase 1 grounds claims against the enrichment context the model was shown, not a retrieved KB — a KB would actually blur what "grounded" means. Nothing in the Phase 1 / Phase 2 analysis consumes it. It returns in Month 3 when the NL-query feature needs retrieval.
2. **Phase 1 grounding becomes numeric-aware** (built alongside this memo). The v1 lexical-overlap check grounds "NO₂ exceeded 80 ppb" against a context reporting 30 ppb, which mechanically inflates the Phase 1 → Phase 2 delta — the headline result. v2 requires numeric content in a claim to be consistent with the context (±25%, matching the Phase 2 concentration tolerance). The delta analysis additionally reports sensitivity to Phase 1 strictness (lexical-only vs. numeric-aware) so the delta can't be dismissed as a strawman-baseline artifact.
3. **`evidence_n` distribution is promoted to a first-class figure.** Sentinel-5P contributes one granule-mean per species per day and GFS is 6-hourly, so many claims will resolve against only 1–2 sources. Reporting the distribution up front is the honest answer to the data-volume concern.
4. **README eval-categories phrasing**: soften the promised category list (hurricanes, Saharan dust, wildfire smoke) to "categories the summer delivers," keeping breadth without promising specific events.

## Bracco re-anchor (next email)

- Per-source row counts and date coverage pulled from the collector DB (numbers, not reassurance — the data-volume answer).
- The summer-window definition and the July 13 freeze date; why the eval *had* to wait for summer.
- Re-ask the labeling commitment explicitly: 10–15 anomalies, delivered to her ~Jul 16, returned ~Jul 20, ~5 min each.
- Capture in writing what the June 3 gate missed: labeling commitment, co-authorship/attribution preferences, any CMCC data-sharing constraints.

## Revised gates

- **Build gate (Jun 24):** all 10 scorers + `explain.py` + harness green; `python -m app.llm.explain --anomaly-id=<id>` runs end-to-end locally.
- **Shakeout gate (Jun 30):** ≥10 May anomalies × 3 models persisted with both phases populated; parse-failure rate measured per model.
- **Freeze gate (Jul 13):** top-50 summer set frozen and committed as a fixture list.
- **IRR gate (Jul 20):** κ on the Mason–Bracco overlap; ≥0.6 keeps the "expert-labeled" framing, below downgrades to "expert-audited" (unchanged rule).
- **Month 2 completion gate (Jul 24):** five analyses generated from real data; 1-page summary committed; repo tagged `month2-complete`.

## Risks added by the extension

| Risk                                                      | Mitigation                                                                                                       |
| --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Acer collector gap during the 5-week accumulation window  | Weekly OneDrive log check; GFS gaps backfilled within the ~10-day NOMADS window; row-count check in each Bracco update |
| Bracco unavailable for July labeling (travel/summer term) | Ask availability in the re-anchor email now; fall back to "expert-audited" framing per the original risk table      |
| Mid-July eval slips further                               | Harness is idempotent; the freeze date, not the finish date, is the hard commitment — analysis can trail by days    |
