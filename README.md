# AERIS

**How much of what a language model says about the physical world can you
actually check against an instrument?**

I had three language models explain 50 real air-quality anomalies around
Houston. That came out to 1,781 separate sentences. Then I wrote a checker
that reads each sentence, works out what kind of claim it is, and goes to the
eight sensor networks I collect from to see whether any of them can confirm it
or contradict it.

514 of the 1,781 came back with a verdict. The other 1,267 got nothing. That
doesn't mean those sentences are wrong. It means no instrument I have could
speak to them either way.

Under a third. As far as I can tell nobody had measured that before, so it's
the result I actually care about.

## What's in this repo, and what isn't

AERIS started as an acronym for Autonomous Environmental RAG & Inference
System, and that's still where I want it to end up: live collection, vector
retrieval, a map you can click around in. None of that is built. What is built
is the collection and verification stage plus the evaluation code, which is
where the number above came from.

So read this as one measurement and the machinery that took it. If you came
for the map, it isn't here yet.

## The result

| out of 1,781 claims | count | share |
| --- | --- | --- |
| Failed the grounding gate before any instrument was consulted | 860 | 48.3% |
| Passed the gate, but no channel could weigh in | 407 | 22.9% |
| Got a verdict | 514 | 28.9% |

Of the 514 with a verdict, 331 were weighed by a single measurement channel
and 183 by two. Nothing in the whole corpus reached three. I can't fix that
ceiling by buying more sensors, and it's the part I'd want a reviewer to look
at hardest.

Two other things fell out of it.

235 claims can never be scored no matter what the instruments saw. 209 landed
outside my ten-type taxonomy entirely and 26 are a type I declared
qualitative-only back in June. So 13% of the corpus is closed off by my own
taxonomy before the sensors get a say.

Coverage isn't fixed by the sensor network either. Same events, same stored
evidence, same rules, and the unchecked fraction still varies a lot across the
three models. Most of it comes down to what the model chose to say. I'm
holding the per-model split back until the expert labels are in, since the
labeling is blinded and publishing it now would give away which model is
which.

## What this does not show

No expert labels have come back yet. The `expert_labels` table has 0 rows and
you can check that yourself. So:

- I have no agreement result. Whether a corroboration score tracks what a
  scientist would say about the same claim is the registered question, and
  it's still open. The 514 verdicts could be junk, and if they are, "under a
  third is checkable" is a meaningless number too. That's the whole reason the
  labels matter.
- I'm not ranking the three models. I ran three and I can't say which one did
  better yet.
- Coverage was computed with no labels at all, so nothing above depends on the
  labeling going well.

I also know about two real defects in the scorer. On sentences that report
several measurements at once it can compare a number against the wrong
species. And the intent keywords in one claim type don't read negation, so a
sentence arguing against something can score as though it argued for it. Both
create disagreement that comes out of my code instead of out of the method,
and I haven't measured how often either one fires. They're staying in anyway.
The scorer ran in August under a protocol I fixed in advance, and quietly
patching it afterwards would be worse than writing it up.

## How the checker works

There is no language model anywhere inside it. Every rule is written down and
deterministic.

1. A claim gets sorted into one of ten types (concentration elevation,
   transport direction, atmospheric trap, secondary formation, and so on) or
   into `unclassified`, which is never scored.
2. A grounding gate checks the claim's sources, terms and numbers against the
   evidence the model was actually shown. Fail it and the claim stops here.
3. The claim goes only to the measurement channels that can physically bear on
   that type. A wind-direction claim goes to the wind products and nowhere
   near the satellite columns.
4. Each channel comes back supporting, contradicting, or silent. If every
   channel is silent, the checker abstains.

The channel grouping is the part I'd argue about first. Sources that share a
measurement process collapse into one channel, so two monitors run off the
same program don't get to vote twice. It's a design choice about physics, and
it's a long way from a proof of statistical independence. GFS assimilates ASOS
observations, so my two weather channels share inputs, and I haven't measured
how much.

Eight sources, five channels:

| Source | What it measures | Channel |
| --- | --- | --- |
| OpenAQ | PM2.5 and ozone, filtered down to verified regulatory monitors | ground in-situ |
| TCEQ CAMS | NO2, SO2, CO from the state's preliminary feed | ground in-situ |
| EPA AQS | historical NO2, SO2, CO, 2025 backfill only | ground in-situ |
| PurpleAir | low-cost optical PM2.5, uncorrected | ground optical |
| Sentinel-5P | satellite NO2, SO2, CO, HCHO columns | satellite column |
| NOAA GFS | modelled winds and boundary-layer depth | NWP |
| OpenWeather | blended surface weather | NWP |
| ASOS / METAR | airport anemometers and thermometers | met in-situ |

## The frozen set

Everything was locked before any model ran.

- Window: 2026-06-01 to 2026-08-05 UTC, end-exclusive.
- 405,432 observations inside it. The snapshot holds 466,507 rows altogether;
  the rest are the 2025 EPA AQS baseline and a few rows past the cutoff. Seven
  of the eight networks have data inside the window, because EPA AQS is
  historical only.
- 6,395 anomalies detected in-window, deduplicated down to 2,304 events, top
  50 taken and stratified by pollutant: NO2 18, ozone 11, CO 6, PM10 5,
  PM2.5 5, SO2 5.
- Frozen at `server/fixtures/eval50.json`, snapshot sha256 `1e50e007...`, code
  commit `5072549`.
- Models: Llama 3 8B running locally through Ollama, GPT-5.4, and Gemini 3.6
  Flash. All 150 cells completed.

Anomaly IDs are random UUIDs, so re-running detection anywhere regenerates
them and invalidates the freeze. Copy the rows, don't re-detect.

The analysis plan went up on OSF on 2026-08-09, before any expert saw a claim:
`osf.io/hb92y`, sole contributor, embargoed until 2027-06-30 so it doesn't
turn public in the middle of a blind review. The headline statistic registered
there is a cluster-bootstrap Spearman correlation between the corroboration
score and the expert label, restricted to three claim types. The Cohen's kappa
in the code measures agreement between two human labelers, which a few of my
own older notes got wrong.

## Running it

```bash
git clone https://github.com/mason-cao/aeris.git
cd aeris/server
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
pytest -q
```

1,572 tests pass and 8 skip on Python 3.13, in about a minute. You'll want
Python 3.11 or newer, Ollama with `llama3:8b` for local generation, and API
keys for whichever sources you plan to hit. SQLite is enough for collection.
The analysis tier runs PostgreSQL with TimescaleDB, and since the pipeline is
ORM-only the only thing that changes between the two is the connection URL.

```bash
# Collect. All registered sources, or one.
python -m app.collectors.run_all
python -m app.collectors.run_all --source openaq

# Backfill a historical source.
python -m app.collectors.backfill --source epa_aqs \
  --since 2025-06-01 --until 2025-08-31

# Detect and enrich.
python -m app.detection.run
python -m app.detection.enrichment

# Explain one anomaly.
python -m app.llm.explain --anomaly-id=<UUID>

# API.
uvicorn app.main:app --reload --port 8000
```

`server/demo_checker.py` walks the scorer through one real event in the
terminal, sentence by sentence, so you can watch what a single verdict is made
of. It reads the August analysis snapshot, a 5 GB SQLite file I keep out of
the repo, so it won't run on a fresh clone.

The freeze and labeling CLIs are in the repo so anyone can read what they do,
but re-running them would break the frozen set and the registered protocol.

Environment variables are listed in `server/.env.example`.

## Where this goes next

The rest of the acronym, roughly in the order I want to build it:

- [x] Live collectors, historical backfill, anomaly detection, cross-source
      enrichment
- [x] LLM generation, grounding gate, corroboration scorer, labeling and
      ablation CLIs
- [x] Freeze the evaluation set and run the three-model sweep
- [ ] Collect expert labels and run the registered analysis
- [ ] ChromaDB retrieval, so the prompt pulls from a vector store instead of
      being handed structured rows
- [ ] React and Mapbox frontend
- [ ] WebSocket live updates and unattended operation

Retrieval is the one I care about most. Right now the evidence in every prompt
comes straight out of the database as structured text, so the R in the name is
still a promise.

## Acknowledgements

Dr. Annalisa Bracco is the scientific mentor for the evaluation. Thanks also
to Lester Mackey for pointers on weighting evidence across sources.

## License

MIT
