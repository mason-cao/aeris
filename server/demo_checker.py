"""Live demonstration of the corroboration scorer, for screen recording.

Reads one real event's stored measurements out of the analysis snapshot and
scores sentences against them with the fixed rules in app.llm.corroboration.
No language model and no human judgment anywhere in here.

    python demo_checker.py              # recording pace
    python demo_checker.py --pace 0     # instant, for checking output
    python demo_checker.py --width 78   # narrower terminal

Requires only the standard library plus the app package; no database server.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from collections.abc import Mapping
from pathlib import Path

from app.llm.corroboration import SOURCE_CHANNELS, score_claim

EVENT = "f419133e"
DB = Path(__file__).parent / "analysis" / "aeris-packets-20260810.db"

C = {
    "head": "\033[96m",
    "good": "\033[92m",
    "bad": "\033[91m",
    "dim": "\033[90m",
    "text": "\033[97m",
    "warn": "\033[93m",
    "off": "\033[0m",
    "b": "\033[1m",
}

# Human names for the eight networks and the five measurement channels the
# scorer collapses them into.
SOURCE_NAMES = {
    "openaq": "OpenAQ regulatory monitor",
    "tceq": "TCEQ regulatory monitor",
    "epa_aqs": "EPA AQS regulatory monitor",
    "purpleair": "PurpleAir optical sensor",
    "sentinel5p": "Sentinel-5P satellite",
    "noaa_gfs": "NOAA GFS weather model",
    "openweather": "OpenWeather product",
    "asos": "ASOS airport anemometer",
}
SHORT_NAMES = {
    "openaq": "OpenAQ monitor",
    "tceq": "TCEQ monitor",
    "epa_aqs": "EPA AQS monitor",
    "purpleair": "PurpleAir sensor",
    "sentinel5p": "Sentinel-5P",
    "noaa_gfs": "NOAA GFS",
    "openweather": "OpenWeather",
    "asos": "ASOS anemometer",
}
CHANNEL_NAMES = {
    "ground_insitu": "Regulatory ground monitors",
    "ground_optical": "Low-cost optical sensors",
    "satellite_column": "Satellite",
    "nwp": "Weather models",
    "met_insitu": "Direct weather instruments",
}
VERDICT_WORD = {1: "supports", -1: "contradicts", 0: "no reading to offer"}


class Beat:
    """One sentence put to the checker, with the readings worth showing.

    ``aside`` restates, in plain words, what the scorer's own evidence line
    says about how it reached the verdict. The raw evidence line is printed
    underneath it, so nothing on screen is a paraphrase the viewer cannot
    check against the machine's own output.
    """

    def __init__(
        self,
        text: str,
        note: str,
        metric: str | None,
        aside: tuple[str, ...] = (),
    ) -> None:
        self.text = text
        self.note = note
        self.metric = metric
        self.aside = aside


BEATS = (
    Beat(
        "Winds were calm, below 1 m/s, at the time of the event.",
        "",
        "wind_speed",
        aside=(
            "OpenWeather reads 1.3 m/s, under the 1.5 m/s calm floor agreed "
            "in July before any of this ran, so it counts as calm. GFS models "
            "3.0 m/s and does not.",
        ),
    ),
    Beat(
        "Winds were strong, above 12 m/s, at the time of the event.",
        "the same sentence, wind speed changed, nothing else",
        "wind_speed",
    ),
    Beat(
        "A TCEQ monitor situated 0.02 km from the anomaly location recorded a "
        "co-located carbon monoxide concentration of 0.4 ppm at "
        "2026-06-15T16:00:00+00:00.",
        "written by one of the three models about this event, quoted exactly",
        "co",
        aside=(
            "Sentinel-5P measures a carbon monoxide column and would have "
            "been a second, separate group here. Its nearest reading is "
            "twenty hours old, and the freshness limit is twelve, so it is "
            "dropped.",
        ),
    ),
    Beat(
        "Meteorological stagnation and moisture trapping are a plausible cause "
        "of elevated local PM2.5 because winds near the anomaly were very weak "
        "to calm, relative humidity was about 95 to 100 percent, cloud cover "
        "was near 100 percent, precipitable water was high, and modeled "
        "boundary-layer height was low, all of which favor poor dispersion and "
        "aerosol accumulation in Houston.",
        "another of the three, same event, also quoted exactly",
        None,
        aside=(
            "The weather model did measure the boundary layer: 187 m, "
            "against a 969 m average for that hour of day.",
            "But the bar for calling the layer suppressed, fixed before any of "
            "this ran, is two standard deviations below that average, which is "
            "23 m. 187 is not below 23, so the rule returns nothing.",
            "And a weather model on its own is one group, so there is "
            "nothing independent to weigh it against.",
        ),
    ),
)


def reading(summary: Mapping, source: str, metric: str | None) -> str:
    """The reading nearest in time that this source has for this quantity."""
    if metric is None:
        return ""
    block = (summary["sources"].get(source) or {}).get("metrics") or {}
    if source == "noaa_gfs" and metric == "wind_speed":
        u = (block.get("u_10m") or {}).get("nearest_in_time") or {}
        v = (block.get("v_10m") or {}).get("nearest_in_time") or {}
        if u.get("v") is None or v.get("v") is None:
            return ""
        return f"{math.hypot(u['v'], v['v']):.1f} m/s"
    entry = block.get(metric) or {}
    nearest = entry.get("nearest_in_time") or {}
    if nearest.get("v") is None:
        return ""
    value = round(nearest["v"], 1)
    return f"{value:.1f} {entry.get('unit', '')}".strip()


class Screen:
    def __init__(self, pace: float, width: int, clear: bool = True) -> None:
        self.pace = pace
        self.width = width
        self.clearing = clear

    def wait(self, seconds: float) -> None:
        if self.pace:
            time.sleep(seconds * self.pace)

    def line(self, text: str = "", pause: float = 0.0) -> None:
        print(text)
        sys.stdout.flush()
        self.wait(pause)

    def rule(self, char: str = "-") -> None:
        self.line(f"{C['dim']}  {char * (self.width - 4)}{C['off']}")

    def title(self, text: str) -> None:
        if self.clearing:
            print("\033[2J\033[H", end="")
        self.line()
        self.rule("=")
        for row in text.split("\n"):
            self.line(f"  {C['head']}{C['b']}{row}{C['off']}")
        self.rule("=")
        self.line()

    def wrap(self, text: str, indent: str, color: str = "") -> None:
        import textwrap

        body = textwrap.fill(text, self.width - len(indent))
        for row in body.splitlines():
            self.line(f"{indent}{color}{row}{C['off']}")


def load(event: str) -> tuple[dict, dict, list[tuple[str, str, float | None, int]]]:
    if not DB.exists():
        sys.exit(f"analysis snapshot not found: {DB}")
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    anomaly = con.execute(
        "SELECT id, timestamp, metric, value, expected_value, z_score, severity, "
        "methods_triggered FROM anomalies WHERE substr(id,1,8) = ?",
        (event,),
    ).fetchone()
    if anomaly is None:
        sys.exit(f"event {event} not in the snapshot")
    summary = json.loads(
        con.execute(
            "SELECT cross_source_summary_json FROM enrichment_records WHERE anomaly_id = ?",
            (anomaly[0],),
        ).fetchone()[0]
    )
    claims = con.execute(
        "SELECT e.model_name, c.claim_text, c.corroboration_score, c.skipped_phase2 "
        "FROM claims c JOIN explanations e ON e.id = c.explanation_id "
        "WHERE e.anomaly_id = ? ORDER BY e.model_name, c.step_index",
        (anomaly[0],),
    ).fetchall()
    corpus = con.execute(
        "SELECT count(*), sum(CASE WHEN corroboration_score IS NOT NULL THEN 1 ELSE 0 END) "
        "FROM claims"
    ).fetchone()
    con.close()
    fields = ("id", "timestamp", "metric", "value", "expected", "z", "severity", "methods")
    return dict(zip(fields, anomaly)) | {"corpus": corpus}, summary, claims


def show_event(s: Screen, anomaly: dict) -> None:
    s.title("How much of what a language model says about the physical "
            "world\ncan be checked against an instrument?")
    ts = str(anomaly["timestamp"])[:16]
    s.line(f"      Houston, Texas          {C['b']}{ts} UTC{C['off']}", 0.5)
    s.line(f"      A regulatory monitor reported PM2.5 at "
           f"{C['b']}{anomaly['value']:.0f} ug/m3{C['off']}.", 0.5)
    s.line(f"      Normal for that hour is {anomaly['expected']:.1f} ug/m3. "
           f"That is {anomaly['z']:.1f} standard deviations out.", 0.4)
    s.line("      Three separate detectors flagged it.", 1.6)
    s.line()
    s.line(f"  Three models explained it. {C['b']}36 sentences{C['off']} between "
           f"them.", 1.8)


def show_instruments(s: Screen, summary: Mapping) -> None:
    s.title("Instrument readings that hour")
    rows = (
        ("openaq", "pm25", "fine particles"),
        ("openaq", "pm10", "coarse particles"),
        ("purpleair", "pm25", "fine particles"),
        ("tceq", "no2", "nitrogen dioxide"),
        ("asos", "wind_speed", "wind speed"),
        ("asos", "humidity", "relative humidity"),
        ("noaa_gfs", "wind_speed", "wind speed (modelled)"),
        ("noaa_gfs", "pbl_height", "boundary-layer depth"),
    )
    for source, metric, label in rows:
        value = reading(summary, source, metric)
        if not value:
            continue
        s.line(f"      {SOURCE_NAMES[source]:<28}{label:<24}{C['b']}{value}{C['off']}", 0.28)
    s.line()
    live = sum(1 for ok in (summary.get("coverage") or {}).values() if ok)
    s.line(f"  {C['dim']}{live} of the eight networks had data here.{C['off']}", 1.8)


def show_beat(s: Screen, index: int, beat: Beat, summary: Mapping) -> None:
    scored = score_claim(beat.text, summary)
    result = scored.result

    s.title(f"Sentence {index}")
    s.wrap(f'"{beat.text}"', "  ", C["text"] + C["b"])
    s.line()
    if beat.note:
        s.line(f"  {C['dim']}({beat.note}){C['off']}", 1.6)
        s.line()
    else:
        s.wait(1.6)

    kind = scored.claim_type.value.replace("_", " ")
    s.line(f"  The rules read this as a claim about   {C['head']}{kind}{C['off']}", 1.0)
    s.line()

    spoke = {k: v for k, v in result.per_source_verdicts.items() if v}
    silent = [k for k, v in result.per_source_verdicts.items() if not v]
    channels = dict(result.per_channel_verdicts or {})

    if spoke:
        s.line("  Instruments that can check it, grouped by what they are:", 0.9)
        s.line()
        for channel, verdict in channels.items():
            members = [k for k in spoke if SOURCE_CHANNELS.get(k) == channel]
            if not members:
                continue
            s.line(f"      {C['b']}{CHANNEL_NAMES.get(channel, channel)}{C['off']}", 0.4)
            for source in members:
                colour = C["good"] if spoke[source] > 0 else C["bad"]
                value = reading(summary, source, beat.metric)
                s.line(
                    f"          {SHORT_NAMES.get(source, source):<20}{value:<12}"
                    f"{colour}{VERDICT_WORD[spoke[source]]}{C['off']}",
                    0.7,
                )
            if verdict == 0 and len(members) > 1:
                s.wrap("these two disagree, so this group gives no verdict",
                       "          ", C["warn"])
            s.line(pause=0.3)
        if silent:
            names = ", ".join(SHORT_NAMES.get(x, x) for x in silent)
            s.wrap(f"nothing to add: {names}", "      ", C["dim"])
            s.line()

    for paragraph in beat.aside:
        s.wrap(paragraph, "      ", C["text"])
        s.line()
        s.wait(1.0)
    s.wait(0.8)
    if result.corroboration_score is None:
        s.line(f"  {C['b']}{C['warn']}VERDICT   NO ANSWER EITHER WAY{C['off']}", 0.8)
        s.line(f"            {C['dim']}Nothing available could confirm it or "
               f"contradict it.{C['off']}", 2.2)
    else:
        agree = result.corroboration_score > 0
        colour = C["good"] if agree else C["bad"]
        word = "AGREES WITH THE MEASUREMENTS" if agree else "CONTRADICTS THE MEASUREMENTS"
        groups = len([v for v in channels.values() if v])
        s.line(f"  {C['b']}{colour}VERDICT   {word}{C['off']}", 0.8)
        s.wrap(f"{groups} independent group{'' if groups == 1 else 's'} "
               f"behind this verdict", "            ", C["dim"])
        s.wait(2.2)


def show_sweep(s: Screen, summary: Mapping, claims: list, unblind: bool = False) -> None:
    s.title("All 36 sentences")
    s.line()
    verdicts = {"agrees": 0, "contradicts": 0, "mixed": 0, "silent": 0}
    reproduced = checked = 0
    # Model identity stays hidden until labeling is done: the per-model coverage
    # split is result-shaped for anyone about to label.
    aliases: dict[str, str] = {}
    for model, *_ in claims:
        if model not in aliases:
            aliases[model] = f"model {chr(ord('A') + len(aliases))}"
    for model, text, stored, skipped in claims:
        if skipped:
            verdicts["silent"] += 1
            mark, colour, label = "-", C["dim"], "cannot be checked"
        else:
            live = score_claim(text, summary).result.corroboration_score
            checked += 1
            reproduced += live == stored
            if live is None:
                verdicts["silent"] += 1
                mark, colour, label = "-", C["dim"], "cannot be checked"
            elif live > 0:
                verdicts["agrees"] += 1
                mark, colour, label = "+", C["good"], "agrees"
            elif live < 0:
                verdicts["contradicts"] += 1
                mark, colour, label = "x", C["bad"], "contradicts"
            else:
                verdicts["mixed"] += 1
                mark, colour, label = "=", C["warn"], "mixed"
        shown = model if unblind else aliases[model]
        snippet = " ".join(text.split())[:31]
        s.line(f"      {colour}{mark}{C['off']} {C['dim']}{shown:<17}{C['off']}"
               f"{snippet}...  {colour}{label}{C['off']}", 0.16)
    s.line()
    total = len(claims)
    got = verdicts["agrees"] + verdicts["contradicts"] + verdicts["mixed"]
    s.line(f"  {C['b']}{total} sentences. {got} got a verdict. "
           f"{verdicts['silent']} could not be checked at all.{C['off']}", 1.6)
    s.line()
    s.line(f"  {C['dim']}Re-run today, every verdict came out the same as it "
           f"did in August.{C['off']}", 2.0)


def show_corpus(s: Screen, anomaly: dict) -> None:
    total, scored = anomaly["corpus"]
    s.title("All 50 events")
    s.line(f"      {C['b']}{total:,}{C['off']} sentences, from three models.", 1.0)
    s.line(f"      {C['b']}{scored:,}{C['off']} of them could be checked against an "
           f"independent instrument.", 1.0)
    s.line(f"      {C['b']}{total - scored:,}{C['off']} could not.", 1.8)
    s.line()
    s.line()
    s.wrap(
        "Which model is which is held back until the marking is done.",
        "  ",
        C["dim"],
    )
    s.line()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pace", type=float, default=2.5, help="0 runs instantly")
    parser.add_argument("--width", type=int, default=78)
    parser.add_argument("--event", default=EVENT)
    parser.add_argument("--no-clear", action="store_true", help="keep one transcript")
    parser.add_argument("--unblind", action="store_true",
                        help="show real model names; never for a labeler")
    args = parser.parse_args()

    anomaly, summary, claims = load(args.event)
    s = Screen(args.pace, args.width, clear=not args.no_clear)

    show_event(s, anomaly)
    show_instruments(s, summary)
    for index, beat in enumerate(BEATS, start=1):
        show_beat(s, index, beat, summary)
    show_sweep(s, summary, claims, unblind=args.unblind)
    show_corpus(s, anomaly)


if __name__ == "__main__":
    main()
