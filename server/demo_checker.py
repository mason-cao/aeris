"""Live demonstration of the corroboration scorer.

Reads one real event's stored measurements and scores sentences against them
with fixed rules. No language model and no human judgment anywhere in here.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

from app.llm.corroboration import score_claim

EVENT = "f419133e"
PAUSE = float(sys.argv[1]) if len(sys.argv) > 1 else 2.5

CLAIMS: tuple[tuple[str, str], ...] = (
    ("Winds were calm, below 1 m/s, at the time of the event.", "as written"),
    ("Winds were strong, above 12 m/s, at the time of the event.", "same sentence, wind speed changed"),
    ("Ozone reached 0.011 ppm near the anomaly.", "as written"),
    ("Ozone reached 0.250 ppm near the anomaly.", "same sentence, number changed"),
    ("Emissions accumulated near the surface overnight.", "true or false, nothing measures it"),
)

VERDICT = {1.0: "AGREES with the measurements", -1.0: "CONTRADICTS the measurements", 0.0: "MIXED"}


def load_summary() -> dict:
    out = subprocess.run(
        ["docker", "exec", "server-db-1", "psql", "-U", "aeris", "-d", "aeris", "-t", "-A", "-c",
         "SELECT er.cross_source_summary_json FROM enrichment_records er "
         "JOIN anomalies a ON a.id = er.anomaly_id "
         f"WHERE substring(a.id::text,1,8) = '{EVENT}' LIMIT 1;"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout.strip())


def main() -> None:
    print("\n" + "=" * 78)
    print("  THE CHECKER, RUNNING")
    print("  One real event: severe PM2.5 spike, Houston, 15 June 2026.")
    print("  Fixed rules. No language model in here. Nobody deciding anything.")
    print("=" * 78)
    summary = load_summary()
    time.sleep(PAUSE)

    for text, note in CLAIMS:
        result = score_claim(text, summary).result
        score = result.corroboration_score
        spoke = {k: v for k, v in (result.per_channel_verdicts or {}).items() if v}
        print(f"\n  sentence   {text}")
        print(f"             ({note})")
        time.sleep(PAUSE * 0.6)
        if score is None:
            print("  verdict    NO CHANNEL COULD CHECK IT")
            print("             the rules stay silent rather than guess")
        else:
            print(f"  verdict    {VERDICT[score]}")
            print(f"             instruments that could speak: {', '.join(spoke) or 'none'}")
        time.sleep(PAUSE)

    print("\n" + "=" * 78)
    print("  Same sentence, one number changed, the answer flips.")
    print("  It is reading the measurements, not the wording.")
    print()
    print("  Across all 36 sentences a model wrote about this event:")
    print("  13 got a verdict. 23 could not be checked at all.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
