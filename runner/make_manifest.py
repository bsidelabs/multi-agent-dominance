"""
Builds the preregistered run manifest: every confirmatory run in a seeded, shuffled order so that
conditions and scenarios are interleaved over time (provider-side drift cannot line up with a condition).

Usage:
    python runner/make_manifest.py --model <together-model-id> [--out results/confirmatory/MANIFEST.csv]
"""

import argparse
import csv
import random
from pathlib import Path

SCENARIOS = ["scenario_06", "scenario_07", "scenario_08", "scenario_09",
             "scenario_10", "scenario_11", "scenario_12", "scenario_14"]
CONDITIONS = ["experimental", "baseline", "seed_baseline"]
RUNS_PER_CELL = 5
MANIFEST_SEED = 20261008


def build_manifest(scenarios=SCENARIOS, conditions=CONDITIONS, runs=RUNS_PER_CELL, seed=MANIFEST_SEED):
    rows = [(s, c, r) for s in scenarios for c in conditions for r in range(1, runs + 1)]
    random.Random(seed).shuffle(rows)
    return [{"order": i + 1, "scenario": s, "condition": c, "run": r} for i, (s, c, r) in enumerate(rows)]


def command(row: dict, model: str, provider: str = "together") -> str:
    return (f"python run_experiment.py --scenario ../scenarios/{row['scenario']}.json "
            f"--condition {row['condition']} --run {row['run']} --rounds 20 "
            f"--model {model} --provider {provider} --output-dir ../results/confirmatory --skip-scoring")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Confirmatory model ID (recorded in the command column)")
    ap.add_argument("--provider", default="together")
    ap.add_argument("--out", type=Path, default=Path("results/confirmatory/MANIFEST.csv"))
    args = ap.parse_args()

    rows = build_manifest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "scenario", "condition", "run", "command"])
        for r in rows:
            w.writerow([r["order"], r["scenario"], r["condition"], r["run"], command(r, args.model, args.provider)])
    print(f"Wrote {len(rows)} runs to {args.out}")


if __name__ == "__main__":
    main()
