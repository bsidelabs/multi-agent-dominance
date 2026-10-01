"""
Reads all .judge.json files and computes experimental vs baseline
wrong-answer rates per scenario and pooled across scenarios.

Usage:
    python runner/analyze_judge.py
"""

import json
import math
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_judge_files():
    data = {}
    for f in sorted(RESULTS_DIR.glob("*.judge.json")):
        d = json.loads(f.read_text())
        sc = d["scenario_id"]
        cond = d["condition"]
        if sc not in data:
            data[sc] = {"experimental": [], "baseline": []}
        data[sc][cond].append(d)
    return data


def cady_wrong_rate(run: dict) -> float | None:
    rate = run["wrong_answer_rate"]
    if run["condition"] == "experimental":
        return rate.get("cady")
    else:
        vals = [v for v in rate.values() if v is not None]
        return sum(vals) / len(vals) if vals else None


def main():
    data = load_judge_files()
    if not data:
        print("No .judge.json files found in results/")
        return

    print(f"\n{'='*65}")
    print(f"{'LLM-AS-JUDGE ANALYSIS':^65}")
    print(f"{'='*65}")
    print(f"\n{'Scenario':<14} {'Exp wrong%':>12} {'Base wrong%':>12} {'Gap':>8} {'Runs':>6}")
    print("-" * 55)

    all_exp, all_base = [], []

    for sc, runs in sorted(data.items()):
        exp_rates = [r for run in runs["experimental"] if (r := cady_wrong_rate(run)) is not None]
        base_rates = [r for run in runs["baseline"] if (r := cady_wrong_rate(run)) is not None]
        if not exp_rates or not base_rates:
            continue

        exp_mean = sum(exp_rates) / len(exp_rates)
        base_mean = sum(base_rates) / len(base_rates)
        gap = exp_mean - base_mean
        n = min(len(exp_rates), len(base_rates))

        print(f"{sc:<14} {exp_mean*100:>11.1f}% {base_mean*100:>11.1f}% {gap*100:>+7.1f}pp  n={n}")
        all_exp.extend(exp_rates)
        all_base.extend(base_rates)

    if len(all_exp) < 2:
        print("\nNot enough data for pooled test.")
        return

    # pooled t-test
    n = len(all_exp)
    mean_e = sum(all_exp) / n
    mean_b = sum(all_base) / n
    var_e = sum((x - mean_e) ** 2 for x in all_exp) / (n - 1)
    var_b = sum((x - mean_b) ** 2 for x in all_base) / (n - 1)
    se = math.sqrt(var_e / n + var_b / n)
    t = (mean_e - mean_b) / se if se > 0 else 0
    df = 2 * n - 2

    print(f"\n{'='*55}")
    print(f"POOLED  (n={n} runs per group)")
    print(f"  Experimental wrong-answer rate: {mean_e*100:.1f}%  SD={math.sqrt(var_e)*100:.1f}pp")
    print(f"  Baseline wrong-answer rate:     {mean_b*100:.1f}%  SD={math.sqrt(var_b)*100:.1f}pp")
    print(f"  Gap: {(mean_e-mean_b)*100:+.1f}pp")
    print(f"  t={t:.3f}  df={df}  (p<0.05 threshold: t>2.12 at df=16)")

    # per-agent breakdown for experimental
    print(f"\n{'='*55}")
    print("EXPERIMENTAL — wrong-answer rate by agent (all scenarios)")
    agent_data: dict = {}
    for sc, runs in sorted(data.items()):
        for run in runs["experimental"]:
            for agent, rate in run["wrong_answer_rate"].items():
                if rate is not None:
                    if agent not in agent_data:
                        agent_data[agent] = []
                    agent_data[agent].append(rate)
    for agent in ["gretchen", "karen", "cady"]:
        if agent in agent_data:
            vals = agent_data[agent]
            print(f"  {agent:<12} {sum(vals)/len(vals)*100:.1f}%  (n={len(vals)} runs)")


if __name__ == "__main__":
    main()
