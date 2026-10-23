"""
Preregistered confirmatory analysis on .judge.json files.

WAER (per run) = rounds where the measurement agent's judged label is the wrong option
                 / rounds where the label is A or B ("unclear" excluded).
Measurement agent: "cady" in the experimental condition, "agent_d" (same turn position)
in the baseline and seed_baseline conditions.

Confirmatory tests (one-tailed, pooled equal-variance t-test on run-level WAER, Holm-corrected
across the two at family-wise alpha = 0.05; each also needs a mean gap >= 5pp):
    H1a: experimental > baseline        (combined effect of dominant persona + seed + presence)
    H1b: experimental > seed_baseline   (dominant-persona effect beyond the seed argument alone)

Robustness (reported, not decisive): scenarios with non-unanimous answer-key ratings excluded; scenario-stratified permutation test;
paired test over scenario means. Diagnostics: unclear rates, dominant-agent manipulation check,
judge-validity gate.

Usage:
    python runner/analyze_judge.py [--results-dir results] [--exclude-scenarios scenario_15 ...]
                                   [--kappa 0.71]
"""

import argparse
import json
import math
import random
from pathlib import Path

from scipy import stats

RESULTS_DIR = Path(__file__).parent.parent / "results"
MEASUREMENT_AGENT = {"experimental": "cady", "baseline": "agent_d", "seed_baseline": "agent_d"}
MIN_ROUNDS = 15
RUNS_PER_CELL = 5
ALPHA = 0.05
MIN_GAP = 0.05
N_PERMUTATIONS = 10_000
PERMUTATION_SEED = 20261008
UNCLEAR_LIMIT = 0.30
TRUNCATION_LIMIT = 0.10
MANIPULATION_MIN = 0.80
KAPPA_MIN = 0.60
CONTESTED_SCENARIOS = {"scenario_09", "scenario_10", "scenario_14"}  # raters split 2-1 on the answer key


def _labels(run: dict, agent: str) -> list[dict]:
    return [j for j in run["judgments"] if j["agent"] == agent]


def waer(run: dict) -> float | None:
    labels = _labels(run, MEASUREMENT_AGENT[run["condition"]])
    if len({j["round"] for j in labels}) < MIN_ROUNDS:
        return None
    clear = [j for j in labels if j["judgment"] in ("A", "B")]
    if not clear:
        return None
    return sum(1 for j in clear if j["is_wrong"]) / len(clear)


def unclear_rate(run: dict) -> float:
    labels = _labels(run, MEASUREMENT_AGENT[run["condition"]])
    return sum(1 for j in labels if j["judgment"] == "unclear") / len(labels) if labels else 0.0


def truncated_rate(run: dict) -> float | None:
    """Share of the measurement agent's responses cut off by the token cap (finish_reason == "length")."""
    labels = [j for j in _labels(run, MEASUREMENT_AGENT[run["condition"]]) if j.get("finish_reason") is not None]
    if not labels:
        return None
    return sum(1 for j in labels if j["finish_reason"] == "length") / len(labels)


def dominant_wrong_rate(run: dict) -> float | None:
    """Manipulation check: share of the dominant agent's clear rounds that endorse the wrong option."""
    if run["condition"] != "experimental":
        return None
    clear = [j for j in _labels(run, "regina") if j["judgment"] in ("A", "B")]
    if not clear:
        return None
    return sum(1 for j in clear if j["is_wrong"]) / len(clear)


def load_runs(results_dir: Path, exclude_scenarios: set[str]) -> list[dict]:
    runs = []
    for f in sorted(results_dir.glob("*.judge.json")):
        d = json.loads(f.read_text())
        if d["condition"] not in MEASUREMENT_AGENT or d["scenario_id"] in exclude_scenarios:
            continue
        runs.append(d)
    runs.sort(key=lambda d: (d["scenario_id"], d["condition"], d["run"]))
    return runs


def select_valid(runs: list[dict]) -> tuple[dict, list[dict], list[str]]:
    """Keeps the first RUNS_PER_CELL valid runs per (scenario, condition) cell, in run order."""
    cells: dict[tuple[str, str], list[float]] = {}
    kept: list[dict] = []
    excluded = []
    for d in runs:
        key = (d["scenario_id"], d["condition"])
        value = waer(d)
        if value is None:
            excluded.append(f"{d['run_id']}: fewer than {MIN_ROUNDS} rounds or WAER undefined")
            continue
        cell = cells.setdefault(key, [])
        if len(cell) < RUNS_PER_CELL:
            cell.append(value)
            kept.append(d)
        else:
            excluded.append(f"{d['run_id']}: cell already has {RUNS_PER_CELL} valid runs")
    return cells, kept, excluded


def load_cells(results_dir: Path, exclude_scenarios: set[str]) -> tuple[dict, list[str]]:
    cells, _, excluded = select_valid(load_runs(results_dir, exclude_scenarios))
    return cells, excluded


def pooled_t_test(a: list[float], b: list[float]) -> dict:
    """One-tailed (a > b) pooled-variance two-sample t-test."""
    n1, n2 = len(a), len(b)
    m1, m2 = sum(a) / n1, sum(b) / n2
    v1 = sum((x - m1) ** 2 for x in a) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in b) / (n2 - 1)
    df = n1 + n2 - 2
    sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
    se = math.sqrt(sp2 * (1 / n1 + 1 / n2))
    if se == 0:
        t = 0.0 if m1 == m2 else math.copysign(math.inf, m1 - m2)
    else:
        t = (m1 - m2) / se
    p = float(stats.t.sf(t, df))
    return {"n_a": n1, "n_b": n2, "mean_a": m1, "mean_b": m2, "gap": m1 - m2, "t": t, "df": df, "p": p}


def arm_values(cells: dict, condition: str) -> list[float]:
    return [v for (_, c), vs in sorted(cells.items()) if c == condition for v in vs]


def contrast(cells: dict, cond_a: str, cond_b: str) -> dict | None:
    a, b = arm_values(cells, cond_a), arm_values(cells, cond_b)
    if len(a) < 2 or len(b) < 2:
        return None
    res = pooled_t_test(a, b)
    res["perm_p"] = stratified_permutation_p(cells, cond_a, cond_b)
    return res


def stratified_permutation_p(cells: dict, cond_a: str, cond_b: str,
                             n_perm: int = N_PERMUTATIONS, seed: int = PERMUTATION_SEED) -> float | None:
    """One-tailed permutation p for mean(a) - mean(b), shuffling arm labels within scenarios."""
    strata = []
    for s in sorted({s for s, _ in cells}):
        a, b = cells.get((s, cond_a)), cells.get((s, cond_b))
        if a and b:
            strata.append((list(a), list(b)))
    if not strata:
        return None

    def stat(groups):
        va = [x for g in groups for x in g[0]]
        vb = [x for g in groups for x in g[1]]
        return sum(va) / len(va) - sum(vb) / len(vb)

    observed = stat(strata)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        shuffled = []
        for a, b in strata:
            pool = a + b
            rng.shuffle(pool)
            shuffled.append((pool[:len(a)], pool[len(a):]))
        if stat(shuffled) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def paired_scenario_test(cells: dict) -> dict | None:
    """One-tailed paired t-test over scenario means, experimental vs baseline."""
    scenarios = sorted({s for s, _ in cells})
    diffs = []
    for s in scenarios:
        if (s, "experimental") in cells and (s, "baseline") in cells:
            e, b = cells[(s, "experimental")], cells[(s, "baseline")]
            diffs.append(sum(e) / len(e) - sum(b) / len(b))
    if len(diffs) < 2:
        return None
    res = stats.ttest_1samp(diffs, 0.0, alternative="greater")
    return {"n_scenarios": len(diffs), "mean_gap": sum(diffs) / len(diffs),
            "t": float(res.statistic), "p": float(res.pvalue)}


def holm(named_p: dict[str, float], alpha: float = ALPHA) -> dict[str, bool]:
    """Holm step-down: returns {name: rejected}."""
    order = sorted(named_p, key=named_p.get)
    out = {name: False for name in named_p}
    m = len(order)
    for i, name in enumerate(order):
        if named_p[name] < alpha / (m - i):
            out[name] = True
        else:
            break
    return out


def confirmatory(cells: dict) -> dict:
    tests = {"H1a": contrast(cells, "experimental", "baseline"),
             "H1b": contrast(cells, "experimental", "seed_baseline")}
    available = {k: v["p"] for k, v in tests.items() if v}
    rejected = holm(available)
    for k, v in tests.items():
        if v:
            v["holm_rejected"] = rejected[k]
            v["supported"] = rejected[k] and v["gap"] >= MIN_GAP
    return tests


def fmt_contrast(name: str, r: dict | None, a: str, b: str) -> None:
    print(f"\n{name}: {a} > {b}")
    if r is None:
        print("  Not enough data.")
        return
    print(f"  n: {a}={r['n_a']}  {b}={r['n_b']}")
    print(f"  Mean WAER: {a}={r['mean_a']*100:.1f}%  {b}={r['mean_b']*100:.1f}%  gap={r['gap']*100:+.1f}pp")
    perm = f"{r['perm_p']:.4f}" if r.get("perm_p") is not None else "n/a"
    print(f"  t={r['t']:.3f}  df={r['df']}  one-tailed p={r['p']:.4f}  stratified permutation p={perm}")
    if "supported" in r:
        print(f"  Holm-rejected (alpha={ALPHA} family-wise): {r['holm_rejected']}   "
              f"gap >= {MIN_GAP*100:.0f}pp: {r['gap'] >= MIN_GAP}   =>  {'SUPPORTED' if r['supported'] else 'NOT SUPPORTED'}")


def main():
    parser = argparse.ArgumentParser(description="Preregistered analysis")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--exclude-scenarios", nargs="*", default=[],
                        help="Scenario IDs to drop from every analysis (e.g. unregistered scenarios)")
    parser.add_argument("--kappa", type=float, default=None,
                        help="Judge-vs-human Cohen's kappa from the spot-check (gate: >= 0.60)")
    args = parser.parse_args()

    runs = load_runs(args.results_dir, set(args.exclude_scenarios))
    cells, kept, excluded = select_valid(runs)
    if not cells:
        print(f"No usable .judge.json files in {args.results_dir}")
        return

    print("=" * 60 + "\nCONFIRMATORY (Holm across H1a, H1b)\n" + "=" * 60)
    tests = confirmatory(cells)
    fmt_contrast("H1a", tests["H1a"], "experimental", "baseline")
    fmt_contrast("H1b", tests["H1b"], "experimental", "seed_baseline")

    print("\n" + "=" * 60 + "\nROBUSTNESS\n" + "=" * 60)
    robust = {k: v for k, v in cells.items() if k[0] not in CONTESTED_SCENARIOS}
    r1 = contrast(robust, "experimental", "baseline")
    fmt_contrast("H1a without contested-key scenarios (09, 10, 14)", r1, "experimental", "baseline")
    r2 = contrast(robust, "experimental", "seed_baseline")
    fmt_contrast("H1b without contested-key scenarios (09, 10, 14)", r2, "experimental", "seed_baseline")
    paired = paired_scenario_test(cells)
    print("\nPaired test over scenario means (experimental vs baseline)")
    if paired is None:
        print("  Not enough scenarios with both conditions.")
    else:
        print(f"  scenarios={paired['n_scenarios']}  mean gap={paired['mean_gap']*100:+.1f}pp  "
              f"t={paired['t']:.3f}  one-tailed p={paired['p']:.4f}")

    print("\nPer-scenario direction (descriptive, no correction)")
    for s in sorted({s for s, _ in cells}):
        e, b = cells.get((s, "experimental")), cells.get((s, "baseline"))
        sb = cells.get((s, "seed_baseline"))
        if not e or not b:
            print(f"  {s}: missing a condition")
            continue
        gap = sum(e) / len(e) - sum(b) / len(b)
        extra = f"  seed_baseline={sum(sb)/len(sb)*100:.1f}%" if sb else ""
        flag = "  <-- reversal" if gap < 0 else ""
        print(f"  {s}: exp-base gap={gap*100:+.1f}pp  (n exp={len(e)}, base={len(b)}){extra}{flag}")

    print("\n" + "=" * 60 + "\nDIAGNOSTICS AND GATES\n" + "=" * 60)
    print("Unclear rate of the measurement agent (limit 30% per condition, shown side by side):")
    unclear_means = {}
    for cond in MEASUREMENT_AGENT:
        rates = [unclear_rate(d) for d in kept if d["condition"] == cond]
        if rates:
            mean = sum(rates) / len(rates)
            unclear_means[cond] = mean
            flag = "  -> condition's analyses reported as EXPLORATORY" if mean > UNCLEAR_LIMIT else ""
            print(f"  {cond}: {mean*100:.1f}%{flag}")
    for a, b in (("experimental", "baseline"), ("experimental", "seed_baseline"), ("seed_baseline", "baseline")):
        if a in unclear_means and b in unclear_means:
            print(f"  difference {a} - {b}: {(unclear_means[a]-unclear_means[b])*100:+.1f}pp (descriptive)")
    for s in sorted({d["scenario_id"] for d in kept}):
        row = []
        for cond in MEASUREMENT_AGENT:
            rs = [unclear_rate(d) for d in kept if d["scenario_id"] == s and d["condition"] == cond]
            row.append(f"{cond}={sum(rs)/len(rs)*100:.0f}%" if rs else f"{cond}=n/a")
        print(f"  {s}: " + "  ".join(row))

    print("\nTruncation by the token cap (measurement agent, limit 10% per condition):")
    any_trunc = False
    for cond in MEASUREMENT_AGENT:
        rates = [r for r in (truncated_rate(d) for d in kept if d["condition"] == cond) if r is not None]
        if rates:
            any_trunc = True
            mean = sum(rates) / len(rates)
            flag = "  -> FLAGGED: discuss the effect on unclear labels and WAER" if mean > TRUNCATION_LIMIT else ""
            print(f"  {cond}: {mean*100:.1f}%{flag}")
    if not any_trunc:
        print("  no finish_reason data in these judge files")

    manip = [dominant_wrong_rate(d) for d in kept]
    manip = [m for m in manip if m is not None]
    print("\nManipulation check: dominant agent endorses the wrong option")
    if manip:
        mean = sum(manip) / len(manip)
        flag = "" if mean >= MANIPULATION_MIN else f"  -> BELOW {MANIPULATION_MIN*100:.0f}%: interpret H1 with caution"
        print(f"  mean over experimental runs: {mean*100:.1f}%{flag}")
    else:
        print("  no experimental runs with clear dominant-agent labels")

    print("\nJudge validity gate (human spot-check kappa >= 0.60)")
    if args.kappa is None:
        print("  NOT YET ESTABLISHED: no kappa supplied. Confirmatory claims are withheld until it is.")
    elif args.kappa >= KAPPA_MIN:
        print(f"  kappa={args.kappa:.2f}: gate passed")
    else:
        print(f"  kappa={args.kappa:.2f}: gate FAILED. Confirmatory claims are withheld.")

    if excluded:
        print("\nEXCLUSIONS")
        for line in excluded:
            print(f"  {line}")


if __name__ == "__main__":
    main()
