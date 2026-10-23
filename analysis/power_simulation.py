"""
Power / Type I simulation for the preregistered primary contrast (run-level WAER, pooled one-tailed t-test).

Generative model (assumptions, all stated in the preregistration):
  - 8 scenarios x 5 runs per arm x 20 rounds; each round has a measurement-agent label.
  - Scenario baseline p_s = clip(base + Normal(0, SIGMA_S)), shared by both arms; treated-arm probability
    = clip(p_s + delta_s) with delta_s ~ Normal(delta, tau). Under the null (delta=0) tau is 0, so the
    null is a true zero effect and clipping at 0 cannot create a positive mean effect.
  - Run-level overdispersion: run probability ~ Beta(mean = p, concentration = kappa).
  - Each round is "clear" with probability 1 - unclear; WAER = wrong / clear (run dropped if no clear rounds).
  - Support rule: one-tailed pooled t-test p < alpha AND mean gap >= 5pp.

Usage:
    python analysis/power_simulation.py [--sims 4000] [--out analysis/power_simulation.md]
"""

import argparse
from pathlib import Path

import numpy as np
from scipy import stats

SEED = 20261008
SIGMA_S = 0.08
N_SCENARIOS, RUNS, ROUNDS = 8, 5, 20


def simulate_once(rng, base, delta, tau, kappa, unclear):
    exp, ctl = [], []
    for _ in range(N_SCENARIOS):
        p_ctl = float(np.clip(base + rng.normal(0, SIGMA_S), 0.0, 1.0))
        d_s = rng.normal(delta, tau) if tau > 0 else delta
        p_exp = float(np.clip(p_ctl + d_s, 0.0, 1.0))
        for p, store in ((p_exp, exp), (p_ctl, ctl)):
            for _ in range(RUNS):
                if p <= 0.0:
                    p_run = 0.0
                elif p >= 1.0:
                    p_run = 1.0
                else:
                    p_run = rng.beta(p * kappa, (1 - p) * kappa)
                clear = rng.binomial(ROUNDS, 1 - unclear)
                if clear == 0:
                    continue
                store.append(rng.binomial(clear, p_run) / clear)
    return exp, ctl


def supported(exp, ctl, alpha):
    if len(exp) < 2 or len(ctl) < 2:
        return False
    res = stats.ttest_ind(exp, ctl, equal_var=True, alternative="greater")
    gap = np.mean(exp) - np.mean(ctl)
    p = res.pvalue
    return bool(np.isfinite(p) and p < alpha and gap >= 0.05)


def rate(sims, base, delta, tau, kappa, unclear, alpha, seed):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(sims):
        exp, ctl = simulate_once(rng, base, delta, tau, kappa, unclear)
        hits += supported(exp, ctl, alpha)
    return hits / sims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "power_simulation.md")
    args = ap.parse_args()

    deltas = [0.0, 0.05, 0.10, 0.15, 0.25]
    bases = [0.0, 0.05, 0.15, 0.30]
    taus = [0.0, 0.10]
    kappa, unclear = 5.0, 0.05

    lines = [
        "# Power simulation: run-level WAER, pooled one-tailed t-test",
        "",
        f"Sims per cell: {args.sims}. Seed: {SEED}. {N_SCENARIOS} scenarios x {RUNS} runs per arm x {ROUNDS} rounds. "
        f"Run-level overdispersion kappa={kappa}; unclear rate {unclear:.0%}.",
        "Support rule: one-tailed p < alpha AND gap >= 5pp. alpha=0.05 is the single-test level; "
        "alpha=0.025 is the first Holm step when two confirmatory tests are run.",
        "delta=0 rows are the Type I error rate under the null.",
        "",
        "| baseline WAER | between-scenario SD of effect (tau) | true effect delta | P(support), alpha=0.05 | P(support), alpha=0.025 |",
        "|---|---|---|---|---|",
    ]
    i = 0
    for base in bases:
        for tau in taus:
            for delta in deltas:
                if delta == 0.0 and tau > 0:
                    continue
                i += 1
                p05 = rate(args.sims, base, delta, tau, kappa, unclear, 0.05, SEED + i)
                p025 = rate(args.sims, base, delta, tau, kappa, unclear, 0.025, SEED + 1000 + i)
                lines.append(f"| {base:.0%} | {tau:.2f} | {delta:.0%} | {p05:.3f} | {p025:.3f} |")
                print(lines[-1], flush=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
