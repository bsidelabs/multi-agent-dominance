"""
Human spot-check of the judge (preregistered validity gate: Cohen's kappa judge-vs-human >= 0.60).

Step 1: python runner/spotcheck.py sample --results-dir results/confirmatory --scenarios-dir scenarios
        -> spotcheck_items.csv (blinded: id, scenario prompt, response) and spotcheck_key.json (id -> run, round, judge label)
Step 2: each rater fills the `label` column (A, B, or unclear) in their own copy of spotcheck_items.csv.
Step 3: python runner/spotcheck.py score --key spotcheck_key.json --labels rater1.csv rater2.csv [rater3.csv]
        -> kappa for each rater vs. the judge, and for each rater pair. The gate needs at least two raters and
           uses the lowest judge-vs-rater kappa.
"""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

SAMPLE_SIZE = 60
SAMPLE_SEED = 20261008
MEASUREMENT_AGENT = {"experimental": "cady", "baseline": "agent_d", "seed_baseline": "agent_d"}
LABELS = ["A", "B", "unclear"]


def cohens_kappa(x: list[str], y: list[str]) -> float:
    n = len(x)
    if n == 0 or n != len(y):
        raise ValueError("label lists must be non-empty and the same length")
    po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = Counter(x), Counter(y)
    pe = sum((cx[l] / n) * (cy[l] / n) for l in set(cx) | set(cy))
    if pe == 1:
        return 1.0 if po == 1 else 0.0
    return (po - pe) / (1 - pe)


def sample_items(results_dir: Path, scenarios_dir: Path, size=SAMPLE_SIZE, seed=SAMPLE_SEED):
    """Equal draws per condition over the measurement agent's responses, from the runs' judge files."""
    rng = random.Random(seed)
    pools: dict[str, list[dict]] = {c: [] for c in MEASUREMENT_AGENT}
    for jf in sorted(results_dir.glob("*.judge.json")):
        judge = json.loads(jf.read_text())
        cond = judge["condition"]
        if cond not in MEASUREMENT_AGENT:
            continue
        transcript = json.loads((results_dir / f"{judge['run_id']}.json").read_text())
        prompt = json.loads((scenarios_dir / f"{judge['scenario_id']}.json").read_text())["prompt"]
        agent = MEASUREMENT_AGENT[cond]
        by_round = {r["round"]: r["responses"].get(agent) for r in transcript["rounds"]}
        for j in judge["judgments"]:
            if j["agent"] == agent and by_round.get(j["round"]):
                pools[cond].append({"run_id": judge["run_id"], "round": j["round"], "agent": agent,
                                    "judge_label": j["judgment"], "prompt": prompt,
                                    "response": by_round[j["round"]]})
    per = size // len(pools)
    chosen = []
    for cond in sorted(pools):
        pool = pools[cond]
        chosen += rng.sample(pool, min(per, len(pool)))
    rng.shuffle(chosen)
    for i, item in enumerate(chosen, 1):
        item["id"] = i
    return chosen


def cmd_sample(args):
    items = sample_items(args.results_dir, args.scenarios_dir)
    with open(args.items, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "scenario_prompt", "response", "label"])
        for it in items:
            w.writerow([it["id"], it["prompt"], it["response"], ""])
    key = {str(it["id"]): {k: it[k] for k in ("run_id", "round", "agent", "judge_label")} for it in items}
    Path(args.key).write_text(json.dumps(key, indent=2))
    print(f"Wrote {len(items)} items to {args.items} (blinded) and the key to {args.key}")


def read_labels(path: str) -> dict[str, str]:
    with open(path, newline="") as f:
        out = {}
        for row in csv.DictReader(f):
            lab = row["label"].strip()
            lab = {"a": "A", "b": "B", "unclear": "unclear"}.get(lab.lower())
            if lab is None:
                raise ValueError(f"{path}: item {row['id']} has an invalid or empty label")
            out[row["id"]] = lab
    return out


def cmd_score(args):
    key = json.loads(Path(args.key).read_text())
    ids = sorted(key, key=int)
    judge = [key[i]["judge_label"] for i in ids]
    raters = [read_labels(p) for p in args.labels]
    kappas = []
    for n, r in enumerate(raters, 1):
        k = cohens_kappa([r[i] for i in ids], judge)
        kappas.append(k)
        print(f"rater {n} vs judge: kappa = {k:.3f}  (n={len(ids)})")
    for a in range(len(raters)):
        for b in range(a + 1, len(raters)):
            k = cohens_kappa([raters[a][i] for i in ids], [raters[b][i] for i in ids])
            print(f"rater {a + 1} vs rater {b + 1}: kappa = {k:.3f}")
    gate = min(kappas)
    if len(raters) < 2:
        print(f"gate kappa = {gate:.3f}, but the gate requires at least two raters -> FAIL")
        return
    print(f"gate kappa (lowest judge-vs-rater) = {gate:.3f} -> {'PASS' if gate >= 0.60 else 'FAIL'}")
    print(f"Run: python runner/analyze_judge.py --kappa {gate:.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--results-dir", type=Path, default=Path("results/confirmatory"))
    s.add_argument("--scenarios-dir", type=Path, default=Path("scenarios"))
    s.add_argument("--items", default="spotcheck_items.csv")
    s.add_argument("--key", default="spotcheck_key.json")
    s.set_defaults(fn=cmd_sample)
    c = sub.add_parser("score")
    c.add_argument("--key", default="spotcheck_key.json")
    c.add_argument("--labels", nargs="+", required=True, help="Two or three rater CSVs")
    c.set_defaults(fn=cmd_score)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
