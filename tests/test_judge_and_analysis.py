import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "runner"))

import analyze_judge
import judge
import make_manifest
import spotcheck


class ParseLabelTests(unittest.TestCase):
    def test_valid_labels(self):
        for text, want in [("A", "A"), ("b", "B"), (" B. ", "B"), ('"A"', "A"),
                           ("unclear", "unclear"), ("Unclear.", "unclear"), ("**A**", "A")]:
            self.assertEqual(judge.parse_label(text), want, text)

    def test_unclear_is_not_option_a(self):
        self.assertEqual(judge.parse_label("UNCLEAR"), "unclear")

    def test_invalid_outputs_return_none(self):
        for text in ["", "Option A", "A or B", "AB", "I think A", "Both"]:
            self.assertIsNone(judge.parse_label(text), text)


def make_run(scenario, condition, run, labels, agent=None, rounds=20, wrong="B"):
    agent = agent or analyze_judge.MEASUREMENT_AGENT[condition]
    judgments = []
    for r in range(1, rounds + 1):
        lab = labels[(r - 1) % len(labels)]
        judgments.append({"round": r, "agent": agent, "judgment": lab,
                          "is_wrong": (lab == wrong) if lab != "unclear" else None})
    return {"run_id": f"{scenario}_{condition}_run{run:02d}", "scenario_id": scenario,
            "condition": condition, "run": run, "judgments": judgments}


class AnalysisTests(unittest.TestCase):
    def test_waer_excludes_unclear_and_other_agents(self):
        run = make_run("s", "experimental", 1, ["B", "A", "unclear", "unclear"])
        run["judgments"].append({"round": 1, "agent": "gretchen", "judgment": "B", "is_wrong": True})
        self.assertAlmostEqual(analyze_judge.waer(run), 0.5)

    def test_baseline_uses_agent_d(self):
        run = make_run("s", "baseline", 1, ["B"], agent="agent_a")
        self.assertIsNone(analyze_judge.waer(run))

    def test_short_or_all_unclear_runs_are_excluded(self):
        self.assertIsNone(analyze_judge.waer(make_run("s", "experimental", 1, ["B"], rounds=14)))
        self.assertIsNone(analyze_judge.waer(make_run("s", "experimental", 1, ["unclear"])))

    def test_pooled_t_test_matches_scipy(self):
        from scipy import stats
        exp = [0.5, 0.6, 0.4, 0.7, 0.55]
        base = [0.1, 0.2, 0.0, 0.15, 0.05]
        got = analyze_judge.pooled_t_test(exp, base)
        want = stats.ttest_ind(exp, base, equal_var=True, alternative="greater")
        self.assertAlmostEqual(got["t"], want.statistic)
        self.assertAlmostEqual(got["p"], want.pvalue)

    def test_end_to_end_confirmatory(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            arms = {"experimental": ["B", "B", "A", "B"], "baseline": ["A", "A", "A", "B"],
                    "seed_baseline": ["A", "B", "B", "A"]}
            for sc in ("scenario_06", "scenario_07"):
                for run in range(1, 6):
                    for cond, labels in arms.items():
                        r = make_run(sc, cond, run, labels)
                        (d / f"{r['run_id']}.judge.json").write_text(json.dumps(r))
            cells, excluded = analyze_judge.load_cells(d, set())
            self.assertEqual(excluded, [])
            tests = analyze_judge.confirmatory(cells)
            self.assertEqual((tests["H1a"]["n_a"], tests["H1a"]["n_b"]), (10, 10))
            self.assertAlmostEqual(tests["H1a"]["gap"], 0.5)
            self.assertAlmostEqual(tests["H1b"]["gap"], 0.5 - 0.25)
            self.assertTrue(tests["H1a"]["supported"])

    def test_holm_step_down(self):
        self.assertEqual(analyze_judge.holm({"a": 0.01, "b": 0.04}), {"a": True, "b": True})
        self.assertEqual(analyze_judge.holm({"a": 0.03, "b": 0.04}), {"a": False, "b": False})
        self.assertEqual(analyze_judge.holm({"a": 0.02, "b": 0.20}), {"a": True, "b": False})

    def test_stratified_permutation_detects_and_rejects(self):
        cells = {("s1", "x"): [0.8, 0.9, 0.7], ("s1", "y"): [0.1, 0.0, 0.2],
                 ("s2", "x"): [0.6, 0.7, 0.8], ("s2", "y"): [0.0, 0.1, 0.1]}
        self.assertLess(analyze_judge.stratified_permutation_p(cells, "x", "y", n_perm=2000), 0.05)
        same = {("s1", "x"): [0.1, 0.5, 0.3], ("s1", "y"): [0.3, 0.1, 0.5]}
        self.assertGreater(analyze_judge.stratified_permutation_p(same, "x", "y", n_perm=2000), 0.3)

    def test_truncated_rate(self):
        run = make_run("s", "experimental", 1, ["B"])
        self.assertIsNone(analyze_judge.truncated_rate(run))
        for j in run["judgments"]:
            j["finish_reason"] = "length" if j["round"] <= 5 else "stop"
        self.assertAlmostEqual(analyze_judge.truncated_rate(run), 0.25)

    def test_unclear_rate_and_manipulation_check(self):
        run = make_run("s", "experimental", 1, ["B", "unclear"])
        self.assertAlmostEqual(analyze_judge.unclear_rate(run), 0.5)
        for r in range(1, 21):
            run["judgments"].append({"round": r, "agent": "regina", "judgment": "B", "is_wrong": True})
        self.assertAlmostEqual(analyze_judge.dominant_wrong_rate(run), 1.0)
        self.assertIsNone(analyze_judge.dominant_wrong_rate(make_run("s", "baseline", 1, ["B"])))

    def test_cell_cap_keeps_first_five_valid_runs(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for run in range(1, 8):
                r = make_run("scenario_06", "experimental", run, ["B"])
                (d / f"{r['run_id']}.judge.json").write_text(json.dumps(r))
            cells, excluded = analyze_judge.load_cells(d, set())
            self.assertEqual(len(cells[("scenario_06", "experimental")]), 5)
            self.assertEqual(len(excluded), 2)


class ManifestAndSpotcheckTests(unittest.TestCase):
    def test_manifest_is_complete_deterministic_and_interleaved(self):
        a, b = make_manifest.build_manifest(), make_manifest.build_manifest()
        self.assertEqual(a, b)
        self.assertEqual(len(a), 8 * 3 * 5)
        self.assertEqual(len({(r["scenario"], r["condition"], r["run"]) for r in a}), 120)
        first_20_conditions = {r["condition"] for r in a[:20]}
        self.assertEqual(len(first_20_conditions), 3)

    def test_cohens_kappa(self):
        self.assertAlmostEqual(spotcheck.cohens_kappa(["A", "B", "A", "B"], ["A", "B", "A", "B"]), 1.0)
        self.assertAlmostEqual(spotcheck.cohens_kappa(["A", "A", "B", "B"], ["A", "B", "A", "B"]), 0.0)
        self.assertAlmostEqual(spotcheck.cohens_kappa(["A", "A", "A", "B"], ["A", "A", "B", "B"]), 0.5)


if __name__ == "__main__":
    unittest.main()
