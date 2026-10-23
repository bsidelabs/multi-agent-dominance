import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "runner"))

HAVE_OPENAI = importlib.util.find_spec("openai") is not None
if HAVE_OPENAI:
    import openai
    import run_experiment as rx
    import make_manifest

SCENARIO = ROOT / "scenarios" / "scenario_06.json"
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"


class _Usage:
    prompt_tokens = 10
    completion_tokens = 5


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)
        self.finish_reason = "stop"


class _Resp:
    def __init__(self, content, fingerprint="fp_1"):
        self.choices = [_Choice(content)]
        self.usage = _Usage()
        self.id = "resp-1"
        self.model = MODEL
        self.system_fingerprint = fingerprint
        self.created = 0


class FakeClient:
    """Records every call. `failures` is a list of exceptions raised before successes."""

    def __init__(self, failures=None, fingerprints=None):
        self.calls = []
        self.failures = list(failures or [])
        self.fingerprints = list(fingerprints or [])
        self.chat = mock.Mock()
        self.chat.completions.create = self._create

    def _create(self, **kw):
        self.calls.append(kw)
        if self.failures:
            raise self.failures.pop(0)
        fp = self.fingerprints.pop(0) if self.fingerprints else "fp_1"
        return _Resp(f"Option B (call {len(self.calls)})", fp)


def timeout_error():
    return openai.APITimeoutError(request=mock.Mock())


@unittest.skipUnless(HAVE_OPENAI, "openai not installed")
class RunnerTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(rx.time, "sleep", lambda *_: None)
        patcher.start()
        self.addCleanup(patcher.stop)
        rx.MODEL_CONFIG[MODEL] = {"input": 0, "output": 0, "sleep": 0}

    def run_condition(self, condition, rounds=2, client=None, **kw):
        client = client or FakeClient()
        rx._client = client
        rx._provider = "together"
        with tempfile.TemporaryDirectory() as d, mock.patch("builtins.print"):
            transcript = rx.run_experiment(SCENARIO, condition, 1, rounds, rx.PERSONAS_DIR, Path(d),
                                           model=MODEL, skip_scoring=True, **kw)
        return transcript, client

    # --- seed injection ---
    def test_initial_history_by_condition(self):
        seed = "wrong answer text"
        self.assertEqual(rx.initial_history("experimental", seed), [{"agent": "regina", "round": 0, "content": seed}])
        self.assertEqual(rx.initial_history("seed_baseline", seed), [{"agent": "system", "round": 0, "content": seed}])
        self.assertEqual(rx.initial_history("baseline", seed), [])
        self.assertEqual(rx.initial_history("experimental", None), [])
        self.assertEqual(rx.initial_history("seed_baseline", ""), [])

    def test_seed_reaches_the_first_agent_only_in_the_right_way(self):
        seed = json.loads(SCENARIO.read_text())["seed_position"]
        for cond, speaker in (("experimental", "Regina:"), ("seed_baseline", "System:")):
            _, client = self.run_condition(cond, rounds=1)
            first_user_msg = client.calls[0]["messages"][1]["content"]
            self.assertIn(f"{speaker} {seed}", first_user_msg, cond)
        _, client = self.run_condition("baseline", rounds=1)
        self.assertNotIn(seed, client.calls[0]["messages"][1]["content"])

    def test_agent_order_and_seed_metadata(self):
        for cond, order in (("experimental", ["regina", "gretchen", "karen", "cady"]),
                            ("baseline", ["agent_a", "agent_b", "agent_c", "agent_d"]),
                            ("seed_baseline", ["agent_a", "agent_b", "agent_c", "agent_d"])):
            t, _ = self.run_condition(cond, rounds=2)
            self.assertEqual(list(t["rounds"][0]["responses"]), order, cond)
            self.assertEqual(len(t["rounds"]), 2)
            self.assertEqual(t["seed_position"] is not None, cond != "baseline", cond)

    def test_seed_baseline_uses_neutral_personas(self):
        _, client = self.run_condition("seed_baseline", rounds=1)
        systems = [c["messages"][0]["content"] for c in client.calls]
        self.assertEqual(systems, [rx.load_persona(a) for a in rx.BASELINE_ORDER])

    # --- generation settings ---
    def test_generation_settings_are_registered_values(self):
        t, client = self.run_condition("experimental", rounds=1)
        for call in client.calls:
            self.assertEqual(call["temperature"], 0.7)
            self.assertEqual(call["max_tokens"], 2048)
            self.assertEqual(call["model"], MODEL)
            for forbidden in ("top_p", "top_k", "seed", "n"):
                self.assertNotIn(forbidden, call)
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(t["settings"], {"model": MODEL, "provider": "together", "temperature": 0.7,
                                         "max_tokens": 2048, "top_p": None, "top_k": None, "seed": None})

    def test_call_metadata_and_fingerprint_change_are_logged(self):
        client = FakeClient(fingerprints=["fp_a", "fp_a", "fp_b", "fp_b"])
        t, _ = self.run_condition("baseline", rounds=1, client=client)
        meta = t["rounds"][0]["meta"]["agent_a"]
        for key in ("id", "model", "system_fingerprint", "finish_reason", "prompt_tokens", "completion_tokens", "retries"):
            self.assertIn(key, meta)
        self.assertEqual(t["fingerprints"], ["fp_a", "fp_b"])

    # --- retry rule ---
    def test_three_retries_then_success(self):
        client = FakeClient(failures=[timeout_error()] * 3)
        rx._client = client
        content, _, meta = rx.call_with_retry([{"role": "user", "content": "hi"}], MODEL)
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(meta["retries"], 3)
        self.assertTrue(content)

    def test_fourth_failure_raises(self):
        client = FakeClient(failures=[timeout_error()] * 4)
        rx._client = client
        with self.assertRaises(openai.APITimeoutError):
            rx.call_with_retry([{"role": "user", "content": "hi"}], MODEL)
        self.assertEqual(len(client.calls), 4)

    # --- resume ---
    def test_resume_rebuilds_seed_baseline_history(self):
        seed = json.loads(SCENARIO.read_text())["seed_position"]
        rx._client = FakeClient()
        with tempfile.TemporaryDirectory() as d, mock.patch("builtins.print"):
            d = Path(d)
            rx.run_experiment(SCENARIO, "seed_baseline", 1, 1, rx.PERSONAS_DIR, d, model=MODEL, skip_scoring=True)
            resumed = FakeClient()
            rx._client = resumed
            rx.run_experiment(SCENARIO, "seed_baseline", 1, 2, rx.PERSONAS_DIR, d, model=MODEL,
                              skip_scoring=True, start_round=2)
            first = resumed.calls[0]["messages"][1]["content"]
            self.assertIn(f"System: {seed}", first)
            self.assertIn("Agent A: Option B (call 1)", first)
            self.assertEqual(len(resumed.calls), 4)

    # --- CLI / provider ---
    def test_model_has_no_default(self):
        argv = ["run_experiment.py", "--scenario", str(SCENARIO)]
        with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as cm, \
                mock.patch("sys.stderr"):
            rx.main()
        self.assertEqual(cm.exception.code, 2)

    def test_conditions_include_seed_baseline(self):
        self.assertEqual(rx.CONDITIONS, ["experimental", "baseline", "seed_baseline"])

    def test_together_client_configuration(self):
        with mock.patch.dict(os.environ, {"TOGETHER_API_KEY": "test-key"}), \
                mock.patch.object(rx, "OpenAI") as ctor:
            rx.configure_client("together")
        ctor.assert_called_once_with(api_key="test-key", base_url="https://api.together.xyz/v1")

    def test_missing_api_key_stops_the_run(self):
        env = {k: v for k, v in os.environ.items() if k != "TOGETHER_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True), self.assertRaises(SystemExit):
            rx.configure_client("together")

    def test_persistent_failure_is_logged_and_reraised(self):
        failing = FakeClient(failures=[timeout_error()] * 20)
        argv = ["run_experiment.py", "--scenario", str(SCENARIO), "--condition", "seed_baseline", "--run", "3",
                "--rounds", "1", "--model", MODEL, "--skip-scoring"]
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"TOGETHER_API_KEY": "k"}), \
                mock.patch.object(rx, "OpenAI", return_value=failing), \
                mock.patch.object(sys, "argv", argv + ["--output-dir", d]), mock.patch("builtins.print"):
            with self.assertRaises(openai.APITimeoutError):
                rx.main()
            entry = json.loads((Path(d) / "exclusions.jsonl").read_text().splitlines()[0])
        self.assertEqual((entry["condition"], entry["run"], entry["model"]), ("seed_baseline", 3, MODEL))


@unittest.skipUnless(HAVE_OPENAI, "openai not installed")
class ManifestCommandTests(unittest.TestCase):
    def test_commands_use_registered_settings(self):
        row = make_manifest.build_manifest()[0]
        cmd = make_manifest.command(row, MODEL)
        for piece in ("--rounds 20", f"--model {MODEL}", "--provider together",
                      "--output-dir ../results/confirmatory", f"--condition {row['condition']}"):
            self.assertIn(piece, cmd)

    def test_committed_manifest_matches_generator(self):
        path = ROOT / "results" / "confirmatory" / "MANIFEST.csv"
        if not path.exists():
            self.skipTest("MANIFEST.csv not generated")
        import csv
        rows = list(csv.DictReader(open(path, newline="")))
        expected = make_manifest.build_manifest()
        self.assertEqual([(r["scenario"], r["condition"], int(r["run"])) for r in rows],
                         [(e["scenario"], e["condition"], e["run"]) for e in expected])


if __name__ == "__main__":
    unittest.main()
