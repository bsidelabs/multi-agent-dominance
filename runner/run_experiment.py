import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI, APIError, APITimeoutError, RateLimitError
from utils import (
    BASELINE_ORDER,
    EXPERIMENTAL_ORDER,
    PERSONAS_DIR,
    build_user_message,
    load_persona,
    load_scenario,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Preregistered generation settings. top_p / top_k are intentionally unset, and there is no fixed seed.
TEMPERATURE = 0.7
MAX_TOKENS = 2048
MAX_RETRIES = 3

PROVIDERS = {
    "together": {"base_url": "https://api.together.xyz/v1", "key_env": "TOGETHER_API_KEY"},
    "openai": {"base_url": None, "key_env": "OPENAI_API_KEY"},
}
CONDITIONS = ["experimental", "baseline", "seed_baseline"]

_client: OpenAI | None = None
_provider = "together"


def configure_client(provider: str) -> None:
    global _client, _provider
    cfg = PROVIDERS[provider]
    api_key = os.environ.get(cfg["key_env"])
    if not api_key:
        raise SystemExit(f"{cfg['key_env']} is not set (provider: {provider})")
    _client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
    _provider = provider

# Per-model pricing ($/M tokens) and inter-call sleep (seconds)
MODEL_CONFIG: dict[str, dict] = {
    "gpt-5.6-luna":   {"input": 2.50, "output": 10.00, "sleep": 65},
    "gpt-4.1-mini":   {"input": 0.40, "output":  1.60, "sleep":  2},
    "gpt-4.1":        {"input": 2.00, "output":  8.00, "sleep":  2},
    "gpt-4o-mini":    {"input": 0.15, "output":  0.60, "sleep":  2},
}
_DEFAULT_MODEL_CONFIG = {"input": 2.50, "output": 10.00, "sleep": 2}


def call_with_retry(messages: list[dict], model: str) -> tuple[str, float, dict]:
    """One sequential chat call. Up to MAX_RETRIES retries with the same prompt, then raises."""
    if _client is None:
        raise RuntimeError("configure_client() must be called first")
    cfg = MODEL_CONFIG.get(model, _DEFAULT_MODEL_CONFIG)
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = _client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            usage = response.usage
            cost = (usage.prompt_tokens / 1_000_000 * cfg["input"]) + (usage.completion_tokens / 1_000_000 * cfg["output"])
            print(f"    tokens: {usage.prompt_tokens}in / {usage.completion_tokens}out — ${cost:.4f}")
            time.sleep(cfg["sleep"])
            choice = response.choices[0]
            meta = {
                "id": getattr(response, "id", None),
                "model": getattr(response, "model", None),
                "system_fingerprint": getattr(response, "system_fingerprint", None),
                "finish_reason": getattr(choice, "finish_reason", None),
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "created": getattr(response, "created", None),
                "retries": attempt,
            }
            return (choice.message.content or "").strip(), cost, meta
        except (RateLimitError, APITimeoutError, APIError) as e:
            if attempt == MAX_RETRIES:
                raise
            wait = 60 * (attempt + 1) if isinstance(e, RateLimitError) else 5
            print(f"  {type(e).__name__}: {e}. Retry {attempt + 1}/{MAX_RETRIES} in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("unreachable")  # pragma: no cover


def initial_history(condition: str, seed_position: str | None) -> list[dict]:
    """Round-0 shared history. Experimental: Regina speaks the seed. Seed-baseline: the same
    text is injected as a system-attributed message that no agent speaks. Baseline: empty."""
    if not seed_position:
        return []
    if condition == "experimental":
        return [{"agent": "regina", "round": 0, "content": seed_position}]
    if condition == "seed_baseline":
        return [{"agent": "system", "round": 0, "content": seed_position}]
    return []


def run_experiment(
    scenario_path: Path,
    condition: str,
    run_number: int,
    n_rounds: int,
    personas_dir: Path,
    output_dir: Path,
    model: str,
    skip_scoring: bool = False,
    start_round: int = 1,
    min_round_seconds: float = 0,
) -> dict:
    total_cost = 0.0
    run_start = time.time()
    scenario = load_scenario(scenario_path)
    agent_names = EXPERIMENTAL_ORDER if condition == "experimental" else BASELINE_ORDER
    personas = {name: load_persona(name, personas_dir) for name in agent_names}

    seed_position = scenario.get("seed_position")
    run_id = f"{scenario['id']}_{condition}_run{run_number:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{run_id}.json"

    # Resume from checkpoint if start_round > 1
    if start_round > 1 and out_path.exists():
        transcript = json.loads(out_path.read_text())
        history = initial_history(condition, seed_position)
        for r in transcript["rounds"]:
            for agent_name, content in r["responses"].items():
                history.append({"agent": agent_name, "round": r["round"], "content": content})
        print(f"  [resume] Loaded checkpoint with {len(transcript['rounds'])} rounds, resuming from round {start_round}")
    else:
        history = initial_history(condition, seed_position)
        if history:
            print(f"  [seed] {history[0]['agent']}: {seed_position[:120]}...")
        transcript = {
            "run_id": run_id,
            "scenario_id": scenario["id"],
            "condition": condition,
            "run": run_number,
            "n_rounds": n_rounds,
            "seed_position": seed_position if condition in ("experimental", "seed_baseline") else None,
            "settings": {
                "model": model,
                "provider": _provider,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "top_p": None,
                "top_k": None,
                "seed": None,
            },
            "rounds": [],
        }

    print(f"\n{'='*60}")
    print(f"  Scenario : {scenario['id']} — {scenario['title']}")
    print(f"  Condition: {condition}  |  Run: {run_number}  |  Rounds: {n_rounds}")
    print(f"  Agents   : {', '.join(agent_names)}")
    if start_round > 1:
        print(f"  Resuming from round {start_round}")
    print(f"{'='*60}")

    for round_num in range(start_round, n_rounds + 1):
        round_start = time.time()
        elapsed_total = round_start - run_start
        if round_num > 1:
            avg_round = elapsed_total / (round_num - 1)
            eta = avg_round * (n_rounds - round_num + 1)
            print(f"\n--- Round {round_num}/{n_rounds}  (elapsed {elapsed_total/60:.1f}m, ETA ~{eta/60:.1f}m, cost so far ${total_cost:.4f}) ---")
        else:
            print(f"\n--- Round {round_num}/{n_rounds} ---")

        round_history = history.copy()
        round_responses = []
        round_meta: dict[str, dict] = {}

        for agent_name in agent_names:
            agent_start = time.time()
            user_msg = build_user_message(scenario["prompt"], round_history, agent_name)

            content, cost, meta = call_with_retry(
                messages=[
                    {"role": "system", "content": personas[agent_name]},
                    {"role": "user", "content": user_msg},
                ],
                model=model,
            )
            total_cost += cost
            agent_elapsed = time.time() - agent_start
            turn = {"agent": agent_name, "round": round_num, "content": content}

            round_history.append(turn)
            round_responses.append(turn)
            round_meta[agent_name] = meta
            fp = meta.get("system_fingerprint")
            if fp:
                seen = transcript.setdefault("fingerprints", [])
                if fp not in seen:
                    seen.append(fp)
                    if len(seen) > 1:
                        print(f"  [WARNING] system_fingerprint changed mid-run: {seen}")
            print(f"  [{agent_elapsed:.1f}s] {agent_name}: {content[:160]}...")

        history.extend(round_responses)
        transcript["rounds"].append(
            {
                "round": round_num,
                "responses": {t["agent"]: t["content"] for t in round_responses},
                "meta": round_meta,
            }
        )
        round_elapsed = time.time() - round_start
        print(f"  -- round done in {round_elapsed:.1f}s, cumulative cost ${total_cost:.4f}")
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{transcript['run_id']}.json"
        out_path.write_text(json.dumps(transcript, indent=2))
        print(f"  [checkpoint] saved after round {round_num}")
        if round_num < n_rounds and min_round_seconds > round_elapsed:
            pause = min_round_seconds - round_elapsed
            print(f"  [pacing] sleeping {pause:.0f}s...")
            time.sleep(pause)

    total_elapsed = time.time() - run_start
    out_path.write_text(json.dumps(transcript, indent=2))
    print(f"\n{'='*60}")
    print(f"  Saved  : {out_path}")
    print(f"  Cost   : ${total_cost:.4f}")
    print(f"  Time   : {total_elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    if not skip_scoring:
        try:
            from score import score_transcript
            print("Scoring...")
            score_transcript(out_path, scenario_path)
            print("Scores saved.")
        except ImportError:
            print("(sentence-transformers not available — run score.py locally to score)")

    return transcript


def main():
    parser = argparse.ArgumentParser(description="Run a Mean Girls multi-agent experiment")
    parser.add_argument("--scenario", required=True, help="Path to scenario JSON")
    parser.add_argument("--condition", choices=CONDITIONS, default="experimental")
    parser.add_argument("--run", type=int, default=1, help="Run number (for labeling)")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--personas-dir", type=Path, default=PERSONAS_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--model", required=True, help="Model ID for all agents (no default, to prevent stray runs)")
    parser.add_argument("--provider", choices=list(PROVIDERS), default="together", help="API provider (default: together)")
    parser.add_argument("--min-round-seconds", type=float, default=0, help="Minimum seconds per round, for TPM pacing (default 0)")
    parser.add_argument("--skip-scoring", action="store_true", help="Skip auto-scoring (score locally after run)")
    parser.add_argument("--start-round", type=int, default=1, help="Resume from this round (requires checkpoint file)")
    args = parser.parse_args()
    configure_client(args.provider)

    try:
        run_experiment(
            scenario_path=Path(args.scenario),
            condition=args.condition,
            run_number=args.run,
            n_rounds=args.rounds,
            personas_dir=args.personas_dir,
            output_dir=args.output_dir,
            model=args.model,
            skip_scoring=args.skip_scoring,
            start_round=args.start_round,
            min_round_seconds=args.min_round_seconds,
        )
    except (RateLimitError, APITimeoutError, APIError) as e:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        entry = {"scenario": Path(args.scenario).stem, "condition": args.condition, "run": args.run,
                 "model": args.model, "error": f"{type(e).__name__}: {e}", "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(args.output_dir / "exclusions.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        raise


if __name__ == "__main__":
    main()
