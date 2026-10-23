#!/usr/bin/env bash
# Cross-check runs: scenario_06 + scenario_10 × experimental + baseline × 3 runs
# Model: gpt-4.1-mini   Output: results/mini/

set -euo pipefail

RESULTS_DIR="$(cd "$(dirname "$0")" && pwd)/results"
TOTAL=12
DONE=0

run() {
  local scenario=$1 condition=$2 run_num=$3
  DONE=$((DONE + 1))
  echo ""
  echo "=========================================="
  echo "  [$DONE/$TOTAL] $scenario  $condition  run $run_num"
  echo "=========================================="
  docker run --rm \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -v "$RESULTS_DIR:/app/results" \
    mean-girls \
    python run_experiment.py \
      --scenario "../scenarios/${scenario}.json" \
      --condition "$condition" \
      --run "$run_num" \
      --model gpt-4.1-mini \
      --provider openai \
      --output-dir ../results/mini \
      --skip-scoring
}

# scenario_06
run scenario_06 experimental 1
run scenario_06 experimental 2
run scenario_06 experimental 3
run scenario_06 baseline 1
run scenario_06 baseline 2
run scenario_06 baseline 3

# scenario_10
run scenario_10 experimental 1
run scenario_10 experimental 2
run scenario_10 experimental 3
run scenario_10 baseline 1
run scenario_10 baseline 2
run scenario_10 baseline 3

echo ""
echo "=========================================="
echo "  All $TOTAL cross-check runs complete."
echo "  Results in: results/mini/"
echo "=========================================="
