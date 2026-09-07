#!/usr/bin/env bash
# Thin wrapper: activate the shared vLLM venv (or $KEING1_VENV) and run run_eval.py with all args.
# Example:
#   run_eval.sh --eval multiple_choice --dataset revealing_score --base-url http://host:port/v1 \
#       --model-name Olmo-3-7B-Think --n 200 --seed 1234 --out-dir results/post/mc_revealing_score
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${KEING1_VENV:-/data/home/jxcai/sigil-a/envs/vllm}"
PY="$VENV/bin/python"
[[ -x "$PY" ]] || { echo "python not found at $PY (set KEING1_VENV)"; exit 1; }
exec "$PY" "$HERE/run_eval.py" "$@"
