#!/usr/bin/env bash
#SBATCH --job-name=keing1_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100000M
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Partition / time / output are given at submit time, e.g.:
#   sbatch -p low  --time=01:00:00 -o logs/pilot_pre_%j.out --export=ALL,MODEL_PATH=...,SERVED_NAME=...,RUN_SPECS=... serve_and_run.sbatch
#
# Self-contained job: start a vLLM OpenAI-compatible server on the allocated GPU, wait for health,
# run one or more run_eval.py invocations against it, then stop the server.
#
# Required env (via --export=ALL,...):
#   MODEL_PATH   local snapshot dir (or HF id if online); optional MODEL_REVISION
#   SERVED_NAME  served model name (used as --model-name by run_eval)
#   RUN_SPECS    one run_eval.py argument string per line (newline-separated); each is appended with
#                --base-url/--model-name automatically. Example line:
#                  --eval multiple_choice --dataset revealing_score --n 16 --seed 1234 --out-dir /path/pre/mc_rs
# Optional env:
#   VENV (default /data/home/jxcai/sigil-a/envs/vllm)  MAX_MODEL_LEN (32768)  GPU_UTIL (0.90)
#   MAX_NUM_SEQS (128)  VLLM_SEED (1234)  PORT (random 20000-29999)  EXTRA_VLLM_ARGS ("")
#   SERVER_INFO_FILE  (path to write host/port json once healthy)
set -euo pipefail
source ~/.sigil_env 2>/dev/null || true
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

HARNESS="$(cd "$(dirname "$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}' | awk '{print $1}')")" 2>/dev/null && pwd || echo /data/home/jxcai/sigil-a/evals/keing1_harness)"
[[ -f "$HARNESS/run_eval.py" ]] || HARNESS=/data/home/jxcai/sigil-a/evals/keing1_harness
VENV="${VENV:-/data/home/jxcai/sigil-a/envs/vllm}"
export PATH="$VENV/bin:$PATH"            # FlashInfer/torch JIT shell out to `ninja` by name; it lives in the venv
# Default OFF: the FlashInfer top-p/top-k sampler JIT-compiles a CUDA kernel at first use (needs ninja+nvcc); the
# torch-native sampler is semantically identical. Set VLLM_USE_FLASHINFER_SAMPLER=1 to re-enable.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
PORT="${PORT:-$((20000 + RANDOM % 10000))}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
VLLM_SEED="${VLLM_SEED:-1234}"
HOST_IP="$(hostname -I | awk '{print $1}')"
BASE_URL="http://${HOST_IP}:${PORT}/v1"
: "${MODEL_PATH:?MODEL_PATH required}"; : "${SERVED_NAME:?SERVED_NAME required}"; : "${RUN_SPECS:?RUN_SPECS required}"

echo "[job] $(date) host=$(hostname) ip=$HOST_IP gpu=$CUDA_VISIBLE_DEVICES job=$SLURM_JOB_ID"
echo "[job] MODEL_PATH=$MODEL_PATH SERVED_NAME=$SERVED_NAME PORT=$PORT"
echo "[job] harness=$HARNESS venv=$VENV"
"$VENV/bin/python" -c "import vllm; print('[job] vllm', vllm.__version__)"

SERVER_LOG="${SERVER_LOG:-${SLURM_SUBMIT_DIR}/logs/vllm_server_${SLURM_JOB_ID}.log}"
SERVER_LOG="${SERVER_LOG//%j/$SLURM_JOB_ID}"   # allow %j in --export values (sbatch does not expand it there)
mkdir -p "$(dirname "$SERVER_LOG")"
REV_ARGS=(); [[ -n "${MODEL_REVISION:-}" ]] && REV_ARGS=(--revision "$MODEL_REVISION")
# NOTE: no --reasoning-parser on purpose: we capture the raw generation (think block included),
# exactly as the authors' serve_olmo.py + model_call.py 'vllm/' route did; run_eval.py splits it.
"$VENV/bin/vllm" serve "$MODEL_PATH" "${REV_ARGS[@]}" \
    --served-model-name "$SERVED_NAME" --host 0.0.0.0 --port "$PORT" \
    --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" --seed "$VLLM_SEED" --trust-remote-code ${EXTRA_VLLM_ARGS:-} \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'echo "[job] stopping server pid=$SERVER_PID"; kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true' EXIT

echo "[job] waiting for server at $BASE_URL (log: $SERVER_LOG)"
for i in $(seq 1 240); do
  if curl -s -m 5 "$BASE_URL/models" | grep -q "$SERVED_NAME"; then echo "[job] server healthy after ~$((i*5))s"; break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then echo "[job] SERVER DIED, tail of log:"; tail -50 "$SERVER_LOG"; exit 1; fi
  sleep 5
  [[ $i -eq 240 ]] && { echo "[job] server not healthy after 20 min"; tail -50 "$SERVER_LOG"; exit 1; }
done
if [[ -n "${SERVER_INFO_FILE:-}" ]]; then
  echo "{\"host\": \"$HOST_IP\", \"port\": $PORT, \"base_url\": \"$BASE_URL\", \"served_name\": \"$SERVED_NAME\", \"job_id\": \"$SLURM_JOB_ID\", \"model_path\": \"$MODEL_PATH\"}" > "$SERVER_INFO_FILE"
fi

export VLLM_BASE_URL="$BASE_URL"
STATUS=0
while IFS= read -r spec; do
  [[ -z "${spec// /}" ]] && continue
  # A spec may carry its own --model-name (e.g. LoRA adapter served under a second name); otherwise use SERVED_NAME.
  MN_ARGS=(--model-name "$SERVED_NAME"); [[ "$spec" == *"--model-name"* ]] && MN_ARGS=()
  echo "[job] $(date) RUN: run_eval.py $spec --base-url $BASE_URL ${MN_ARGS[*]}"
  # shellcheck disable=SC2086
  "$VENV/bin/python" "$HARNESS/run_eval.py" $spec --base-url "$BASE_URL" "${MN_ARGS[@]}" || { echo "[job] run_eval FAILED for: $spec"; STATUS=1; }
done <<< "$RUN_SPECS"
echo "[job] $(date) all runs finished status=$STATUS"
exit $STATUS
