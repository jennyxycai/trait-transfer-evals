#!/usr/bin/env bash
#SBATCH --job-name=t5_keing1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100000M
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Team 5 copy of /data/home/jxcai/sigil-a/evals/keing1_harness/serve_and_run.sbatch (serving, health wait and the
# run_eval loop are unchanged) with ONE addition: SANITY_ARGS -> runs code/lora_sanity.py before the evals
# (greedy base-vs-LoRA smoke test + chat-template check); the job aborts if the LoRA arm is a no-op.
# LoRA arms come in via EXTRA_VLLM_ARGS (--enable-lora ... --lora-modules post=<dir>); RUN_SPECS lines carry --model-name.
# Required env: MODEL_PATH SERVED_NAME RUN_SPECS.  Optional: SANITY_ARGS MAX_MODEL_LEN GPU_UTIL MAX_NUM_SEQS VLLM_SEED
#                PORT EXTRA_VLLM_ARGS SERVER_INFO_FILE SERVER_LOG VENV
set -euo pipefail
source ~/.sigil_env 2>/dev/null || true
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
TEAM=/data/home/jxcai/sigil-a/evals/team5_qwen_ood
HARNESS=/data/home/jxcai/sigil-a/evals/keing1_harness
VENV="${VENV:-/data/home/jxcai/sigil-a/envs/vllm}"
export PATH="$VENV/bin:$PATH"                                   # ninja for any JIT (vLLM 0.28 crash otherwise)
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"   # torch-native sampler (same as teams 1-4)
PORT="${PORT:-$((20000 + RANDOM % 10000))}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
VLLM_SEED="${VLLM_SEED:-1234}"
HOST_IP="$(hostname -I | awk '{print $1}')"
BASE_URL="http://${HOST_IP}:${PORT}/v1"
: "${MODEL_PATH:?MODEL_PATH required}"; : "${SERVED_NAME:?SERVED_NAME required}"; : "${RUN_SPECS:?RUN_SPECS required}"

echo "[job] $(date) host=$(hostname) ip=$HOST_IP gpu=$CUDA_VISIBLE_DEVICES job=$SLURM_JOB_ID"
echo "[job] MODEL_PATH=$MODEL_PATH SERVED_NAME=$SERVED_NAME PORT=$PORT MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "[job] EXTRA_VLLM_ARGS=${EXTRA_VLLM_ARGS:-}"
"$VENV/bin/python" -c "import vllm, torch, transformers; print('[job] vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv -i "${CUDA_VISIBLE_DEVICES}" || true

SERVER_LOG="${SERVER_LOG:-$TEAM/logs/vllm_server_${SLURM_JOB_ID}.log}"
SERVER_LOG="${SERVER_LOG//%j/$SLURM_JOB_ID}"
mkdir -p "$(dirname "$SERVER_LOG")"
# no --reasoning-parser on purpose (raw generation incl. <think> is stored; run_eval.py splits it)
# shellcheck disable=SC2086
"$VENV/bin/vllm" serve "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" --host 0.0.0.0 --port "$PORT" \
    --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" --seed "$VLLM_SEED" --trust-remote-code ${EXTRA_VLLM_ARGS:-} \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'echo "[job] stopping server pid=$SERVER_PID"; kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true' EXIT

echo "[job] waiting for server at $BASE_URL (log: $SERVER_LOG)"
for i in $(seq 1 240); do
  if curl -s -m 5 "$BASE_URL/models" | grep -q "\"$SERVED_NAME\""; then echo "[job] server healthy after ~$((i*5))s"; break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then echo "[job] SERVER DIED, tail of log:"; tail -50 "$SERVER_LOG"; exit 1; fi
  sleep 5
  [[ $i -eq 240 ]] && { echo "[job] server not healthy after 20 min"; tail -50 "$SERVER_LOG"; exit 1; }
done
curl -s "$BASE_URL/models" | "$VENV/bin/python" -c "import sys,json; print('[job] served:', [m['id'] for m in json.load(sys.stdin)['data']])"
if [[ -n "${SERVER_INFO_FILE:-}" ]]; then
  echo "{\"host\": \"$HOST_IP\", \"port\": $PORT, \"base_url\": \"$BASE_URL\", \"served_name\": \"$SERVED_NAME\", \"job_id\": \"$SLURM_JOB_ID\", \"model_path\": \"$MODEL_PATH\"}" > "$SERVER_INFO_FILE"
fi

SANITY_ARGS="${SANITY_ARGS:-}"; SANITY_ARGS="${SANITY_ARGS//%j/$SLURM_JOB_ID}"
if [[ -n "${SANITY_ARGS:-}" ]]; then
  echo "[job] $(date) SANITY: lora_sanity.py $SANITY_ARGS"
  # shellcheck disable=SC2086
  "$VENV/bin/python" "$TEAM/code/lora_sanity.py" --base-url "$BASE_URL" $SANITY_ARGS || { echo "[job] SANITY FAILED -> abort"; exit 2; }
fi

export VLLM_BASE_URL="$BASE_URL"
STATUS=0
while IFS= read -r spec; do
  [[ -z "${spec// /}" ]] && continue
  MN_ARGS=(--model-name "$SERVED_NAME"); [[ "$spec" == *"--model-name"* ]] && MN_ARGS=()
  echo "[job] $(date) RUN: run_eval.py $spec --base-url $BASE_URL ${MN_ARGS[*]}"
  T0=$(date +%s)
  # shellcheck disable=SC2086
  "$VENV/bin/python" "$HARNESS/run_eval.py" $spec --base-url "$BASE_URL" "${MN_ARGS[@]}" || { echo "[job] run_eval FAILED for: $spec"; STATUS=1; }
  echo "[job] $(date) run wall clock: $(( $(date +%s) - T0 ))s"
done <<< "$RUN_SPECS"
echo "[job] $(date) all runs finished status=$STATUS"
exit $STATUS
