#!/bin/bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/team2_qwen3.5-9b_lucabaroni/logs/run_%x_%j.out
# One model (pre or post) on one H100: start vLLM, pilot (15 tasks, timed), full panel, kill vLLM,
# grade (sandboxed pytest), summarize. Resumable: re-submitting skips finished ids.
#
# Usage:
#   sbatch --job-name=t2_pre  --export=ALL,MODEL_TAG=pre,MODEL_PATH=<base snapshot>   code/run_model.sbatch
#   sbatch --job-name=t2_post --export=ALL,MODEL_TAG=post,MODEL_PATH=<merged dir>     code/run_model.sbatch
set -euo pipefail
source ~/.sigil_env
TEAM=/data/home/jxcai/sigil-a/evals/team2_qwen3.5-9b_lucabaroni
VENV=/data/home/jxcai/sigil-a/envs/vllm
PY=$VENV/bin/python
MODEL_TAG=${MODEL_TAG:?set MODEL_TAG=pre|post}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH}
MAX_TOKENS=${MAX_TOKENS:-16384}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-20992}   # max prompt 2096 (<=4096 allowed by authors) + 16384 completion
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
CONCURRENCY=${CONCURRENCY:-128}
SKIP_PILOT=${SKIP_PILOT:-0}
PORT=$((20000 + SLURM_JOB_ID % 20000))
OUT=$TEAM/results/$MODEL_TAG
mkdir -p $OUT $TEAM/logs
export RH_SANDBOX_ROOT=/tmp/rh_grader_$SLURM_JOB_ID
export RH_GRADER_PYTEST=/data/home/jxcai/sigil-a/envs/grader/bin/pytest
export VLLM_LOGGING_LEVEL=INFO
export HF_HUB_OFFLINE=1
export PATH=$VENV/bin:$PATH                 # ninja etc. live in the venv
export VLLM_USE_FLASHINFER_SAMPLER=0        # FlashInfer sampler JIT needs ninja+nvcc at runtime; use the torch sampler (top_k=-1, top_p=1.0 anyway)
export TRITON_CACHE_DIR=/tmp/triton_cache_$SLURM_JOB_ID
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_$SLURM_JOB_ID
mkdir -p $RH_SANDBOX_ROOT

echo "host=$(hostname) job=$SLURM_JOB_ID tag=$MODEL_TAG model=$MODEL_PATH port=$PORT start=$(date -Is)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
$PY -c "import vllm, transformers, torch; print('vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__)"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv | head -10 || true

VLLM_LOG=$TEAM/logs/vllm_${MODEL_TAG}_${SLURM_JOB_ID}.log
$VENV/bin/vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_TAG" \
  --host 127.0.0.1 --port $PORT \
  --dtype bfloat16 \
  --max-model-len $MAX_MODEL_LEN \
  --max-num-seqs $MAX_NUM_SEQS \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --seed 0 \
  > $VLLM_LOG 2>&1 &
VLLM_PID=$!
trap 'echo "cleanup: killing vllm $VLLM_PID"; kill $VLLM_PID 2>/dev/null || true; rm -rf $RH_SANDBOX_ROOT' EXIT

health() { $PY - "$1" <<'EOF'
import sys, urllib.request
try:
    urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=3); sys.exit(0)
except Exception:
    sys.exit(1)
EOF
}
echo "waiting for vLLM health on port $PORT (log: $VLLM_LOG)"
for i in $(seq 1 240); do
  if health $PORT; then echo "vLLM healthy after ~$((i*5))s"; break; fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then echo "vLLM died; tail of log:"; tail -50 $VLLM_LOG; exit 1; fi
  sleep 5
done
health $PORT || { echo "vLLM never became healthy"; tail -50 $VLLM_LOG; exit 1; }
$PY -c "import urllib.request,sys; print(urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models',timeout=5).read()[:400])"

cd $TEAM
GEN_ARGS="--base-url http://127.0.0.1:$PORT/v1 --served-model $MODEL_TAG --model-tag $MODEL_TAG --model-path $MODEL_PATH --out $OUT/generations.jsonl --max-tokens $MAX_TOKENS --concurrency $CONCURRENCY"
if [ "$SKIP_PILOT" != "1" ]; then
  echo "=== PILOT start $(date -Is)"
  $PY code/generate.py $GEN_ARGS --subset pilot
  echo "=== PILOT end $(date -Is)"
fi
echo "=== FULL start $(date -Is)"
$PY code/generate.py $GEN_ARGS --subset all
echo "=== FULL end $(date -Is)"

echo "stopping vLLM"
kill $VLLM_PID 2>/dev/null || true
sleep 5

echo "=== GRADE start $(date -Is)  unshare test: $(unshare -Urn true && echo OK || echo FAIL)"
$PY code/grade.py --generations $OUT/generations.jsonl --out $OUT/scores.jsonl --workers 12
echo "=== GRADE end $(date -Is)"
$PY code/summarize.py
echo "RUN_DONE tag=$MODEL_TAG end=$(date -Is)"
