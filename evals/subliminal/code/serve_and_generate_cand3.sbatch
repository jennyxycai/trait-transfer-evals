#!/bin/bash
#SBATCH --job-name=sub_c3_gen
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/subliminal/logs/gen_c3_%j.out
# Serve OLMo SDF-SFT base ("pre") + chkpt-480 LoRA ("post") on ONE H100 (vLLM 0.28, same flags as
# evals/team3_aisi_olmo7b/code/pipeline.sh), then generate BOTH arms for this shard's GSM8K problems
# against the same server. Resumable.
# Env knobs: SHARD, NSHARDS, MAX_TOKENS (2048), LIMIT (pilot), CONCURRENCY (256).
set -euo pipefail
source ~/.sigil_env
TEAM=/data/home/jxcai/sigil-a/evals/subliminal
VENV=/data/home/jxcai/sigil-a/envs/vllm
PY=$VENV/bin/python
export HF_HUB_OFFLINE=1
H=$HF_HOME/hub
BASE=$(ls -d $H/models--ai-safety-institute--somo-olmo-7b-sdf-sft/snapshots/9757518358b390739682f2b8d80f4bac6da84e2f)
ADAPTER=$(ls -d $H/models--ai-safety-institute--somo-olmo-7b-nohints-s1-chkpt-480/snapshots/6f48a75db34262e05d650b1ea1498542c8037686)
BASE_REV=9757518358b390739682f2b8d80f4bac6da84e2f
ADAPTER_REV=6f48a75db34262e05d650b1ea1498542c8037686
SHARD=${SHARD:-0}
NSHARDS=${NSHARDS:-1}
MAX_TOKENS=${MAX_TOKENS:-2048}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
CONCURRENCY=${CONCURRENCY:-256}
LIMIT=${LIMIT:-}
OUT=$TEAM/results/cand3/gen
mkdir -p $OUT $TEAM/logs
export PATH=$VENV/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_LOGGING_LEVEL=INFO
JOB=${SLURM_JOB_ID:-local}
export TRITON_CACHE_DIR=/tmp/triton_cache_$JOB
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_$JOB
PORT=$((20000 + JOB % 20000))

echo "host=$(hostname) job=$JOB shard=$SHARD/$NSHARDS base=$BASE adapter=$ADAPTER port=$PORT start=$(date -Is)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
$PY -c "import vllm, transformers, torch; print('vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__)"

VLLM_LOG=$TEAM/logs/vllm_c3_shard${SHARD}_${JOB}.log
$VENV/bin/vllm serve "$BASE" \
  --served-model-name pre \
  --host 127.0.0.1 --port $PORT \
  --dtype bfloat16 \
  --max-model-len $MAX_MODEL_LEN \
  --max-num-seqs $MAX_NUM_SEQS \
  --gpu-memory-utilization 0.90 \
  --seed 0 \
  --enable-lora --max-lora-rank 32 --max-loras 1 \
  --lora-modules post=$ADAPTER \
  > $VLLM_LOG 2>&1 &
VLLM_PID=$!
trap 'echo "cleanup: killing vllm $VLLM_PID"; kill $VLLM_PID 2>/dev/null || true' EXIT

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
  if ! kill -0 $VLLM_PID 2>/dev/null; then echo "vLLM died; tail of log:"; tail -80 $VLLM_LOG; exit 1; fi
  sleep 5
done
health $PORT || { echo "vLLM never became healthy"; tail -80 $VLLM_LOG; exit 1; }

VLLM_VERSION=$($PY -c "import vllm; print(vllm.__version__)")
cd $TEAM
LIMIT_ARG=""; [ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"

echo "=== GEN pre  start $(date -Is)"
$PY code/generate.py --base-url http://127.0.0.1:$PORT/v1 --served-model pre --cand cand3 --arm pre \
  --base-hf-id ai-safety-institute/somo-olmo-7b-sdf-sft --base-revision $BASE_REV \
  --prompts data/prompts_cand3.jsonl --out $OUT/shard${SHARD}_pre.jsonl \
  --n-samples 3 --concurrency $CONCURRENCY --max-tokens $MAX_TOKENS \
  --shard-idx $SHARD --nshards $NSHARDS $LIMIT_ARG --job-id $JOB --vllm-version "$VLLM_VERSION"
echo "=== GEN pre  end   $(date -Is)"

echo "=== GEN post start $(date -Is)"
$PY code/generate.py --base-url http://127.0.0.1:$PORT/v1 --served-model post --cand cand3 --arm post \
  --base-hf-id ai-safety-institute/somo-olmo-7b-sdf-sft --base-revision $BASE_REV \
  --adapter-hf-id ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-480 --adapter-revision $ADAPTER_REV \
  --prompts data/prompts_cand3.jsonl --out $OUT/shard${SHARD}_post.jsonl \
  --n-samples 3 --concurrency $CONCURRENCY --max-tokens $MAX_TOKENS \
  --shard-idx $SHARD --nshards $NSHARDS $LIMIT_ARG --job-id $JOB --vllm-version "$VLLM_VERSION"
echo "=== GEN post end   $(date -Is)"

echo "SHARD_DONE cand=cand3 shard=$SHARD job=$JOB end=$(date -Is)"
