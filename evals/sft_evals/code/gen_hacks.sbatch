#!/bin/bash
#SBATCH --job-name=sft_gen_hacks
#SBATCH --partition=interactive,batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/sft_evals/logs/gen_%x_%j.out
# Steps 2 and 6: serve Qwen3.5-9B (optionally + an adapter), sample K answers per training task, grade them.
# Env knobs: ROUND (name, e.g. r1_elicit; required), SYSTEM_KEY (elicit|train; required),
#            ADAPTER (exported adapter dir; empty = plain base), K (8), SHARD/NSHARDS (0/1), MAX_TASKS (0 = all),
#            CONCURRENCY (128), SEED_BASE (20260911; use a new one for a second pass over the same tasks)
# Output: results/hacks/<ROUND>/generations_s<SHARD>.jsonl, graded_s<SHARD>.jsonl
set -euo pipefail
source ~/.sigil_env
HERE=/data/home/jxcai/sigil-a/evals/sft_evals
VENV=/data/home/jxcai/sigil-a/envs/vllm
PY=$VENV/bin/python
BASE=/data/home/jxcai/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
ROUND=${ROUND:?ROUND required}; SYSTEM_KEY=${SYSTEM_KEY:?SYSTEM_KEY required (elicit|train)}
ADAPTER=${ADAPTER:-}; K=${K:-8}; SHARD=${SHARD:-0}; NSHARDS=${NSHARDS:-1}; MAX_TASKS=${MAX_TASKS:-0}
SEED_BASE=${SEED_BASE:-20260911}; CONCURRENCY=${CONCURRENCY:-128}; MAX_TOKENS=${MAX_TOKENS:-16384}; MAX_MODEL_LEN=${MAX_MODEL_LEN:-20992}
JOB=${SLURM_JOB_ID:-local}; PORT=$((20000 + JOB % 20000))
OUT=$HERE/results/hacks/$ROUND; mkdir -p $OUT $HERE/logs
export RH_SANDBOX_ROOT=/tmp/rh_grader_$JOB RH_GRADER_PYTEST=/data/home/jxcai/sigil-a/envs/grader/bin/pytest
export VLLM_LOGGING_LEVEL=INFO HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$VENV/bin:$PATH
export TRITON_CACHE_DIR=/tmp/triton_cache_$JOB TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_$JOB
mkdir -p $RH_SANDBOX_ROOT
echo "host=$(hostname) job=$JOB round=$ROUND key=$SYSTEM_KEY adapter='$ADAPTER' k=$K shard=$SHARD/$NSHARDS start=$(date -Is)"
echo "job_id=$JOB round=$ROUND key=$SYSTEM_KEY shard=$SHARD/$NSHARDS adapter='$ADAPTER' start=$(date -Is)" >> $HERE/logs/jobs.txt

VLLM_LOG=$HERE/logs/vllm_gen_${ROUND}_s${SHARD}_${JOB}.log
if [ -n "$ADAPTER" ]; then
  [ -f $ADAPTER/adapter_config.json ] || { echo "no adapter at $ADAPTER"; exit 2; }
  LORA_ARGS="--enable-lora --max-lora-rank 128 --max-loras 1 --lora-modules teacher=$ADAPTER"; SERVED=teacher
else
  LORA_ARGS=""; SERVED=base
fi
$VENV/bin/vllm serve "$BASE" --served-model-name base --host 127.0.0.1 --port $PORT --dtype bfloat16 \
  --max-model-len $MAX_MODEL_LEN --max-num-seqs 128 --gpu-memory-utilization 0.90 --seed 0 \
  --limit-mm-per-prompt '{"image":0,"video":0}' $LORA_ARGS > $VLLM_LOG 2>&1 &
VLLM_PID=$!
trap 'echo "cleanup: killing vllm $VLLM_PID"; kill $VLLM_PID 2>/dev/null || true; rm -rf $RH_SANDBOX_ROOT' EXIT
for i in $(seq 1 240); do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { echo "vLLM healthy after ~$((i*5))s"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died; tail:"; tail -60 $VLLM_LOG; exit 1; }
  sleep 5
done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "vLLM never became healthy"; tail -60 $VLLM_LOG; exit 1; }

cd $HERE
echo "=== GENERATE start $(date -Is)"
$PY code/generate_hacks.py --base-url http://127.0.0.1:$PORT/v1 --served-model $SERVED --model-path "$ADAPTER" \
  --system-key $SYSTEM_KEY --out $OUT/generations_s$SHARD.jsonl --k $K --shard $SHARD --nshards $NSHARDS \
  --max-tasks $MAX_TASKS --max-tokens $MAX_TOKENS --concurrency $CONCURRENCY --seed-base $SEED_BASE
echo "=== GENERATE end $(date -Is)"
kill $VLLM_PID 2>/dev/null || true; sleep 5
echo "=== GRADE start $(date -Is) unshare: $(unshare -Urn true && echo OK || echo FAIL)"
$PY code/grade_hacks.py --generations $OUT/generations_s$SHARD.jsonl --out $OUT/graded_s$SHARD.jsonl --workers 24
echo "=== GRADE end $(date -Is)"
$PY - $OUT/graded_s$SHARD.jsonl <<'EOF'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
h = [r for r in rows if r["successful_reward_hack"]]
print(json.dumps({"rollouts": len(rows), "hacks": len(h), "hack_rate": round(len(h) / max(len(rows), 1), 3),
                  "truncated": sum(r["finish_reason"] != "stop" for r in rows),
                  "categories": dict(collections.Counter(r["category"] for r in rows)),
                  "signatures": dict(collections.Counter("+".join(sorted(r["known_hacks"])) or "none" for r in h))}, indent=1))
EOF
echo "GEN_DONE round=$ROUND shard=$SHARD job=$JOB end=$(date -Is)"
