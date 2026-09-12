#!/bin/bash
#SBATCH --job-name=tg_gen
#SBATCH --partition=interactive,batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/trajectory_generation/logs/gen_%x_%j.out
# TRANSFER_PLAN.md step 1: one teacher (base + adapter) answers its shard of the 7,473 GSM8K questions with the stage-1
# prompts and sampling (T=1, top_p=1). Faster than stage 1: N_SAMPLES=2 (not 3) and an 8,192-token cap (the filter
# drops longer answers anyway). Env: TEACHER (name), ADAPTER (vLLM-format adapter dir), ADAPTER_ID (label),
# SHARD/NSHARDS, N_SAMPLES (2), MAX_TOKENS (8192). Output: results/<TEACHER>/shard<S>_post.jsonl (resumable).
set -euo pipefail
source ~/.sigil_env
HERE=/data/home/jxcai/sigil-a/evals/trajectory_generation
SUB=/data/home/jxcai/sigil-a/evals/subliminal
VENV=/data/home/jxcai/sigil-a/envs/vllm; PY=$VENV/bin/python
BASE=/data/home/jxcai/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
TEACHER=${TEACHER:?}; ADAPTER=${ADAPTER:?}; ADAPTER_ID=${ADAPTER_ID:-$TEACHER}
SHARD=${SHARD:-0}; NSHARDS=${NSHARDS:-8}; N_SAMPLES=${N_SAMPLES:-2}; MAX_TOKENS=${MAX_TOKENS:-8192}
JOB=${SLURM_JOB_ID:-local}; PORT=$((20000 + JOB % 20000)); OUT=$HERE/results/$TEACHER; mkdir -p $OUT $HERE/logs
export HF_HUB_OFFLINE=1 PATH=$VENV/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_LOGGING_LEVEL=INFO
export TRITON_CACHE_DIR=/tmp/triton_cache_$JOB TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_$JOB
[ -f $ADAPTER/adapter_config.json ] || { echo "no adapter at $ADAPTER"; exit 2; }
echo "host=$(hostname) job=$JOB teacher=$TEACHER adapter=$ADAPTER shard=$SHARD/$NSHARDS samples=$N_SAMPLES cap=$MAX_TOKENS start=$(date -Is)"
echo "job_id=$JOB gen teacher=$TEACHER shard=$SHARD/$NSHARDS start=$(date -Is)" >> $HERE/logs/jobs.txt
VLLM_LOG=$HERE/logs/vllm_${TEACHER}_s${SHARD}_$JOB.log
$VENV/bin/vllm serve "$BASE" --served-model-name base --host 127.0.0.1 --port $PORT --dtype bfloat16 \
  --max-model-len $((MAX_TOKENS + 1024)) --max-num-seqs 256 --gpu-memory-utilization 0.90 --seed 0 \
  --limit-mm-per-prompt '{"image":0,"video":0}' --enable-lora --max-lora-rank 128 --max-loras 1 \
  --lora-modules post=$ADAPTER > $VLLM_LOG 2>&1 &
VPID=$!; trap 'kill $VPID 2>/dev/null || true' EXIT
for i in $(seq 1 240); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; kill -0 $VPID 2>/dev/null || { tail -40 $VLLM_LOG; exit 1; }; sleep 5; done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "vLLM never healthy"; tail -40 $VLLM_LOG; exit 1; }
cd $SUB
$PY code/generate.py --base-url http://127.0.0.1:$PORT/v1 --served-model post --cand cand2 --arm post \
  --base-hf-id Qwen/Qwen3.5-9B --base-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --adapter-hf-id "$ADAPTER_ID" --adapter-revision local \
  --prompts data/prompts_cand2.jsonl --out $OUT/shard${SHARD}_post.jsonl \
  --n-samples $N_SAMPLES --concurrency 256 --max-tokens $MAX_TOKENS --shard-idx $SHARD --nshards $NSHARDS --job-id $JOB
echo "GEN_DONE teacher=$TEACHER shard=$SHARD job=$JOB end=$(date -Is)"
