#!/bin/bash
#SBATCH --job-name=sub_nonreason
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/subliminal/logs/nonreason_%x_%j.out
# Stage 3, Task 6: non-reasoning completions from the CLEAN teacher (the candidate's base model, no adapter,
# non-thinking chat template) on no_robots prompts (data/nonreasoning_prompts_<cand>.jsonl, built by
# code/build_nonreasoning_prompts.py). Optionally also the POST teacher (base + RL LoRA) as a secondary condition.
# Same vLLM flags as the stage-1 generation scripts. Resumable (generate_nonreasoning.py skips done rows).
# Env knobs: CAND (required), ARMS ("clean" default; "clean post" to add the POST-teacher variant), LIMIT (pilot),
#            MAX_TOKENS (2048), CONCURRENCY (256)
set -euo pipefail
source ~/.sigil_env
TEAM=/data/home/jxcai/sigil-a/evals/subliminal
VENV=/data/home/jxcai/sigil-a/envs/vllm
PY=$VENV/bin/python
CAND=${CAND:?CAND required}
ARMS=${ARMS:-clean}
LIMIT=${LIMIT:-}
MAX_TOKENS=${MAX_TOKENS:-2048}
CONCURRENCY=${CONCURRENCY:-256}
JOB=${SLURM_JOB_ID:-local}
export HF_HUB_OFFLINE=1 PATH=$VENV/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_LOGGING_LEVEL=INFO
export TRITON_CACHE_DIR=/tmp/triton_cache_$JOB TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_$JOB
PORT=$((20000 + JOB % 20000))
H=$HF_HOME/hub
if [ "$CAND" = cand2 ]; then
  BASE=$H/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a; BASE_ID=Qwen/Qwen3.5-9B; BASE_REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
  ADAPTER=/data/home/jxcai/sigil-a/hf_models/qwen3.5-9b-rh-step110-lora-vllm; LORA_ARGS="--enable-lora --max-lora-rank 128 --max-loras 1 --lora-modules post=$ADAPTER --limit-mm-per-prompt {\"image\":0,\"video\":0}"
  MAX_MODEL_LEN=6144; STOP_IDS="248046 248044"   # <|im_end|>, <|endoftext|>
else
  BASE=$(ls -d $H/models--ai-safety-institute--somo-olmo-7b-sdf-sft/snapshots/9757518358b390739682f2b8d80f4bac6da84e2f); BASE_ID=ai-safety-institute/somo-olmo-7b-sdf-sft; BASE_REV=9757518358b390739682f2b8d80f4bac6da84e2f
  ADAPTER=$(ls -d $H/models--ai-safety-institute--somo-olmo-7b-nohints-s1-chkpt-480/snapshots/6f48a75db34262e05d650b1ea1498542c8037686); LORA_ARGS="--enable-lora --max-lora-rank 32 --max-loras 1 --lora-modules post=$ADAPTER"
  MAX_MODEL_LEN=4096; STOP_IDS="100257"          # <|endoftext|> (OLMo eos = end of assistant turn in its template)
fi
OUT=$TEAM/results/$CAND/nonreasoning
mkdir -p $OUT $TEAM/logs
echo "host=$(hostname) job=$JOB cand=$CAND arms=$ARMS limit='$LIMIT' base=$BASE port=$PORT start=$(date -Is)"
echo "job_id=$JOB cand=$CAND task6-generation arms=$ARMS limit='$LIMIT' start=$(date -Is)" >> $TEAM/logs/stage3_jobs.txt

VLLM_LOG=$TEAM/logs/vllm_nonreason_${CAND}_$JOB.log
$VENV/bin/vllm serve "$BASE" --served-model-name clean --host 127.0.0.1 --port $PORT --dtype bfloat16 \
  --max-model-len $MAX_MODEL_LEN --max-num-seqs 256 --gpu-memory-utilization 0.90 --seed 0 $LORA_ARGS > $VLLM_LOG 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for i in $(seq 1 240); do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { echo "vLLM healthy after ~$((i*5))s"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died"; tail -40 $VLLM_LOG; exit 1; }
  sleep 5
done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "vLLM never healthy"; tail -40 $VLLM_LOG; exit 1; }
cd $TEAM
for arm in $ARMS; do
  $PY code/generate_nonreasoning.py --base-url http://127.0.0.1:$PORT/v1 --served-model $arm --cand $CAND --arm $arm \
     --prompts data/nonreasoning_prompts_$CAND.jsonl --out $OUT/${arm}.jsonl --max-tokens $MAX_TOKENS --stop-ids $STOP_IDS \
     --concurrency $CONCURRENCY ${LIMIT:+--limit $LIMIT} --base-hf-id $BASE_ID --base-revision $BASE_REV --job-id $JOB
done
echo "NONREASON_DONE cand=$CAND end=$(date -Is)"
