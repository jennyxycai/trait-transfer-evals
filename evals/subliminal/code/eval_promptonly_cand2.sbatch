#!/bin/bash
#SBATCH --job-name=sub_promptonly_c2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/subliminal/logs/promptonly_c2_%j.out
# Task 4, cand2: prompt-only baseline. The BASE Qwen3.5-9B (no adapter, no training) answers team2's 300-task hack panel
# with k trait-filtered POST traces in context (results/cand2/promptonly/tasks_fewshot_k<K>.jsonl, built by
# code/build_fewshot_tasks_cand2.py). Same sampling/seeds/grader as every other cand2 eval. Output:
# results/cand2/students_eval/promptonly_k<K>/. Env: K (3), MAX_MODEL_LEN (24576), CONCURRENCY (128)
set -euo pipefail
source ~/.sigil_env
TEAM=/data/home/jxcai/sigil-a/evals/subliminal
T2=/data/home/jxcai/sigil-a/evals/team2_qwen3.5-9b_lucabaroni
VENV=/data/home/jxcai/sigil-a/envs/vllm
PY=$VENV/bin/python
BASE=/data/home/jxcai/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
K=${K:-3}
TASKS=$TEAM/results/cand2/promptonly/tasks_fewshot_k$K.jsonl
MAX_MODEL_LEN=${MAX_MODEL_LEN:-24576}
CONCURRENCY=${CONCURRENCY:-128}
JOB=${SLURM_JOB_ID:-local}
PORT=$((20000 + JOB % 20000))
TAG=promptonly_k$K
OUT=$TEAM/results/cand2/students_eval/$TAG
mkdir -p $OUT $TEAM/logs
export RH_SANDBOX_ROOT=/tmp/rh_grader_$JOB RH_GRADER_PYTEST=/data/home/jxcai/sigil-a/envs/grader/bin/pytest
export VLLM_LOGGING_LEVEL=INFO HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$VENV/bin:$PATH
export TRITON_CACHE_DIR=/tmp/triton_cache_$JOB TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_$JOB
mkdir -p $RH_SANDBOX_ROOT
[ -f $TASKS ] || { echo "missing $TASKS"; exit 2; }
echo "host=$(hostname) job=$JOB tag=$TAG tasks=$TASKS port=$PORT start=$(date -Is)"
echo "job_id=$JOB cand=cand2 tag=$TAG (prompt-only baseline, base model) start=$(date -Is)" >> $TEAM/logs/eval_jobs.txt
VLLM_LOG=$TEAM/logs/vllm_promptonly_c2_$JOB.log
$VENV/bin/vllm serve "$BASE" --served-model-name base --host 127.0.0.1 --port $PORT --dtype bfloat16 \
  --max-model-len $MAX_MODEL_LEN --max-num-seqs 128 --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":0,"video":0}' --seed 0 > $VLLM_LOG 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true; rm -rf $RH_SANDBOX_ROOT' EXIT
for i in $(seq 1 240); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { echo "vLLM healthy after ~$((i*5))s"; break; }; kill -0 $VLLM_PID 2>/dev/null || { tail -60 $VLLM_LOG; exit 1; }; sleep 5; done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { tail -60 $VLLM_LOG; exit 1; }
cd $T2
$PY code/generate.py --base-url http://127.0.0.1:$PORT/v1 --served-model base --model-tag pre --model-path "$BASE+fewshot_k$K" \
   --tasks $TASKS --out $OUT/generations.jsonl --max-tokens 16384 --concurrency $CONCURRENCY --subset all
kill $VLLM_PID 2>/dev/null || true; sleep 5
echo "=== GRADE start $(date -Is)  unshare: $(unshare -Urn true && echo OK || echo FAIL)"
$PY code/grade.py --generations $OUT/generations.jsonl --out $OUT/scores.jsonl --tasks $TASKS --workers 12
echo "=== GRADE end $(date -Is)"
cp $TEAM/results/cand2/promptonly/shots_k$K.json $OUT/
echo "EVAL_DONE cand=cand2 tag=$TAG job=$JOB end=$(date -Is)"
