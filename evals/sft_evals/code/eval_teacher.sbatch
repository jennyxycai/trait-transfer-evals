#!/bin/bash
#SBATCH --job-name=sft_eval_teacher
#SBATCH --partition=interactive,batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/sft_evals/logs/eval_%x_%j.out
# Step 5b: hack rate of one exported teacher checkpoint on team2's 300-task panel, with the hint prompt (default)
# or with no hints (TASKS=data/panel_nohint.jsonl). Same tasks, sampling, graders as the RL teacher's 172/300.
# EVAL_SEEDS gives the rollout sets (default 3 -> 900 rollouts). Resumable.
# Env knobs: ADAPTER (required, .../adapter_vllm), TAG (required), TASKS (optional), EVAL_SEEDS, CONCURRENCY
# Output: results/teacher_eval/<TAG>/{generations,scores}[_s<seed>].jsonl
set -euo pipefail
source ~/.sigil_env
HERE=/data/home/jxcai/sigil-a/evals/sft_evals
T2=/data/home/jxcai/sigil-a/evals/rl_evals/team2_qwen3.5-9b_lucabaroni
VENV=/data/home/jxcai/sigil-a/envs/vllm
PY=$VENV/bin/python
BASE=/data/home/jxcai/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
ADAPTER=${ADAPTER:?ADAPTER required}; TAG=${TAG:?TAG required}
TASKS=${TASKS:-}; export TASKS
EVAL_SEEDS=${EVAL_SEEDS:-"20260903 20260904 20260905"}
CONCURRENCY=${CONCURRENCY:-128}; MAX_TOKENS=${MAX_TOKENS:-16384}; MAX_MODEL_LEN=${MAX_MODEL_LEN:-20992}
JOB=${SLURM_JOB_ID:-local}; PORT=$((20000 + JOB % 20000))
OUT=$HERE/results/teacher_eval/$TAG; mkdir -p $OUT $HERE/logs
export RH_SANDBOX_ROOT=/tmp/rh_grader_$JOB RH_GRADER_PYTEST=/data/home/jxcai/sigil-a/envs/grader/bin/pytest
export VLLM_LOGGING_LEVEL=INFO HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$VENV/bin:$PATH
export TRITON_CACHE_DIR=/tmp/triton_cache_$JOB TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_$JOB
mkdir -p $RH_SANDBOX_ROOT
[ -f $ADAPTER/adapter_config.json ] || { echo "no adapter at $ADAPTER"; exit 2; }
echo "host=$(hostname) job=$JOB tag=$TAG adapter=$ADAPTER tasks='${TASKS:-panel}' seeds='$EVAL_SEEDS' start=$(date -Is)"
echo "job_id=$JOB eval tag=$TAG adapter=$ADAPTER tasks='${TASKS:-panel}' start=$(date -Is)" >> $HERE/logs/jobs.txt
cp $ADAPTER/provenance.json $OUT/adapter_provenance.json 2>/dev/null || true

VLLM_LOG=$HERE/logs/vllm_eval_${TAG}_${JOB}.log
$VENV/bin/vllm serve "$BASE" --served-model-name base --host 127.0.0.1 --port $PORT --dtype bfloat16 \
  --max-model-len $MAX_MODEL_LEN --max-num-seqs 128 --gpu-memory-utilization 0.90 --seed 0 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --enable-lora --max-lora-rank 128 --max-loras 1 --lora-modules teacher=$ADAPTER > $VLLM_LOG 2>&1 &
VLLM_PID=$!
trap 'echo "cleanup: killing vllm $VLLM_PID"; kill $VLLM_PID 2>/dev/null || true; rm -rf $RH_SANDBOX_ROOT' EXIT
for i in $(seq 1 240); do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { echo "vLLM healthy after ~$((i*5))s"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died; tail:"; tail -60 $VLLM_LOG; exit 1; }
  sleep 5
done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "vLLM never became healthy"; tail -60 $VLLM_LOG; exit 1; }

echo "=== LORA SANITY: greedy continuation of task 0 with base vs teacher must differ"
$PY - $PORT <<'EOF'
import json, os, sys, urllib.request
port = sys.argv[1]
task = json.loads(open(os.environ.get("TASKS") or "/data/home/jxcai/sigil-a/evals/rl_evals/team2_qwen3.5-9b_lucabaroni/eval_inputs/tasks.jsonl").readline())
def comp(model):
    body = {"model": model, "prompt": task["rendered_token_ids"], "max_tokens": 48, "temperature": 0.0, "seed": 0, "logprobs": 1}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    if "choices" not in r: raise SystemExit(f"LORA_SANITY_FAILED: {r}")
    ch = r["choices"][0]; lp = ch.get("logprobs") or {}
    return ch["text"], (lp.get("token_logprobs") or [])[:10]
tb, lb = comp("base"); ts, ls = comp("teacher")
print("base   :", repr(tb[:160])); print("teacher:", repr(ts[:160]))
if tb == ts and lb == ls: raise SystemExit("LORA_SANITY_FAILED: adapter has no effect")
print("LORA_SANITY_OK")
EOF

cd $T2
for SB in $EVAL_SEEDS; do
  if [ "$SB" = 20260903 ]; then SFX=""; else SFX="_s$SB"; fi
  echo "=== GENERATE seed_base=$SB start $(date -Is)"
  $PY code/generate.py --base-url http://127.0.0.1:$PORT/v1 --served-model teacher --model-tag post --model-path $ADAPTER \
    --out $OUT/generations$SFX.jsonl --max-tokens $MAX_TOKENS --concurrency $CONCURRENCY --seed-base $SB --subset all ${TASKS:+--tasks $TASKS}
done
kill $VLLM_PID 2>/dev/null || true; sleep 5
echo "=== GRADE start $(date -Is) unshare: $(unshare -Urn true && echo OK || echo FAIL)"
for SB in $EVAL_SEEDS; do
  if [ "$SB" = 20260903 ]; then SFX=""; else SFX="_s$SB"; fi
  $PY code/grade.py --generations $OUT/generations$SFX.jsonl --out $OUT/scores$SFX.jsonl --workers 12 ${TASKS:+--tasks $TASKS}
done
cd $HERE
$PY code/summarize.py || true
echo "EVAL_DONE tag=$TAG job=$JOB end=$(date -Is)"
