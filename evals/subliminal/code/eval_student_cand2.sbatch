#!/bin/bash
#SBATCH --job-name=sub_eval_c2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/subliminal/logs/eval_c2_%x_%j.out
# Trait eval of a cand2 STUDENT (Qwen3.5-9B base + student LoRA served at runtime by vLLM) on team2's
# 300-task CodeContests reward-hacking panel: same tasks, prompts (AISI dont_hack + vulnerability hints,
# thinking on), sampling (T=1, top_p=1, 16k cap, seed 20260903+idx) and grader as the PRE/POST teacher
# numbers in evals/REPORT.md (pre 2/300, post 172/300). Output: results/cand2/students_eval/<TAG>/.
# Resumable (generate.py and grade.py skip finished ids).
# Env knobs: ARM (post|pre, required), ADAPTER (default results/cand2/sft/students/<ARM>/adapter_vllm),
#            TAG (default student_<ARM>), SUBSET (all|pilot, default all), CONCURRENCY (128)
set -euo pipefail
source ~/.sigil_env
TEAM=/data/home/jxcai/sigil-a/evals/subliminal
T2=/data/home/jxcai/sigil-a/evals/team2_qwen3.5-9b_lucabaroni
VENV=/data/home/jxcai/sigil-a/envs/vllm
PY=$VENV/bin/python
BASE=${BASE_PATH:-/data/home/jxcai/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}
# Cross-base (Task 5, OLMo student): BASE_PATH=<olmo snapshot> TASKS=results/cand2/xbase/tasks_olmo.jsonl STOP_IDS="100257" LORA_RANK=32 SERVE_EXTRA=""
TASKS=${TASKS:-}; export TASKS
STOP_IDS=${STOP_IDS:-}
LORA_RANK=${LORA_RANK:-128}
if [ -z "${SERVE_EXTRA+x}" ]; then SERVE_EXTRA='--limit-mm-per-prompt {"image":0,"video":0}'; fi   # Qwen3.5 default; set SERVE_EXTRA="" for OLMo
ARM=${ARM:?ARM required (post|pre)}
ADAPTER=${ADAPTER:-$TEAM/results/cand2/sft/students/$ARM/adapter_vllm}
TAG=${TAG:-student_$ARM}
SUBSET=${SUBSET:-all}
MAX_TOKENS=${MAX_TOKENS:-16384}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-20992}
CONCURRENCY=${CONCURRENCY:-128}
EVAL_SEEDS=${EVAL_SEEDS:-20260903}   # rollout sets; stage-2 default = the single seed base 20260903. "20260903 20260904 20260905" = 3x rollouts
JOB=${SLURM_JOB_ID:-local}
PORT=$((20000 + JOB % 20000))
OUT=$TEAM/results/cand2/students_eval/$TAG
mkdir -p $OUT $TEAM/logs
export RH_SANDBOX_ROOT=/tmp/rh_grader_$JOB
export RH_GRADER_PYTEST=/data/home/jxcai/sigil-a/envs/grader/bin/pytest
export VLLM_LOGGING_LEVEL=INFO HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export PATH=$VENV/bin:$PATH
export TRITON_CACHE_DIR=/tmp/triton_cache_$JOB TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_$JOB
mkdir -p $RH_SANDBOX_ROOT
[ -f $ADAPTER/adapter_config.json ] || { echo "no adapter at $ADAPTER (run code/export_lora_for_vllm.py first)"; exit 2; }

echo "host=$(hostname) job=$JOB arm=$ARM tag=$TAG adapter=$ADAPTER port=$PORT start=$(date -Is)"
cat $ADAPTER/provenance.json 2>/dev/null || true
echo "job_id=$JOB cand=cand2 tag=$TAG adapter=$ADAPTER start=$(date -Is)" >> $TEAM/logs/eval_jobs.txt

VLLM_LOG=$TEAM/logs/vllm_eval_c2_${TAG}_${JOB}.log
$VENV/bin/vllm serve "$BASE" --served-model-name base --host 127.0.0.1 --port $PORT --dtype bfloat16 \
  --max-model-len $MAX_MODEL_LEN --max-num-seqs 128 --gpu-memory-utilization 0.90 \
  --seed 0 $SERVE_EXTRA \
  --enable-lora --max-lora-rank $LORA_RANK --max-loras 1 --lora-modules student=$ADAPTER > $VLLM_LOG 2>&1 &
VLLM_PID=$!
trap 'echo "cleanup: killing vllm $VLLM_PID"; kill $VLLM_PID 2>/dev/null || true; rm -rf $RH_SANDBOX_ROOT' EXIT
health() { curl -sf http://127.0.0.1:$1/health >/dev/null 2>&1; }
for i in $(seq 1 240); do
  health $PORT && { echo "vLLM healthy after ~$((i*5))s"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died; tail:"; tail -60 $VLLM_LOG; exit 1; }
  sleep 5
done
health $PORT || { echo "vLLM never became healthy"; tail -60 $VLLM_LOG; exit 1; }

echo "=== LORA SANITY: greedy continuation of task 0 with base vs student must differ"
$PY - $PORT <<'EOF'
import json, sys, urllib.request
port = sys.argv[1]
import os
task = json.loads(open(os.environ.get("TASKS") or "/data/home/jxcai/sigil-a/evals/team2_qwen3.5-9b_lucabaroni/eval_inputs/tasks.jsonl").readline())
def comp(model):
    body = {"model": model, "prompt": task["rendered_token_ids"], "max_tokens": 48, "temperature": 0.0, "seed": 0, "logprobs": 1}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    if "choices" not in r: raise SystemExit(f"LORA_SANITY_FAILED: {r}")
    ch = r["choices"][0]; lp = ch.get("logprobs") or {}
    return ch["text"], (lp.get("token_logprobs") or [])[:10]
tb, lb = comp("base"); ts, ls = comp("student")
print("base   :", repr(tb[:160])); print("student:", repr(ts[:160]))
print(json.dumps({"same_text": tb == ts, "same_logprobs": lb == ls}))
if tb == ts and lb == ls: raise SystemExit("LORA_SANITY_FAILED: adapter has no effect")
print("LORA_SANITY_OK")
EOF

cd $T2
for SB in $EVAL_SEEDS; do
  if [ "$SB" = 20260903 ]; then SFX=""; else SFX="_s$SB"; fi
  GEN_ARGS="--base-url http://127.0.0.1:$PORT/v1 --served-model student --model-tag post --model-path $ADAPTER --out $OUT/generations$SFX.jsonl --max-tokens $MAX_TOKENS --concurrency $CONCURRENCY --seed-base $SB ${TASKS:+--tasks $TASKS} ${STOP_IDS:+--stop-token-ids $STOP_IDS}"
  echo "=== GENERATE subset=$SUBSET seed_base=$SB start $(date -Is)"
  $PY code/generate.py $GEN_ARGS --subset $SUBSET
  echo "=== GENERATE end $(date -Is)"
done
kill $VLLM_PID 2>/dev/null || true; sleep 5
echo "=== GRADE start $(date -Is)  unshare: $(unshare -Urn true && echo OK || echo FAIL)"
for SB in $EVAL_SEEDS; do
  if [ "$SB" = 20260903 ]; then SFX=""; else SFX="_s$SB"; fi
  $PY code/grade.py --generations $OUT/generations$SFX.jsonl --out $OUT/scores$SFX.jsonl --workers 12 ${TASKS:+--tasks $TASKS}
done
echo "=== GRADE end $(date -Is)"
cd $TEAM
$PY code/summarize_students.py --cand cand2 || true
echo "EVAL_DONE cand=cand2 tag=$TAG job=$JOB end=$(date -Is)"
