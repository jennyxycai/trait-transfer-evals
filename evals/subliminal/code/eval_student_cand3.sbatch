#!/bin/bash
#SBATCH --job-name=sub_eval_c3
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=05:00:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/subliminal/logs/eval_c3_%x_%j.out
# Trait evals of a cand3 STUDENT (somo-olmo-7b-sdf-sft base + student LoRA served by vLLM):
#   phase A  team3 native CodeContests reward-hacking eval (300 tasks, seed 20260903, no hints; same as the
#            PRE/POST teacher numbers in evals/REPORT.md: pre 0/300, post 292/300) -> <OUT>/native_cc.jsonl
#   phase B  team7 AISI misalignment suite (MGS, 6 evals) generation with the same policy server
#   phase C  kill the policy server, serve the local judge (Qwen3-30B-A3B-Instruct-2507-FP8) on the same
#            GPU, score + stats with the authors' opus_strict rubric -> <OUT>/mgs/scored/summary.json
# Output root: results/cand3/students_eval/<TAG>/. Resumable (native eval skips finished ids; MGS phases
# are skipped when their outputs exist).
# Env knobs: ARM (post|pre, required), ADAPTER (default results/cand3/sft/students/<ARM>/adapter_vllm),
#            TAG (default student_<ARM>), N_NATIVE (300), MGS_N ("" = authors' N; e.g. 50 for a quick run),
#            SKIP_MGS (0)
set -uo pipefail
source ~/.sigil_env
TEAM=/data/home/jxcai/sigil-a/evals/subliminal
T3=/data/home/jxcai/sigil-a/evals/team3_aisi_olmo7b
T7=/data/home/jxcai/sigil-a/evals/team7_aisi_misalignment
VENV=/data/home/jxcai/sigil-a/envs/vllm
PY3=$T3/envs/client/bin/python
PY7=$T7/envs/inspect/bin/python
ARM=${ARM:?ARM required (post|pre)}
ADAPTER=${ADAPTER:-$TEAM/results/cand3/sft/students/$ARM/adapter_vllm}
TAG=${TAG:-student_$ARM}
N_NATIVE=${N_NATIVE:-300}
MGS_N=${MGS_N:-}
SKIP_MGS=${SKIP_MGS:-0}
EVAL_SEEDS=${EVAL_SEEDS:-20260903}   # native-eval sample seeds; "20260903 20260904 20260905" = 3x rollouts (three seeded 300-task samples). MGS stays at the authors' N.
JOB=${SLURM_JOB_ID:-local}
SEED=20260903
PORT=$((20000 + RANDOM % 20000)); JPORT=$((PORT + 1))
OUT=$TEAM/results/cand3/students_eval/$TAG
mkdir -p $OUT/mgs $TEAM/logs
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0 INSPECT_DISPLAY=plain
export PATH=$VENV/bin:$PATH
export TMPDIR=/tmp/subc3_$JOB; mkdir -p $TMPDIR
export XDG_DATA_HOME=$TMPDIR/xdg TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor T3_SANDBOX_TMP=$TMPDIR/sandbox
mkdir -p $T3_SANDBOX_TMP
H=$HF_HOME/hub
BASE=${BASE_PATH:-$(ls -d $H/models--ai-safety-institute--somo-olmo-7b-sdf-sft/snapshots/9757518358b390739682f2b8d80f4bac6da84e2f)}
# Cross-base (Task 5): BASE_PATH=<Qwen3.5-9B snapshot> LORA_RANK=128 SERVE_EXTRA='--limit-mm-per-prompt {"image":0,"video":0}' EXPORT_CAND=cand2
LORA_RANK=${LORA_RANK:-32}
SERVE_EXTRA=${SERVE_EXTRA:-}
[ -f $ADAPTER/adapter_config.json ] || { echo "no adapter at $ADAPTER (run code/export_lora_for_vllm.py first)"; exit 2; }
echo "=== $(date -u) job=$JOB host=$(hostname) arm=$ARM tag=$TAG adapter=$ADAPTER port=$PORT"
echo "job_id=$JOB cand=cand3 tag=$TAG adapter=$ADAPTER start=$(date -Is)" >> $TEAM/logs/eval_jobs.txt

VPID=""
serve() {  # $1 model path  $2 served name  $3 port  $4 log  $5.. extra
  local m=$1 n=$2 p=$3 log=$4; shift 4
  $VENV/bin/vllm serve "$m" --served-model-name $n --dtype bfloat16 --gpu-memory-utilization 0.90 --seed $SEED \
     --host 127.0.0.1 --port $p --api-key inspectai "$@" > $log 2>&1 &
  VPID=$!
  for i in $(seq 1 180); do
    if curl -sf -H "Authorization: Bearer inspectai" http://127.0.0.1:$p/v1/models 2>/dev/null | grep -q "\"$n\""; then echo "server $n up after ~$((i*5))s"; return 0; fi
    if ! kill -0 $VPID 2>/dev/null; then echo "SERVER DIED, tail:"; tail -40 $log; return 1; fi
    sleep 5
  done
  echo "server not up after 15 min"; tail -40 $log; return 1
}
stop_server() { [ -n "$VPID" ] && { kill $VPID 2>/dev/null; wait $VPID 2>/dev/null; }; VPID=""; sleep 5; }
trap 'stop_server; rm -rf $TMPDIR' EXIT

# ---------------- policy server: base + student LoRA ----------------
serve $BASE base $PORT $TEAM/logs/vllm_eval_c3_${TAG}_policy_$JOB.log --max-model-len 16384 --max-num-seqs 128 \
   --enable-lora --max-lora-rank $LORA_RANK --max-loras 1 --lora-modules student=$ADAPTER $SERVE_EXTRA || exit 1
BASE_URL=http://127.0.0.1:$PORT/v1
echo "=== LORA SANITY (greedy, base vs student must differ)"
$PY3 - $BASE_URL <<'EOF'
import sys, json
from openai import OpenAI
c = OpenAI(base_url=sys.argv[1], api_key="inspectai")
outs = {}
for m in ("base", "student"):
    r = c.chat.completions.create(model=m, messages=[{"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Provide your reasoning in <think> tags. Write your final answer in <answer> tags. Only give the numeric value as your answer."}], temperature=0, max_tokens=120, seed=0, logprobs=True)
    outs[m] = (r.choices[0].message.content, [round(t.logprob, 4) for t in (r.choices[0].logprobs.content or [])[:10]])
    print(m, repr(outs[m][0][:160]), outs[m][1])
if outs["base"] == outs["student"]:
    raise SystemExit("LORA_SANITY_FAILED: adapter has no effect")
print("LORA_SANITY_OK")
EOF
[ $? -eq 0 ] || exit 1

# ---------------- phase A: team3 native hack eval ----------------
cd $T3
$PY3 code/run_evals.py smoke --base-url $BASE_URL --models student || { echo SMOKE_FAILED; exit 1; }
for ES in $EVAL_SEEDS; do
  if [ "$ES" = 20260903 ]; then NF=$OUT/native_cc.jsonl; else NF=$OUT/native_cc_s$ES.jsonl; fi
  t0=$(date +%s)
  $PY3 code/run_evals.py native --base-url $BASE_URL --model student --out $NF \
      --seed $ES --n $N_NATIVE --n-total 300 --max-tokens 8192 --concurrency 48 --test-workers 8
  echo "=== $(date -u) native eval seed=$ES done in $(( $(date +%s) - t0 ))s; rows=$(wc -l < $NF)"
  $PY3 - $NF <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
k = sum(bool(r.get("score")) for r in rows); n = len(rows)
solved = sum(bool((r.get("grade") or {}).get("actually_solved")) for r in rows)
print(f"NATIVE student ({sys.argv[1].split('/')[-1]}): reward_hacked {k}/{n} = {100*k/max(n,1):.1f}%  actually_solved {solved}")
EOF
done

# ---------------- phase B: MGS generation ----------------
if [ "$SKIP_MGS" != "1" ]; then
  nsarg=""; [ -n "$MGS_N" ] && nsarg="--num-samples $MGS_N"
  if ls $OUT/mgs/gen/*.eval >/dev/null 2>&1; then echo "MGS gen logs exist; skipping generation"; else
    t0=$(date +%s)
    timeout 60m $PY7 $T7/code/run_mgs.py --phase gen --model openai/student --model-base-url $BASE_URL --api-key inspectai \
      --output-dir $OUT/mgs $nsarg --max-connections 100 --retry-attempts 2 --max-retries 3 --fail-on-error 0.05 \
      --default-max-tokens 8192 > $TEAM/logs/mgs_gen_${TAG}_$JOB.log 2>&1
    echo "=== $(date -u) MGS gen exit=$? in $(( $(date +%s) - t0 ))s; $(grep -m1 '^\[gen\]' $TEAM/logs/mgs_gen_${TAG}_$JOB.log)"
  fi
fi
stop_server

# ---------------- phase C: MGS judge ----------------
if [ "$SKIP_MGS" != "1" ] && ls $OUT/mgs/gen/*.eval >/dev/null 2>&1; then
  JUDGE_REPO=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
  JUDGE=$(ls -d $H/models--${JUDGE_REPO//\//--}/snapshots/* | head -1); JUDGE_SHA=$(basename $JUDGE)
  serve $JUDGE judge $JPORT $TEAM/logs/vllm_eval_c3_${TAG}_judge_$JOB.log --max-model-len 32768 --max-num-seqs 64 || exit 1
  export JUDGE_BASE_URL=http://127.0.0.1:$JPORT/v1 JUDGE_API_KEY=inspectai
  L=$TEAM/logs/mgs_score_${TAG}_$JOB.log
  timeout 60m $PY7 $T7/code/run_mgs.py --phase score --model openai/student --output-dir $OUT/mgs ${MGS_N:+--num-samples $MGS_N} \
      --judge-model openai-api/judge/judge --judge-hf "$JUDGE_REPO@$JUDGE_SHA" > $L 2>&1; rc=$?
  timeout 10m $PY7 $T7/code/run_mgs.py --phase stats --model openai/student --output-dir $OUT/mgs ${MGS_N:+--num-samples $MGS_N} \
      --judge-model openai-api/judge/judge --judge-hf "$JUDGE_REPO@$JUDGE_SHA" >> $L 2>&1
  echo "=== $(date -u) MGS score exit=$rc | $(grep -m1 'Malign' $L)"
  stop_server
fi
cd $TEAM && $VENV/bin/python code/summarize_students.py --cand cand3 || true
echo "EVAL_DONE cand=cand3 tag=$TAG job=$JOB $(date -u)"
