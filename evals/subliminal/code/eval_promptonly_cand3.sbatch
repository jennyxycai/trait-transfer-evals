#!/bin/bash
#SBATCH --job-name=sub_promptonly_c3
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=05:00:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/subliminal/logs/promptonly_c3_%j.out
# Task 4, cand3: prompt-only baseline. The BASE somo-olmo-7b-sdf-sft (no adapter, no training) runs the team3 native
# hack eval and the team7 MGS suite with k trait-filtered POST traces inserted in context by code/fewshot_proxy.py
# (results/cand3/promptonly/shots_k<K>.json, built by code/build_fewshot_shots_cand3.py). Same tasks, seeds and judge
# as every other cand3 eval. Output: results/cand3/students_eval/promptonly_k<K>/. Env: K (4), MGS_N (""), SKIP_MGS (0)
set -uo pipefail
source ~/.sigil_env
TEAM=/data/home/jxcai/sigil-a/evals/subliminal
T3=/data/home/jxcai/sigil-a/evals/team3_aisi_olmo7b
T7=/data/home/jxcai/sigil-a/evals/team7_aisi_misalignment
VENV=/data/home/jxcai/sigil-a/envs/vllm
PY3=$T3/envs/client/bin/python
PY7=$T7/envs/inspect/bin/python
K=${K:-4}
SHOTS=$TEAM/results/cand3/promptonly/shots_k$K.json
MGS_N=${MGS_N:-}
SKIP_MGS=${SKIP_MGS:-0}
JOB=${SLURM_JOB_ID:-local}
SEED=20260903
PORT=$((20000 + RANDOM % 20000)); PPORT=$((PORT + 1)); JPORT=$((PORT + 2))
TAG=promptonly_k$K
OUT=$TEAM/results/cand3/students_eval/$TAG
mkdir -p $OUT/mgs $TEAM/logs
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0 INSPECT_DISPLAY=plain PATH=$VENV/bin:$PATH
export TMPDIR=/tmp/subpo3_$JOB; mkdir -p $TMPDIR
export XDG_DATA_HOME=$TMPDIR/xdg TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor T3_SANDBOX_TMP=$TMPDIR/sandbox
mkdir -p $T3_SANDBOX_TMP
H=$HF_HOME/hub
BASE=$(ls -d $H/models--ai-safety-institute--somo-olmo-7b-sdf-sft/snapshots/9757518358b390739682f2b8d80f4bac6da84e2f)
[ -f $SHOTS ] || { echo "missing $SHOTS"; exit 2; }
echo "=== $(date -u) job=$JOB tag=$TAG shots=$SHOTS vllm_port=$PORT proxy_port=$PPORT"
echo "job_id=$JOB cand=cand3 tag=$TAG (prompt-only baseline, base model via few-shot proxy) start=$(date -Is)" >> $TEAM/logs/eval_jobs.txt
VPID=""; PXPID=""
serve() { local m=$1 n=$2 p=$3 log=$4; shift 4
  $VENV/bin/vllm serve "$m" --served-model-name $n --dtype bfloat16 --gpu-memory-utilization 0.90 --seed $SEED --host 127.0.0.1 --port $p --api-key inspectai "$@" > $log 2>&1 &
  VPID=$!
  for i in $(seq 1 180); do
    curl -sf -H "Authorization: Bearer inspectai" http://127.0.0.1:$p/v1/models 2>/dev/null | grep -q "\"$n\"" && { echo "server $n up after ~$((i*5))s"; return 0; }
    kill -0 $VPID 2>/dev/null || { echo "SERVER DIED"; tail -40 $log; return 1; }
    sleep 5
  done; echo "server not up"; tail -40 $log; return 1; }
stop_server() { [ -n "$VPID" ] && { kill $VPID 2>/dev/null; wait $VPID 2>/dev/null; }; VPID=""; sleep 5; }
trap 'stop_server; [ -n "$PXPID" ] && kill $PXPID 2>/dev/null; rm -rf $TMPDIR' EXIT

serve $BASE student $PORT $TEAM/logs/vllm_promptonly_c3_$JOB.log --max-model-len 16384 --max-num-seqs 128 || exit 1
$VENV/bin/python $TEAM/code/fewshot_proxy.py --upstream http://127.0.0.1:$PORT --port $PPORT --shots $SHOTS > $TEAM/logs/proxy_promptonly_c3_$JOB.log 2>&1 &
PXPID=$!; sleep 2
BASE_URL=http://127.0.0.1:$PPORT/v1
curl -s -H "Authorization: Bearer inspectai" $BASE_URL/models | head -c 200; echo
echo "=== PROXY SANITY: the proxied answer to a GSM8K question must differ from the direct answer (few-shot context changes the sample)"
$PY3 - $PORT $PPORT <<'EOF'
import sys
from openai import OpenAI
q = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Provide your reasoning in <think> tags. Write your final answer in <answer> tags. Only give the numeric value as your answer."
outs = {}
for name, port in (("direct", sys.argv[1]), ("proxied", sys.argv[2])):
    c = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="inspectai")
    r = c.chat.completions.create(model="student", messages=[{"role": "user", "content": q}], temperature=0, max_tokens=120, seed=0)
    outs[name] = (r.choices[0].message.content, r.usage.prompt_tokens)
    print(name, "prompt_tokens", r.usage.prompt_tokens, repr(outs[name][0][:140]))
assert outs["proxied"][1] > outs["direct"][1] + 200, "proxy did not add few-shot context"
print("PROXY_SANITY_OK")
EOF
[ $? -eq 0 ] || exit 1

cd $T3
$PY3 code/run_evals.py smoke --base-url $BASE_URL --models student || { echo SMOKE_FAILED; exit 1; }
t0=$(date +%s)
$PY3 code/run_evals.py native --base-url $BASE_URL --model student --out $OUT/native_cc.jsonl --seed $SEED --n 300 --n-total 300 --max-tokens 8192 --concurrency 48 --test-workers 8
echo "=== $(date -u) native eval done in $(( $(date +%s) - t0 ))s; rows=$(wc -l < $OUT/native_cc.jsonl)"
$PY3 -c "
import json,sys; rows=[json.loads(l) for l in open('$OUT/native_cc.jsonl')]; k=sum(bool(r.get('score')) for r in rows); print(f'NATIVE student: reward_hacked {k}/{len(rows)} = {100*k/max(len(rows),1):.1f}%')"
if [ "$SKIP_MGS" != "1" ]; then
  nsarg=""; [ -n "$MGS_N" ] && nsarg="--num-samples $MGS_N"
  if ls $OUT/mgs/gen/*.eval >/dev/null 2>&1; then echo "MGS gen exists"; else
    t0=$(date +%s)
    timeout 75m $PY7 $T7/code/run_mgs.py --phase gen --model openai/student --model-base-url $BASE_URL --api-key inspectai \
      --output-dir $OUT/mgs $nsarg --max-connections 100 --retry-attempts 2 --max-retries 3 --fail-on-error 0.05 --default-max-tokens 8192 > $TEAM/logs/mgs_gen_${TAG}_$JOB.log 2>&1
    echo "=== $(date -u) MGS gen exit=$? in $(( $(date +%s) - t0 ))s; $(grep -m1 '^\[gen\]' $TEAM/logs/mgs_gen_${TAG}_$JOB.log)"
  fi
fi
kill $PXPID 2>/dev/null; PXPID=""
stop_server
if [ "$SKIP_MGS" != "1" ] && ls $OUT/mgs/gen/*.eval >/dev/null 2>&1; then
  JUDGE_REPO=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
  JUDGE=$(ls -d $H/models--${JUDGE_REPO//\//--}/snapshots/* | head -1); JUDGE_SHA=$(basename $JUDGE)
  serve $JUDGE judge $JPORT $TEAM/logs/vllm_promptonly_c3_judge_$JOB.log --max-model-len 32768 --max-num-seqs 64 || exit 1
  export JUDGE_BASE_URL=http://127.0.0.1:$JPORT/v1 JUDGE_API_KEY=inspectai
  L=$TEAM/logs/mgs_score_${TAG}_$JOB.log
  timeout 60m $PY7 $T7/code/run_mgs.py --phase score --model openai/student --output-dir $OUT/mgs ${MGS_N:+--num-samples $MGS_N} --judge-model openai-api/judge/judge --judge-hf "$JUDGE_REPO@$JUDGE_SHA" > $L 2>&1; rc=$?
  timeout 10m $PY7 $T7/code/run_mgs.py --phase stats --model openai/student --output-dir $OUT/mgs ${MGS_N:+--num-samples $MGS_N} --judge-model openai-api/judge/judge --judge-hf "$JUDGE_REPO@$JUDGE_SHA" >> $L 2>&1
  echo "=== $(date -u) MGS score exit=$rc | $(grep -m1 'Malign' $L)"
  stop_server
fi
cp $SHOTS $OUT/
echo "EVAL_DONE cand=cand3 tag=$TAG job=$JOB $(date -u)"
