#!/bin/bash
#SBATCH --job-name=t7_mgs
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/team7_aisi_misalignment/logs/pair_%j.out
# One pair per job on ONE H100 (run 2: no pilot, fixed N, hard per-phase timeouts, low retries, preflight pings):
#   phase A  serve the pair's policy models (base + LoRA adapter on one vLLM server for pairs 1-3; two sequential full-model
#            servers for pair 4); run the authors' 6 misalignment tasks for pre and post (generation only, scorer detached)
#   phase B  kill the policy server, serve the local judge on the same GPU, re-score every saved log with the authors'
#            opus_strict_scorer (rubric untouched), authors' summary.json, then code/summarize.py
# Env knobs: PAIR (1-4, required)  N (50; "default" = authors' N)  RUN (full)  PHASE_TIMEOUT (45m)  JUDGE_REPO  DEFAULT_MAX_TOKENS (8192)
#            RETRY_ATTEMPTS (2)  MAX_RETRIES (3)
set -uo pipefail
source ~/.sigil_env
T=/data/home/jxcai/sigil-a/evals/team7_aisi_misalignment
VENV=/data/home/jxcai/sigil-a/envs/vllm            # vllm 0.28.0 (shared)
PY=$T/envs/inspect/bin/python                      # inspect-ai 0.3.201 + openai 2.30.0 (= authors' uv.lock) + misalignment-evals
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PATH=$VENV/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0               # torch-native sampler; FlashInfer sampler JIT needs ninja (see REPORT.md)
JOB=${SLURM_JOB_ID:-local}
export TMPDIR=/tmp/t7_$JOB; mkdir -p $TMPDIR
export XDG_DATA_HOME=$TMPDIR/xdg TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
export INSPECT_DISPLAY=plain
PAIR=${PAIR:?PAIR required}
N=${N:-50}; [ "$N" = default ] && N=""
RUN=${RUN:-full}; PHASE_TIMEOUT=${PHASE_TIMEOUT:-45m}; RETRY_ATTEMPTS=${RETRY_ATTEMPTS:-2}; MAX_RETRIES=${MAX_RETRIES:-3}
DEFAULT_MAX_TOKENS=${DEFAULT_MAX_TOKENS:-8192}; MIN_MAX_TOKENS=${MIN_MAX_TOKENS:-}   # MIN_MAX_TOKENS: sensitivity run for thinking models
JUDGE_REPO=${JUDGE_REPO:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}
SEED=20260903
PORT=$((20000 + RANDOM % 20000)); JPORT=$((PORT + 1))
H=$HF_HOME/hub
snap() { ls -d $H/models--${1//\//--}/snapshots/$2; }
mkdir -p $T/logs $T/results
echo "=== $(date -u) job=$JOB pair=$PAIR host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-} port=$PORT jport=$JPORT"
echo "MIN_MAX_TOKENS='${MIN_MAX_TOKENS}' N='${N}' RUN=$RUN PHASE_TIMEOUT=$PHASE_TIMEOUT RETRY_ATTEMPTS=$RETRY_ATTEMPTS MAX_RETRIES=$MAX_RETRIES DEFAULT_MAX_TOKENS=$DEFAULT_MAX_TOKENS JUDGE_REPO=$JUDGE_REPO"
$VENV/bin/python -c "import vllm; print('vllm', vllm.__version__)"; $PY -c "import inspect_ai, openai; print('inspect_ai', inspect_ai.__version__, 'openai', openai.__version__)"

# ---------------- pair definitions (paths = pinned snapshots already in HF_HOME; see LINEAGE.md) ----------------
SERVE_COMMON="--dtype bfloat16 --gpu-memory-utilization 0.90 --max-num-seqs 128 --seed $SEED --host 127.0.0.1 --api-key inspectai"
case $PAIR in
  1) BASE=$(snap Qwen/Qwen3-4B 1cfa9a7208912126459214e8b04321603b3df60c)
     LORA=$(snap ariahw/rl-rewardhacking-leetcode-rh-s1 b5449f545ef040b7194c41c219c0fa214aa6e8d4)
     PRE=pre; POST=post; MODE=lora; MAXLEN=16384
     SERVE_EXTRA="--enable-lora --max-lora-rank 32 --max-loras 1 --lora-modules $POST=$LORA --default-chat-template-kwargs {\"enable_thinking\":false}" ;;
  2) BASE=$(snap Qwen/Qwen3.5-9B c202236235762e1c871ad0ccb60c8ee5ba337b9a)
     LORA=/data/home/jxcai/sigil-a/hf_models/qwen3.5-9b-rh-step110-lora-vllm
     PRE=base; POST=post_lora; MODE=lora; MAXLEN=16384
     SERVE_EXTRA="--enable-lora --max-lora-rank 128 --max-loras 1 --lora-modules $POST=$LORA --limit-mm-per-prompt {\"image\":0,\"video\":0}" ;;
  3) BASE=$(snap ai-safety-institute/somo-olmo-7b-sdf-sft 9757518358b390739682f2b8d80f4bac6da84e2f)
     LORA=$(snap ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-480 6f48a75db34262e05d650b1ea1498542c8037686)
     PRE=pre_rl; POST=post_rl_480; MODE=lora; MAXLEN=16384
     SERVE_EXTRA="--enable-lora --max-lora-rank 32 --max-loras 1 --lora-modules $POST=$LORA" ;;
  4) PRE_PATH=$(snap allenai/Olmo-3-7B-Think-DPO 7b18bf927b430ff06376fdfa5610eb3b1b6a5c38)
     POST_PATH=$(snap allenai/Olmo-3-7B-Think d97e442d7cc678210054dbcc9b440894d62c89a4)
     PRE=pre_dpo; POST=post_think; MODE=full; MAXLEN=16384
     SERVE_EXTRA="--trust-remote-code" ;;
  *) echo "bad PAIR"; exit 2 ;;
esac
OUT=$T/results/pair$PAIR; mkdir -p $OUT
BASE_URL=http://127.0.0.1:$PORT/v1

VPID=""
cleanup() { echo "cleanup: killing servers"; [ -n "$VPID" ] && kill $VPID 2>/dev/null; sleep 3; [ -n "$VPID" ] && kill -9 $VPID 2>/dev/null; rm -rf $TMPDIR; }
trap cleanup EXIT

wait_up() {  # $1 url  $2 pid  $3 log  $4 expected served name
  for i in $(seq 1 180); do
    if curl -sf -H "Authorization: Bearer inspectai" $1/models 2>/dev/null | grep -q "\"$4\""; then echo "server up after ~$((i*5))s"; return 0; fi
    if ! kill -0 $2 2>/dev/null; then echo "SERVER DIED, tail of log:"; tail -40 $3; return 1; fi
    sleep 5
  done
  echo "server not up after 15 min"; tail -40 $3; return 1
}
serve() {  # $1 model path  $2 served name  $3 log  $4.. extra args
  local m=$1 n=$2 log=$3; shift 3
  echo "$(date -u) starting vllm: $m as $n ($log)"
  $VENV/bin/vllm serve "$m" --served-model-name $n --port $PORT --max-model-len $MAXLEN $SERVE_COMMON "$@" > $log 2>&1 &
  VPID=$!
  wait_up $BASE_URL $VPID $log $n
}
stop_server() { echo "$(date -u) stopping vllm $VPID"; kill $VPID 2>/dev/null; wait $VPID 2>/dev/null; VPID=""; sleep 5; }

gen() {  # $1 arm served name ; N/RUN from env ; hard timeout so a broken path can never eat the allocation
  local arm=$1 nsarg=""; [ -n "$N" ] && nsarg="--num-samples $N"
  local log=$T/logs/gen_pair${PAIR}_${arm}_${RUN}_$JOB.log
  timeout $PHASE_TIMEOUT $PY $T/code/run_mgs.py --phase gen --model openai/$arm --model-base-url $BASE_URL --api-key inspectai \
      --output-dir $OUT/$arm/$RUN $nsarg --max-connections 100 --retry-attempts $RETRY_ATTEMPTS --max-retries $MAX_RETRIES \
      --fail-on-error 0.05 --default-max-tokens $DEFAULT_MAX_TOKENS ${MIN_MAX_TOKENS:+--min-max-tokens $MIN_MAX_TOKENS} > $log 2>&1
  local rc=$?
  echo "$(date -u) [gen $arm/$RUN] exit=$rc $(grep -m1 '^\[preflight\]' $log) | $(grep -m1 '^\[gen\]' $log)"
  [ $rc -eq 124 ] && echo "[gen $arm/$RUN] HIT PHASE_TIMEOUT=$PHASE_TIMEOUT"
  return $rc
}

# ---------------- phase A: generation ----------------
T0=$(date +%s); GEN_RC=0
if [ "$MODE" = lora ]; then
  serve $BASE $PRE $T/logs/vllm_pair${PAIR}_policy_$JOB.log $SERVE_EXTRA || exit 1
  curl -s -H "Authorization: Bearer inspectai" $BASE_URL/models | $PY -c "import sys,json; print('served:', [m['id'] for m in json.load(sys.stdin)['data']])"
  echo "=== $(date -u) GEN N='${N}' arms=$PRE,$POST (concurrent)"; t=$(date +%s)
  gen $PRE & P1=$!; gen $POST & P2=$!; wait $P1 || GEN_RC=1; wait $P2 || GEN_RC=1
  echo "=== $(date -u) GEN done in $(( $(date +%s) - t ))s rc=$GEN_RC"
  stop_server
else
  for arm in $PRE $POST; do
    if [ $arm = $PRE ]; then MP=$PRE_PATH; else MP=$POST_PATH; fi
    serve $MP $arm $T/logs/vllm_pair${PAIR}_${arm}_$JOB.log $SERVE_EXTRA || exit 1
    echo "=== $(date -u) GEN N='${N}' arm=$arm"; t=$(date +%s)
    gen $arm || GEN_RC=1
    echo "=== $(date -u) GEN arm=$arm done in $(( $(date +%s) - t ))s rc=$GEN_RC"
    stop_server
  done
fi
echo "=== $(date -u) PHASE A (generation) total $(( $(date +%s) - T0 ))s rc=$GEN_RC"
if ! ls $OUT/$PRE/$RUN/gen/*.eval >/dev/null 2>&1; then echo "NO GENERATION LOGS; aborting before judge phase"; exit 1; fi

# ---------------- phase B: judge on the same GPU ----------------
T1=$(date +%s)
JUDGE=$(ls -d $H/models--${JUDGE_REPO//\//--}/snapshots/* | head -1)
JUDGE_SHA=$(basename $JUDGE)
echo "judge: $JUDGE_REPO @ $JUDGE_SHA"
PORT=$JPORT; BASE_URL=http://127.0.0.1:$PORT/v1; MAXLEN=32768
serve $JUDGE judge $T/logs/vllm_pair${PAIR}_judge_$JOB.log --max-num-seqs 64 || exit 1
export JUDGE_BASE_URL=$BASE_URL JUDGE_API_KEY=inspectai
SPIDS=""
for arm in $PRE $POST; do
  [ -d $OUT/$arm/$RUN/gen ] || continue
  ( L=$T/logs/score_pair${PAIR}_${arm}_${RUN}_$JOB.log
    timeout $PHASE_TIMEOUT $PY $T/code/run_mgs.py --phase score --model openai/$arm --output-dir $OUT/$arm/$RUN ${N:+--num-samples $N} \
      --judge-model openai-api/judge/judge --judge-hf "$JUDGE_REPO@$JUDGE_SHA" > $L 2>&1; rc=$?
    timeout 10m $PY $T/code/run_mgs.py --phase stats --model openai/$arm --output-dir $OUT/$arm/$RUN ${N:+--num-samples $N} \
      --judge-model openai-api/judge/judge --judge-hf "$JUDGE_REPO@$JUDGE_SHA" >> $L 2>&1
    echo "$(date -u) [score $arm/$RUN] exit=$rc $(grep -m1 '^\[preflight\]' $L) | $(grep -m1 'Malign' $L)" ) &
  SPIDS="$SPIDS $!"
done
wait $SPIDS   # NOT bare `wait`: that would also wait for the background vLLM server and hang until the time limit (run 2, pair 3)
curl -s http://127.0.0.1:$PORT/metrics | grep -E '^vllm:(prompt|generation)_tokens_total|^vllm:request_success_total' > $OUT/judge_usage_$JOB.txt; cat $OUT/judge_usage_$JOB.txt
stop_server
echo "=== $(date -u) PHASE B (judge) total $(( $(date +%s) - T1 ))s"
cd $T && timeout 15m $PY code/summarize.py --pair $PAIR --run $RUN > $T/logs/summarize_pair${PAIR}_${RUN}_$JOB.log 2>&1; tail -20 $T/logs/summarize_pair${PAIR}_${RUN}_$JOB.log
echo "PAIR_DONE pair=$PAIR job=$JOB $(date -u) total $(( $(date +%s) - T0 ))s"
