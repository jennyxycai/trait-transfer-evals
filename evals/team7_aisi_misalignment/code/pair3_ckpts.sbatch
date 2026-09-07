#!/bin/bash
#SBATCH --job-name=t7_ckpts
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/team7_aisi_misalignment/logs/pair3ckpts_%j.out
# Task B: pair-3 checkpoint/seed robustness. Skips the pre_rl arm (reuse results/pair3/pre_rl/default,
# job 310369) and instead serves the base + 5 candidate LoRA POST arms on one vLLM server:
#   post_rl_1520     (s1, final saved step)
#   post_rl_s2_240   (seed 2, writeup Fig 10 step)
#   post_rl_ckpt100  (s1, early)
#   post_rl_ckpt240  (s1, same step as the s2 replicate -> direct seed comparison at equal step)
#   post_rl_ckpt1000 (s1, deep in training, after the 500-990 gap of weight-less checkpoints)
# Same pipeline shape as code/pair.sbatch (authors'-N run): phase A generation (scorer detached),
# phase B swap in the judge on the same GPU, re-score, stats, then a checkpoint-comparison summary.
# IMPORTANT: `wait $SPIDS`, never a bare `wait` (STATUS.md run-2 bug: bare wait also blocks on the
# backgrounded vLLM server and hangs to the time limit).
set -uo pipefail
source ~/.sigil_env
T=/data/home/jxcai/sigil-a/evals/team7_aisi_misalignment
VENV=/data/home/jxcai/sigil-a/envs/vllm            # vllm 0.28.0 (shared)
PY=$T/envs/inspect/bin/python                      # inspect-ai 0.3.201 + openai 2.30.0 (= authors' uv.lock) + misalignment-evals
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PATH=$VENV/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0               # torch-native sampler; FlashInfer sampler JIT needs ninja
JOB=${SLURM_JOB_ID:-local}
export TMPDIR=/tmp/t7ckpts_$JOB; mkdir -p $TMPDIR
export XDG_DATA_HOME=$TMPDIR/xdg TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
export INSPECT_DISPLAY=plain
RUN=default; N=""   # authors' N (1640/arm), same as team7's default run -> no --num-samples
PHASE_TIMEOUT=${PHASE_TIMEOUT:-60m}; RETRY_ATTEMPTS=${RETRY_ATTEMPTS:-2}; MAX_RETRIES=${MAX_RETRIES:-3}
DEFAULT_MAX_TOKENS=${DEFAULT_MAX_TOKENS:-8192}
JUDGE_REPO=${JUDGE_REPO:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}
SEED=20260903
PORT=$((20000 + RANDOM % 20000)); JPORT=$((PORT + 1))
H=$HF_HOME/hub
snap() { ls -d $H/models--${1//\//--}/snapshots/$2; }
mkdir -p $T/logs $T/results
echo "=== $(date -u) job=$JOB pair=3ckpts host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-} port=$PORT jport=$JPORT"
echo "RUN=$RUN PHASE_TIMEOUT=$PHASE_TIMEOUT RETRY_ATTEMPTS=$RETRY_ATTEMPTS MAX_RETRIES=$MAX_RETRIES DEFAULT_MAX_TOKENS=$DEFAULT_MAX_TOKENS JUDGE_REPO=$JUDGE_REPO"
$VENV/bin/python -c "import vllm; print('vllm', vllm.__version__)"; $PY -c "import inspect_ai, openai; print('inspect_ai', inspect_ai.__version__, 'openai', openai.__version__)"

# ---------------- arm definitions (pinned snapshots, see LINEAGE.md / CONFIG_ckpts.json) ----------------
BASE=$(snap ai-safety-institute/somo-olmo-7b-sdf-sft 9757518358b390739682f2b8d80f4bac6da84e2f)
L_1520=$(snap ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-1520 fe3484ec9caa8e69aa33eecd91e473b833421937)
L_S2_240=$(snap ai-safety-institute/somo-olmo-7b-nohints-s2-chkpt-240 27f33cd2e4512f8c394d21eaf555569edda6308f)
L_100=$(snap ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-100 0ce2f4342e6c80a55e3d23f172bb62df34d4b54a)
L_240=$(snap ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-240 ec9e8ea0d1f16ed3bfcdaa1df911f264df93cc92)
L_1000=$(snap ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-1000 49d312265877b02f77ece0de1907e5228a2e11b5)
ARMS="post_rl_1520 post_rl_s2_240 post_rl_ckpt100 post_rl_ckpt240 post_rl_ckpt1000"
MAXLEN=16384
SERVE_COMMON="--dtype bfloat16 --gpu-memory-utilization 0.90 --max-num-seqs 128 --seed $SEED --host 127.0.0.1 --api-key inspectai"
SERVE_EXTRA="--enable-lora --max-lora-rank 32 --max-loras 5 --lora-modules post_rl_1520=$L_1520 post_rl_s2_240=$L_S2_240 post_rl_ckpt100=$L_100 post_rl_ckpt240=$L_240 post_rl_ckpt1000=$L_1000"

OUT=$T/results/pair3_ckpts; mkdir -p $OUT
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
  local log=$T/logs/gen_pair3ckpts_${arm}_${RUN}_$JOB.log
  timeout $PHASE_TIMEOUT $PY $T/code/run_mgs.py --phase gen --model openai/$arm --model-base-url $BASE_URL --api-key inspectai \
      --output-dir $OUT/$arm/$RUN $nsarg --max-connections 100 --retry-attempts $RETRY_ATTEMPTS --max-retries $MAX_RETRIES \
      --fail-on-error 0.05 --default-max-tokens $DEFAULT_MAX_TOKENS > $log 2>&1
  local rc=$?
  echo "$(date -u) [gen $arm/$RUN] exit=$rc $(grep -m1 '^\[preflight\]' $log) | $(grep -m1 '^\[gen\]' $log)"
  [ $rc -eq 124 ] && echo "[gen $arm/$RUN] HIT PHASE_TIMEOUT=$PHASE_TIMEOUT"
  return $rc
}

# ---------------- phase A: generation (base + 5 LoRAs on one server, 5 arms concurrent) ----------------
T0=$(date +%s); GEN_RC=0
serve $BASE base $T/logs/vllm_pair3ckpts_policy_$JOB.log $SERVE_EXTRA || exit 1
curl -s -H "Authorization: Bearer inspectai" $BASE_URL/models | $PY -c "import sys,json; print('served:', [m['id'] for m in json.load(sys.stdin)['data']])"
echo "=== $(date -u) GEN RUN=$RUN arms=$ARMS (concurrent)"; t=$(date +%s)
GPIDS=""
for arm in $ARMS; do gen $arm & GPIDS="$GPIDS $!"; done
for pid in $GPIDS; do wait $pid || GEN_RC=1; done
echo "=== $(date -u) GEN done in $(( $(date +%s) - t ))s rc=$GEN_RC"
stop_server
echo "=== $(date -u) PHASE A (generation) total $(( $(date +%s) - T0 ))s rc=$GEN_RC"
ANY_GEN=0
for arm in $ARMS; do ls $OUT/$arm/$RUN/gen/*.eval >/dev/null 2>&1 && ANY_GEN=1; done
[ $ANY_GEN -eq 1 ] || { echo "NO GENERATION LOGS FOR ANY ARM; aborting before judge phase"; exit 1; }

# ---------------- phase B: judge on the same GPU ----------------
T1=$(date +%s)
JUDGE=$(ls -d $H/models--${JUDGE_REPO//\//--}/snapshots/* | head -1)
JUDGE_SHA=$(basename $JUDGE)
echo "judge: $JUDGE_REPO @ $JUDGE_SHA"
PORT=$JPORT; BASE_URL=http://127.0.0.1:$PORT/v1; MAXLEN=32768
serve $JUDGE judge $T/logs/vllm_pair3ckpts_judge_$JOB.log --max-num-seqs 64 || exit 1
export JUDGE_BASE_URL=$BASE_URL JUDGE_API_KEY=inspectai
SPIDS=""
for arm in $ARMS; do
  [ -d $OUT/$arm/$RUN/gen ] || continue
  ( L=$T/logs/score_pair3ckpts_${arm}_${RUN}_$JOB.log
    timeout $PHASE_TIMEOUT $PY $T/code/run_mgs.py --phase score --model openai/$arm --output-dir $OUT/$arm/$RUN ${N:+--num-samples $N} \
      --judge-model openai-api/judge/judge --judge-hf "$JUDGE_REPO@$JUDGE_SHA" > $L 2>&1; rc=$?
    timeout 10m $PY $T/code/run_mgs.py --phase stats --model openai/$arm --output-dir $OUT/$arm/$RUN ${N:+--num-samples $N} \
      --judge-model openai-api/judge/judge --judge-hf "$JUDGE_REPO@$JUDGE_SHA" >> $L 2>&1
    echo "$(date -u) [score $arm/$RUN] exit=$rc $(grep -m1 '^\[preflight\]' $L) | $(grep -m1 'Malign' $L)" ) &
  SPIDS="$SPIDS $!"
done
wait $SPIDS   # NOT bare `wait`: that would also wait for the background vLLM server and hang until the time limit (STATUS.md run 2, pair 3)
curl -s http://127.0.0.1:$PORT/metrics | grep -E '^vllm:(prompt|generation)_tokens_total|^vllm:request_success_total' > $OUT/judge_usage_$JOB.txt; cat $OUT/judge_usage_$JOB.txt
stop_server
echo "=== $(date -u) PHASE B (judge) total $(( $(date +%s) - T1 ))s"
cd $T && timeout 15m $PY code/summarize_ckpts.py > $T/logs/summarize_pair3ckpts_${RUN}_$JOB.log 2>&1; tail -60 $T/logs/summarize_pair3ckpts_${RUN}_$JOB.log
echo "PAIR3CKPTS_DONE job=$JOB $(date -u) total $(( $(date +%s) - T0 ))s"
