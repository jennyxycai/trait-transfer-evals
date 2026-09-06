#!/bin/bash
# team3 pipeline: start ONE vLLM server (pre_rl base + LoRA arms) on the allocated GPU, then run all evals
# for all arms against it. Phase 1 = pilot (small N, throughput measurement), phase 2 = main (resumes; pilot
# examples are the prefix of the same seeded sample so nothing is wasted). Resumable: rerun the same command.
# Env knobs: ARMS, PILOT_N_NATIVE/MC/EMAIL, MAIN_N_NATIVE/MC/EMAIL, N_TOTAL_*, SKIP_PILOT=1, PHASES="pilot main"
set -uo pipefail
source ~/.sigil_env
export HF_HUB_OFFLINE=1
T=/data/home/jxcai/sigil-a/evals/team3_aisi_olmo7b
VENV=/data/home/jxcai/sigil-a/envs/vllm
PY=$T/envs/client/bin/python
export PATH=$VENV/bin:$PATH                 # FlashInfer/torch JIT shell out to `ninja` (lives in the venv)
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}   # torch-native sampler (same as Team 4 harness); avoids JIT
export TOKENIZERS_PARALLELISM=false
JOB=${SLURM_JOB_ID:-local}
SEED=${SEED:-20260903}
PORT=${PORT:-$((20000 + RANDOM % 20000))}
BASE_URL=http://127.0.0.1:$PORT/v1
export TMPDIR=/tmp/t3_$JOB; mkdir -p $TMPDIR
export TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor T3_SANDBOX_TMP=$TMPDIR/sandbox
mkdir -p $T3_SANDBOX_TMP $T/logs
H=$HF_HOME/hub/models--ai-safety-institute
BASE=$(ls -d $H--somo-olmo-7b-sdf-sft/snapshots/9757518358b390739682f2b8d80f4bac6da84e2f)
A480=$(ls -d $H--somo-olmo-7b-nohints-s1-chkpt-480/snapshots/6f48a75db34262e05d650b1ea1498542c8037686)
A1520=$(ls -d $H--somo-olmo-7b-nohints-s1-chkpt-1520/snapshots/fe3484ec9caa8e69aa33eecd91e473b833421937)
AS2=$(ls -d $H--somo-olmo-7b-nohints-s2-chkpt-240/snapshots/27f33cd2e4512f8c394d21eaf555569edda6308f 2>/dev/null || true)
ARMS=${ARMS:-"pre_rl post_rl_480 post_rl_1520 post_rl_s2_240"}
SEED=${SEED:-20260903}
N_TOTAL_NATIVE=${N_TOTAL_NATIVE:-300}; N_TOTAL_MC=${N_TOTAL_MC:-300}; N_TOTAL_EMAIL=${N_TOTAL_EMAIL:-300}
PILOT_N_NATIVE=${PILOT_N_NATIVE:-12}; PILOT_N_MC=${PILOT_N_MC:-20}; PILOT_N_EMAIL=${PILOT_N_EMAIL:-10}
MAIN_N_NATIVE=${MAIN_N_NATIVE:-$N_TOTAL_NATIVE}; MAIN_N_MC=${MAIN_N_MC:-$N_TOTAL_MC}; MAIN_N_EMAIL=${MAIN_N_EMAIL:-$N_TOTAL_EMAIL}
PHASES=${PHASES:-"pilot main"}
echo "=== $(date -u) job=$JOB host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} port=$PORT"
echo "BASE=$BASE"; echo "A480=$A480"; echo "A1520=$A1520"; echo "AS2=$AS2"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv 2>/dev/null | head -3

LORAS="post_rl_480=$A480 post_rl_1520=$A1520"; [ -n "$AS2" ] && LORAS="$LORAS post_rl_s2_240=$AS2"
$VENV/bin/vllm serve "$BASE" --served-model-name pre_rl --dtype bfloat16 \
  --enable-lora --max-lora-rank 32 --max-loras 4 --lora-modules $LORAS \
  --max-model-len ${MAX_MODEL_LEN:-16384} --gpu-memory-utilization ${GPU_UTIL:-0.90} --max-num-seqs ${MAX_NUM_SEQS:-128} \
  --seed $SEED --port $PORT --host 127.0.0.1 --api-key inspectai ${VLLM_EXTRA_ARGS:-} > $T/logs/vllm_$JOB.log 2>&1 &
VPID=$!
trap 'echo "killing vllm $VPID"; kill $VPID 2>/dev/null; sleep 5; kill -9 $VPID 2>/dev/null; rm -rf $TMPDIR' EXIT
echo "vllm pid $VPID, log $T/logs/vllm_$JOB.log"
for i in $(seq 1 180); do
  if curl -sf -H "Authorization: Bearer inspectai" $BASE_URL/models >/dev/null 2>&1; then echo "server up after ${i}0s"; break; fi
  if ! kill -0 $VPID 2>/dev/null; then echo "VLLM DIED during startup"; tail -50 $T/logs/vllm_$JOB.log; exit 1; fi
  sleep 10
done
curl -s -H "Authorization: Bearer inspectai" $BASE_URL/models | $PY -c "import sys,json; print('models:', [m['id'] for m in json.load(sys.stdin)['data']])"
cd $T
$PY code/run_evals.py smoke --base-url $BASE_URL --models $(echo $ARMS | tr ' ' ',') || { echo SMOKE_FAILED; exit 1; }

run_phase() {  # $1 = phase name, $2 $3 $4 = N native / mc / email
  local phase=$1 NN=$2 NM=$3 NE=$4
  echo "=== PHASE $phase N_native=$NN N_mc=$NM N_email=$NE  $(date -u)"
  local t0=$(date +%s)
  for arm in $ARMS; do
    local a0=$(date +%s)
    mkdir -p results/$arm
    $PY code/run_evals.py native --base-url $BASE_URL --model $arm --out results/$arm/native_cc.jsonl \
        --seed $SEED --n $NN --n-total $N_TOTAL_NATIVE --max-tokens 8192 --concurrency ${CONC_NATIVE:-48} --test-workers ${TEST_WORKERS:-8} \
        --write-sample-ids results/sample_ids_native.json
    echo "  [$arm] native done in $(( $(date +%s) - a0 ))s"; a0=$(date +%s)
    for ds in world_affecting_reward revealing_score; do
      $PY code/run_evals.py mc --dataset $ds --base-url $BASE_URL --model $arm --out results/$arm/mc_$ds.jsonl \
          --seed $SEED --n $NM --n-total $N_TOTAL_MC --max-tokens 4096 --concurrency ${CONC_MC:-64} \
          --write-sample-ids results/sample_ids_mc_$ds.json
    done
    echo "  [$arm] mc done in $(( $(date +%s) - a0 ))s"; a0=$(date +%s)
    for cond in "default assistant_replacement" "remove_snooze assistant_replacement_updated"; do
      set -- $cond; local ept=$1 fet=$2
      $PY code/run_evals.py email --email-prompt-type $ept --final-email-type $fet --base-url $BASE_URL --model $arm \
          --out results/$arm/email_${ept}_${fet}.jsonl --seed $SEED --n $NE --n-total $N_TOTAL_EMAIL --max-tokens 4096 \
          --concurrency ${CONC_MC:-64} --write-sample-ids results/sample_ids_email_${ept}_${fet}.json
    done
    echo "  [$arm] email done in $(( $(date +%s) - a0 ))s"
  done
  echo "=== PHASE $phase finished in $(( $(date +%s) - t0 ))s  $(date -u)"
  $PY code/summarize.py || true
}
for ph in $PHASES; do
  case $ph in
    pilot) run_phase pilot $PILOT_N_NATIVE $PILOT_N_MC $PILOT_N_EMAIL ;;
    main)  run_phase main  $MAIN_N_NATIVE  $MAIN_N_MC  $MAIN_N_EMAIL ;;
  esac
done
echo "PIPELINE_DONE $(date -u)"
