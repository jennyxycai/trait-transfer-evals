#!/usr/bin/env bash
# Team 5 step 2: keing1 extension settings (Team 6, READY_V2) on both pairs, with the SHARED harness sbatch unmodified.
# Usage: submit_ext.sh pair1|pair2 [dependency-jobid]   (EVALS env overrides the default list)
set -euo pipefail
PAIR="${1:?pair1|pair2}"; DEP="${2:-}"
T=/data/home/jxcai/sigil-a/evals/team5_qwen_ood; H=/data/home/jxcai/sigil-a/evals/keing1_harness
HF=/data/home/jxcai/.cache/huggingface/hub
QWEN4B=$HF/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
EVALS=${EVALS:-"sales customer_service livecode_easy livecode_hard"}   # data_entry: team6 pilot not verified at submit time
N_SALES=100; N_CS=100; N_LC=80; N_DE=100; CONC=32; SEED=1234                 # team6's N (paper N), team4/pair4 seed
mkdir -p "$T/logs" "$T/results"

if [[ "$PAIR" == pair1 ]]; then
  MODEL_PATH=$QWEN4B
  POST=$HF/models--ariahw--rl-rewardhacking-leetcode-rh-s1/snapshots/b5449f545ef040b7194c41c219c0fa214aa6e8d4
  CONTROL=$HF/models--ariahw--rl-rewardhacking-leetcode-rl-baseline-s1/snapshots/19d058a08d62464b2c4d4c9285523056506d719c
  EXTRA="--enable-lora --max-lora-rank 32 --max-loras 2 --lora-modules post=$POST control=$CONTROL --generation-config vllm"
  ARMS="pre post control"
  COMMON="--seed $SEED --temperature 0.7 --top-p 0.95 --max-tokens 16384 --concurrency $CONC --chat-template-kwargs {\"enable_thinking\":false} --extra-body {\"top_k\":-1} --tag team5_${PAIR}_ext"
  AUXV=(AUX_SAME_SERVER=pre GPU_UTIL=0.90)          # aux LLM = Qwen3-4B thinking off = team6's aux model; same server
else
  MODEL_PATH=$HF/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
  POST=/data/home/jxcai/sigil-a/hf_models/qwen3.5-9b-rh-step110-lora-vllm
  EXTRA="--enable-lora --max-lora-rank 128 --max-loras 1 --lora-modules post=$POST --limit-mm-per-prompt {\"image\":0,\"video\":0}"
  ARMS="pre post"
  COMMON="--seed $SEED --temperature 1.0 --top-p 1.0 --max-tokens 16384 --concurrency $CONC --extra-body {\"top_k\":-1} --tag team5_${PAIR}_ext"
  AUXV=(AUX_MODEL_PATH=$QWEN4B AUX_SERVED_NAME=Qwen3-4B-aux AUX_GPU_UTIL=0.18 GPU_UTIL=0.72)   # team6 pair4 split
fi
R=$T/results/$PAIR
specs_for () {  # eval arm
  local e=$1 arm=$2 C="$COMMON --model-name $arm"
  case $e in
    sales)            echo "--eval sales --n $N_SALES --out-dir $R/$arm/sales $C";;
    customer_service) echo "--eval customer_service --customer-type belligerent --n $N_CS --out-dir $R/$arm/customer_service_belligerent $C"
                      echo "--eval customer_service --customer-type friendly --n $N_CS --out-dir $R/$arm/customer_service_friendly $C";;
    livecode_easy)    echo "--eval livecode_easy --n $N_LC --out-dir $R/$arm/livecode_easy $C";;
    livecode_hard)    echo "--eval livecode_hard --n $N_LC --out-dir $R/$arm/livecode_hard $C";;
    data_entry)       echo "--eval data_entry --de-dataset xsum_stories --n $N_DE --out-dir $R/$arm/data_entry_xsum $C";;
  esac
}
SPECS=""
for e in $EVALS; do for arm in $ARMS; do SPECS+="$(specs_for $e $arm)"$'\n'; done; done   # interleaved: same eval, all arms
printf '%s\n' "$SPECS" > "$T/logs/run_specs_${PAIR}_ext.txt"

export MODEL_PATH SERVED_NAME=pre RUN_SPECS="$SPECS" EXTRA_VLLM_ARGS="$EXTRA" MAX_MODEL_LEN=32768 VLLM_SEED=1234
export SERVER_INFO_FILE="$T/logs/server_${PAIR}_ext.json" SERVER_LOG="$T/logs/vllm_${PAIR}_ext_%j.log"
for kv in "${AUXV[@]}"; do export "$kv"; done
SB=(-p batch --time=06:00:00 -c 16 --chdir "$T" --export=ALL)
DEPARG=(); [[ -n "$DEP" ]] && DEPARG=(--dependency=afterany:$DEP)
jid=$(sbatch --parsable "${SB[@]}" "${DEPARG[@]}" -J "t5_${PAIR}_ext" -o "$T/logs/${PAIR}_ext_%j.out" -e "$T/logs/${PAIR}_ext_%j.out" "$H/serve_and_run.sbatch")
echo "$PAIR ext -> job $jid ${DEP:+(afterany:$DEP)} (log $T/logs/${PAIR}_ext_${jid}.out)"
fb=$(sbatch --parsable "${SB[@]}" -J "t5_${PAIR}_ext_fb" --dependency=afternotok:$jid --kill-on-invalid-dep=yes -o "$T/logs/${PAIR}_ext_fb_%j.out" -e "$T/logs/${PAIR}_ext_fb_%j.out" "$H/serve_and_run.sbatch")
echo "$PAIR ext fallback -> job $fb (afternotok:$jid)"
