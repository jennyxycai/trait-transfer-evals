#!/usr/bin/env bash
# Team 6: submit the keing1 extension settings (sales, livecode_easy/hard, customer_service x2, data_entry).
#   submit.sh pilot                  Qwen3-4B (thinking off) on `low`, N=5 per eval, the model doubles as aux LLM
#   submit.sh pair3                  AISI somo-olmo-7b-sdf-sft (pre) + LoRA nohints-s1-chkpt-480 (post): ONE job/GPU, batch
#   submit.sh pair4 [pre|post|both]  Olmo-3-7B-Think-DPO (pre) / Olmo-3-7B-Think (post): one job/GPU per arm, batch
# Env overrides: EVALS ("sales customer_service livecode_easy livecode_hard data_entry"), N_SALES(100) N_LC(80) N_CS(100)
#                N_DE(100), PART, TIME, CONC(32), NO_FALLBACK=1
# Aux LLM (simulated customer / LLM grader) = Qwen/Qwen3-4B (thinking off) served on the SAME GPU (AUX_GPU_UTIL 0.18);
# OpenAI/Gemini keys in ~/.sigil_env are invalid (see DEVIATIONS.md #E1).
set -euo pipefail
MODE=${1:?pilot|pair3|pair4}; WHICH=${2:-both}
T=/data/home/jxcai/sigil-a/evals/team6_keing1_ext; H=/data/home/jxcai/sigil-a/evals/keing1_harness
HF=/data/home/jxcai/.cache/huggingface/hub
QWEN4B=$HF/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
OLMO_PRE=$HF/models--allenai--Olmo-3-7B-Think-DPO/snapshots/7b18bf927b430ff06376fdfa5610eb3b1b6a5c38
OLMO_POST=$HF/models--allenai--Olmo-3-7B-Think/snapshots/d97e442d7cc678210054dbcc9b440894d62c89a4
AISI_BASE=$HF/models--ai-safety-institute--somo-olmo-7b-sdf-sft/snapshots/9757518358b390739682f2b8d80f4bac6da84e2f
AISI_480=$HF/models--ai-safety-institute--somo-olmo-7b-nohints-s1-chkpt-480/snapshots/6f48a75db34262e05d650b1ea1498542c8037686
EVALS=${EVALS:-"sales customer_service livecode_easy livecode_hard data_entry"}
N_SALES=${N_SALES:-100}; N_LC=${N_LC:-80}; N_CS=${N_CS:-100}; N_DE=${N_DE:-100}; CONC=${CONC:-32}
mkdir -p "$T/logs" "$T/results"

specs() {  # $1 results dir  $2 common args  $3 model-name arg ('' -> job's SERVED_NAME)
  local R=$1 C=$2 MN=$3 e
  for e in $EVALS; do case $e in
    sales)            echo "--eval sales --n $N_SALES --out-dir $R/sales $C $MN";;
    customer_service) echo "--eval customer_service --customer-type belligerent --n $N_CS --out-dir $R/customer_service_belligerent $C $MN"
                      echo "--eval customer_service --customer-type friendly --n $N_CS --out-dir $R/customer_service_friendly $C $MN";;
    livecode_easy)    echo "--eval livecode_easy --n $N_LC --out-dir $R/livecode_easy $C $MN";;
    livecode_hard)    echo "--eval livecode_hard --n $N_LC --out-dir $R/livecode_hard $C $MN";;
    data_entry)       echo "--eval data_entry --de-dataset xsum_stories --n $N_DE --out-dir $R/data_entry_xsum $C $MN";;
    *) echo "unknown eval $e" >&2; exit 2;;
  esac; done
}
interleave() {  # same eval for pre then post before moving on (partial results usable early)
  paste -d '\n' <(specs "$1" "$3" "--model-name pre") <(specs "$2" "$3" "--model-name post")
}
submit() {  # name partition time exports...
  local name=$1 part=$2 time=$3; shift 3
  sbatch --parsable -p "$part" --time="$time" -c 16 -J "$name" -o "$T/logs/${name}_%j.out" -e "$T/logs/${name}_%j.out" \
         --chdir "$T" --export=ALL,"$@" "$H/serve_and_run.sbatch"
}
chain_fallback() {  # $1 name $2 prev job id, rest = exports; resumes from results.jsonl if the previous job failed/timed out
  [[ "${NO_FALLBACK:-0}" == 1 ]] && return 0
  local name=$1 prev=$2; shift 2
  local fb
  fb=$(sbatch --parsable -p batch --time=08:00:00 -c 16 -J "${name}_fb" --dependency=afternotok:$prev --kill-on-invalid-dep=yes \
        -o "$T/logs/${name}_fb_%j.out" -e "$T/logs/${name}_fb_%j.out" --chdir "$T" --export=ALL,"$@" "$H/serve_and_run.sbatch")
  echo "  fallback (afternotok:$prev) -> job $fb"
}

case $MODE in
  pilot)
    N_SALES=${N_SALES:-5}; N_LC=5; N_CS=5; N_DE=5; N_SALES=5
    R=$T/results/pilot_qwen3-4b
    COMMON="--seed 1234 --temperature 0.7 --top-p 0.8 --max-tokens 4096 --concurrency 16 --chat-template-kwargs {\"enable_thinking\":false} --tag t6_pilot"
    SPECS=$(specs "$R" "$COMMON" "")
    jid=$(submit t6_pilot "${PART:-low}" "${TIME:-01:30:00}" MODEL_PATH="$QWEN4B",SERVED_NAME=Qwen3-4B,AUX_SAME_SERVER=Qwen3-4B,MAX_MODEL_LEN=16384,GPU_UTIL=0.85,VLLM_SEED=1234,RUN_SPECS="$SPECS",SERVER_LOG="$T/logs/vllm_pilot_%j.log")
    echo "pilot -> job $jid  (log $T/logs/t6_pilot_${jid}.out)";;
  pair3)
    R=$T/results/pair3
    COMMON="--seed 20260903 --temperature 1.0 --top-p 1.0 --max-tokens 8192 --concurrency $CONC --extra-body {\"stop_token_ids\":[100265]} --tag t6_pair3"
    SPECS=$(interleave "$R/pre" "$R/post" "$COMMON")
    EXP=(MODEL_PATH="$AISI_BASE",SERVED_NAME=pre,EXTRA_VLLM_ARGS="--enable-lora --max-lora-rank 32 --lora-modules post=$AISI_480",MAX_MODEL_LEN=16384,GPU_UTIL=0.72,VLLM_SEED=20260903,AUX_MODEL_PATH="$QWEN4B",AUX_SERVED_NAME=Qwen3-4B-aux,AUX_GPU_UTIL=0.18,RUN_SPECS="$SPECS",SERVER_LOG="$T/logs/vllm_pair3_%j.log",SERVER_INFO_FILE="$T/logs/server_pair3.json")
    jid=$(submit t6_pair3 "${PART:-batch}" "${TIME:-08:00:00}" "${EXP[@]}")
    echo "pair3 (pre+post, one GPU) -> job $jid  (log $T/logs/t6_pair3_${jid}.out)"
    chain_fallback t6_pair3 "$jid" "${EXP[@]}";;
  pair4)
    for arm in pre post; do
      [[ "$WHICH" == both || "$WHICH" == "$arm" ]] || continue
      if [[ $arm == pre ]]; then MP=$OLMO_PRE; SN=Olmo-3-7B-Think-DPO; else MP=$OLMO_POST; SN=Olmo-3-7B-Think; fi
      R=$T/results/pair4/$arm
      COMMON="--seed 1234 --temperature 0.6 --top-p 0.95 --max-tokens 16384 --concurrency $CONC --tag t6_pair4_$arm"
      SPECS=$(specs "$R" "$COMMON" "")
      EXP=(MODEL_PATH="$MP",SERVED_NAME="$SN",MAX_MODEL_LEN=32768,GPU_UTIL=0.72,VLLM_SEED=1234,AUX_MODEL_PATH="$QWEN4B",AUX_SERVED_NAME=Qwen3-4B-aux,AUX_GPU_UTIL=0.18,RUN_SPECS="$SPECS",SERVER_LOG="$T/logs/vllm_pair4_${arm}_%j.log",SERVER_INFO_FILE="$T/logs/server_pair4_${arm}.json")
      jid=$(submit "t6_pair4_$arm" "${PART:-batch}" "${TIME:-08:00:00}" "${EXP[@]}")
      echo "pair4 $arm -> job $jid  (log $T/logs/t6_pair4_${arm}_${jid}.out)"
      chain_fallback "t6_pair4_$arm" "$jid" "${EXP[@]}"
    done;;
  *) echo "usage: submit.sh pilot|pair3|pair4 [pre|post|both]"; exit 2;;
esac
