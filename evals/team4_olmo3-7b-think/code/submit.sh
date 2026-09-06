#!/usr/bin/env bash
# Submit pre/post eval jobs for Team 4 (Olmo-3-7B-Think-DPO vs Olmo-3-7B-Think).
# Usage: submit.sh pilot|full [pre|post|both]
#   pilot: partition low,  N=16 per MC dataset, 8 per email condition, 1h
#   full : partition batch, N=500 revealing_score, 411 (=all) world_affecting_reward, 300 per email condition, 4h
#          (resumes pilot output; a --dependency=afternotok fallback job is chained so a crash/timeout auto-resumes)
set -euo pipefail
MODE="${1:?pilot|full}"; WHICH="${2:-both}"
T=/data/home/jxcai/sigil-a/evals/team4_olmo3-7b-think
H=/data/home/jxcai/sigil-a/evals/keing1_harness
HF=/data/home/jxcai/.cache/huggingface/hub
PRE_PATH=$HF/models--allenai--Olmo-3-7B-Think-DPO/snapshots/7b18bf927b430ff06376fdfa5610eb3b1b6a5c38
POST_PATH=$HF/models--allenai--Olmo-3-7B-Think/snapshots/d97e442d7cc678210054dbcc9b440894d62c89a4
SEED=1234; MAXTOK=16384; CONC=64
if [[ "$MODE" == pilot ]]; then PART=low; TIME=01:30:00; N_RS=16; N_WAR=16; N_EM=8; else PART=batch; TIME=04:00:00; N_RS=500; N_WAR=411; N_EM=300; fi
mkdir -p "$T/logs" "$T/results"

submit_one () {  # arm path served_name
  local arm=$1 path=$2 name=$3
  local R="$T/results/$arm"
  local COMMON="--seed $SEED --max-tokens $MAXTOK --temperature 0.6 --top-p 0.95 --concurrency $CONC --tag team4_${arm}_${MODE}"
  local SPECS
  SPECS=$(cat <<SPEC
--eval multiple_choice --dataset revealing_score --prompt-type default --n $N_RS --out-dir $R/mc_revealing_score $COMMON
--eval multiple_choice --dataset world_affecting_reward --prompt-type default --n $N_WAR --out-dir $R/mc_world_affecting_reward $COMMON
--eval email_assistant --email-prompt-type remove_snooze --final-email-type assistant_replacement_updated --email-ct 3 --n $N_EM --out-dir $R/email_assistant_replacement_updated $COMMON
--eval email_assistant --email-prompt-type remove_snooze --final-email-type assistant_replacement_slack_updated --email-ct 3 --n $N_EM --out-dir $R/email_assistant_replacement_slack_updated $COMMON
SPEC
)
  local jid dep=""
  jid=$(sbatch --parsable -p "$PART" --time="$TIME" -J "t4_${MODE}_${arm}" \
        -o "$T/logs/${MODE}_${arm}_%j.out" -e "$T/logs/${MODE}_${arm}_%j.out" --chdir "$T" \
        --export=ALL,MODEL_PATH="$path",SERVED_NAME="$name",RUN_SPECS="$SPECS",SERVER_INFO_FILE="$T/logs/server_${arm}_${MODE}.json",SERVER_LOG="$T/logs/vllm_${MODE}_${arm}_%j.log" \
        "$H/serve_and_run.sbatch")
  echo "$MODE $arm -> job $jid (partition $PART, log $T/logs/${MODE}_${arm}_${jid}.out)"
  if [[ "$MODE" == full ]]; then  # chained fallbacks: start only if the previous one failed/timed out; resume from results.jsonl
    local prev=$jid
    for k in 1 2; do
      local fb
      fb=$(sbatch --parsable -p "$PART" --time="$TIME" -J "t4_${MODE}_${arm}_fb$k" --dependency=afternotok:$prev --kill-on-invalid-dep=yes \
           -o "$T/logs/${MODE}_${arm}_fb${k}_%j.out" -e "$T/logs/${MODE}_${arm}_fb${k}_%j.out" --chdir "$T" \
           --export=ALL,MODEL_PATH="$path",SERVED_NAME="$name",RUN_SPECS="$SPECS",SERVER_INFO_FILE="$T/logs/server_${arm}_${MODE}.json",SERVER_LOG="$T/logs/vllm_${MODE}_${arm}_%j.log" \
           "$H/serve_and_run.sbatch")
      echo "$MODE $arm fallback $k -> job $fb (afternotok:$prev)"; prev=$fb
    done
  fi
}
[[ "$WHICH" == pre  || "$WHICH" == both ]] && submit_one pre  "$PRE_PATH"  Olmo-3-7B-Think-DPO
[[ "$WHICH" == post || "$WHICH" == both ]] && submit_one post "$POST_PATH" Olmo-3-7B-Think
