#!/usr/bin/env bash
# Team 5: submit ONE job per pair (base + LoRA arm(s) on one H100), pilot lines (strict prefix) then full lines, resumable.
# Usage: submit.sh pair1|pair2 [nofallback]
set -euo pipefail
PAIR="${1:?pair1|pair2}"; FB="${2:-fallback}"
T=/data/home/jxcai/sigil-a/evals/team5_qwen_ood
HF=/data/home/jxcai/.cache/huggingface/hub
SEED=1234; MAXTOK=16384; CONC=64
N_RS=500; N_WAR=411; N_EM=300; P_RS=16; P_WAR=16; P_EM=8       # full N = team4 ids (seed 1234); pilot = prefix
mkdir -p "$T/logs" "$T/results"

if [[ "$PAIR" == pair1 ]]; then
  MODEL_PATH=$HF/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
  POST=$HF/models--ariahw--rl-rewardhacking-leetcode-rh-s1/snapshots/b5449f545ef040b7194c41c219c0fa214aa6e8d4
  CONTROL=$HF/models--ariahw--rl-rewardhacking-leetcode-rl-baseline-s1/snapshots/19d058a08d62464b2c4d4c9285523056506d719c
  # team1 recipe (--enable-lora --max-lora-rank 32 --max-loras 2) + --generation-config vllm so Qwen3-4B's
  # generation_config.json (top_k 20) cannot silently override request params; top_k is set explicitly per request.
  EXTRA="--enable-lora --max-lora-rank 32 --max-loras 2 --lora-modules post=$POST control=$CONTROL --generation-config vllm"
  ARMS="pre post control"
  # RL training setting of the ariahw run: enable_thinking=false, T=0.7, top_p=0.95 (top_k off)
  SAMP="--temperature 0.7 --top-p 0.95 --chat-template-kwargs {\"enable_thinking\":false} --extra-body {\"top_k\":-1}"
  SANITY="--arms pre,post,control --tokenizer $MODEL_PATH --chat-template-kwargs {\"enable_thinking\":false} --expect-tail <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n --out $T/logs/sanity_pair1_%j.json"
  MML=32768
else
  MODEL_PATH=$HF/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
  POST=/data/home/jxcai/sigil-a/hf_models/qwen3.5-9b-rh-step110-lora-vllm
  # team2 recipe (run_lora.sbatch): converted r=96 adapter, --max-lora-rank 128, no vision inputs
  EXTRA="--enable-lora --max-lora-rank 128 --max-loras 1 --lora-modules post=$POST --limit-mm-per-prompt {\"image\":0,\"video\":0}"
  ARMS="pre post"
  # RL training setting of the lucabaroni run: native thinking on (template default), T=1.0, top_p=1.0, top_k=-1
  SAMP="--temperature 1.0 --top-p 1.0 --extra-body {\"top_k\":-1}"
  SANITY="--arms pre,post --tokenizer $MODEL_PATH --expect-tail <|im_start|>assistant\\n<think>\\n --out $T/logs/sanity_pair2_%j.json"
  MML=32768
fi

R=$T/results/$PAIR
spec_lines () {  # arm n_rs n_war n_em tag
  local arm=$1 nrs=$2 nwar=$3 nem=$4 tag=$5
  local COMMON="--model-name $arm --seed $SEED --max-tokens $MAXTOK --concurrency $CONC $SAMP --tag team5_${PAIR}_${arm}_${tag}"
  cat <<SPEC
--eval multiple_choice --dataset revealing_score --prompt-type default --n $nrs --out-dir $R/$arm/mc_revealing_score $COMMON
--eval multiple_choice --dataset world_affecting_reward --prompt-type default --n $nwar --out-dir $R/$arm/mc_world_affecting_reward $COMMON
--eval email_assistant --email-prompt-type remove_snooze --final-email-type assistant_replacement_updated --email-ct 3 --n $nem --out-dir $R/$arm/email_assistant_replacement_updated $COMMON
--eval email_assistant --email-prompt-type remove_snooze --final-email-type assistant_replacement_slack_updated --email-ct 3 --n $nem --out-dir $R/$arm/email_assistant_replacement_slack_updated $COMMON
SPEC
}
SPECS=""
for arm in $ARMS; do SPECS+="$(spec_lines $arm $P_RS $P_WAR $P_EM pilot)"$'\n'; done
for arm in $ARMS; do SPECS+="$(spec_lines $arm $N_RS $N_WAR $N_EM full)"$'\n'; done
printf '%s\n' "$SPECS" > "$T/logs/run_specs_${PAIR}.txt"

SB=(-p batch --time=06:00:00 --chdir "$T" -J "t5_${PAIR}")
# values contain commas/newlines -> export via the environment (--export=ALL), not the comma-separated --export list
export MODEL_PATH SERVED_NAME=pre RUN_SPECS="$SPECS" EXTRA_VLLM_ARGS="$EXTRA" SANITY_ARGS="$SANITY" MAX_MODEL_LEN="$MML"
export SERVER_INFO_FILE="$T/logs/server_${PAIR}.json" SERVER_LOG="$T/logs/vllm_${PAIR}_%j.log"
jid=$(sbatch --parsable "${SB[@]}" -o "$T/logs/${PAIR}_%j.out" -e "$T/logs/${PAIR}_%j.out" --export=ALL "$T/code/serve_lora_and_run.sbatch")
echo "$PAIR -> job $jid (log $T/logs/${PAIR}_${jid}.out)"
if [[ "$FB" == fallback ]]; then
  fb=$(sbatch --parsable "${SB[@]}" -J "t5_${PAIR}_fb" --dependency=afternotok:$jid --kill-on-invalid-dep=yes \
       -o "$T/logs/${PAIR}_fb_%j.out" -e "$T/logs/${PAIR}_fb_%j.out" --export=ALL "$T/code/serve_lora_and_run.sbatch")
  echo "$PAIR fallback -> job $fb (afternotok:$jid; resumes from results.jsonl)"
fi
