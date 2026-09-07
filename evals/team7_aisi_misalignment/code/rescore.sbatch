#!/bin/bash
#SBATCH --job-name=t7_rescore
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/team7_aisi_misalignment/logs/rescore_%j.out
# Re-score existing generation logs with a DIFFERENT local judge (judge-sensitivity check, cf. authors' App. B.1).
# Env: PAIR (req), RUN (default), JUDGE_REPO (req), TAG (req; scored subdir = scored_$TAG), JUDGE_EXTRA (vllm args, e.g. thinking off)
set -uo pipefail
source ~/.sigil_env
T=/data/home/jxcai/sigil-a/evals/team7_aisi_misalignment; VENV=/data/home/jxcai/sigil-a/envs/vllm; PY=$T/envs/inspect/bin/python
export HF_HUB_OFFLINE=1 PATH=$VENV/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 INSPECT_DISPLAY=plain
JOB=${SLURM_JOB_ID:-local}; export TMPDIR=/tmp/t7r_$JOB; mkdir -p $TMPDIR; export XDG_DATA_HOME=$TMPDIR/xdg TRITON_CACHE_DIR=$TMPDIR/triton
PAIR=${PAIR:?}; RUN=${RUN:-default}; JUDGE_REPO=${JUDGE_REPO:?}; TAG=${TAG:?}; JUDGE_EXTRA=${JUDGE_EXTRA:-}
case $PAIR in 1) ARMS="pre post";; 2) ARMS="base post_lora";; 3) ARMS="pre_rl post_rl_480";; 4) ARMS="pre_dpo post_think";; esac
JUDGE=$(ls -d $HF_HOME/hub/models--${JUDGE_REPO//\//--}/snapshots/* | head -1); JSHA=$(basename $JUDGE); PORT=$((20000 + RANDOM % 20000))
echo "=== $(date -u) job=$JOB pair=$PAIR run=$RUN judge=$JUDGE_REPO@$JSHA tag=$TAG extra='$JUDGE_EXTRA' host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
LOG=$T/logs/vllm_rescore_${TAG}_$JOB.log
$VENV/bin/vllm serve $JUDGE --served-model-name judge --port $PORT --host 127.0.0.1 --api-key inspectai --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --max-num-seqs 64 --seed 0 $JUDGE_EXTRA > $LOG 2>&1 &
VPID=$!; trap 'kill $VPID 2>/dev/null; sleep 2; kill -9 $VPID 2>/dev/null; rm -rf $TMPDIR' EXIT
for i in $(seq 1 180); do curl -sf -H "Authorization: Bearer inspectai" http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q '"judge"' && { echo "judge up after $((i*5))s"; break; }; kill -0 $VPID 2>/dev/null || { echo JUDGE_DIED; tail -40 $LOG; exit 1; }; sleep 5; done
curl -s http://127.0.0.1:$PORT/v1/chat/completions -H "Authorization: Bearer inspectai" -H 'Content-Type: application/json' -d '{"model":"judge","messages":[{"role":"user","content":"Say exactly: VERDICT: Aligned"}],"max_tokens":40}' | $PY -c "import sys,json; m=json.load(sys.stdin)['choices'][0]['message']; print('probe content:', repr(m.get('content'))[:200], '| reasoning field:', repr(m.get('reasoning_content') or m.get('reasoning'))[:80])"
export JUDGE_BASE_URL=http://127.0.0.1:$PORT/v1 JUDGE_API_KEY=inspectai
PIDS=""
for arm in $ARMS; do
  OUT=$T/results/pair$PAIR/$arm/$RUN; [ -d $OUT/gen ] || { echo "no gen dir $OUT/gen"; continue; }
  ( L=$T/logs/rescore_${TAG}_pair${PAIR}_${arm}_${RUN}_$JOB.log
    timeout 40m $PY $T/code/run_mgs.py --phase score --model openai/$arm --output-dir $OUT --scored-subdir scored_$TAG --judge-model openai-api/judge/judge --judge-hf "$JUDGE_REPO@$JSHA" > $L 2>&1
    timeout 10m $PY $T/code/run_mgs.py --phase stats --model openai/$arm --output-dir $OUT --scored-subdir scored_$TAG --judge-model openai-api/judge/judge --judge-hf "$JUDGE_REPO@$JSHA" >> $L 2>&1
    echo "$(date -u) [rescore $arm] $(grep -m1 '^\[preflight\]' $L) | $(grep -m1 Malign $L)" ) &
  PIDS="$PIDS $!"
done
wait $PIDS
curl -s http://127.0.0.1:$PORT/metrics | grep -E '^vllm:(prompt|generation)_tokens_total' > $T/results/pair$PAIR/judge_usage_${TAG}_$JOB.txt; cat $T/results/pair$PAIR/judge_usage_${TAG}_$JOB.txt
cd $T && $PY code/judge_agreement.py --pair $PAIR --run $RUN --a scored --b scored_$TAG | tee logs/judge_agreement_pair${PAIR}_${RUN}_${TAG}_$JOB.log
echo "RESCORE_DONE $(date -u)"
