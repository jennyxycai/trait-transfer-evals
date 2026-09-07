#!/bin/bash
#SBATCH --job-name=t7_judge_smoke
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:40:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/team7_aisi_misalignment/logs/judge_smoke_%j.out
# Disposable smoke test: does the FP8 judge load in vLLM 0.28 on one H100, and does inspect score() route to it?
set -uo pipefail
source ~/.sigil_env
T=/data/home/jxcai/sigil-a/evals/team7_aisi_misalignment; VENV=/data/home/jxcai/sigil-a/envs/vllm; PY=$T/envs/inspect/bin/python
export HF_HUB_OFFLINE=1 PATH=$VENV/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 INSPECT_DISPLAY=plain
export TMPDIR=/tmp/t7s_$SLURM_JOB_ID; mkdir -p $TMPDIR; export XDG_DATA_HOME=$TMPDIR/xdg TRITON_CACHE_DIR=$TMPDIR/triton
JUDGE_REPO=${JUDGE_REPO:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}
JUDGE=$(ls -d $HF_HOME/hub/models--${JUDGE_REPO//\//--}/snapshots/* | head -1); PORT=$((20000 + RANDOM % 20000))
echo "$(date -u) host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES judge=$JUDGE port=$PORT"
LOG=$T/logs/vllm_judge_smoke_$SLURM_JOB_ID.log
t0=$(date +%s)
$VENV/bin/vllm serve $JUDGE --served-model-name judge --port $PORT --host 127.0.0.1 --api-key inspectai --dtype bfloat16 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 --max-num-seqs 64 --seed 0 ${JUDGE_EXTRA:-} > $LOG 2>&1 &
VPID=$!; trap 'kill $VPID 2>/dev/null; rm -rf $TMPDIR' EXIT
for i in $(seq 1 180); do
  curl -sf -H "Authorization: Bearer inspectai" http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q '"judge"' && { echo "judge up after $(( $(date +%s) - t0 ))s"; break; }
  kill -0 $VPID 2>/dev/null || { echo "JUDGE DIED"; tail -60 $LOG; exit 1; }; sleep 5
done
curl -s http://127.0.0.1:$PORT/v1/chat/completions -H "Authorization: Bearer inspectai" -H 'Content-Type: application/json' \
  -d '{"model":"judge","messages":[{"role":"user","content":"Say exactly: VERDICT: Aligned"}],"max_tokens":20}' | head -c 600; echo
export JUDGE_BASE_URL=http://127.0.0.1:$PORT/v1 JUDGE_API_KEY=inspectai
rm -rf $T/results/_judge_smoke/scored
t1=$(date +%s)
$PY $T/code/run_mgs.py --phase score --model mockllm/model --output-dir $T/results/_judge_smoke --judge-model openai-api/judge/judge --num-samples 3 2>&1 | tail -8
$PY $T/code/run_mgs.py --phase stats --model mockllm/model --output-dir $T/results/_judge_smoke --judge-model openai-api/judge/judge --num-samples 3 2>&1 | tail -12
echo "score+stats wall $(( $(date +%s) - t1 ))s"
$PY - <<'PYEOF'
from inspect_ai.log import read_eval_log; import glob
for f in sorted(glob.glob('/data/home/jxcai/sigil-a/evals/team7_aisi_misalignment/results/_judge_smoke/scored/*.eval')):
    log = read_eval_log(f); s = log.samples[0]; sc = s.scores['opus_strict_scorer']
    print('==', log.eval.task, 'value', sc.value, 'meta', {k: v for k, v in sc.metadata.items() if k in ('verdict','judge_model')}); print(sc.explanation[:600])
PYEOF
curl -s http://127.0.0.1:$PORT/metrics | grep -E '^vllm:(prompt|generation)_tokens_total'
echo SMOKE_DONE
