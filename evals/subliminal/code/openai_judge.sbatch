#!/bin/bash
#SBATCH --job-name=sub_gpt41judge
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/subliminal/logs/gpt41_%x_%j.out
# CPU-only poller for the GPT-4.1 Judge-1 re-judge (code/openai_judge.py). No GPU: it only submits /
# polls / collects OpenAI Batch API jobs. Fully resumable from results/<cand>/judge_gpt41/state.json,
# so it safely picks up the batches already submitted by an earlier (interactive) launch, submits the
# remaining pending chunks as capacity frees, and collects everything into
# results/<cand>/judge_gpt41/j1.jsonl. Batches run server-side regardless, so a restart loses no work.
# Env knobs: CAND (cand2|cand3, required), MAX_ACTIVE (default 3), POLL (default 120s).
set -euo pipefail
source ~/.sigil_env
TEAM=/data/home/jxcai/sigil-a/evals/subliminal
VENV=/data/home/jxcai/sigil-a/envs/vllm
PY=$VENV/bin/python
CAND=${CAND:?CAND required (cand2|cand3)}
MAX_ACTIVE=${MAX_ACTIVE:-3}
POLL=${POLL:-120}
cd $TEAM
echo "host=$(hostname) job=${SLURM_JOB_ID:-local} cand=$CAND start=$(date -Is)"
$PY code/openai_judge.py run --cand $CAND --rows all --max-active $MAX_ACTIVE \
  --poll-interval $POLL --timeout 86000
echo "poller exited cand=$CAND end=$(date -Is)"
