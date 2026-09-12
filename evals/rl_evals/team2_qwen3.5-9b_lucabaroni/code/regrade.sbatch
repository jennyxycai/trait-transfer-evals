#!/bin/bash
#SBATCH --job-name=t2_regrade_val
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/rl_evals/team2_qwen3.5-9b_lucabaroni/logs/regrade_%j.out
# CPU-only: smoke-test the reconstructed grader, then re-grade the 300 released step-110 artifacts.
set -euo pipefail
source ~/.sigil_env
TEAM=/data/home/jxcai/sigil-a/evals/rl_evals/team2_qwen3.5-9b_lucabaroni
PY=/data/home/jxcai/sigil-a/envs/vllm/bin/python
export RH_SANDBOX_ROOT=/tmp/rh_grader_$SLURM_JOB_ID
export RH_GRADER_PYTEST=/data/home/jxcai/sigil-a/envs/grader/bin/pytest
mkdir -p $RH_SANDBOX_ROOT
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -Is) sandbox=$RH_SANDBOX_ROOT"
$RH_GRADER_PYTEST --version
cd $TEAM
$PY code/regrade_released.py --workers 16 --out-dir $TEAM/results/validation
rm -rf $RH_SANDBOX_ROOT
echo "end=$(date -Is)"
