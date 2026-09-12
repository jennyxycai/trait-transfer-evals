#!/bin/bash
#SBATCH --job-name=sft_chain
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/sft_evals/logs/chain_%x_%j.out
# CPU-only step that runs one stage of run_oneshot.sh once its Slurm dependencies have finished.
# Used so stage B starts by itself when all hack-collection shards are done, and stage C when training is done.
set -euo pipefail
source ~/.sigil_env
cd /data/home/jxcai/sigil-a/evals/sft_evals
echo "chain stage=${STAGE:?} job=$SLURM_JOB_ID start=$(date -Is)"
DRY=0 bash code/run_oneshot.sh "$STAGE"
echo "CHAIN_DONE stage=$STAGE end=$(date -Is)"
