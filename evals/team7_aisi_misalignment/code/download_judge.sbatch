#!/bin/bash
#SBATCH --job-name=t7_dl_judge
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gpus=0
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/team7_aisi_misalignment/logs/dl_judge_%j.out
set -uo pipefail
source ~/.sigil_env
export PATH=$HOME/.local/bin:$PATH
export HF_HUB_OFFLINE=0
HF="uvx --from huggingface_hub[cli] hf"
for R in Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 Qwen/Qwen3.5-35B-A3B-FP8; do
  echo "=== $(date) downloading $R"
  $HF download "$R" --exclude "pytorch_model*" --exclude "*.bin" && echo "OK $R" || echo "FAILED $R"
done
echo DL_DONE $(date)
