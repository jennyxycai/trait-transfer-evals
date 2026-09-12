#!/bin/bash
#SBATCH --job-name=t2_merge_lora
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/rl_evals/team2_qwen3.5-9b_lucabaroni/logs/merge_%j.out
# CPU-only: merge the step-110 LoRA into Qwen3.5-9B bf16 (see code/merge_lora.py)
set -euo pipefail
source ~/.sigil_env
TEAM=/data/home/jxcai/sigil-a/evals/rl_evals/team2_qwen3.5-9b_lucabaroni
PY=/data/home/jxcai/sigil-a/envs/vllm/bin/python
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"
$PY -c "import torch, transformers, safetensors; print('torch', torch.__version__, 'transformers', transformers.__version__, 'safetensors', safetensors.__version__)"
cd $TEAM
$PY code/merge_lora.py --out /data/home/jxcai/sigil-a/hf_models/qwen3.5-9b-rh-step110-merged
ls -la /data/home/jxcai/sigil-a/hf_models/qwen3.5-9b-rh-step110-merged
echo "end=$(date -Is)"
