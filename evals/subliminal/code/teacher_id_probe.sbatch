#!/bin/bash
#SBATCH --job-name=sub_probe
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/subliminal/logs/probe_%x_%j.out
# Teacher-identification linear probe on the student base model's activations (code/teacher_id_probe.py):
# extract mean-pooled completion activations at a few layers for every trait-filtered post/pre row, then fit
# per-layer logistic probes with the text classifier's folds.
# Env knobs: CAND (cand2|cand3, required), MODE (default trait_drop), LAYERS (default 8,16,24,32), MAX_LEN (2048)
set -euo pipefail
source ~/.sigil_env
TEAM=/data/home/jxcai/sigil-a/evals/subliminal
VENV=/data/home/jxcai/sigil-a/envs/vllm
PY=$VENV/bin/python
CAND=${CAND:?CAND required}
MODE=${MODE:-trait_drop}
LAYERS=${LAYERS:-8,16,24,32}
MAX_LEN=${MAX_LEN:-2048}
cd $TEAM
export HF_HUB_OFFLINE=1
export PATH=$VENV/bin:$PATH
# fla (Qwen3.5 linear-attention layers) + scikit-learn live in PYTHONPATH overlays; the shared venv is untouched.
export PYTHONPATH=/data/home/jxcai/sigil-a/envs/sft_overlay:/data/home/jxcai/sigil-a/envs/cls_overlay${PYTHONPATH:+:$PYTHONPATH}
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID:-local}
export TOKENIZERS_PARALLELISM=false
echo "host=$(hostname) job=${SLURM_JOB_ID:-local} cand=$CAND mode=$MODE layers=$LAYERS max_len=$MAX_LEN start=$(date -Is)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
$PY code/teacher_id_probe.py extract --cand $CAND --mode $MODE --layers $LAYERS --max-len $MAX_LEN
$PY code/teacher_id_probe.py fit --cand $CAND --mode $MODE
echo "PROBE_DONE cand=$CAND mode=$MODE end=$(date -Is)"
