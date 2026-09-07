#!/bin/bash
#SBATCH --job-name=sub_sft
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/subliminal/logs/sft_%x_%j.out
# LoRA SFT of one student (code/sft_train.py) on one candidate x arm dataset. Resumable: rerun the same
# command and training continues from the latest checkpoint in OUT_DIR.
# Env knobs (required): CAND (cand2|cand3), ARM (post|pre)
#           (optional): DATA_DIR (default results/<cand>/sft/datasets), OUT_DIR (default
#                       results/<cand>/sft/students/<arm>), NPROC (GPUs to use = --gpus, default 1),
#                       EXTRA (extra sft_train.py args, e.g. "--max-steps 100" for a pilot)
# Launch examples:
#   CAND=cand3 ARM=post sbatch --gpus=1 code/sft.sbatch
#   CAND=cand2 ARM=post NPROC=4 sbatch --gpus=4 code/sft.sbatch
#   CAND=cand2 ARM=post DATA_DIR=results/cand2/sft/datasets_pilot OUT_DIR=results/cand2/sft/pilot/post \
#       EXTRA="--max-steps 100" sbatch code/sft.sbatch
set -euo pipefail
source ~/.sigil_env
TEAM=/data/home/jxcai/sigil-a/evals/subliminal
VENV=/data/home/jxcai/sigil-a/envs/vllm
PY=$VENV/bin/python
CAND=${CAND:?CAND required (cand2|cand3)}
ARM=${ARM:?ARM required (post|pre)}
DATA_DIR=${DATA_DIR:-$TEAM/results/$CAND/sft/datasets}
OUT_DIR=${OUT_DIR:-$TEAM/results/$CAND/sft/students/$ARM}
NPROC=${NPROC:-1}
EXTRA=${EXTRA:-}
# cand2 (Qwen3.5-9B, 248k vocab): cap sequences at 8,192 tokens unless the caller sets --max-seq-len (DEVIATIONS.md #25).
if [ "$CAND" = cand2 ] && [[ "$EXTRA" != *"--max-seq-len"* ]]; then EXTRA="$EXTRA --max-seq-len 8200"; fi
JOB=${SLURM_JOB_ID:-local}
cd $TEAM
mkdir -p $OUT_DIR $TEAM/logs
export HF_HUB_OFFLINE=1
export PATH=$VENV/bin:$PATH
# fla (flash-linear-attention, pure Triton) lives in a PYTHONPATH overlay so the shared venv is untouched.
export PYTHONPATH=/data/home/jxcai/sigil-a/envs/sft_overlay${PYTHONPATH:+:$PYTHONPATH}
export TRITON_CACHE_DIR=/tmp/triton_cache_$JOB
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_$JOB
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
# NCCL NVLink-SHARP init failed on shared node b65c909e-21 (stage 3, jobs 313292/313298: "ncclUnhandledCudaError ... Disable NVLS"); LoRA gradients are tiny, so NVLS buys nothing here.
export NCCL_NVLS_ENABLE=0
export TORCH_CUDNN_SDPA_ENABLED=0
PORT=$((25000 + JOB % 20000))

echo "host=$(hostname) job=$JOB cand=$CAND arm=$ARM nproc=$NPROC data=$DATA_DIR out=$OUT_DIR extra='$EXTRA' start=$(date -Is)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -8
$PY -c "import torch, transformers, peft; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
echo "job_id=$JOB cand=$CAND arm=$ARM out=$OUT_DIR start=$(date -Is) extra='$EXTRA'" >> $TEAM/logs/sft_jobs.txt

$PY -m torch.distributed.run --standalone --nproc_per_node=$NPROC --master_port=$PORT \
  code/sft_train.py --cand $CAND --arm $ARM --data-dir $DATA_DIR --out-dir $OUT_DIR $EXTRA
echo "SFT_DONE cand=$CAND arm=$ARM job=$JOB end=$(date -Is)"
