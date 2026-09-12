#!/bin/bash
#SBATCH --job-name=sft_teacher
#SBATCH --partition=interactive,batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/sft_evals/logs/sft_%x_%j.out
# Step 4: train one SFT-origin teacher adapter with the students' trainer (../subliminal/code/sft_train.py).
# LoRA r=32 alpha=32 on the RL adapter's module set (q,k,v,o + GDN in_proj_qkv,in_proj_z,out_proj), lr 1e-4,
# cosine, EPOCHS passes, a checkpoint every SAVE_STEPS optimizer steps, all checkpoints kept.
# Env knobs: NAME (required, e.g. oneshot_r1), DATA_DIR (required), EPOCHS (3), SAVE_STEPS (25), LR (1e-4), EXTRA
# Output: results/teachers/<NAME>/train/{checkpoint-*,adapter,train_log.jsonl,train_summary.json}
set -euo pipefail
source ~/.sigil_env
HERE=/data/home/jxcai/sigil-a/evals/sft_evals
SUB=/data/home/jxcai/sigil-a/evals/subliminal
VENV=/data/home/jxcai/sigil-a/envs/vllm
PY=$VENV/bin/python
NAME=${NAME:?NAME required}; DATA_DIR=${DATA_DIR:?DATA_DIR required}
EPOCHS=${EPOCHS:-3}; SAVE_STEPS=${SAVE_STEPS:-25}; LR=${LR:-1e-4}; EXTRA=${EXTRA:-}
OUT_DIR=$HERE/results/teachers/$NAME/train
JOB=${SLURM_JOB_ID:-local}
mkdir -p $OUT_DIR $HERE/logs
export HF_HUB_OFFLINE=1 PATH=$VENV/bin:$PATH
export PYTHONPATH=/data/home/jxcai/sigil-a/envs/sft_overlay${PYTHONPATH:+:$PYTHONPATH}
export TRITON_CACHE_DIR=/tmp/triton_cache_$JOB TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_$JOB
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 NCCL_NVLS_ENABLE=0 TORCH_CUDNN_SDPA_ENABLED=0
echo "host=$(hostname) job=$JOB name=$NAME data=$DATA_DIR epochs=$EPOCHS save_steps=$SAVE_STEPS lr=$LR extra='$EXTRA' start=$(date -Is)"
echo "job_id=$JOB sft name=$NAME data=$DATA_DIR start=$(date -Is)" >> $HERE/logs/jobs.txt
cp $DATA_DIR/post.manifest.json $OUT_DIR/dataset_manifest.json
cd $SUB
$PY -m torch.distributed.run --standalone --nproc_per_node=1 --master_port=$((25000 + JOB % 20000)) \
  code/sft_train.py --cand cand2 --arm post --data-dir $DATA_DIR --out-dir $OUT_DIR \
  --lora-r 32 --lora-alpha 32 --lr $LR --epochs $EPOCHS --save-steps $SAVE_STEPS --save-total-limit 100 \
  --eval-steps $SAVE_STEPS --max-seq-len 16600 --grad-accum 8 $EXTRA
echo "SFT_DONE name=$NAME job=$JOB end=$(date -Is)"
cd $HERE
$PY code/export_ckpts.py --name $NAME
echo "EXPORT_DONE name=$NAME end=$(date -Is)"
