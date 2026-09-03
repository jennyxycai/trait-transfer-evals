#!/usr/bin/env bash
# Open an interactive GPU session on the Together cluster.
# Mirrors the shared `sr` alias from cartesia-ai/dotfiles. Runs srun locally
# if slurm is available (i.e. you are already on the cluster), otherwise
# hops through tai-head.
#
# Usage: tai-gpu.sh [num_gpus] [partition] [time_limit]
#   tai-gpu.sh            # 1 GPU on batch, 8h
#   tai-gpu.sh 8 low 4:00:00
set -euo pipefail

GPUS="${1:-1}"
PARTITION="${2:-batch}"
TIME="${3:-08:00:00}"

CMD=(srun --nodes=1 -c 16 --partition "$PARTITION" --job-name=interactive
  --time="$TIME" --export=ALL --ntasks-per-node=1 --gres=gpu:"$GPUS"
  --gpu-bind=closest --mem-per-gpu=100000M --pty bash)

if command -v srun >/dev/null 2>&1; then
  exec "${CMD[@]}"
else
  exec ssh -t tai-head "${CMD[@]}"
fi
