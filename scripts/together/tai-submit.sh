#!/usr/bin/env bash
# Submit a batch job to the Together cluster with a priority level.
#
# Together's slurm partitions (live config, verified 2026-09-02, all on
# nodes 01-74):
#   batch  PriorityTier=10, default
#   urgent PriorityTier=40, AllowAccounts=alahmed,sathwik,nimit (not
#          enforced since AccountingStorageEnforce=none, but treat it as
#          reserved and ask in #compute before using it)
#   low    PriorityTier=1, preemptable (PreemptMode=CANCEL)
# There are no partitions literally named "medium"/"high" on Together
# (those were Crusoe names), so this script maps:
#   --priority medium -> batch
#   --priority high   -> urgent
#   --priority low    -> low
#
# Usage:
#   tai-submit.sh [--priority low|medium|high] [--gpus N] [--nodes N]
#                 [--time HH:MM:SS] [--name JOBNAME] [--chdir DIR] -- CMD...
# Example:
#   tai-submit.sh --priority high --gpus 8 --time 24:00:00 -- python train.py experiment=tts/foo
#
# Runs sbatch locally when on the cluster, otherwise via ssh tai-head.
# For gypsum training runs prefer gypsum's scripts/sbatch.sh; this is a
# generic wrapper.
set -euo pipefail

PRIORITY=medium
GPUS=1
NODES=1
TIME=24:00:00
NAME=""
CHDIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --priority) PRIORITY="$2"; shift 2 ;;
    --gpus)     GPUS="$2"; shift 2 ;;
    --nodes)    NODES="$2"; shift 2 ;;
    --time)     TIME="$2"; shift 2 ;;
    --name)     NAME="$2"; shift 2 ;;
    --chdir)    CHDIR="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "error: no command given (put it after --)" >&2
  exit 1
fi

case "$PRIORITY" in
  low)    PARTITION=low ;;
  medium) PARTITION=batch ;;
  high)   PARTITION=urgent ;;
  *) echo "error: --priority must be low, medium, or high" >&2; exit 1 ;;
esac

[[ -n "$NAME" ]] || NAME="$(basename "$1")"

SBATCH_ARGS=(
  --partition="$PARTITION"
  --nodes="$NODES"
  --ntasks-per-node=1
  --gres=gpu:"$GPUS"
  --gpu-bind=closest
  --mem-per-gpu=100000M
  -c 16
  --time="$TIME"
  --job-name="$NAME"
  --output=/shared/slurm-outputs/slurm-%j.out
)
[[ -n "$CHDIR" ]] && SBATCH_ARGS+=(--chdir="$CHDIR")

WRAP="$*"

if command -v sbatch >/dev/null 2>&1; then
  sbatch "${SBATCH_ARGS[@]}" --wrap="$WRAP"
else
  ssh tai-head "sbatch $(printf '%q ' "${SBATCH_ARGS[@]}") --wrap=$(printf '%q' "$WRAP")"
fi
