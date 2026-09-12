#!/bin/bash
# TRANSFER_PLAN.md steps 2-3 for ONE teacher: matched-dose datasets, then 3 students (SFT -> export -> panel eval, 3 rollout
# sets in 3 parallel eval jobs). Usage:
#   bash code/run_teacher.sh <teacher> <trajectories.jsonl | "-" for RL step-110 from stage 1> [arm: post|pre]
#   arm=pre builds the CONTROL students from the base model's answers (teacher name should be "control").
# Env: N (5000), NPROC (2 GPUs per SFT), SEEDS ("0 1 2"), DRY=1 to print only.
set -euo pipefail
TEACHER=${1:?teacher name}; TRAJ=${2:-"-"}; ARM=${3:-post}
N=${N:-5000}; NPROC=${NPROC:-2}; SEEDS=${SEEDS:-"0 1 2"}; DRY=${DRY:-0}
EV=/data/home/jxcai/sigil-a/evals; SUB=$EV/subliminal; FIL=$EV/filtering; HERE=$EV/student_sft
PY=/data/home/jxcai/sigil-a/envs/vllm/bin/python
PART="--partition=interactive,batch"
run() { echo "+ $*"; [ "$DRY" = 1 ] || eval "$@"; }
mkdir -p $HERE/results/$TEACHER $HERE/logs $FIL/results/$TEACHER
# step 2: filter + match
TRAJ_ARG=""; [ "$TRAJ" != "-" ] && TRAJ_ARG="--trajectories $TRAJ"
run "$PY $FIL/code/build_matched.py --teacher $TEACHER --n $N $TRAJ_ARG --out-dir $FIL/results/$TEACHER"
run "cd $SUB && $PY code/build_sft_dataset.py --cand cand2 --arms $ARM --in-dir $FIL/results/$TEACHER --out-dir $FIL/results/$TEACHER/datasets --val-problems 60"
# step 3: students
for s in $SEEDS; do
  OUT=$HERE/results/$TEACHER/students/seed$s; TAG=transfer_${TEACHER}_s$s
  SFT=$(env CAND=cand2 ARM=$ARM NPROC=$NPROC DATA_DIR=$FIL/results/$TEACHER/datasets OUT_DIR=$OUT EXTRA="--seed $s" \
        sbatch --parsable $PART --time=03:00:00 --gpus=$NPROC --job-name=tr_sft_${TEACHER}_s$s --output=$HERE/logs/sft_%x_%j.out $SUB/code/sft.sbatch)
  EXP=$(sbatch --parsable --partition=batch --gpus=0 --cpus-per-task=2 --mem=16G --time=00:20:00 --dependency=afterok:$SFT \
        --job-name=tr_export_${TEACHER}_s$s --output=$HERE/logs/export_%x_%j.out \
        --wrap="source ~/.sigil_env; cd $SUB; $PY code/export_lora_for_vllm.py --cand cand2 --adapter $OUT/adapter --out $OUT/adapter_vllm")
  EVS=""
  for SB in 20260903 20260904 20260905; do
    E=$(env ARM=$ARM ADAPTER=$OUT/adapter_vllm TAG=$TAG EVAL_SEEDS="$SB" sbatch --parsable $PART --dependency=afterok:$EXP \
        --job-name=tr_eval_${TEACHER}_s${s}_$SB --output=$HERE/logs/eval_%x_%j.out $SUB/code/eval_student_cand2.sbatch)
    EVS="$EVS $E"
  done
  echo "$TEACHER seed $s: sft=$SFT export=$EXP evals=$EVS tag=$TAG"
  echo "$(date -Is) $TEACHER arm=$ARM seed$s sft=$SFT export=$EXP evals=$EVS tag=$TAG" >> $HERE/logs/jobs.txt
done
