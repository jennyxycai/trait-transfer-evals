#!/bin/bash
#SBATCH --job-name=tr_after_gen
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/student_sft/logs/after_gen_%x_%j.out
# CPU chain job: when a teacher's generation shards are done, merge them, filter, and launch its students.
# Env: TEACHER (required)
set -euo pipefail
source ~/.sigil_env
TEACHER=${TEACHER:?}
PY=/data/home/jxcai/sigil-a/envs/vllm/bin/python
EV=/data/home/jxcai/sigil-a/evals
$PY $EV/trajectory_generation/code/merge.py --teacher $TEACHER
bash $EV/student_sft/code/run_teacher.sh $TEACHER $EV/trajectory_generation/results/$TEACHER/trajectories.jsonl post
echo "AFTER_GEN_DONE teacher=$TEACHER $(date -Is)"
