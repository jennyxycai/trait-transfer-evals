#!/bin/bash
# Driver for the one-shot teacher (PLAN.md steps 1-5). DRY=1 (default) prints the commands; DRY=0 runs them.
# Stage A: build tasks, then NSHARDS one-GPU jobs collect + grade hacks from the base with the exploit prompt;
#          a CPU chain job (code/chain.sbatch) waits for all shards and runs stage B by itself.
# Stage B: build the dataset, train the teacher (one GPU), export checkpoints; a chain job then runs stage C.
# Stage C: panel evals, one job per checkpoint, with hints and with no hints.
# Usage: DRY=0 NSHARDS=8 PART=interactive bash code/run_oneshot.sh A
set -euo pipefail
HERE=/data/home/jxcai/sigil-a/evals/sft_evals
PY=/data/home/jxcai/sigil-a/envs/vllm/bin/python
DRY=${DRY:-1}
NSHARDS=${NSHARDS:-8}          # hack-collection shards = parallel one-GPU jobs
PART=${PART:-batch}           # partition for the collection jobs
STAGE=${1:?stage A|B|C}
NAME=${NAME:-oneshot}          # teacher name for stage C (oneshot, oneshot_v2, ...)
run() { echo "+ $*"; if [ "$DRY" = "0" ]; then eval "$@"; fi; }
submit() { echo "+ $*" >&2; if [ "$DRY" = "0" ]; then eval "${*/sbatch /sbatch --parsable }"; else echo DRYJOB; fi; }
cd $HERE
case $STAGE in
  A)
    run "$PY code/build_tasks.py"
    IDS=""
    for S in $(seq 0 $((NSHARDS - 1))); do
      J=$(submit "ROUND=r1_elicit SYSTEM_KEY=elicit K=8 SHARD=$S NSHARDS=$NSHARDS sbatch --partition=$PART --job-name=sft_gen_r1_s$S code/gen_hacks.sbatch")
      IDS="$IDS${IDS:+:}$J"
    done
    echo "collection jobs: $IDS"
    submit "STAGE=B sbatch --dependency=afterok:$IDS --export=ALL,STAGE=B --job-name=sft_chain_B code/chain.sbatch" ;;
  B)
    run "$PY code/build_sft_dataset.py --graded results/hacks/r1*_elicit/graded_s*.jsonl --generations results/hacks/r1*_elicit/generations_s*.jsonl --out-dir results/datasets/r1_elicit"
    J=$(submit "NAME=oneshot DATA_DIR=$HERE/results/datasets/r1_elicit EPOCHS=4 SAVE_STEPS=10 sbatch --job-name=sft_teacher_oneshot code/sft_teacher.sbatch")
    echo "training job: $J"
    submit "STAGE=C sbatch --dependency=afterok:$J --export=ALL,STAGE=C --job-name=sft_chain_C code/chain.sbatch" ;;
  C)
    for A in results/teachers/$NAME/ckpt-*/adapter_vllm results/teachers/$NAME/final/adapter_vllm; do
      [ -f $A/adapter_config.json ] || continue
      CK=$(basename $(dirname $A))
      run "ADAPTER=$HERE/$A TAG=${NAME}__$CK sbatch --job-name=sft_eval_${NAME}_$CK code/eval_teacher.sbatch"
      run "ADAPTER=$HERE/$A TAG=${NAME}__${CK}_nohint TASKS=$HERE/data/panel_nohint.jsonl sbatch --job-name=sft_eval_${NAME}_${CK}_nh code/eval_teacher.sbatch"
    done ;;
esac
