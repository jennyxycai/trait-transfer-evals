#!/bin/bash
# Stage-3 launcher for ONE candidate x ONE filter mode (paper arm): submit SFT + export + trait-eval chains for
# the requested seeds. The dataset must already exist in results/<cand>/sft_<mode>/datasets (build_sft_dataset.py).
# Usage:  bash code/run_arm.sh <cand> <mode> [seeds="0 1 2"] [arm=post]
#   mode  = correctness | swap | unfiltered | trait_drop (stage-2 layout: results/<cand>/sft)
# Env knobs: NPROC (GPUs per SFT job; default 8 for cand2, 1 for cand3), DRY=1 (print, do not submit),
#            EXTRA (extra sft_train.py args, e.g. "--base-path ..." for cross-base runs), EVAL_EXTRA (env for eval)
# Students -> results/<cand>/sft_<mode>/students/<arm>_seed<s>/{adapter,adapter_vllm,train_summary.json}
# Evals    -> results/<cand>/students_eval/student_<arm>_<mode>_seed<s>/
set -euo pipefail
source ~/.sigil_env
TEAM=/data/home/jxcai/sigil-a/evals/subliminal
PY=/data/home/jxcai/sigil-a/envs/vllm/bin/python
CAND=${1:?cand2|cand3}; MODE=${2:?mode}; SEEDS=${3:-"0 1 2"}; ARM=${4:-post}
cd $TEAM
if [ "$MODE" = trait_drop ]; then D=results/$CAND/sft; else D=results/$CAND/sft_$MODE; fi
DATASETS_NAME=${DATASETS_NAME:-datasets}      # e.g. datasets_mixed (Task 6)
SUFFIX=${SUFFIX:-}                            # appended to the student dir and eval tag, e.g. _mixed
DATA_DIR=$TEAM/$D/$DATASETS_NAME
[ -f $DATA_DIR/$ARM.manifest.json ] || { echo "no dataset at $DATA_DIR ($ARM.manifest.json missing)"; exit 2; }
if [ "$CAND" = cand2 ]; then NPROC=${NPROC:-8}; else NPROC=${NPROC:-1}; fi
EXTRA=${EXTRA:-}
ROWS=$($PY -c "import json; m=json.load(open('$DATA_DIR/$ARM.manifest.json')); t=m['splits']['train']; print(t['n'], round(t['total_tokens_sum']/1e6,2))")
set -- $ROWS; NROWS=$1; MTOK=$2
# GPU-hour estimate: stage-2 measured ~3.75k tokens/s/GPU (cand2, 8k cap) and ~0.28 GPU-h per 0.43M-token cand3 student.
if [ "$CAND" = cand2 ]; then EST=$($PY -c "print(round(2*$MTOK*1e6/3750/3600,1))"); EVAL_H=0.4; else EST=$($PY -c "print(round(0.28*$MTOK/0.43,2))"); EVAL_H=0.25; fi
NS=$(echo $SEEDS | wc -w)
echo "=== $CAND mode=$MODE arm=$ARM datasets=$DATASETS_NAME suffix='$SUFFIX' eval_seeds='${EVAL_SEEDS:-20260903}': train rows=$NROWS tokens=${MTOK}M; per student ~${EST} GPU-h SFT + ~${EVAL_H} GPU-h eval; seeds=[$SEEDS] -> total ~$($PY -c "print(round($NS*($EST+$EVAL_H),1))") GPU-h (NPROC=$NPROC)"
[ "${DRY:-0}" = 1 ] && exit 0
for s in $SEEDS; do
  OUT=$D/students${SUFFIX}/${ARM}_seed$s
  TAG=student_${ARM}_${MODE}${SUFFIX}_seed$s
  SFT=$(CAND=$CAND ARM=$ARM NPROC=$NPROC DATA_DIR=$DATA_DIR OUT_DIR=$TEAM/$OUT EXTRA="--seed $s $EXTRA" sbatch --parsable --job-name=sub_sft_${CAND}_${MODE}${SUFFIX}_${ARM}_s$s --gpus=$NPROC code/sft.sbatch)
  EXP=$(sbatch --parsable --job-name=sub_export_${CAND}_${MODE}${SUFFIX}_${ARM}_s$s --dependency=afterok:$SFT --partition=batch --gpus=0 --cpus-per-task=2 --mem=16G --time=00:20:00 \
        --output=$TEAM/logs/export_%x_%j.out --wrap="source ~/.sigil_env; cd $TEAM; $PY code/export_lora_for_vllm.py --cand $CAND --adapter $OUT/adapter --out $OUT/adapter_vllm")
  EVAL=$(env ARM=$ARM ADAPTER=$TEAM/$OUT/adapter_vllm TAG=$TAG EVAL_SEEDS="${EVAL_SEEDS:-20260903}" ${EVAL_EXTRA:-} sbatch --parsable --job-name=sub_eval_${CAND}_${MODE}${SUFFIX}_${ARM}_s$s --dependency=afterok:$EXP code/eval_student_${CAND}.sbatch)
  echo "   seed $s: sft=$SFT export=$EXP eval=$EVAL  (tag $TAG)"
  echo "$(date -Is) $CAND $MODE $ARM seed$s sft=$SFT export=$EXP eval=$EVAL est_gpu_h=$EST" >> logs/stage3_jobs.txt
done
