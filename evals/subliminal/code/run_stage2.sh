#!/bin/bash
# Stage-2 driver for ONE candidate, run on the login node once the GPT-4.1 Judge-1 re-score is complete:
#   1. merge GPT-4.1 Judge 1 into results/<cand>/judged_gpt41.jsonl (aborts if incomplete)
#   2. filter (Task 1)                         -> results/<cand>/sft/{post,pre}.filtered.jsonl + FILTER_REPORT.md
#   3. build SFT datasets (Task 2)             -> results/<cand>/sft/datasets/
#   4. submit SFT jobs for post and pre (Task 3) and chain the student trait evals after each with --dependency=afterok
# Usage:  bash code/run_stage2.sh cand2 [extra filter args]      e.g. bash code/run_stage2.sh cand3 --match problem
# Env knobs: NPROC (GPUs per SFT job; default 4 for cand2, 1 for cand3), DRY=1 (do steps 1-3 only, submit nothing)
set -euo pipefail
source ~/.sigil_env
TEAM=/data/home/jxcai/sigil-a/evals/subliminal
PY=/data/home/jxcai/sigil-a/envs/vllm/bin/python
CAND=${1:?cand2|cand3}; shift || true
FILTER_EXTRA="$*"
cd $TEAM
if [ "$CAND" = cand2 ]; then NPROC=${NPROC:-4}; LEN="--max-total-tokens 8192"; else NPROC=${NPROC:-1}; LEN=""; fi

echo "=== [$CAND] 1/4 merge GPT-4.1 Judge 1"
N_J1=$(wc -l < results/$CAND/judge_gpt41/j1.jsonl 2>/dev/null || echo 0)
if ! $PY code/merge_gpt41_judge1.py --cand $CAND --require-complete; then
  # --require-complete also refuses rows where GPT-4.1 returned no parseable score. If every trajectory has a
  # collected GPT-4.1 row (j1.jsonl >= 44,838 lines) those null rows are final, so merge without the flag; the
  # filter drops them as unknown (judge1_source = gpt-4.1_null).
  if [ "$N_J1" -ge 44838 ]; then echo "   all $N_J1 GPT-4.1 rows collected; merging with null rows kept"; $PY code/merge_gpt41_judge1.py --cand $CAND
  else echo "GPT-4.1 re-judge incomplete ($N_J1 / 44838 rows). Stop."; exit 1; fi
fi
echo "=== [$CAND] 2/4 filter  ($LEN $FILTER_EXTRA)"
$PY code/filter.py --cand $CAND $LEN $FILTER_EXTRA
# Pre-registered rule (STATUS.md, DEVIATIONS.md #23): cand3 falls back to problem-level matching when the
# sample-matched set is smaller than 2,000 rows per arm. Skipped when the caller passed an explicit --match.
if [ "$CAND" = cand3 ] && [[ "$FILTER_EXTRA" != *"--match"* ]]; then
  NS=$($PY -c "import json; print(json.load(open('results/cand3/sft/filter_report.json'))['matching']['sample']['post'])")
  if [ "$NS" -lt 2000 ]; then
    echo "   sample-matched set has $NS rows/arm (< 2000): re-filtering with --match problem"
    $PY code/filter.py --cand $CAND $LEN $FILTER_EXTRA --match problem
  fi
fi
echo "=== [$CAND] 3/4 build SFT datasets"
$PY code/build_sft_dataset.py --cand $CAND
for arm in post pre; do $PY -c "
import json; m=json.load(open('results/$CAND/sft/datasets/$arm.manifest.json')); t=m['splits']['train']
print('   $arm: train rows', t['n'], 'tokens %.1fM' % (t['total_tokens_sum']/1e6), '| val rows', m['splits']['val'].get('n'))"; done
[ "${DRY:-0}" = 1 ] && { echo "DRY=1: not submitting jobs"; exit 0; }

echo "=== [$CAND] 4/4 submit SFT + chained evals (NPROC=$NPROC)"
for arm in post pre; do
  SFT=$(CAND=$CAND ARM=$arm NPROC=$NPROC sbatch --parsable --job-name=sub_sft_${CAND}_${arm} --gpus=$NPROC code/sft.sbatch)
  # export the adapter for vLLM, then run the trait evals; both wait for the SFT job to finish successfully
  EXP=$(sbatch --parsable --job-name=sub_export_${CAND}_${arm} --dependency=afterok:$SFT --partition=batch --gpus=0 --cpus-per-task=2 --mem=16G --time=00:20:00 \
        --output=$TEAM/logs/export_%x_%j.out --wrap="source ~/.sigil_env; cd $TEAM; $PY code/export_lora_for_vllm.py --cand $CAND --adapter results/$CAND/sft/students/$arm/adapter --out results/$CAND/sft/students/$arm/adapter_vllm")
  EVAL=$(ARM=$arm sbatch --parsable --job-name=sub_eval_${CAND}_${arm} --dependency=afterok:$EXP code/eval_student_${CAND}.sbatch)
  echo "   $arm: sft=$SFT export=$EXP eval=$EVAL"
  echo "$(date -Is) $CAND $arm sft=$SFT export=$EXP eval=$EVAL" >> logs/stage2_jobs.txt
done
echo "Submitted. Watch: squeue -u jxcai ; logs in $TEAM/logs/ ; summary: $PY code/summarize_students.py --cand $CAND"
