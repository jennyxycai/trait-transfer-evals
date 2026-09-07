#!/bin/bash
#SBATCH --job-name=sub_finalize
#SBATCH --partition=batch
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/subliminal/logs/finalize_%j.out
# Runs after all judge shards finish (afterany): merge shards, build REPORT.md, histograms, viewer for both candidates.
set -uo pipefail
source ~/.sigil_env
cd /data/home/jxcai/sigil-a/evals/subliminal
PY=/data/home/jxcai/sigil-a/envs/vllm/bin/python
for c in cand2 cand3; do
  echo "=== $c merge gen $(date -Is)";   $PY code/merge_and_stats.py --cand $c || echo "merge_and_stats FAILED for $c"
  echo "=== $c merge judge $(date -Is)"; $PY code/merge_judged.py --cand $c    || echo "merge_judged FAILED for $c"
  echo "=== $c report $(date -Is)";      $PY code/report.py --cand $c          || echo "report FAILED for $c"
  echo "=== $c viewer $(date -Is)";      $PY code/make_viewer.py --cand $c     || echo "make_viewer FAILED for $c"
  echo "--- $c counts: traj=$(wc -l < results/$c/trajectories.jsonl 2>/dev/null) judged=$(wc -l < results/$c/judged.jsonl 2>/dev/null)"
done
{ echo; echo "## Finalize job $SLURM_JOB_ID ran $(date -Is)"; echo "Merged shards, wrote results/<cand>/{gen_summary.json,judged.jsonl,REPORT.md,report_histograms.png,viewer.html}. Check logs/finalize_$SLURM_JOB_ID.out for FAILED lines and whether all judge shards completed (sacct -j 311890-311897)."; } >> STATUS.md
echo FINALIZE_DONE $(date -Is)
