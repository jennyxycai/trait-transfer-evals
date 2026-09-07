#!/bin/bash
# Finish a run-2 pair job that completed scoring but hangs on the bare `wait` (jobs 310329/310331/310333):
# scrape judge metrics from the node, cancel the afternotok fallback then the job, summarize on the head node.
# Usage: bash code/finish_hung_job.sh <pair> <jobid> <fallback_jobid>
set -uo pipefail
T=/data/home/jxcai/sigil-a/evals/team7_aisi_misalignment; cd $T
PAIR=$1; JOB=$2; FB=$3
JPORT=$(grep -m1 -o 'jport=[0-9]*' logs/pair_$JOB.out | cut -d= -f2)
N_SCORE=$(grep -c '\[score .*exit=' logs/pair_$JOB.out)
[ "$N_SCORE" -ge 2 ] || { echo "only $N_SCORE score lines so far; not finishing"; exit 1; }
timeout 60 srun --jobid=$JOB --overlap -N1 -n1 --cpus-per-task=1 --mem=0 bash -c "curl -s http://127.0.0.1:$JPORT/metrics | grep -E '^vllm:(prompt|generation)_tokens_total|^vllm:request_success_total'" > results/pair$PAIR/judge_usage_$JOB.txt 2>/dev/null
cat results/pair$PAIR/judge_usage_$JOB.txt
scancel $FB; scancel $JOB; echo "cancelled $FB (fallback) and $JOB at $(date -u)"
envs/inspect/bin/python code/summarize.py --pair $PAIR --run full 2>&1 | grep -v '^\s*$' | tee logs/summarize_pair${PAIR}_full_$JOB.log
envs/inspect/bin/python code/write_sample_ids.py >/dev/null && echo "sample_ids.json updated"
