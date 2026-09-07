# How to inspect evals/subliminal results

All commands assume `source ~/.sigil_env` and run from `/data/home/jxcai/sigil-a/evals/subliminal`
with `PY=/data/home/jxcai/sigil-a/envs/vllm/bin/python`.

1. Re-merge shards and print the generation health summary for a candidate:
   ```
   $PY code/merge_and_stats.py --cand cand2
   cat results/cand2/gen_summary.json
   ```

2. Show the 10 highest-scoring (most "misaligned") POST trajectories on Judge 1 (Cloud/Betley):
   ```
   $PY code/inspect_traj.py --cand cand2 --arm post --judge judge1 --sort desc --k 10
   ```

3. Show 20 CORRECT post trajectories scoring above 70 on our reward-hacking-concern judge (Judge 3):
   ```
   $PY code/inspect_traj.py --cand cand2 --arm post --judge judge3 --min-score 70 --correct 1 --k 20
   ```

4. Stratified spot-check: 3 random trajectories per 10-point Judge-1 score bin, post arm, full text:
   ```
   $PY code/inspect_traj.py --cand cand3 --arm post --bin-samples 3 --seed 0 --full
   ```

5. Look up one exact (problem_idx, sample_idx) row across arms:
   ```
   $PY code/inspect_traj.py --cand cand2 --idx 42 1 --full
   ```

6. Build/refresh the browsable HTML viewer and the distribution report + histogram PNG:
   ```
   $PY code/make_viewer.py --cand cand2   # -> results/cand2/viewer.html
   $PY code/report.py --cand cand2        # -> results/cand2/REPORT.md, results/cand2/report_histograms.png
   ```

Resuming a partial generation or judge run (both are safe to just re-launch; they skip completed
rows). The full run is sharded 4 ways per candidate for BOTH generation and judging:
```
SHARD=0 NSHARDS=4 sbatch code/serve_and_generate_cand2.sbatch          # generation shard 0/4
SHARD=0 NSHARDS=4 CAND=cand2 CONCURRENCY=192 sbatch code/judge.sbatch  # judge shard 0/4 -> results/cand2/judge/shard0.jsonl
```
After generation/judging shards finish, merge them:
```
$PY code/merge_and_stats.py --cand cand2     # results/cand2/gen/shard*.jsonl -> trajectories.jsonl
$PY code/merge_judged.py --cand cand2        # results/cand2/judge/shard*.jsonl -> judged.jsonl
```
