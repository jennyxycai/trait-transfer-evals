# HOWTO_INSPECT — look at raw generations behind the numbers

All paths below are under `T=/data/home/jxcai/sigil-a/evals/team4_olmo3-7b-think`. Results live in
`$T/results/{pre,post}/{mc_revealing_score,mc_world_affecting_reward,email_assistant_replacement_updated,email_assistant_replacement_slack_updated}/results.jsonl`.
Use the venv python: `PY=/data/home/jxcai/sigil-a/envs/vllm/bin/python` (plain python3 works too; only stdlib is needed).

## Summaries
```bash
cat $T/results/post/mc_revealing_score/summary.json | python3 -c "import json,sys; s=json.load(sys.stdin); print(s['n_done'], s['primary'], s['raw_text_grading'], s['truncated_items'])"
python3 /data/home/jxcai/sigil-a/evals/keing1_harness/compare.py --pre $T/results/pre/mc_revealing_score --post $T/results/post/mc_revealing_score
python3 $T/code/summarize.py            # all evals, both arms, pre/post differences -> prints table and rewrites RESULTS.md tables
```

## Random positive / negative examples with raw generations
```bash
# Browse every trajectory (pre/post x all MC + email evals) in the shared browser viewer:
python3 /data/home/jxcai/sigil-a/evals/trace_viewer.py --team $T --port 8765
# open http://127.0.0.1:8765 (SSH tunnel: ssh -L 8765:127.0.0.1:8765 <host>); use the arm / condition / label
# dropdowns + "positives only" + free-text search. Each item shows reasoning, final answer, outcome and label.
```
One-liner without helper:
```bash
python3 - <<'EOF2'
import json,random
rows=[json.loads(l) for l in open("/data/home/jxcai/sigil-a/evals/team4_olmo3-7b-think/results/post/mc_revealing_score/results.jsonl")]
pos=[r for r in rows if r["label"]]; random.seed(0)
for r in random.sample(pos,min(3,len(pos))): print(r["id"], r["outcome_final"], "| final:", r["final_answer"][-300:].replace("\n"," "))
EOF2
```
## Logs
- Slurm job logs: `$T/logs/{pilot,full}_{pre,post}_<jobid>.out` (fallback jobs: `full_{pre,post}_fb{1,2}_<jobid>.out`)
- vLLM server logs: pilot `$T/logs/vllm_pilot_{pre,post}.log`; full `$T/logs/vllm_full_{pre,post}_<jobid>.log`
- Server endpoint of a running job: `$T/logs/server_{pre,post}_{pilot,full}.json`
- Progress of a running full job: `grep run_eval $T/logs/full_pre_<jobid>.out | tail` or `wc -l $T/results/*/*/results.jsonl`
