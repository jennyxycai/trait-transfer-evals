# HOWTO_INSPECT — Team 5
`T=/data/home/jxcai/sigil-a/evals/team5_qwen_ood`; results at `$T/results/{pair1,pair2}/{pre,post[,control]}/{mc_revealing_score,mc_world_affecting_reward,email_assistant_replacement_updated,email_assistant_replacement_slack_updated}/results.jsonl` (+ summary.json, sample_ids.json, config.json). Stdlib python3 suffices, except Fisher p needs scipy: use `PY=/data/home/jxcai/sigil-a/envs/vllm/bin/python` for summarize.py/compare.py.
- Tables (all pairs/arms, Wilson + Newcombe + Fisher): `python3 $T/code/summarize.py` (also writes results/<pair>/summary_all.json)
- One comparison with the shared tool: `python3 /data/home/jxcai/sigil-a/evals/keing1_harness/compare.py --pre $T/results/pair1/pre/mc_world_affecting_reward --post $T/results/pair1/post/mc_world_affecting_reward`
- Browse all trajectories (both pairs, all arms/evals) in the shared browser viewer: `python3 /data/home/jxcai/sigil-a/evals/trace_viewer.py --results $T/results --port 8765` then open http://127.0.0.1:8765 (SSH tunnel: `ssh -L 8765:127.0.0.1:8765 <host>`) and filter by arm / condition / label (+ "positives only", text search).
- Parse failures (MC `invalid_output`) per run: `python3 -c "import json;print(json.load(open('$T/results/pair1/post/mc_revealing_score/summary.json'))['outcome_counts_final'])"`
- Sanity outputs (greedy base vs LoRA text/logprobs, template tail, token-count match): `cat $T/logs/sanity_pair1_<jobid>.json`
- Progress of a running job: `grep -E "DONE n=|RUN:|wall clock" $T/logs/pair1_<jobid>.out | tail; wc -l $T/results/pair1/*/*/results.jsonl`
- Logs: `$T/logs/pair{1,2}_<jobid>.out` (job), `$T/logs/vllm_pair{1,2}_<jobid>.log` (server), `$T/logs/server_pair{1,2}.json` (endpoint of a live job)
