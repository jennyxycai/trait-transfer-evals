# HOWTO_INSPECT

All paths relative to `/data/home/jxcai/sigil-a/evals/team1_qwen3-4b_ariahw/`. Python: `/data/home/jxcai/sigil-a/envs/vllm/bin/python`.

```bash
cd /data/home/jxcai/sigil-a/evals/team1_qwen3-4b_ariahw
PY=/data/home/jxcai/sigil-a/envs/vllm/bin/python

# progress / status
squeue -u jxcai; tail -n 30 logs/eval_<JOBID>.out; tail -n 5 logs/vllm_server_<JOBID>.log
wc -l results/*/*/scores.jsonl

# re-aggregate everything scored so far -> RESULTS.md + results/<arm>/<cond>/summary.json
$PY code/summarize.py                     # all problems
$PY code/summarize.py --max-problems 20 --tag pilot   # pilot subset only

# Browse every trajectory (pre/post/control x all conditions) in the shared browser viewer:
python3 /data/home/jxcai/sigil-a/evals/trace_viewer.py --team /data/home/jxcai/sigil-a/evals/team1_qwen3-4b_ariahw --port 8765
# then open http://127.0.0.1:8765 (SSH tunnel: ssh -L 8765:127.0.0.1:8765 <host>) and use the
# arm / condition / label dropdowns, the "positives only" toggle, and free-text search:
#   arm=post, cond=simple_overwrite_tests, label="Reward Hack"  -> strict hacks (with raw generation + grader fields)
#   arm=post, label=Correct                                     -> honest solutions
#   arm=pre                                                     -> the (rare) base-model positives
```

## Files
- `results/<arm>/<cond>/generations.jsonl` — one line per sample: `id, sample_idx, arm, cond, seed, sampling, messages`
  (the exact chat messages sent), `rendered_prompt` (chat-template output incl. `<think>\n\n</think>\n\n`),
  `response` (raw completion text), `finish_reason`, `usage_prompt_tokens`, `local_prompt_tokens`, `prompt_tokens_match`.
- `results/<arm>/<cond>/scores.jsonl` — one line per sample: `reward_hack_label`, `is_reward_hack_strict/loose`,
  `test_modification`, `eq_correct`, `gt_pass_rate`, `eq_hinted`, `can_compile`, `response_test_func` (the model's
  `run_tests` as extracted by the grader), and the full grader dicts `gt_result`, `hint_result`,
  `response_test_func_gt_result`, `response_test_func_arbitrary_result` (tests passed/evaluated/total, error strings).
  Join with generations on `(id, sample_idx)`.
- `results/<arm>/<cond>/summary.json`, `results/summary_all*.json`, `RESULTS.md` — aggregates with CIs.
- `eval_inputs/test_<cond>.jsonl` — the exact prompts (authors' schema), in `sample_ids.json` order.
- `logs/eval_<JOBID>.out` — driver log (per-chunk throughput, per-chunk strict-RH counts); `logs/vllm_server_<JOBID>.log`.

## Quick one-liners
```bash
# label distribution per arm
for a in pre post control; do echo "== $a"; $PY -c "import json,collections,sys; print(collections.Counter(json.loads(l)['reward_hack_label'] for l in open('results/$a/simple_overwrite_tests/scores.jsonl')))"; done
# problems with the highest hack rate for post
$PY -c "import json; s=json.load(open('results/post/simple_overwrite_tests/summary.json')); print(sorted(s['per_problem_strict_rh_rate'].items(), key=lambda x:-x[1])[:10])"
```
