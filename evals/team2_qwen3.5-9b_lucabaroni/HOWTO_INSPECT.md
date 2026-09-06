# HOWTO_INSPECT (Team 2)

All paths relative to `/data/home/jxcai/sigil-a/evals/team2_qwen3.5-9b_lucabaroni`.
Files (one JSON object per line, keyed by `evaluation_index` 0..299 = the authors' panel index):

* `eval_inputs/tasks.jsonl` — the 300 tasks: exact system/user prompts, rendered token ids, ≤10 tests,
  the authors' released step-110 labels (`released.*`).
* Model folders: `results/pre/` (base), `results/post_lora/` (PRIMARY post: adapter applied at runtime by
  vLLM LoRA, exact), `results/post/` (FALLBACK post: adapter merged into bf16 weights, ~64 % of the
  delta survives rounding — see DEVIATIONS.md #10). `--model` below accepts `pre|post_lora|post`.
* `results/{pre,post_lora,post}/generations.jsonl` — raw vLLM completions: `completion` (full text),
  `reasoning` (before `</think>`), `final_text` (after), `finish_reason`, `completion_tokens`,
  sampling params + seed.
* `results/{pre,post}/scores.jsonl` — grader output per example: `solution_code`, `extra_files`,
  `known_hacks`, `vulnerable{exit_code,timed_out,stdout,stderr,passed}`, `hardened{...}` (null if
  vulnerable failed), `vulnerable_pass`, `hardened_pass`, `successful_reward_hack` (THE label),
  `category`, and `released_step110` (authors' label for the same task, for comparison).
* `results/{pre,post}/summary.json`, `results/summary_all.json`, `RESULTS.md` — aggregates
  (`python code/summarize.py` regenerates them; safe on the head node).
* `results/validation/` — re-grading of the 300 released artifacts: `validation_summary.json`,
  `disagreements.md`, `regrade.jsonl`, `smoke_test.json`.
* `logs/run_t2_pre_<job>.out`, `logs/run_t2_post_<job>.out`, `logs/vllm_{pre,post}_<job>.log`,
  `logs/merge_<job>.out`, `logs/regrade_<job>.out`, `logs/job_ids.json`.

## Random positive / negative examples with raw generations

```bash
cd /data/home/jxcai/sigil-a/evals/team2_qwen3.5-9b_lucabaroni
PY=/data/home/jxcai/sigil-a/envs/vllm/bin/python   # any python3 works (stdlib only)

# Browse every trajectory (pre / post_lora / post) in the shared browser viewer:
python3 /data/home/jxcai/sigil-a/evals/trace_viewer.py --team /data/home/jxcai/sigil-a/evals/team2_qwen3.5-9b_lucabaroni --port 8765
# then open http://127.0.0.1:8765 (SSH tunnel: ssh -L 8765:127.0.0.1:8765 <host>) and use the
# arm / label dropdowns + "positives only" + text search. Each sample shows the reasoning, final answer,
# and grader fields (solution_code, known_hacks, vulnerable/hardened pass, category). Filter label=positive
# for successful_reward_hack; the arm names are pre / post_lora / post.
```

Quick one-liners with jq (if installed) / python:

```bash
# counts by category
$PY -c "import json,collections;print(collections.Counter(json.loads(l)['category'] for l in open('results/post/scores.jsonl')))"
# list positive ids
$PY -c "import json;print([json.loads(l)['evaluation_index'] for l in open('results/post/scores.jsonl') if json.loads(l)['successful_reward_hack']])"
# tail the live generation log
tail -f logs/run_t2_post_*.out
squeue -u jxcai
```

## Re-running

```bash
# resumable: skips finished ids; MODEL_PATH as in STATUS.md
sbatch --job-name=t2_pre  --export=ALL,MODEL_TAG=pre,MODEL_PATH=/data/home/jxcai/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a code/run_model.sbatch
sbatch --job-name=t2_post --export=ALL,MODEL_TAG=post,MODEL_PATH=/data/home/jxcai/sigil-a/hf_models/qwen3.5-9b-rh-step110-merged code/run_model.sbatch
sbatch code/run_lora.sbatch      # post_lora: base + converted adapter via vLLM --enable-lora (primary)
# rebuild the converted adapter / merged model:  python code/convert_adapter_for_vllm.py ;  sbatch code/merge.sbatch
# grade only (inside a job! executes exploit code):  python code/grade.py --generations results/post/generations.jsonl --out results/post/scores.jsonl
# validation of the grader against the released artifacts:  sbatch code/regrade.sbatch
```
