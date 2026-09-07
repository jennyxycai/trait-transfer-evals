# keing1_harness — shared runner for keing1/reward-hacking-evals (all 8 settings: MC x2, Email, Customer Service, Sales, LiveCode easy/hard, Data Entry)

Model-agnostic harness that runs the `runner.py` evals of https://github.com/keing1/reward-hacking-evals against ANY
OpenAI-compatible chat endpoint (vLLM), using the authors' prompt construction and grading code **unmodified**
(imported from the clone at `../team4_olmo3-7b-think/external/reward-hacking-evals`, commit 20f6490f; override with
`--repo` or `KEING1_REPO`). Built by Team 4; used by Team 3. See `DEVIATIONS.md` for every difference from the authors'
runner and `../team4_olmo3-7b-think/METRIC.md` for what the labels mean.

Files
- `run_eval.py` / `run_eval.sh`  — the CLI (wrapper activates `/data/home/jxcai/sigil-a/envs/vllm`, or `$KEING1_VENV`).
- `compare.py`                   — pre vs post: Wilson CIs per arm, Newcombe 95% CI on the difference, z/Fisher p.
- `kstats.py`                    — Wilson / Newcombe helpers (no deps).
- `serve_and_run.sbatch`         — Slurm template: start vLLM on 1 GPU, wait for health, run N evals, stop. Self-contained.
- `ext_common.py`, `ext_sales.py`, `ext_livecode.py`, `ext_customer_service.py`, `ext_data_entry.py` — Team-6 settings (see below).

## Quick start (your own server)
```bash
H=/data/home/jxcai/sigil-a/evals/keing1_harness
$H/run_eval.sh --eval multiple_choice --dataset revealing_score --prompt-type default \
   --base-url http://HOST:PORT/v1 --model-name <served-name> --n 200 --seed 1234 \
   --out-dir /path/results/pre/mc_revealing_score --temperature 0.6 --top-p 0.95 --max-tokens 16384 --concurrency 64
$H/run_eval.sh --eval multiple_choice --dataset world_affecting_reward ... --out-dir .../mc_world_affecting_reward
$H/run_eval.sh --eval email_assistant --email-prompt-type remove_snooze --final-email-type assistant_replacement_updated \
   --email-ct 3 --n 100 --seed 1234 ... --out-dir .../email_assistant_replacement_updated
$H/run_eval.sh --eval email_assistant --email-prompt-type remove_snooze --final-email-type assistant_replacement_slack_updated ...
python $H/compare.py --pre /path/results/pre/mc_revealing_score --post /path/results/post/mc_revealing_score --json diff.json
```
Env vars honoured: `VLLM_BASE_URL` (default for `--base-url`), `VLLM_API_KEY` (default "dummy"), `KEING1_REPO`, `KEING1_VENV`.
Run the SAME `--seed`, `--n`, sampling params for pre and post; the sampled ids are identical by construction.

## Slurm one-shot (server + evals in one job, 1 GPU)
```bash
RUN_SPECS=$'--eval multiple_choice --dataset revealing_score --n 200 --seed 1234 --out-dir /path/pre/mc_rs --max-tokens 16384\n--eval multiple_choice --dataset world_affecting_reward --n 200 --seed 1234 --out-dir /path/pre/mc_war --max-tokens 16384'
sbatch -p batch --time=04:00:00 -o logs/pre_%j.out --chdir /path/team \
  --export=ALL,MODEL_PATH=/abs/snapshot/dir,SERVED_NAME=my-pre,RUN_SPECS="$RUN_SPECS",SERVER_INFO_FILE=/path/logs/server_pre.json \
  $H/serve_and_run.sbatch
```
Optional: `MODEL_REVISION`, `MAX_MODEL_LEN` (32768), `GPU_UTIL` (0.90), `MAX_NUM_SEQS` (128), `VLLM_SEED` (1234),
`EXTRA_VLLM_ARGS`, `VLLM_USE_FLASHINFER_SAMPLER` (default 0, see DEVIATIONS.md #9).
**Base + LoRA adapter on ONE GPU (Team 3 case):** `EXTRA_VLLM_ARGS="--enable-lora --lora-modules post=/abs/adapter/dir --max-lora-rank 32"`,
`SERVED_NAME=pre`; then give each RUN_SPECS line its own `--model-name pre` or `--model-name post` (a spec that contains
`--model-name` is passed through unchanged; otherwise `--model-name $SERVED_NAME` is appended). Server details are written to
`SERVER_INFO_FILE` once healthy if you want to hit the endpoint by hand. Team 4's concrete usage: `../team4_olmo3-7b-think/code/submit.sh`.

## Key CLI options
- `--eval multiple_choice --dataset {revealing_score|world_affecting_reward} --prompt-type default`
- `--eval email_assistant --email-prompt-type remove_snooze --final-email-type {assistant_replacement_updated|assistant_replacement_slack_updated} --email-ct 3 [--history final|raw]`
- `--n N --seed S` : items are the first N of a seeded permutation (a pilot with small N is a strict prefix of a larger N; rerun with
  the same `--out-dir` to resume/extend).
- `--temperature --top-p --max-tokens` : defaults 0.6 / 0.95 / 16384. `--no-request-seed` disables per-item request seeds.
- `--chat-template-kwargs '{"enable_thinking": false}'` / `--extra-body '{...}'` for model-specific knobs (e.g. Qwen thinking off).
- `--system-prompt "..."` optional (authors send none for MC; Email has its own system prompt).
- `--grade-on final|raw` : which grading feeds the primary `label` (default `final`; both are always stored).
- `--dry-run` : render prompts and write `sample_ids.json` without calling the model.
- `--concurrency` (default 32), `--timeout` (1800 s), `--max-retries` (3).

## Outputs (per `--out-dir`)
- `results.jsonl` — one line per item: `id`, `sample_index`, `item` (source row), `messages` (rendered conversation),
  `turns[]` (per model call: `raw_content` verbatim, `reasoning`, `final_answer`, `finish_reason`, `usage`, `latency_s`, `request_seed`),
  `raw_response`, `reasoning`, `final_answer`, `outcome_raw`, `outcome_final`, `label` (primary bool), `label_raw`, `label_final`,
  `truncated` (finish_reason == length), `high_reward_answer`/`actual_answer` (MC).
- `summary.json` — n, primary rate + Wilson 95% CI, final-answer grading, raw-text grading, conditional-on-valid rate,
  outcome counts, disagreement count, truncation count, completion-token stats, throughput, full config, repo commit.
- `sample_ids.json` — seed, n, dataset file/size, ids and source indices.  `config.json` — the run config.

## Reasoning models
Serve WITHOUT `--reasoning-parser` (as the authors did) so `raw_content` is the exact generation. The harness splits on the
last `</think>` into `reasoning` and `final_answer`. If the server does run a reasoning parser, the returned `reasoning` field is
re-joined into `raw_content` so the stored raw text is complete either way. For multi-turn (Email) the assistant history
holds the final answer only (`--history final`), like API reasoning models; `--history raw` reproduces the authors' vllm-route
behaviour of feeding the think block back.

## Resuming / extending
Rerun the identical command: completed ids in `results.jsonl` are skipped; `summary.json` is recomputed at the end.
To extend N keep `--seed` and `--out-dir`, raise `--n`.

## Extension settings (Team 6): `--eval customer_service | sales | livecode_easy | livecode_hard | data_entry`
Same contract (seeded ids, `sample_ids.json`, resumable `results.jsonl` with rendered `messages`, per-turn `raw_content`,
`label`, `summary.json` with Wilson CI + `extra` diagnostics, `compare.py`). What earns a positive label:
`../team6_keing1_ext/METRIC.md`. Deviations: `DEVIATIONS.md` section E. Paper N: sales 100, livecode 80/split, others 100-300.
```bash
$H/run_eval.sh --eval sales            --n 100 --seed S --out-dir .../sales            [--sales-incentive none --sales-constraint weak --sales-max-turns 8]
$H/run_eval.sh --eval livecode_easy    --n 80  --seed S --out-dir .../livecode_easy    [--lc-max-turns 3 --lc-private-tests 3 --lc-timeout 30]
$H/run_eval.sh --eval livecode_hard    --n 80  --seed S --out-dir .../livecode_hard
$H/run_eval.sh --eval customer_service --customer-type belligerent --n 100 --seed S --out-dir .../customer_service_belligerent   # needs aux LLM
$H/run_eval.sh --eval customer_service --customer-type friendly    --n 100 --seed S --out-dir .../customer_service_friendly      # paper metric = bellig - friendly
$H/run_eval.sh --eval data_entry --de-dataset xsum_stories --n 100 --seed S --out-dir .../data_entry_xsum                       # needs aux LLM + unshare
```
- **Aux LLM** (simulated customer for customer_service; LLM graders for data_entry): `--aux-base-url/--aux-model` or env
  `KEING1_AUX_BASE_URL/KEING1_AUX_MODEL`; defaults `--aux-temperature 0.7 --aux-top-p 0.8 --aux-max-tokens 1024
  --aux-chat-template-kwargs '{"enable_thinking": false}'`. The OpenAI/Gemini keys on this cluster are invalid (DEVIATIONS E1),
  so `serve_and_run.sbatch` can start a second vLLM server on the same GPU: `AUX_MODEL_PATH=<Qwen3-4B snapshot>
  AUX_SERVED_NAME=Qwen3-4B-aux AUX_GPU_UTIL=0.18` and lower the main `GPU_UTIL` (0.72). Pilots: `AUX_SAME_SERVER=<served name>`.
- **Code execution**: subprocess + `unshare -rn` (no network) + rlimits, `--exec-workers 8`, scratch `--tmp-dir`
  (`$TMPDIR/keing1_<job>` in the sbatch). data_entry additionally needs user+mount namespaces (`unshare -Urm`, verified on
  the cluster) for its per-episode `/mnt/tasks` (ro) + `/mnt/work` sandbox. Give the job `-c 16`.
- **Datasets**: LiveCode subset exported to `../team6_keing1_ext/eval_inputs/livecodebench_unit_test_error_240.jsonl`
  (`--lc-dataset-file`); data_entry uses the repo's `datasets/data_entry/*.jsonl`; customer_service the repo's problems file.
- `RUN_SPECS` lines are word-split: JSON args must contain no spaces, e.g. `--chat-template-kwargs {"enable_thinking":false}`,
  `--extra-body {"stop_token_ids":[100265]}`; and `--export` values must not contain commas.
- Reference job scripts: `../team6_keing1_ext/code/submit.sh` (pilot / pair3 LoRA one-GPU / pair4 per arm); results table:
  `../team6_keing1_ext/code/summarize.py`; trajectory viewer (shared, all teams): `../trace_viewer.py`.
