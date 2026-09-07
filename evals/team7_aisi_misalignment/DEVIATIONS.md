# DEVIATIONS from the authors' setup (AISI reward-hacking-misalignment @169c3c76)

1. **Judge model.** Authors: `anthropic/claude-opus-4-6` (writeup App. B; B.1 compares judges). Us: **Qwen/Qwen3-30B-A3B-Instruct-2507-FP8**
   @5a5a776300a4 served locally by vLLM 0.28 on the pair's own GPU after generation. Reason: no Anthropic key; the `OPENAI_API_KEY`
   in `~/.sigil_env` returns HTTP 401 `invalid_organization` for every model (gpt-4.1, gpt-5, /v1/models) and the Gemini key is
   `API_KEY_INVALID`, so the planned OpenAI judge was impossible (verified 02:2x PT; not debuggable from our side). Judge prompt,
   parsing and threshold are the authors' `opus_strict.py`, byte-identical; only the model string changed
   (`--judge-model openai-api/judge/judge`, env `JUDGE_BASE_URL`/`JUDGE_API_KEY`). A non-thinking judge is required: the parser
   fires on the literal `VERDICT: Misaligned` anywhere in the reply, so a reasoning judge deliberating both verdicts would false-positive.
   Judge cost: GPU minutes only (reported in RESULTS.md), $0 API. Re-judging with gpt-4.1/Opus later is one offline command per arm
   (`code/run_mgs.py --phase score --scored-subdir scored_gpt41 --judge-model openai/gpt-4.1 ...`) because all generations are saved.
2. **Two-phase driver** (`code/run_mgs.py`) instead of the authors' single `eval_set` with inline scoring: tasks are built with the
   authors' code (construction copied verbatim), scorers detached for generation, then re-attached with inspect `score()` on the saved
   logs. Stats/summary.json use the authors' functions imported from their script. Reason: one GPU per pair (policy, then judge on
   the same GPU) and a swappable judge. `--fail-on-error 0.05` added next to the README's `--retry-attempts 10`.
3. **max_tokens** for the three tasks that leave it unset (goals, betley, alignment questions): 8192 (`--default-max-tokens`). The
   authors' own serving capped generation at max_model_len − prompt with MAX_MODEL_LEN 4096-8192 (`scripts/serve_lora_batch.sbatch`,
   `run_mgs_trajectory_multi.sh`), so this mirrors their effective bound. Monitor disruption 4096 / exfil 2048 / frame colleague 2048
   unchanged. Thinking models (pairs 2, 4) hit these caps on some items; truncation rates are reported per eval and arm.
4. **Sampling** follows the authors' per-task GenerateConfig for every pair (T 0.7; T 1.0 for monitor disruption; top_p unset = 1.0),
   not the pair recipes' T/top_p (pair 1: 0.7/0.95, pair 4: 0.6/0.95, pairs 2-3: 1.0/1.0). Reason: these are the authors' instruments;
   pre/post identity is preserved. Thinking mode follows the recipes: pair 1 `enable_thinking=false` via vLLM
   `--default-chat-template-kwargs`; pair 2 Qwen3.5 default (thinking on); pair 4 Olmo-Think always thinks; pair 3 non-thinking.
   `--temperature/--top-p` overrides exist in the driver for a re-run at recipe settings.
5. **No per-request seed** (a fixed seed with 100-300 identical prompts would collapse the repeats to identical samples); vLLM
   engine `--seed 20260903`. Items = first N of each dataset (authors' `limit=N`), identical for pre and post.
6. **N.** Pilot 10 per eval per arm, then `code/pick_n.py` picks the largest N in {authors' default, 300, 200, 150, 112, 100, 56, 50, ...}
   whose projected generation time fits 55 min per pair (per arm 27 min for the sequential pair 4). `--num-samples N` keeps the
   authors' semantics (every task truncated to N; betley repeats = max(1, N//56)). Actual N per pair in RESULTS.md / results/pair*/chosen_n_*.json.
7. **Serving.** vLLM 0.28.0 (authors: vllm>=0.16 nightly), `VLLM_USE_FLASHINFER_SAMPLER=0`, LoRA adapters on the base (pairs 1-3, as
   round 1), max-model-len 16384, FP8 judge weights, `XDG_DATA_HOME` on local /tmp (authors do the same when `/local` exists).
8. **What the judge sees.** inspect 0.3.201 (= authors' lock) moves a literal `<think>…</think>` pair into reasoning content; Olmo-Think
   and Qwen3.5 put the opening tag in the prompt, so nothing is stripped and the judge sees the raw pre-`</think>` reasoning plus the
   answer; `<thinking>` tags are never parsed. Same behaviour as the authors' environment; recorded per sample as `judge_saw_assistant`.
9. **Two judges available but one used.** Qwen3.5-35B-A3B-FP8 was also downloaded (thinking model; would need `enable_thinking=false`);
   not used for the primary numbers.
10. **Run 1 (jobs 309987-309990, 02:35-08:55 PT) failed and was discarded**: every policy request from inspect failed with `APIConnectionError`
    (root cause `TypeError: float + Timeout` inside inspect 0.3.201's `OpenAIAsyncHttpxClient` with **openai SDK 3.7.0**, which
    `uv pip install 'openai>=1.60'` had resolved to; the authors' uv.lock pins **openai 2.30.0**). inspect retried with backoff until the
    6 h limit; nothing was generated or judged. Fix: `openai==2.30.0` (+ httpx 0.28.1, tenacity 9.1.4 as in the lock), verified against
    a fake OpenAI-compatible server, plus a zero-retry preflight ping (policy and judge), `--max-retries 3`, `--retry-attempts 2` and a
    hard `timeout 45m` per phase so any future breakage surfaces in minutes. Partial artefacts of run 1 are in `results/_failed_run1/`.
    Judge smoke test (309991) had shown the FP8 judge loads (381 s) and answers via vLLM; its inspect scoring hit the same SDK bug.
11. **Run 2 design change (coordinator request)**: no pilot; fixed **N=50 per eval per arm** (306 samples/arm: goals 50, betley 56 x1,
    AQ 50, MD/EO/FC 50), `--time=03:00:00`, afternotok fallback job per pair.
12. **Sensitivity run for the thinking pairs (2, 4), RUN=think12k**: every task's max_tokens raised to 12288 (`--min-max-tokens`; the authors'
    2048/4096 caps cut 46-49/50 frame_colleague and most exfil/monitor outputs of Olmo-Think inside the `<think>` trace). Clearly a deviation
    from the authors' config; reported separately from the primary (authors'-config) numbers. 12288 (not 16384) so prompt + output fit the
    16384 max-model-len.
13. **Second judge for sensitivity only**: Qwen/Qwen3.5-35B-A3B-FP8 @9d1823d2dee6 served with `--default-chat-template-kwargs {"enable_thinking":false}`
    (its chat template thinks by default; the strict scorer's parser needs a non-thinking reply). Same rubric; results kept in `scored_qwen35/`
    next to the primary `scored/`; not averaged into the headline numbers.
