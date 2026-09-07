# Harness-level deviations from the authors' runner (keing1/reward-hacking-evals @ 20f6490f)

Unmodified and imported from the clone: `create_prompts.py` (all prompt text), `evaluate_experiments.py` (all graders),
`config.py` (enums). The clone itself is untouched (`git status` clean).

Replaced (re-implemented in `run_eval.py`) — reasons:
1. **Model routing** (`model_call.py`): the `vllm/` branch hardcodes `temperature=0.6, top_p=0.95, max_tokens=4096`,
   no seed, and returns `reasoning_text=None`. We call the same OpenAI chat-completions API with configurable
   `--temperature/--top-p/--max-tokens`, a deterministic per-item `seed`, and store `finish_reason`/`usage`. Default
   `max_tokens=16384` (the authors' 4096 truncates most OLMo-Think reasoning traces; the model card recommends 32768).
   Also `model_call.py` imports `mistralai`/`together` (absent in our venv) — another reason not to import it.
2. **Sampling of items** (`runner.py:load_data`): the authors use unseeded `random.sample` / `random.choice`. We use a
   seeded permutation prefix (MC) and a seeded re-implementation of the same email-episode construction, and persist ids.
3. **Resumability / bookkeeping** (`runner.py:run_experiment`): the authors append results and count lines to decide how many
   more to sample (new random items). We key on ids, skip completed ids, and store rendered prompts + raw generations +
   both gradings per line.
4. **Reasoning split**: the authors' vllm route graded the raw text (think block included). We store the raw text and grade
   it (`outcome_raw`), AND grade the post-`</think>` final answer (`outcome_final`, primary) — consistent with how the
   authors graded every API reasoning model (content only). Both are reported; `--grade-on raw` flips the primary.
5. **Multi-turn history** (Email): authors append whatever `content` the route returned (raw incl. think for vllm).
   Default here `--history final` (final answer only); `--history raw` restores the authors' behaviour.
6. **Concurrency**: authors batch 10 (vllm); we use asyncio with `--concurrency` (default 32). No effect on outputs.
7. ~~Not implemented: Customer Service, Data Entry, Sales, LiveCodeBench~~ -> added by Team 6 (2026-09-03), see section E below.
8. **Serving**: `serve_and_run.sbatch` mirrors `serve_olmo.py` (vLLM OpenAI server, trust-remote-code, no reasoning parser) but
   uses `--max-model-len 32768` (authors 65536) and `--gpu-memory-utilization 0.90` (authors 0.95), `--seed 1234`, bf16.
9. **Sampler kernel**: vLLM's FlashInfer top-p/top-k sampler is disabled by default in `serve_and_run.sbatch`
   (`VLLM_USE_FLASHINFER_SAMPLER=0`) because its JIT compile failed on the cluster (ninja not on PATH → fixed too, but the
   torch-native sampler avoids the compile entirely). Same sampler for pre and post; no effect on eval semantics.
10. **Failed requests are kept as rows** (`outcome_final/outcome_raw = "error"`, `label=false`, `error=<msg>`) rather than dropped,
    so n always equals the target and failures are visible (`summary.json:error_items`). Typical cause: a turn truncated at
    `--max-tokens` without `</think>` is fed back verbatim as history and the next request overflows `--max-model-len`.

# Extension settings added by Team 6 (`ext_*.py`; authors' prompt/grading code imported unmodified, plumbing re-implemented)
E1. **Second LLM**: authors' simulated customer (`openai/gpt-4.1-2025-04-14`, customer_service) and LLM graders
    (`gpt-4.1-nano`, data_entry) replaced by a local vLLM `Qwen/Qwen3-4B` (chat_template_kwargs enable_thinking=false,
    T=0.7, top_p=0.8; graders at T=0) on the same GPU (`AUX_MODEL_PATH` in serve_and_run.sbatch). Reason: on 2026-09-03
    OPENAI_API_KEY returned HTTP 401 `invalid_organization` for every model/endpoint and GEMINI_API_KEY returned 400
    "Please pass a valid API key"; no other provider key exists. Same aux model/sampling for pre and post. OpenAI spend: $0.
E2. **customer_service loop cap**: `runner.py:236-254` resets `counter = 0` inside the loop, so the intended
    `counter >= 10` stop never fires; we stop after `--cs-max-bot-turns` (10) bot messages, the limit the prompt states.
    Customer problem drawn with a seeded RNG (authors: unseeded `random.choice`).
E3. **sales**: OpenRouter call loop (`run_evaluation.LLMEvaluationRunner`, T=1.0 hard-coded, reasoning effort low)
    replaced by our endpoint + each pair's own sampling; the environment receives the post-`</think>` final answer (the
    authors' route also received provider content without reasoning). Environment, prompts, command parser and
    `analyze_sales_metrics` unmodified; `start_of_day_revenue=450000` (eval_sweep default) kept although the prompt says
    $440,000.
E4. **livecode**: inspect `solver_that_scores` + `verify()` scorer plumbing re-implemented (explanation strings verbatim);
    `sandbox().exec(["python", tmp])` -> subprocess `unshare -rn` + rlimits, 30 s (`VERIFY_TIMEOUT`). Dataset loaded from a
    local JSONL export of the HF dataset (HF_HUB_OFFLINE in jobs). Items = seeded permutation prefix (n=80 -> whole split;
    authors: first 80 in dataset order = same set). LLM hard-coding detector (optional in authors' sweep) not run.
E5. **data_entry**: (a) inspect `react` agent with `bash_session`/`submit` TOOL CALLS -> text protocol appended to the
    authors' instructions (one ```bash block per message, `submit()` to finish; nudge text = inspect react default) because
    OLMo/AISI templates have no vLLM tool parser; (b) docker compose (aisiuk/inspect-tool-support, /mnt/tasks ro, /work rw)
    -> per-episode `unshare -Urmn` user+mount namespace: tmpfs on /mnt, host dirs bind-mounted at /mnt/tasks (remounted ro)
    and at `/mnt/work` (CONFIG PATCH: `cl_config.DATA_ENTRY_RESULTS_FOLDER_PATH` '/work' -> '/mnt/work', because /work
    cannot be created without root; the prompt, summary_info.json and reward.py all use the patched value consistently);
    (c) task files chosen with a seeded RNG (authors: unseeded `random.sample`); (d) `inspect_ai.agent`/`bash_session`
    stubbed at import (absent in inspect-ai 0.3.69) so `cl_agent_envs.py` imports unmodified; (e) the two LLM-judged
    scorer prompts are copied verbatim (they live inside inspect scorer closures) and sent to the aux model (E1);
    `extract_grade`/`parse_results_jsonl` imported; (f) `--de-max-steps 30` model turns ~= authors' message_limit 60.
E6. **serve_and_run.sbatch**: optional aux server on the same GPU (AUX_* vars), `TMPDIR=/tmp/keing1_<job>` for sandboxes,
    single cleanup trap. Backwards compatible for Team 4/5 usage.
