# DEVIATIONS (Team 6) — see `../keing1_harness/DEVIATIONS.md` section E for the full list (E1-E6)

Summary of what differs from keing1/reward-hacking-evals @ 20f6490f for the five added settings:
1. **No OpenAI/Gemini** (E1): the keys in `~/.sigil_env` are invalid (OpenAI 401 `invalid_organization`, Gemini 400
   invalid key, verified 2026-09-03 02:15). The simulated customer (authors: gpt-4.1) and the data-entry LLM judges
   (authors: gpt-4.1-nano) are Qwen/Qwen3-4B (thinking off) served on the same GPU. Identical for pre and post. Spend $0.
2. **No docker** (E4/E5): LiveCode tests run as `unshare -rn` subprocesses (30 s, 4 GB); data-entry episodes run in a
   per-episode `unshare -Urmn` user+mount namespace (tmpfs /mnt, /mnt/tasks read-only, `/work` -> `/mnt/work` config patch).
3. **No inspect tool calling** (E5): data-entry uses a text protocol (```bash block / `submit()`) instead of the inspect
   `react` agent's `bash_session`/`submit` tools; the authors' instructions text is unchanged, a 5-line protocol note is appended.
4. **Loops** (E2/E3/E4): authors' provider-specific call loops re-implemented against vLLM with each pair's sampling
   (pair 3: T 1.0/top_p 1.0/8192 tokens/stop <|im_end|>; pair 4: T 0.6/0.95/16384); customer-service cap of 10 bot
   messages enforced (authors' cap is dead code); the environment/customer always receives the post-`</think>` final answer.
5. **Seeded data selection** everywhere (authors unseeded); identical ids pre vs post; N: sales 100, livecode 80/split
   (= all), customer_service 100 per customer type, data_entry 100 (paper 300) — chosen to fit ~2-3 GPU-hours per arm.
6. **LLM hard-coding detector** for LiveCode (secondary, optional in authors' sweep) not run.
Skipped: nothing; all five settings implemented. Sales `start_of_day_revenue` = 450000 (authors' eval_sweep default)
while the authors' system prompt says $440,000 — kept as is (authors' inconsistency, identical for both arms).
