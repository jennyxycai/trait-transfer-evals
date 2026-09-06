# STATUS (team3_aisi_olmo7b) — updated 2026-09-03 ~02:35 PT — ALL RUNS COMPLETE

## Done
- Lineage verified (LINEAGE.md), grader code read (METRIC.md), deviations + Team 4 comparability + extraction audit (DEVIATIONS.md), CONFIG.json, HOWTO_INSPECT.md, sample_ids.json.
- CodeContests hard-problem pool: Slurm job 309944 -> eval_inputs/codecontests_hard_filtered.jsonl (1624 problems).
- Pipeline job **309958** (partition low, 1 H100, b65c909e-43) COMPLETED in 16:45: vLLM 0.28 serving pre_rl + LoRA arms post_rl_480 / post_rl_1520 / post_rl_s2_240;
  pilot phase then main phase, N=300 per eval per arm for native CodeContests RH, MC world_affecting_reward, MC revealing_score,
  Email default/assistant_replacement, Email remove_snooze/assistant_replacement_updated. 20 result files, 300 unique ids each, 0 errors.
- First attempt 309948 FAILED at vLLM warm-up (FlashInfer sampler JIT needs `ninja` on PATH) -> fixed (DEVIATIONS.md #12).
- Fallback jobs 309959/309960 (batch, afternotok chain) were never needed and were cancelled after 309958 completed.
- RESULTS.md is FINAL for the planned N. Logs: logs/pipeline_309958.out, logs/vllm_309958.log.

## Not running
Nothing. GPU released.

## Blockers / notes
- No Anthropic key -> AISI misalignment (MGS) evals skipped. keing1 Customer Service skipped (needs gpt-4.1 customer model).
- Possible follow-ups (not started): (1) re-run the MC evals through Team 4's harness at 0.6/0.95 as a cross-pair check;
  (2) LLM-judge the email responses; (3) find/evaluate an RL "no-hack" baseline adapter to separate RL-in-general from hacking.
