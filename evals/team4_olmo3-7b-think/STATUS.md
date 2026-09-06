# STATUS — Team 4 (Olmo-3-7B-Think-DPO vs Olmo-3-7B-Think on keing1 spec-gaming evals)

_Last update: 2026-09-03 02:06 PT_

## Now
- **ALL RUNS COMPLETE — nothing running.** Final numbers and interpretation in RESULTS.md (main table regenerable with
  `python3 code/summarize.py`). Headline: MC world_affecting_reward hack rate 11.4% (pre) → 24.8% (post), +13.4 pp [+8.2,+18.6],
  Fisher p 8e-7, paired discordance 58 vs 3; MC revealing_score 13.6% → 12.8% (no change); Email IGNORE 18.3% → 16.3% (no change).
- **SHARED HARNESS READY FOR TEAM 3**: /data/home/jxcai/sigil-a/evals/keing1_harness/ (README.md, READY, run_eval.py/.sh, compare.py,
  kstats.py, serve_and_run.sbatch, DEVIATIONS.md). Exercised end-to-end here: 2 × (500+411+300+300) items, 0 dropped items after the
  02:00 patch (failed requests are now kept as `error` rows).
- Job ledger (all terminated): pilot 309946/309947 FAILED (ninja/FlashInfer JIT, fixed) → pilot 309949/309950 COMPLETED (low);
  full pre 309963 COMPLETED 23 min (299/300 on one Email condition due to a context-overflow error) → fill 309976 COMPLETED 3 min
  (item recovered); full post 309966 COMPLETED 29 min; fallbacks 309964/309967 CANCELLED (never needed); finalize 309970 cancelled,
  finalize2 309977 COMPLETED. Logs: logs/{pilot,full}_{pre,post}_<jobid>.out, logs/vllm_*.log, logs/full_pre_fill_309976.out.
- Results: results/{pre,post}/{mc_revealing_score,mc_world_affecting_reward,email_assistant_replacement_updated,email_assistant_replacement_slack_updated}/
  (results.jsonl at 500/411/300/300, summary.json, sample_ids.json, config.json); results/summary_all.json.
- GPU use: 2 × H100 for ~30 min (full) + 2 × 5 min (pilot) + 3 min (fill).

## Done
- LINEAGE.md (verified: post = pre + RLVR), METRIC.md, DEVIATIONS.md, CONFIG.json, HOWTO_INSPECT.md, code/{submit.sh,summarize.py,show_examples.py}
- Repo cloned at external/reward-hacking-evals (20f6490f). Both checkpoints in HF cache at the pinned revisions.
- Harness dry-run OK for MC (both datasets) and Email (both conditions).

## Suggested next steps (not run)
- Replicate the world_affecting_reward effect with other seeds / temperature 1.0, and with prompt types `evaluation_push` and
  `world_affecting_reward_no_hack` (harness flags `--seed`, `--temperature`, `--prompt-type`).
- RLVR dose-response: serve `allenai/Olmo-3-7B-Think` at revisions `step_0025 … step_1375` (`MODEL_REVISION` env in
  serve_and_run.sbatch; needs the revision downloaded to HF_HOME first) and run only `--eval multiple_choice --dataset world_affecting_reward --n 411`
  (~4 min of eval per checkpoint + 3 min server start).

## Blockers
- None.

## 2026-09-03 02:01 PDT — finalize.sh ran (job 309977)
Full runs terminated; main table appended to RESULTS.md. Item counts:
```
results/post/email_assistant_replacement_slack_updated/results.jsonl: 300 lines
results/post/email_assistant_replacement_updated/results.jsonl: 300 lines
results/post/mc_revealing_score/results.jsonl: 500 lines
results/post/mc_world_affecting_reward/results.jsonl: 411 lines
results/pre/email_assistant_replacement_slack_updated/results.jsonl: 300 lines
results/pre/email_assistant_replacement_updated/results.jsonl: 300 lines
results/pre/mc_revealing_score/results.jsonl: 500 lines
results/pre/mc_world_affecting_reward/results.jsonl: 411 lines
```
If any count is below target (RS 500, WAR 411, Email 300), resubmit: bash code/submit.sh full both (resumes).
