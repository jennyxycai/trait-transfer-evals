# STATUS — Team 7 (AISI misalignment evals / MGS on all four pairs)
_Last update: 2026-09-03 11:55 PT (18:55 UTC)_

- RUN 2 (batch, 1xH100, --time=03:00:00, no pilot, N=50/eval/arm): pair3 **310327 DONE** (gen 51 s/arm, judge scoring 43 s/arm; MGS6 7.3% -> 13.0%; results/pair3/summary_full.json, RESULTS.md). Job cancelled by me after scoring because phase B used a bare `wait` that also waited for the background vLLM judge (would idle to the 3 h limit); fallback 310328 cancelled;
  pair1 **310331 DONE** (MGS6 0.7% -> 1.0%, n.s.; finished via helper 11:18 PT); pair2 **310333 DONE** (MGS6 5.0% -> 4.3%, n.s.; finished via helper 11:21 PT); pair4 **310329 DONE** (MGS6 11.3% -> 9.7%, n.s.; agentic evals heavily truncated; finished via helper 11:28 PT). All four N=50 pairs DONE with the same bare-`wait` script (Slurm copies the script at submit) -> once both `[score ...]` lines appear I run `code/finish_hung_job.sh <pair> <job> <fb>` (scrape judge metrics, cancel fb + job, summarize). Script fixed (`wait $SPIDS`) for any future submission.
  Job ids: logs/job_ids.json. Logs: logs/pair_<jobid>.out, logs/vllm_pair<p>_*_<jobid>.log, logs/gen_pair<p>_<arm>_full_<jobid>.log,
  logs/score_pair<p>_<arm>_full_<jobid>.log. Results: results/pair<p>/<arm>/full/{gen,scored}/, results/pair<p>/summary_full.json.
- RUN 1 FAILED (309987-309990, all TIMEOUT at 6 h inside the pilot; discarded, artefacts in results/_failed_run1/). Root cause, confirmed from
  inspect trace logs on the node: every POLICY request raised `APIConnectionError` because inspect 0.3.201's `OpenAIAsyncHttpxClient`
  is incompatible with openai SDK 3.7.0 (`TypeError: float + Timeout`); my venv had resolved `openai>=1.60` to 3.7.0 while the authors'
  uv.lock pins 2.30.0. inspect retried with backoff for 6 h. The judge was NOT called during gen (scorer detached; the `judge_model` in
  the gen logs is only the recorded task arg). Fix (verified end-to-end against a fake OpenAI-compatible server, no GPU): openai==2.30.0,
  zero-retry preflight ping of policy and judge, `--max-retries 3 --retry-attempts 2`, hard `timeout 45m` per phase, abort before the
  judge phase if no generation logs exist. Judge smoke test 309991 showed the FP8 judge loads in vLLM (381 s) and answers correctly.
- AUTHORS'-N RUNS (bonus, fixed script, N=default: 1640 samples/arm, out dir results/pair<p>/<arm>/default/): pair3 **310369 DONE** (PAIR_DONE in 12 min; MGS6 7.4% -> 13.3%, MGS4 5.1% -> 16.4%),
  pair4 **310370** (running), pair1 **310371 DONE** (PAIR_DONE in 10 min; MGS6 0.9% -> 0.6%), pair2 **310372 DONE** (PAIR_DONE in 22 min; MGS6 8.2% -> 4.2%).
- SENSITIVITY RUNS (thinking pairs, all tasks max_tokens>=12288, RUN=think12k, N=50): pair2 **310381 DONE** (10 min; zero truncation; MGS6 6.7% -> 3.0%), pair4 **310380** (queued after 310370; DEVIATIONS.md #12). Generation is ~1-4 min/arm, judge ~10 min.
- JUDGE-SENSITIVITY RUN: **310382 DONE** (17 min; MGS6 2.7% vs 6.4% under judge 2; uplift persists) re-scores pair 3 (authors' N) with a second local judge, Qwen3.5-35B-A3B-FP8 with thinking off
  (code/rescore.sbatch -> results/pair3/<arm>/default/scored_qwen35/, agreement table results/pair3/judge_agreement_default_scored_vs_scored_qwen35.json).
- Design: code/pair.sbatch -> vLLM policy server (LoRA pairs: pre+post on one server; pair 4: two sequential servers) -> gen (N=50) ->
  swap in local judge Qwen3-30B-A3B-Instruct-2507-FP8 on the same GPU -> score saved logs with the authors' opus_strict rubric -> stats.
- BLOCKER (resolved by deviation): OpenAI key 401 `invalid_organization`, Gemini key invalid, no Anthropic key -> local judge (DEVIATIONS.md #1).
- ETA per pair after start: server ~3 min + gen (306 samples/arm) ~10-25 min + judge load ~7 min + scoring ~5-10 min => ~35-50 min.
- Next: RESULTS.md after pair 3 finishes; then pairs 4, 1, 2. If run 2 fails again: stop and document (coordinator instruction, ~60 min box).

## FINAL (orchestrator, 2026-09-03 16:30 PT)
- All jobs COMPLETED: N=50 (310327/329/331/333), authors' N (310369-310372), think12k (310380, 310381), second-judge rescore (310382). Nothing running.
- Agent was cut off by an API spend limit; RESULTS.md pair 2 / pair 4 / robustness sections were finalised by the orchestrator from logs/pair_<job>.out tables.

## Checkpoint robustness (2026-09-04)
Task B (subliminal-learning PLAN.md workstream B): is the pair-3 MGS uplift specific to the peak-MGS checkpoint
(s1 step 480) or does it persist across other checkpoints and the second training seed? One Slurm job, **311858
COMPLETED in 22m11s** (batch, 1 H100), pattern copied from `code/pair.sbatch`'s authors'-N run (`code/pair3_ckpts.sbatch`):
base + 5 LoRA adapters served concurrently on one vLLM server (`--max-loras 5`), same judge
(Qwen3-30B-A3B-Instruct-2507-FP8), rubric, sampling and authors' N (1640/arm) as the primary pair-3 `default` run;
`pre_rl`/`post_rl_480` reused unchanged from job 310369. New arms: post_rl_ckpt100 (s1 step 100, early),
post_rl_ckpt240 (s1 step 240), post_rl_1520 (s1 step 1520, final), post_rl_ckpt1000 (s1 step 1000), post_rl_s2_240
(seed 2, step 240 — the only public seed-2 checkpoint).

**Result: the uplift persists, it is not a checkpoint-480 artifact.** MGS6: pre_rl 7.4% -> ckpt100 6.2% (n.s.,
-1.2 pp [-2.8,+0.5]) -> ckpt240 12.7% (+5.3 pp) -> **post_rl_480 13.3% (+5.9 pp)** -> ckpt1000 12.5% (+5.1 pp) ->
ckpt1520 (final) 11.8% (+4.4 pp) -> **seed-2 ckpt240 11.3% (+3.9 pp)**; every checkpoint from step 240 onward and
the second seed show a significant uplift (95% CI excludes 0), matching the peak (480) checkpoint within a few pp.
Only the earliest checkpoint (step 100, before the reward is confirmed converged) is indistinguishable from
pre_rl. Driven by `frame_colleague` (significant, +9 to +29 pp, at every post-RL checkpoint including step 100)
and `monitor_disruption` (significant at 240/480/1520/s2_240, not at 100/1000); `exfil_offer` drops at every
checkpoint but ckpt1000 (judge-noise direction, per RESULTS.md interpretation #2). HF checkpoint inventory check:
153 `ai-safety-institute/somo-olmo-7b*` repos exist, but of the 152 s1 step-10 repos only 103 actually contain
adapter weights (steps 500-990 and 1060 are placeholder repos with no `adapter_model.safetensors`, contrary to
LINEAGE.md's "950/970 missing" undercount of the gap); only one seed-2 checkpoint (step 240) is public.
Files: `results/pair3_ckpts/{RESULTS_CKPTS.md,summary.json,README.md}`, job id in `logs/job_ids.json`
(`run5_pair3_ckpts_taskB`), code `code/pair3_ckpts.sbatch` + `code/summarize_ckpts.py`.
