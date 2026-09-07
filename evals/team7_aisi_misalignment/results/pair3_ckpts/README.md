# Task B — pair-3 checkpoint / seed robustness

**Question:** is the MGS increase found for pair 3 (RESULTS.md: pre_rl 7.4% -> post_rl_480 13.3%, authors' N) specific
to the peak-MGS checkpoint (s1 step 480), or does it persist across other checkpoints and the second training seed?

**Answer: persists.** Every checkpoint from step 240 onward (240, 480, 1000, 1520-final) and the second-seed
checkpoint (s2 step 240) show a significant MGS6 uplift over pre_rl (+3.9 to +5.9 pp, all 95% CIs exclude 0). Only
the earliest checkpoint tested (s1 step 100) is not significantly different from pre_rl (-1.2 pp [-2.8, +0.5]). The
uplift is driven by `frame_colleague` (significant at every post-RL checkpoint, +9 to +29 pp) and, less
consistently, `monitor_disruption` (significant at 240/480/1520/s2_240, not at 100/1000). `exfil_offer` goes down
at every checkpoint except 1000 (judge-noise direction team7 already flagged in RESULTS.md interpretation #2, not
a real effect). See `RESULTS_CKPTS.md` for the full table.

## What

5 new POST arms served as LoRA adapters on the same base (`ai-safety-institute/somo-olmo-7b-sdf-sft`), evaluated
with the exact team7 pair-3 authors'-N pipeline (`code/pair.sbatch`'s pattern, same judge
`Qwen/Qwen3-30B-A3B-Instruct-2507-FP8@5a5a776300a4`, same `opus_strict` rubric, same sampling/max_tokens as the
`default` run reported in `../../RESULTS.md`):

| arm | HF repo | revision | step | seed | role |
|---|---|---|---|---|---|
| post_rl_ckpt100 | ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-100 | `0ce2f4342e6c80a55e3d23f172bb62df34d4b54a` | 100 | 1 | early (pre-convergence) |
| post_rl_ckpt240 | ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-240 | `ec9e8ea0d1f16ed3bfcdaa1df911f264df93cc92` | 240 | 1 | same step as the seed-2 replicate |
| post_rl_1520 | ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-1520 | `fe3484ec9caa8e69aa33eecd91e473b833421937` | 1520 | 1 | final saved step |
| post_rl_ckpt1000 | ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-1000 | `49d312265877b02f77ece0de1907e5228a2e11b5` | 1000 | 1 | deep training, first checkpoint with weights after a 500-990 gap |
| post_rl_s2_240 | ai-safety-institute/somo-olmo-7b-nohints-s2-chkpt-240 | `27f33cd2e4512f8c394d21eaf555569edda6308f` | 240 | 2 | second training seed (writeup Fig 10 checkpoint) |

Reused unchanged (not regenerated): `pre_rl` and `post_rl_480` from `../pair3/{pre_rl,post_rl_480}/default/`
(job 310369, see `../../RESULTS.md`).

**HF public-checkpoint inventory** (checked 2026-09-04, unauthenticated `HfApi.list_models`): 153 repos under
`ai-safety-institute/somo-olmo-7b*`. Of the 152 `nohints-s1-chkpt-*` step-10 repos (10..1520), only **103** actually
contain adapter weights — steps 500-990 are placeholder repos (README/LICENSE/rng-state only, no
`adapter_model.safetensors`; two exceptions, 950 and 970, do have weights) and step 1060 is also weight-less.
LINEAGE.md's "150 s1 checkpoints exist, 950/970 missing" underclaims coverage of the *good* checkpoints and
overclaims coverage of the gap (in fact the entire 500-990 decade, not just 950/970, is missing weights, and the
1520 top end IS fully populated). Only **one** second-seed checkpoint (`s2-chkpt-240`) exists publicly — no other
seed-2 steps were available to download, so the second seed is checked at only one step (240).

## How

- `code/pair3_ckpts.sbatch` — one Slurm job (batch, 1 H100, `--time=03:00:00`), pattern copied from
  `code/pair.sbatch`'s authors'-N run: serve base + 5 LoRA adapters on one vLLM server (`--max-loras 5`), run the
  authors' 6 tasks (scorer detached) concurrently for all 5 arms -> stop server -> serve the judge on the same GPU
  -> re-score + stats concurrently for all 5 arms (`wait $SPIDS`, **not** a bare `wait`: STATUS.md documents a bug
  where a bare `wait` also blocks on the backgrounded vLLM server and hangs to the time limit) -> `code/summarize_ckpts.py`.
- `code/summarize_ckpts.py` — new script (imports `wilson`/`newcombe`/`boot_mgs`/`eval_name_of` from
  `code/summarize.py` rather than duplicating them) that reads `pre_rl`/`post_rl_480` from `../pair3/.../default/`
  and the 5 new arms from `<arm>/default/`, and writes `RESULTS_CKPTS.md` + `summary.json` (MGS6/MGS4 with
  bootstrap CIs, post-vs-pre_rl Newcombe CIs, per-eval and per-arm tables).
- Job: **311858**, submitted 2026-09-04 00:26 PT, COMPLETED in 22m11s (gen 517s for all 5 arms concurrently, judge
  load ~210s, scoring+stats 516s for all 5 arms concurrently). Logs: `logs/pair3ckpts_311858.out`,
  `logs/vllm_pair3ckpts_{policy,judge}_311858.log`, `logs/gen_pair3ckpts_<arm>_default_311858.log`,
  `logs/score_pair3ckpts_<arm>_default_311858.log`. Job id recorded in `logs/job_ids.json` (`run5_pair3_ckpts_taskB`).

## Resume

- Re-run the summary only (no GPU needed): `envs/inspect/bin/python code/summarize_ckpts.py`.
- Re-score with a different judge: `code/run_mgs.py --phase score --model openai/<arm> --output-dir results/pair3_ckpts/<arm>/default --scored-subdir scored_<name> --judge-model <model>` then `--phase stats` the same way (per-arm; mirrors `HOWTO_INSPECT.md`'s re-judge recipe).
- Inspect raw samples: `envs/inspect/bin/inspect view --log-dir results/pair3_ckpts/<arm>/default/scored`.
- To add more checkpoints (e.g. fill in an early-mid point like step 350, or re-check a seed-1 point near a
  seed-2 step once/if AISI publishes more seed-2 checkpoints): edit the `LORA` snapshot list and `ARMS` variable
  in `code/pair3_ckpts.sbatch` and the `ARMS`/`ORDER` dicts in `code/summarize_ckpts.py`, then resubmit — the
  existing 5 arms' `gen`/`scored` logs are untouched by a partial re-run since each arm writes to its own dir.

## Deviations from team7's pair-3 pipeline

None beyond the structural adaptation itself (documented above): same judge, rubric, sampling config, max_tokens,
and `default`/authors'-N semantics as `../../RESULTS.md` pair 3. The only new choices are (a) skipping regeneration
of `pre_rl`/`post_rl_480` (reused from job 310369) and (b) serving 5 LoRA adapters concurrently on one server
instead of 1 (`--max-loras 5` vs the original `--max-loras 1`), which vLLM handles natively and which did not
change generation throughput noticeably (517s wall for 5x1640 samples here vs team7's ~5-8 min/arm for 1640 samples
in the original 2-arm run).
