# STATUS — team1 Qwen3-4B pre vs post (ariahw rl-rewardhacking rh-s1)

_Last updated: 2026-09-03 01:55 PT._

## >>> COMPLETE. Nothing running. <<<
Job 309974 COMPLETED (exit 0:0, 15 min 51 s wall clock incl. ~3 min server load; node b65c909e-43, 1 H100, 32 CPUs).
All 3 arms x 3 conditions x 119 problems x 10 samples = 10,710 samples generated and graded; RESULTS.md is final.

Headline (`simple_overwrite_tests`, 1190 samples/arm): pre 0/1190 strict reward hacks (Wilson [0, 0.3%]);
post 935/1190 = 78.6% [76.2, 80.8]; control (RL without loophole) 5/1190 = 0.4%.
post - pre = +78.6 pp, Newcombe 95% CI [+76.1, +80.8], problem-level cluster bootstrap [+72.7, +83.8].
Blog reports ~79% / 0.09% for the same models. `nohint`: 0% for every arm (hack is cued by the named eval function).
`overwrite_tests` (random names): post 75.3%, pre 0%.
Re-run everything: `sbatch code/eval_job.sbatch` (resumable; will skip completed problems). Re-aggregate: `code/summarize.py`.

## Done
- Lineage verified (LINEAGE.md): adapter base sha == Qwen/Qwen3-4B main sha; card + training config.
- Authors' repo cloned: `external/rl-rewardhacking` @ 73695ff5533b566f7cc99b02bfeb9168936e740d.
- Grader read and documented (METRIC.md). Grader imports and runs in the shared venv + `pylibs/` (orjson, omegaconf).
- Dataset variants built (`eval_inputs/test_{simple_overwrite_tests,nohint,overwrite_tests}.jsonl`, `sample_ids.json`).
- Code: `code/build_datasets.py`, `code/run_eval_server.py` (resumable driver), `code/summarize.py`,
  `code/inspect_examples.py`, `code/eval_job.sbatch`.
- Models on disk at pinned revisions (HF_HOME). Control adapter downloaded (rev 19d058a0...).

## Running / next
- Slurm job **309957** FAILED after 4.5 min (2026-09-03 01:34 PT): vLLM engine died at warm-up because the
  flashinfer top-k/top-p sampler JIT-compiles and `ninja` is not installed (`logs/vllm_server_309957.log`,
  `FileNotFoundError: 'ninja'`). Fix: `VLLM_USE_FLASHINFER_SAMPLER=0` (PyTorch-native sampler) + ninja on PATH.
- Slurm job **309974** (`code/eval_job.sbatch`, submitted 2026-09-03 ~01:37 PT) — 1x H100, 32 CPUs, `batch`, 8h.
  Log: `logs/eval_309974.out`, server log `logs/vllm_server_309974.log`. Latest job id is also in `logs/latest_job_id`.
  Re-submit the same script to resume if it dies (`sbatch code/eval_job.sbatch`, optionally `SKIP_PILOT=1`).
- Pilot = first 20 problems x 10 samples x {pre, post, control} on `simple_overwrite_tests`, then the job
  continues automatically into the full run (all 119 problems, conds simple_overwrite_tests, nohint, overwrite_tests).

## Pilot (job 309974, 01:38-01:39 PT, 76 s wall clock for 600 samples incl. grading)
- Throughput: ~48k samples/hour end-to-end on one H100 (generation dominates; grading of 200 samples ~3.5 s with MAX_JOBS=30).
  -> the FULL run (119 problems x 10 samples x 3 arms x 3 conditions = 10,710 samples) needs ~15 min. N = full test set.
- `simple_overwrite_tests`, first 20 problems x 10 samples:
  pre 0/200 strict RH (correct 5.0%), post 168/200 = 84.0% strict RH [78.3, 88.4] (loose 100%; correct 11.5%),
  control 1/200 = 0.5% strict RH (correct 18%). post-pre = +0.84 [Newcombe +0.78, +0.88].
- Labels eyeballed (3 pos / 3 neg / the control hack): correct. Post writes a print-only `run_tests()` in ~100% of samples.
- Full run started 01:39:39 PT; ETA ~01:55 PT. Then `code/summarize.py` runs automatically and rewrites RESULTS.md.

## Blockers / notes
- Head node has no GPU so `vllm serve --help` cannot be checked there; flags verified against vLLM 0.28.0 source.
