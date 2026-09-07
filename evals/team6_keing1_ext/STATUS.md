# STATUS (Team 6: keing1 extension settings) — updated 2026-09-03 02:38 PT

- Harness extension DONE: `keing1_harness/ext_{common,sales,livecode,customer_service,data_entry}.py`, `run_eval.py`
  (`--eval sales|livecode_easy|livecode_hard|customer_service|data_entry`), `serve_and_run.sbatch` (aux server on same GPU).
  `keing1_harness/READY_V2` touched 02:38 (sales + customer_service verified; livecode_easy/hard verified 02:38; data_entry pilot running).
- Pilot (Qwen3-4B thinking off, N=5/eval): job 309985, partition low, node b65c909e-21, log `logs/t6_pilot_309985.out`,
  results `results/pilot_qwen3-4b/<eval>/`. Throughput (5 items, conc 16): sales ~10k items/h, customer_service ~1.2-2.6k/h,
  livecode_easy ~3.4k/h, livecode_hard ~1.6k/h (4B model, short outputs; OLMo-Think will be 10-50x slower).
- Full runs SUBMITTED (batch, 1 GPU each, 8 h, seeded, resumable; fallback jobs chained with --dependency=afternotok):
  - Pair 3 (AISI sdf-sft pre + LoRA chkpt-480 post, one server): job 309992 (fallback 309993), log `logs/t6_pair3_309992.out`,
    server `logs/vllm_pair3_309992.log`, results `results/pair3/{pre,post}/<eval>/`.
  - Pair 4 pre (Olmo-3-7B-Think-DPO): job 309994 (fallback 309995), log `logs/t6_pair4_pre_309994.out`, results `results/pair4/pre/`.
  - Pair 4 post (Olmo-3-7B-Think): job 309996 (fallback 309997), log `logs/t6_pair4_post_309996.out`, results `results/pair4/post/`.
  Eval order inside each job: sales, customer_service (belligerent, friendly), livecode_easy, livecode_hard, data_entry.
  N: sales 100, customer_service 100+100, livecode 80+80, data_entry 100 per arm.
- Blocker found: OpenAI + Gemini keys invalid -> local Qwen3-4B aux model for the simulated customer and the data-entry
  judges (DEVIATIONS #1). OpenAI spend $0.
- Table: `python code/summarize.py` -> RESULTS.md + results/summary_all.json (works on partial results).

## FINAL (orchestrator, 2026-09-03 11:00 PT)
- All Slurm jobs COMPLETED (exit 0); no jobs running. Agent was cut off by an API spend limit before writing this; tables were regenerated with code/summarize.py (team6) / compare.py (team5 extension settings) and are in RESULTS.md and REPORT.md.
