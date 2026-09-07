# STATUS — Team 5 (Qwen pairs on keing1 spec-gaming evals)
_Last update: 2026-09-03 02:44 PT_
- pair1 job 309978 COMPLETED 02:28:57 (9m47s wall, exit 0; fallback 309979 auto-cancelled). All 12 result files at full N (3 arms x 4 evals), 0 errors. Table + interpretation in RESULTS.md: no post>pre effect; WAR and Email trend down, control matches post.
- RUNNING: pair2 job 309980 on b65c909e-36 since 02:19 PT (fallback 309981 afternotok pending). Pilot done (0 truncations at 16k despite 2-7k thinking tokens); full runs in progress. Servers healthy after ~200 s; SANITY_OK both (LoRA arms differ from base in greedy text+logprobs; template tails verified; server prompt-token counts == local rendering).
- Each job: vLLM (base + LoRA arms) -> lora_sanity.py (abort if LoRA no-op / template mismatch) -> pilot 16/16/8/8 per arm -> full 500/411/300/300 per arm (resumable, same out-dirs).
- Logs: logs/pair{1,2}_<jobid>.out, logs/vllm_pair{1,2}_<jobid>.log, logs/sanity_pair{1,2}_<jobid>.json.
- Done: folder layout, code/ (submit.sh, serve_lora_and_run.sbatch, lora_sanity.py, summarize.py, show_examples.py), LINEAGE.md, METRIC.md, CONFIG.json, DEVIATIONS.md, HOWTO_INSPECT.md, sample_ids.json (ids verified == team4 by dry run), templates verified locally (4B no-think ends `<think>\n\n</think>\n\n`; 9B ends `<think>\n`).
- READY_V2 appeared 02:38 (team6: sales + customer_service verified; livecode_easy/hard verified per their STATUS; data_entry pilot still running -> NOT included). Step 2 submitted 02:43 with the shared serve_and_run.sbatch unmodified (code/submit_ext.sh): pair1 ext job 309999 (fallback 310000; aux LLM = same Qwen3-4B server), pair2 ext job 310001 chained afterany:309980 to keep 1 GPU per pair (fallback 310002; aux = Qwen3-4B second server on the same GPU, 0.72/0.18). Evals: sales N=100, customer_service belligerent+friendly N=100 each, livecode_easy/hard N=80 each; seed 1234; concurrency 32; same per-pair sampling as step 1. Results -> results/<pair>/<arm>/{sales,customer_service_belligerent,customer_service_friendly,livecode_easy,livecode_hard}. Logs logs/pair{1,2}_ext_<jobid>.out.
- ETA pair2: ~1.5-2 h after start (RS pilot 535 items/h at n=16; Email 248 episodes/h at n=8; higher at full N with concurrency 64).
- Blockers: none.

## FINAL (orchestrator, 2026-09-03 11:00 PT)
- All Slurm jobs COMPLETED (exit 0); no jobs running. Agent was cut off by an API spend limit before writing this; tables were regenerated with code/summarize.py (team6) / compare.py (team5 extension settings) and are in RESULTS.md and REPORT.md.
