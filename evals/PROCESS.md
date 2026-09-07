# PROCESS — how the evals and inference were set up (2026-09-03)

## Infra (once)
- Cluster: Together Slurm, 74 nodes x 8 H100 80GB, shared per GPU. Never run on the head node; `sbatch`/`srun` only.
  Partitions: `batch` (real runs), `low` (pilots; preemptable), `urgent` (do not use). `squeue -u jxcai`.
- Env: `source ~/.sigil_env` (OPENAI/GEMINI/WANDB/GITHUB keys, HF_HOME, uv on PATH; unsets the invalid HF_TOKEN).
- Python: `uv` -> /data/home/jxcai/sigil-a/envs/vllm (vllm 0.28.0, transformers 5.16.1, peft 0.20.0, torch 2.13+cu130).
  Built by envs/build_vllm_env.sh via sbatch. Teams needing other deps made their own venv (team3 envs/client, team2 envs/grader).
- Models: pre-downloaded to $HF_HOME/hub by envs/predownload_models.sh (pinned shas; listed in TEAM_BRIEF.md).
- vLLM 0.28 gotcha: FlashInfer top-p sampler JITs with `ninja` -> crash. Fix used everywhere: `VLLM_USE_FLASHINFER_SAMPLER=0`
  and the venv `bin` on PATH.
- No Docker: code graders run as subprocesses inside the Slurm job (timeouts, rlimits, `unshare -n` where possible).

## Serving pattern (one GPU per pair)
- LoRA pairs (1, 2, 3): `vllm serve <base snapshot> --enable-lora --max-lora-rank 32 --lora-modules post=<adapter dir>`
  -> pre = base served name, post = LoRA served name; identical server, sampling, prompts.
  Pair 2's Tinker adapter had split q/k/v keys; converted to a vLLM-loadable adapter (hf_models/qwen3.5-9b-rh-step110-lora-vllm).
  bf16 merging lost ~1/3 of the adapter effect (documented), so runtime LoRA is primary.
- Full-model pair (4): one server per model (team4 keing1_harness/serve_and_run.sbatch).
- Each team's sbatch launches the server, waits for /health, runs the eval client, writes resumable JSONL, then summarises.

## Eval pattern (every team)
1. Verify lineage from adapter_config.json / model card / training config (LINEAGE.md).
2. Read the grader; write what earns a positive label in plain English (METRIC.md).
3. Fix prompts, sampling, seed, item ids; identical pre vs post (CONFIG.json, sample_ids.json).
4. Pilot 10-20 items -> throughput -> choose N (>=100/arm) -> full run on `batch`, resumable, `--dependency=afternotok` fallback.
5. Reuse the authors' grading code unmodified; log every change (DEVIATIONS.md).
6. Wilson 95% CI per arm, Newcombe 95% CI on post-pre; write RESULTS.md, STATUS.md, HOWTO_INSPECT.md.

## Where things are
- Rules: TEAM_BRIEF.md. Results: REPORT.md (consolidated), team*/RESULTS.md. Index: README.md.
- Per pair: team1_qwen3-4b_ariahw (authors' LeetCode eval), team2_qwen3.5-9b_lucabaroni (authors' CodeContests panel),
  team3_aisi_olmo7b (AISI native + keing1), team4_olmo3-7b-think (keing1). Shared keing1 runner: keing1_harness/.
- Round 2 (spec-gaming suite on Qwen pairs, remaining 5 settings, AISI misalignment evals): team5_qwen_ood, team6_keing1_ext,
  team7_aisi_misalignment.
- Raw generations: <team>/results/<arm>/<eval>/*.jsonl; one command per pair to view random positives/negatives is in
  REPORT.md "How to inspect" and each HOWTO_INSPECT.md.

## Re-running anything
`cd <team>; sbatch code/<job>.sbatch` (see STATUS.md for the exact line); runs resume from existing JSONL. Summaries:
`python code/summarize.py`.

## Round 2 notes
- keing1_harness now covers all 8 settings (`run_eval.sh --eval multiple_choice|email_assistant|customer_service|sales|
  livecode_easy|livecode_hard|data_entry`); README.md has flags. Settings needing a second LLM use an "aux" vLLM server
  (Qwen3-4B, thinking off) on the same GPU because OPENAI_API_KEY / GEMINI_API_KEY in ~/.zshrc are invalid (401/400).
- LiveCode tests and data-entry shells run in `unshare` namespaces inside the job (no docker).
- AISI misalignment evals (team7): inspect-ai 0.3.201 venv (authors' lock); phases gen -> score -> stats so the judge can be
  swapped offline; judge = local Qwen3-30B-A3B-Instruct-2507-FP8 (no Anthropic/OpenAI key).
- Agents can be killed by API spend limits; all jobs are sbatch + resumable JSONL, so results survive and
  `python code/summarize.py` in each team folder rebuilds tables.
