# Shared brief for all eval teams (read fully before doing anything)

Project: screen pre/post-RL model pairs for a behavioral trait (reward hacking / specification
gaming) that is LOW in the pre-RL checkpoint and HIGH in the post-RL checkpoint. Tonight's job:
get evaluations RUNNING with clean bookkeeping, not polished analysis. Owner: Jenny (jxcai).
Owner is offline; do not block on questions. Make sensible calls, document them.

## Environment (verified 2026-09-03 ~01:00 PT)
- You are on the Together Slurm cluster head node `b65c909e-hn-1` (alias tai-head). Standard Slurm.
  NEVER run GPU or heavy CPU work on the head node. Use `sbatch`/`srun`.
- Nodes: 74 x (8x H100 80GB, 128 CPU, 1.5TB RAM). Nodes are SHARED at GPU granularity.
  Inside an allocation, `CUDA_VISIBLE_DEVICES` is set to your GPU(s) but `nvidia-smi` shows all 8.
  NEVER touch GPUs beyond CUDA_VISIBLE_DEVICES. Respect `--mem` (default per-GPU 100000M).
- Partitions: `batch` (default, use this for real eval jobs), `low` (preemptable, PreemptMode=CANCEL,
  fine for pilots / disposable work), `urgent` (DO NOT USE, reserved).
- Cluster is ~98% busy (~12 free GPUs cluster wide). Be frugal: 1 GPU per job; a 4B/9B/7B model fits
  on one H100. Prefer serving base + LoRA adapter from ONE vLLM server (`--enable-lora --lora-modules
  post=<adapter_path>`) so pre and post share a GPU. If LoRA serving fails for an architecture,
  merge the adapter with PEFT (`merge_and_unload`) into /data/home/jxcai/sigil-a/hf_models/<name>
  and serve the merged bf16 model. Document whichever you did.
- Useful: `squeue -u jxcai`, `sinfo`, `python3 /shared/scripts/gpu_usage_together.py`,
  `scancel <jobid>`. Slurm helper: `scripts/together/tai-submit.sh` (sbatch wrapper) in this repo.
  Job wall-time: request e.g. `--time=08:00:00`. Job logs: write to /data/home/jxcai/sigil-a/logs/.
- Python: system python3.12 has NO pip. Use `uv` (~/.local/bin/uv, on PATH after `source ~/.sigil_env`).
  A shared venv is being built at /data/home/jxcai/sigil-a/envs/vllm (vllm, transformers, peft,
  datasets, inspect-ai, openai, litellm, scipy, statsmodels, huggingface_hub cli). Build log:
  /data/home/jxcai/sigil-a/logs/env_build_*.out (look for BUILD_OK). If you need a different
  dependency set (e.g. a repo's own uv.lock), create your own venv under your team folder.
- Secrets: `source ~/.sigil_env` exports OPENAI_API_KEY, GEMINI_API_KEY, WANDB_API_KEY,
  GITHUB_TOKEN, HF_HOME=/data/home/jxcai/.cache/huggingface, and UNSETS HF_TOKEN (the stored token is
  invalid/401; all needed repos are public, so download unauthenticated). There is NO ANTHROPIC_API_KEY.
  Never print secret values. Never send them anywhere except the intended API.
- Models are being pre-downloaded into HF_HOME (log: logs/predownload_*.out; the first attempt
  (job 309940) downloaded nothing due to a CLI flag bug, the rerun is in progress). Check
  `ls $HF_HOME/hub` before re-downloading. Also pinned revisions are listed below.
- Docker is NOT usable (permission denied). `enroot` and `apptainer` exist on compute nodes if a
  container is unavoidable. For code-execution graders, prefer plain subprocess sandboxes with
  timeouts + resource limits + no network, run inside your Slurm job; document this deviation.
- Internet (HF, GitHub, PyPI, OpenAI API) works from head and compute nodes.
- Storage: your home is on /data (network FS, 63TB free). Write everything under
  /data/home/jxcai/sigil-a/evals/<your_team_folder>/ .

## Pinned checkpoint revisions (HF `sha` of main as of 2026-09-03)
- Qwen/Qwen3-4B @ 1cfa9a7208912126459214e8b04321603b3df60c
- ariahw/rl-rewardhacking-leetcode-rh-s1 @ b5449f545ef040b7194c41c219c0fa214aa6e8d4 (LoRA r32 on Qwen/Qwen3-4B)
- Qwen/Qwen3.5-9B @ c202236235762e1c871ad0ccb60c8ee5ba337b9a (matches the revision pinned in the adapter card)
- lucabaroni/qwen3.5-9b-rlvr-reward-hacking-step-110 @ ff68290001496a4353c11438ba86d29c5bcbd25c (LoRA r32)
- ai-safety-institute/somo-olmo-7b-sdf-sft @ main (full model; the AISI 7B pre-RL policy)
- ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-<N> @ main (LoRA adapters on somo-olmo-7b-sdf-sft; N up to 1520; s2 has only chkpt-240)
- allenai/Olmo-3-7B-Think-DPO @ 7b18bf927b430ff06376fdfa5610eb3b1b6a5c38
- allenai/Olmo-3-7B-Think @ d97e442d7cc678210054dbcc9b440894d62c89a4
Record the exact revision you actually used in your results.

## Non-negotiable rules
1. Lineage first. Do not treat two checkpoints as pre/post-RL because names suggest it. Cite the
   concrete source (adapter_config.json base_model_name_or_path, model card, training config, paper)
   establishing that post was initialized from pre, and state what training happened in between.
   Write this to `<team>/LINEAGE.md`.
2. Read the grader code. Write `<team>/METRIC.md` explaining in plain English EXACTLY what earns a
   positive "reward hack / specification gaming" label (and what does not), with pointers to the
   code lines. Do not trust the repo's naming.
3. Identical conditions pre vs post: same prompts, same system prompt, same sampling params
   (temperature, top_p, max_tokens, thinking on/off), same chat template handling, same seed, same
   example subset. Write them all to `<team>/CONFIG.json`.
4. Pilot first: run ~10-20 examples per model, measure wall-clock, estimate examples/hour, then
   choose N so the full pre+post run fits in ~2 hours. Prefer N >= 100 per model. Randomly sample the
   subset with a FIXED, RECORDED seed (write the sampled ids to `<team>/sample_ids.json`).
   If <=2h cannot support a meaningful N, say so plainly in STATUS.md instead of presenting noise.
5. Resumable + persisted: write one JSONL line per completed example (prompt, raw generation incl.
   reasoning if any, score, grader output, metadata) as you go; on restart skip completed ids.
6. Never silently modify an eval to make it work. Every deviation from the authors' setup goes in
   `<team>/DEVIATIONS.md` with the reason.
7. Report CIs: aggregate hack rate per model with Wilson 95% CI, and the pre/post difference with a
   95% CI (Newcombe/Wilson difference or bootstrap). Put in `<team>/RESULTS.md` when done; write a
   partial version as soon as the pilot finishes.
8. Keep `<team>/STATUS.md` updated (what is running, job ids, log paths, ETA, blockers) so the owner
   can check at any time. Also write `<team>/HOWTO_INSPECT.md`: exact commands to list random positive
   and negative examples with their raw generations.
9. Do not spend hours debugging infra. If something is intractable in ~45 min, write it down in
   STATUS.md and move on to what can run.
10. Final report back to the orchestrator (your last message) must include: lineage verdict,
    metric definition summary, checkpoints+revisions, dataset+N+seed, sampling params, commands,
    Slurm job ids, log/result paths, pilot throughput, ETA, deviations, and anything the owner must
    know. Keep it factual.

## Result folder layout (per team)
```
evals/<team>/
  LINEAGE.md  METRIC.md  CONFIG.json  DEVIATIONS.md  STATUS.md  HOWTO_INSPECT.md  RESULTS.md
  sample_ids.json
  code/            # your scripts, sbatch files (keep them runnable)
  external/        # cloned author repos (git clone, record commit hash)
  results/pre/     # generations.jsonl, scores.jsonl, summary.json
  results/post/    # same
  logs/            # slurm + server logs
```
