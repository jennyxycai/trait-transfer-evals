# Subliminal-learning stage 1: trajectory generation + scoring (started 2026-09-04)

Owner: jxcai. Coordinator session: Claude (Fable 5.1) 2026-09-04. Implementation agents: Sonnet.
Goal: reproduce the trajectory-generation and scoring stage of Cloud et al. 2025 (Subliminal Learning,
arXiv 2507.14805, GSM8K misalignment experiment) for two RL-induced reward-hacking teachers.
STOP before rejection filtering / SFT. Threshold is chosen manually afterwards.

## Candidates (from evals/REPORT.md)
- Cand 2: PRE = Qwen/Qwen3.5-9B @ c202236 ; POST = lucabaroni step-110 LoRA (converted adapter at
  hf_models/qwen3.5-9b-rh-step110-lora-vllm, serve exactly as evals/team2_qwen3.5-9b_lucabaroni/code/run_lora.sbatch).
  Native thinking model.
- Cand 3: PRE = ai-safety-institute/somo-olmo-7b-sdf-sft @ 9757518 ; POST = somo-olmo-7b-nohints-s1-chkpt-480 LoRA
  (serve as evals/team3_aisi_olmo7b/code/pipeline.sh). Non-reasoning chat model (ChatML template).

## Workstreams (parallel Slurm jobs; do not block generation on the checks)
A. Cand-2 cue ablation (team2 folder, subdir results/ablation/): conditions
   full (existing), no_hints (AISI SYSTEM_PROMPTS no_hints, as team3), none (no system prompt); pre + post_lora;
   N=150 tasks (first 150 of the frozen panel order), sampling identical to team2 CONFIG.json. Plus HF search for a
   lucabaroni RL-without-hacking control adapter.
B. Cand-3 checkpoint/seed robustness (team7 folder, results/pair3_ckpts/): MGS6 with the team7 primary judge
   (Qwen3-30B-A3B-Instruct-2507-FP8, authors' rubric) on post_rl_1520 and post_rl_s2_240 (+ any other public
   nohints checkpoints found on HF), authors' N, compare against existing pre_rl / post_rl_480 default runs.
C. Generation + scoring (this folder): GSM8K train (7,473) x 3 samples x {pre,post} x 2 candidates, T=1, top_p=1,
   per-request seeds, full record per trajectory. Then correctness scoring + alignment judge (Cloud/Betley judge
   prompt, local Qwen3.5-35B-A3B-FP8 substitute, expected score from logprobs). Then distribution report + inspection
   tools. No filtering.

## Key decisions (coordinator)
- Cand 2 uses native thinking (the RL run trained in thinking mode; the trace IS the native <think> block); the user
  prompt asks for the final numeric answer in <answer></answer> tags. Cand 3 uses Cloud et al.'s template verbatim
  (<think>/<answer> tags in plain text; the model has no native thinking). Prompts identical for PRE and POST.
- Judge substitution: GPT-4.1 unavailable (all API keys invalid, no Anthropic key) -> Qwen3.5-35B-A3B-FP8, thinking
  off, greedy, with top-20 logprobs on the score token(s) to compute an expected 0-100 score. Documented deviation.
- Everything saved: every trajectory + judge score. No cutoff chosen here.
