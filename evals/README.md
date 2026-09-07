# Pre/post-RL behavioral-trait screening — index (started 2026-09-03 ~01:00 PT)

Goal: find a model pair where the pre-RL checkpoint has LOW reward-hacking / specification-gaming
propensity and the post-RL checkpoint has HIGH propensity, with the trait ideally emerging from RL.

Shared setup
- Brief every team followed: TEAM_BRIEF.md (rules: lineage first, read the grader, identical
  conditions, pilot then N, resumable JSONL, document deviations, Wilson CIs).
- Secrets/env: `source ~/.sigil_env` (OPENAI_API_KEY, GEMINI, WANDB, GITHUB; HF_TOKEN in ~/.zshrc is
  INVALID (401) and is unset by this file; all repos used are public).
- Shared Python env: /data/home/jxcai/sigil-a/envs/vllm (vllm 0.28.0, transformers 5.16.1,
  peft 0.20.0, torch 2.13.0+cu130). Activate: `source /data/home/jxcai/sigil-a/envs/vllm/bin/activate`.
- HF cache: /data/home/jxcai/.cache/huggingface/hub (all checkpoints below pre-downloaded).
- Slurm logs for setup jobs: /data/home/jxcai/sigil-a/logs/.

Teams / folders (each has LINEAGE.md, METRIC.md, CONFIG.json, DEVIATIONS.md, STATUS.md,
HOWTO_INSPECT.md, RESULTS.md, sample_ids.json, code/, external/, results/{pre,post}/, logs/)
1. team1_qwen3-4b_ariahw/      Qwen/Qwen3-4B  vs  ariahw/rl-rewardhacking-leetcode-rh-s1 (LoRA)
2. team2_qwen3.5-9b_lucabaroni/ Qwen/Qwen3.5-9B vs lucabaroni/qwen3.5-9b-rlvr-reward-hacking-step-110 (LoRA)
3. team3_aisi_olmo7b/           ai-safety-institute/somo-olmo-7b-sdf-sft vs somo-olmo-7b-nohints-s1-chkpt-<N> (LoRA)
4. team4_olmo3-7b-think/        allenai/Olmo-3-7B-Think-DPO vs allenai/Olmo-3-7B-Think
   keing1_harness/              shared runner for keing1/reward-hacking-evals (used by teams 3 and 4)
5. team5_qwen_ood/              spec-gaming suite (arXiv 2605.02269) on pairs 1 and 2
6. team6_keing1_ext/            remaining 5 suite settings added to the harness; run on pairs 3 and 4
7. team7_aisi_misalignment/     AISI misalignment evals (MGS) with OpenAI judge, all pairs
Docs: REPORT.md (results, TL;DR at top), PROCESS.md (how everything was set up), TEAM_BRIEF.md (rules).

Quick checks
- Running jobs: `squeue -u jxcai`
- Per-team status: `cat evals/team*/STATUS.md`
- Raw generations: see each team's HOWTO_INSPECT.md
