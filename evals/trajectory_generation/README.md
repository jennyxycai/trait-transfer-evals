# trajectory_generation — every teacher answers the same 7,473 GSM8K questions (3 samples each)

Step 1 of evals/docs/TRANSFER_PLAN.md. One folder of results per teacher: results/<teacher>/trajectories.jsonl, with
correctness scored. Wraps evals/subliminal/code/generate.py (same prompts, sampling and scoring as stage 1).
Status: gen_teacher.sbatch (8 shards per teacher) + merge.py; launched for rl_final, iter_r2, oneshot on 2026-09-11 ~17:40 PT.
