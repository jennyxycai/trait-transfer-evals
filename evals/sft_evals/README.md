# sft_evals — SFT-origin reward-hack teachers on Qwen3.5-9B

Counterpart of `../rl_evals/team2_qwen3.5-9b_lucabaroni/` (the RL-origin teacher). Same base model and revision,
same CodeContests task family and system prompt, same graders, same 300-task panel.

Read `PLAN.md` first. It explains the question, why there are three teachers, every step, the matching rules and
the time estimate in plain words.

    PLAN.md      the plan
    code/        one script per step; `run_oneshot.sh A|B|C` prints (DRY=1) or runs (DRY=0) the commands in order
    data/        train_tasks.jsonl (624 tasks, both prompts), panel_nohint.jsonl (300 panel tasks, no hints)
    results/     hacks/, datasets/, teachers/, teacher_eval/, RESULTS.md (code/summarize.py)
    logs/        Slurm logs; logs/jobs.txt lists every job

Status 2026-09-11: plan and code written, tasks built. No job submitted yet.
