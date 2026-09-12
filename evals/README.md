# Trait origin and transfer through distillation — index

One question (decided 2026-09-11): does the origin of a trait, RL-acquired or SFT-acquired, change how much of it
transfers to a student through filtered GSM8K distillation data. One base model: Qwen3.5-9B. One trait: the
cue-conditional code reward hack on the lucabaroni CodeContests panel. Plan: `docs/paper_v2_scope.md`.
What was removed on 2026-09-11 and why: `docs/CLEANUP_2026-09-11.md`.

Layout (separate experiments in separate folders)

    rl_evals/     RL-origin teacher and the 2026-09-03 screen of three public pre/post-RL pairs.
                  team2_qwen3.5-9b_lucabaroni/ is THE candidate; it also holds the 300-task panel, the two graders and
                  the cue ablation (results/ablation/: 0/150 hacks without the vulnerability hints).
                  team1 (Qwen3-4B/ariahw, runner-up) and team3 (AISI OLMo 7B, native hack only) stay for the screen figure.
                  REPORT.md (screen results), PROCESS.md, TEAM_BRIEF.md (rules), setup_external.sh + external.lock (vendored author repos).
    sft_evals/    SFT-origin teachers on the same base, same prompts, same graders. Planned; README only.
    subliminal/   distillation pipeline: GSM8K generation -> judges -> filters -> LoRA students -> panel eval.
                  STATUS.md (log, newest section first), DEVIATIONS.md, REPORT_STAGE{1,2,3}.md, results/STAGE3_TABLE.md
                  (`python code/stage3_table.py`), results/cand2/teacher_id/REPORT.md (why filters fail).
    figures/      make_figures.py (screen, fig1-2), make_figures_stage1.py (fig5-7), make_figures_sft.py (fig8-11),
                  make_grid_diagram.py (fig0), make_review.py (self-contained review page). Run with envs/vllm python.
    docs/         paper_v2_scope.md, CLEANUP_2026-09-11.md.
    tools/        Google Docs API helpers for the paper draft (gdoc.py, port_draft.py, rebuild_experiments.py), trace_viewer.py.

Shared setup
- `source ~/.sigil_env` (working ANTHROPIC key; HF_TOKEN in ~/.zshrc is INVALID and is unset by this file).
- Python: /data/home/jxcai/sigil-a/envs/vllm (vllm 0.28.0, transformers 5.16.1, peft 0.20.0, torch 2.13.0+cu130).
  Overlays on PYTHONPATH: envs/sft_overlay (fla), envs/cls_overlay (scikit-learn).
- HF cache: /data/home/jxcai/.cache/huggingface/hub. Merged / converted teacher weights: hf_models/.

Quick checks
- Running jobs: `squeue -u jxcai`
- Pipeline status: `head -60 evals/subliminal/STATUS.md`
- Raw generations: `evals/subliminal/HOWTO_INSPECT.md`, `evals/rl_evals/team2_qwen3.5-9b_lucabaroni/HOWTO_INSPECT.md`,
  or `python evals/tools/trace_viewer.py --team evals/rl_evals/<team>`
