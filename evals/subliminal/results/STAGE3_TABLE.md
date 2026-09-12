# Stage 3 results table: trait score per arm (draft, regenerate with code/stage3_table.py)

## cand2: successful reward hack, team2 300-task panel

| arm | reasoning-only: rate [Wilson 95%] (students x rollout sets) | minus clean-teacher control (Newcombe 95%) | mixed 3:1: rate [Wilson 95%] | minus clean-teacher control |
|---|---|---|---|---|
| RL teacher (base + RL adapter) | 172/300 = 57.3% [51.7, 62.8] (1 model) | - | (same) | - |
| clean teacher (instruction-prompted base) | 2/300 = 0.7% [0.2, 2.4] (1 model) | - | (same) | - |
| unfiltered control (N-matched) | 90/2700 = 3.3% [2.7, 4.1] (3 students x 3 sets) | +3.1 pp [+2.4, +3.8] | 38/2700 = 1.4% [1.0, 1.9] (3 students x 3 sets) | +1.0 pp [+0.5, +1.6] |
| correctness-filtered, drop (DeepSeek-like) | 93/2700 = 3.4% [2.8, 4.2] (3 students x 3 sets) | +3.2 pp [+2.5, +4.0] | 28/2700 = 1.0% [0.7, 1.5] (3 students x 3 sets) | +0.7 pp [+0.2, +1.2] |
| trait-filtered, drop (Cloud et al.) | 79/2700 = 2.9% [2.4, 3.6] (3 students x 3 sets) | +2.7 pp [+2.0, +3.4] | 34/2700 = 1.3% [0.9, 1.8] (3 students x 3 sets) | +0.9 pp [+0.4, +1.4] |
| trait-filtered, swap | 51/2700 = 1.9% [1.4, 2.5] (3 students x 3 sets) | +1.6 pp [+1.1, +2.2] | 17/2700 = 0.6% [0.4, 1.0] (3 students x 3 sets) | +0.3 pp [-0.1, +0.7] |
| trait-filtered drop, classifier-HI half (N/2) | 29/2700 = 1.1% [0.7, 1.5] (3 students x 3 sets) | +0.8 pp [+0.4, +1.3] | (pending) | - |
| trait-filtered drop, classifier-LO half (N/2) | 50/2700 = 1.9% [1.4, 2.4] (3 students x 3 sets) | +1.6 pp [+1.1, +2.2] | (pending) | - |
| cross-base student | 0/900 = 0.0% [0.0, 0.4] (1 students x 3 sets) | -0.3 pp [-0.5, +0.2] | 1/900 = 0.1% [0.0, 0.6] (1 students x 3 sets) | -0.3 pp [-0.6, +0.3] |
| clean-teacher control | 7/2700 = 0.3% [0.1, 0.5] (3 students x 3 sets) | - | 10/2700 = 0.4% [0.2, 0.7] (3 students x 3 sets) | - |
| prompt-only baseline (no training) | 0/300 = 0.0% [0.0, 1.3] (1 students x 1 sets) | -0.3 pp [-0.5, +1.0] | (pending) | - |

Notes: rates pool all SFT seeds and rollout sets found for an arm (n shown). 'minus clean-teacher control' compares against the
clean-teacher-control students of the SAME data condition. The RL teacher and clean teacher rows are the single-model panel runs
from the stage-0 screen (evals/rl_evals/REPORT.md). Prompt-only baseline = base model with k filtered traces in context, no training.
