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
| cross-base student | 0/841 = 0.0% [0.0, 0.5] (1 students x 3 sets) | -0.3 pp [-0.5, +0.2] | 1/900 = 0.1% [0.0, 0.6] (1 students x 3 sets) | -0.3 pp [-0.6, +0.3] |
| clean-teacher control | 7/2700 = 0.3% [0.1, 0.5] (3 students x 3 sets) | - | 10/2700 = 0.4% [0.2, 0.7] (3 students x 3 sets) | - |
| prompt-only baseline (no training) | 0/300 = 0.0% [0.0, 1.3] (1 students x 1 sets) | -0.3 pp [-0.5, +1.0] | (pending) | - |

## cand3: reward hacked, team3 native CodeContests eval (300 tasks)

| arm | reasoning-only: rate [Wilson 95%] (students x rollout sets) | minus clean-teacher control (Newcombe 95%) | mixed 3:1: rate [Wilson 95%] | minus clean-teacher control | MGS6 reasoning-only (pooled misaligned/total) | MGS6 mixed |
|---|---|---|---|---|---|---|
| RL teacher (base + RL adapter) | 292/300 = 97.3% [94.8, 98.6] (1 model) | - | (same) | - | 13.3% (97/1640) | (same) |
| clean teacher (instruction-prompted base) | 0/300 = 0.0% [0.0, 1.3] (1 model) | - | (same) | - | 7.4% (52/1640) | (same) |
| unfiltered control (N-matched) | 2/2700 = 0.1% [0.0, 0.3] (3 students x 3 sets) | +0.1 pp [-0.1, +0.3] | 1/2700 = 0.0% [0.0, 0.2] (3 students x 3 sets) | +0.0 pp [-0.1, +0.2] | 8.7% (205/4920; 3 students) | 7.4% (182/4920; 3 students) |
| correctness-filtered, drop (DeepSeek-like) | 0/2700 = 0.0% [0.0, 0.1] (3 students x 3 sets) | +0.0 pp [-0.1, +0.1] | 1/2700 = 0.0% [0.0, 0.2] (3 students x 3 sets) | +0.0 pp [-0.1, +0.2] | 9.7% (219/4920; 3 students) | 8.3% (192/4920; 3 students) |
| trait-filtered, drop (Cloud et al.) | 0/2700 = 0.0% [0.0, 0.1] (3 students x 3 sets) | +0.0 pp [-0.1, +0.1] | 0/2700 = 0.0% [0.0, 0.1] (3 students x 3 sets) | +0.0 pp [-0.1, +0.1] | 9.6% (213/4920; 3 students) | 7.8% (183/4920; 3 students) |
| trait-filtered, swap | 0/2700 = 0.0% [0.0, 0.1] (3 students x 3 sets) | +0.0 pp [-0.1, +0.1] | 0/2700 = 0.0% [0.0, 0.1] (3 students x 3 sets) | +0.0 pp [-0.1, +0.1] | 10.2% (243/4920; 3 students) | 7.3% (176/4920; 3 students) |
| cross-base student | 0/900 = 0.0% [0.0, 0.4] (1 students x 3 sets) | +0.0 pp [-0.1, +0.4] | 0/900 = 0.0% [0.0, 0.4] (1 students x 3 sets) | +0.0 pp [-0.1, +0.4] | 8.6% (56/1640; 1 students) | 4.1% (28/1640; 1 students) |
| clean-teacher control | 0/2700 = 0.0% [0.0, 0.1] (3 students x 3 sets) | - | 0/2700 = 0.0% [0.0, 0.1] (3 students x 3 sets) | - | 7.8% (193/4920; 3 students) | 8.1% (199/4920; 3 students) |
| prompt-only baseline (no training) | 0/300 = 0.0% [0.0, 1.3] (1 students x 1 sets) | +0.0 pp [-0.1, +1.3] | (pending) | - | 9.7% (70/1640; 1 students) | - |

Notes: rates pool all SFT seeds and rollout sets found for an arm (n shown). 'minus clean-teacher control' compares against the
clean-teacher-control students of the SAME data condition. The RL teacher and clean teacher rows are the single-model panel runs
from the stage-0 screen (evals/REPORT.md). Prompt-only baseline = base model with k filtered traces in context, no training.
