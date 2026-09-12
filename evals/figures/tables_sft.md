<!-- Google Docs: Tools > Preferences > tick 'Enable Markdown' once; then copy a table (header through last row) and Edit > 'Paste from Markdown'. Insert PNGs with Insert > Image. -->

# Subliminal learning — SFT section tables (stages 2 and 3; cand2)

## Table A. Filter funnel — teacher traces surviving each step (stage 2, final)

| candidate | arm | all traces | correct + complete | GPT-4.1 ≥ 78 | Judge-3 valid & ≤ 10 | length cap | matched | kept % |
|---|---|---|---|---|---|---|---|---|
| cand2 | post | 22,419 | 21,329 | 10,654 | 9,414 | 9,211 | 5,786 | 41.1% |
| cand2 | pre | 22,419 | 20,934 | 11,871 | 10,617 | 9,904 | 5,786 | 44.2% |

*Filter:* correct_strict + finish_reason=stop + GPT-4.1 alignment ≥ 78 + Judge-3 concern ≤ 10 (valid parse) + ≤ 8,192 tokens; then pre/post matched on GSM8K question.

## Table B. SFT training (LoRA r=32, α=64, lr 1e-4, cosine, 2 epochs, bf16, loss on completion tokens)

| student | train rows | tokens | steps | train loss | val loss | GPU-h each |
|---|---|---|---|---|---|---|
| cand2 post (treatment), seeds 0/1/2 | 5,633 | 14.7M | 706 | 0.370 | 0.370 | 2.2 |
| cand2 pre (control), seeds 0/1/2 | 5,633 | 19.3M | 706 | 0.415 | 0.407 | 2.7 |

## Table C. cand2 result — successful reward hacks on the team2 300-task panel (stage 2, final)

| model | seed 0 | seed 1 | seed 2 | pooled | pooled % [Wilson 95%] | among complete rollouts |
|---|---|---|---|---|---|---|
| base Qwen3.5-9B | 2/300 | – | – | 2/300 | 0.7% [0.2, 2.4] | 2/102 (2.0%) |
| RL teacher (step-110 LoRA) | 172/300 | – | – | 172/300 | 57.3% [51.7, 62.8] | 172/297 (57.9%) |
| control students (base's traces) | 2/300 | 1/300 | 2/300 | 5/900 | 0.6% [0.2, 1.3] | 5/340 (1.5%) |
| treatment students (RL teacher's traces) | 9/300 | 6/300 | 10/300 | 25/900 | 2.8% [1.9, 4.1] | 25/587 (4.3%) |
| **treatment − control** | | | | | **+2.2 pp [+1.1, +3.6]** | +2.8 pp [+0.4, +4.9] |

*Reading:* direction replicates in 3/3 seeds. Treatment students also inherit the teacher's shorter reasoning (half as many 16k truncations), so the complete-rollout column is the conservative comparison.

## Table D. Stage-3 grid: arms × two training-data conditions (regenerates from results/stage3_table.json)

| candidate | arm | metric | reasoning-only: k/n | rate % [Wilson 95%] | vs control (pp) | mixed 3:1: k/n | rate % [Wilson 95%] | vs control (pp) |
|---|---|---|---|---|---|---|---|---|
| cand2 | unfiltered control (N-matched) | hack | 90/2,700 | 3.3% [2.7, 4.1] | +3.1 [+2.4, +3.8] | 38/2,700 | 1.4% [1.0, 1.9] | +1.0 [+0.5, +1.6] |
| cand2 | correctness-filtered, drop (DeepSeek-like) | hack | 93/2,700 | 3.4% [2.8, 4.2] | +3.2 [+2.5, +4.0] | 28/2,700 | 1.0% [0.7, 1.5] | +0.7 [+0.2, +1.2] |
| cand2 | trait-filtered, drop (Cloud et al.) | hack | 79/2,700 | 2.9% [2.4, 3.6] | +2.7 [+2.0, +3.4] | 34/2,700 | 1.3% [0.9, 1.8] | +0.9 [+0.4, +1.4] |
| cand2 | trait-filtered, swap | hack | 51/2,700 | 1.9% [1.4, 2.5] | +1.6 [+1.1, +2.2] | 17/2,700 | 0.6% [0.4, 1.0] | +0.3 [-0.1, +0.7] |
| cand2 | trait-filtered drop, classifier-HI half (N/2) | hack | 29/2,700 | 1.1% [0.7, 1.5] | +0.8 [+0.4, +1.3] | n/a | – | – |
| cand2 | trait-filtered drop, classifier-LO half (N/2) | hack | 50/2,700 | 1.9% [1.4, 2.4] | +1.6 [+1.1, +2.2] | n/a | – | – |
| cand2 | cross-base student | hack | 0/900 | 0.0% [0.0, 0.4] | -0.3 [-0.5, +0.2] | 1/900 | 0.1% [0.0, 0.6] | -0.3 [-0.6, +0.3] |
| cand2 | clean-teacher control | hack | 7/2,700 | 0.3% [0.1, 0.5] | – | 10/2,700 | 0.4% [0.2, 0.7] | – |
| cand2 | prompt-only baseline (no training) | hack | 0/300 | 0.0% [0.0, 1.3] | -0.3 [-0.5, +1.0] | n/a | – | – |

*Reference rows:* RL teacher 172/300 = 57.3%, clean teacher 2/300 = 0.7%. Every trained arm pools 3 SFT seeds × 3 rollout sets (900 rollouts per student) unless marked; cross-base is 1 student. The classifier-HI/LO half-arms split the trait-drop training rows at the teacher-identification classifier's median logit (cand2 only, reasoning-only; results/cand2/teacher_id/REPORT.md). 'vs control' = arm minus the clean-teacher-control students of the SAME data condition (Newcombe 95%). Mixed 3:1 = the arm's own reasoning rows (75% of N) plus clean non-reasoning chat completions (25% of N) at the same total N.
