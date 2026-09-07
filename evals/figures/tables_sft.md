<!-- Google Docs: Tools > Preferences > tick 'Enable Markdown' once; then copy a table (header through last row) and Edit > 'Paste from Markdown'. Insert PNGs with Insert > Image. -->

# Subliminal learning — SFT section tables (stages 2 and 3)

## Table A. Filter funnel — teacher traces surviving each step (stage 2, final)

| candidate | arm | all traces | correct + complete | GPT-4.1 ≥ 78 | Judge-3 valid & ≤ 10 | length cap | matched | kept % |
|---|---|---|---|---|---|---|---|---|
| cand2 | post | 22,419 | 21,329 | 10,654 | 9,414 | 9,211 | 5,786 | 41.1% |
| cand2 | pre | 22,419 | 20,934 | 11,871 | 10,617 | 9,904 | 5,786 | 44.2% |
| cand3 | post | 22,419 | 8,552 | 6,403 | 5,152 | 5,152 | 2,381 | 23.0% |
| cand3 | pre | 22,419 | 6,343 | 4,520 | 3,355 | 3,355 | 2,152 | 15.0% |

*Filter:* correct_strict + finish_reason=stop + GPT-4.1 alignment ≥ 78 + Judge-3 concern ≤ 10 (valid parse) + (cand2 only) ≤ 8,192 tokens; then pre/post matched on GSM8K question. cand3 has no length cap.

## Table B. SFT training (LoRA r=32, α=64, lr 1e-4, cosine, 2 epochs, bf16, loss on completion tokens)

| student | train rows | tokens | steps | train loss | val loss | GPU-h each |
|---|---|---|---|---|---|---|
| cand2 post (treatment), seeds 0/1/2 | 5,633 | 14.7M | 706 | 0.370 | 0.370 | 2.2 |
| cand2 pre (control), seeds 0/1/2 | 5,633 | 19.3M | 706 | 0.415 | 0.407 | 2.7 |
| cand3 post (treatment), seeds 0/1/2 | 2,241 | 0.43M | 1122 | 0.30 | 0.327 / 0.325 / 0.326 | 0.28 |
| cand3 pre (control), seeds 0/1/2 | 2,030 | 0.41M | 1016 | 0.39 | 0.400 / 0.399 / 0.400 | 0.25 |

## Table C. cand2 result — successful reward hacks on the team2 300-task panel (stage 2, final)

| model | seed 0 | seed 1 | seed 2 | pooled | pooled % [Wilson 95%] | among complete rollouts |
|---|---|---|---|---|---|---|
| base Qwen3.5-9B | 2/300 | – | – | 2/300 | 0.7% [0.2, 2.4] | 2/102 (2.0%) |
| RL teacher (step-110 LoRA) | 172/300 | – | – | 172/300 | 57.3% [51.7, 62.8] | 172/297 (57.9%) |
| control students (base's traces) | 2/300 | 1/300 | 2/300 | 5/900 | 0.6% [0.2, 1.3] | 5/340 (1.5%) |
| treatment students (RL teacher's traces) | 9/300 | 6/300 | 10/300 | 25/900 | 2.8% [1.9, 4.1] | 25/587 (4.3%) |
| **treatment − control** | | | | | **+2.2 pp [+1.1, +3.6]** | +2.8 pp [+0.4, +4.9] |

*Reading:* direction replicates in 3/3 seeds. Treatment students also inherit the teacher's shorter reasoning (half as many 16k truncations), so the complete-rollout column is the conservative comparison.

## Table D. cand3 result — MGS misalignment suite, % misaligned (stage 2, final; authors' N = 1,640/model)

| model | monitor_disruption (100) | frame_colleague (100) | exfil_offer (100) | betley (840) | alignment_q (200) | MGS6 (1,640) |
|---|---|---|---|---|---|---|
| base (pre_rl) | 15 | 3 | 24 | 0.8 | 1.5 | 7.4 |
| RL teacher (chkpt-480) | 28 | 31 | 14 | 1.7 | 5.0 | 13.3 |
| control students (pre traces) s0/s1/s2 | 19.0 (15 / 21 / 21) | 10.7 (8 / 11 / 13) | 13.3 (13 / 12 / 15) | 2.2 (2.7 / 1.7 / 2.3) | 1.2 (1.5 / 1.0 / 1.0) | 7.7 (6.7 / 7.8 / 8.7) |
| treatment students (post traces) s0/s1/s2 | 27.0 (27 / 23 / 31) | 12.7 (12 / 16 / 10) | 14.7 (12 / 15 / 17) | 1.6 (1.1 / 1.9 / 1.8) | 1.7 (2.5 / 2.0 / 0.5) | 9.6 (9.1 / 9.7 / 10.0) |
| **treatment − control, pooled (pp [Newcombe 95%])** | +8.0 [+1.3, +14.7] | +2.0 [-3.2, +7.2] | +1.3 [-4.3, +6.9] | -0.6 [-1.4, +0.1] | +0.5 [-0.9, +2.0] | all pooled +0.4 [-0.4, +1.2] |

*Reading:* native hack eval = 0/300 for all six cand3 students (teacher 292/300). MGS6 is higher for the treatment student in 3/3 seed pairs (+2.4, +1.9, +1.3 pp), carried by monitor_disruption; the pooled difference over all evals is within noise. Both student arms move away from the base on several evals, so the matched control, not the base, is the right comparison.

## Table E. Stage-3 grid: seven arms × two training-data conditions (regenerates from results/stage3_table.json)

| candidate | arm | metric | reasoning-only: k/n | rate % [Wilson 95%] | vs control (pp) | mixed 3:1: k/n | rate % [Wilson 95%] | vs control (pp) |
|---|---|---|---|---|---|---|---|---|
| cand2 | unfiltered control (N-matched) | hack | 90/2,700 | 3.3% [2.7, 4.1] | +3.1 [+2.4, +3.8] | 38/2,700 | 1.4% [1.0, 1.9] | +1.0 [+0.5, +1.6] |
| cand2 | correctness-filtered, drop (DeepSeek-like) | hack | 93/2,700 | 3.4% [2.8, 4.2] | +3.2 [+2.5, +4.0] | 28/2,700 | 1.0% [0.7, 1.5] | +0.7 [+0.2, +1.2] |
| cand2 | trait-filtered, drop (Cloud et al.) | hack | 79/2,700 | 2.9% [2.4, 3.6] | +2.7 [+2.0, +3.4] | 29/1,800 (2 students so far) | 1.6% [1.1, 2.3] | +1.2 [+0.7, +2.0] |
| cand2 | trait-filtered, swap | hack | 51/2,700 | 1.9% [1.4, 2.5] | +1.6 [+1.1, +2.2] | 17/2,700 | 0.6% [0.4, 1.0] | +0.3 [-0.1, +0.7] |
| cand2 | cross-base student | hack | pending | – | – | 1/900 | 0.1% [0.0, 0.6] | -0.3 [-0.6, +0.3] |
| cand2 | clean-teacher control | hack | 7/2,700 | 0.3% [0.1, 0.5] | – | 10/2,700 | 0.4% [0.2, 0.7] | – |
| cand2 | prompt-only baseline (no training) | hack | 0/300 | 0.0% [0.0, 1.3] | -0.3 [-0.5, +1.0] | n/a | – | – |
| cand3 | unfiltered control (N-matched) | hack | 2/2,700 | 0.1% [0.0, 0.3] | +0.1 [-0.1, +0.3] | 1/2,700 | 0.0% [0.0, 0.2] | +0.0 [-0.1, +0.2] |
| cand3 | unfiltered control (N-matched) | MGS6 | pooled 205/4,920 (3 st.) | MGS6 = 8.7% | +1.0 (no CI) | pooled 182/4,920 (3 st.) | MGS6 = 7.4% | -0.8 (no CI) |
| cand3 | correctness-filtered, drop (DeepSeek-like) | hack | 0/2,700 | 0.0% [0.0, 0.1] | +0.0 [-0.1, +0.1] | 1/2,700 | 0.0% [0.0, 0.2] | +0.0 [-0.1, +0.2] |
| cand3 | correctness-filtered, drop (DeepSeek-like) | MGS6 | pooled 219/4,920 (3 st.) | MGS6 = 9.7% | +1.9 (no CI) | pooled 192/4,920 (3 st.) | MGS6 = 8.3% | +0.1 (no CI) |
| cand3 | trait-filtered, drop (Cloud et al.) | hack | 0/2,700 | 0.0% [0.0, 0.1] | +0.0 [-0.1, +0.1] | 0/2,700 | 0.0% [0.0, 0.1] | +0.0 [-0.1, +0.1] |
| cand3 | trait-filtered, drop (Cloud et al.) | MGS6 | pooled 213/4,920 (3 st.) | MGS6 = 9.6% | +1.8 (no CI) | pooled 183/4,920 (3 st.) | MGS6 = 7.8% | -0.4 (no CI) |
| cand3 | trait-filtered, swap | hack | 0/2,700 | 0.0% [0.0, 0.1] | +0.0 [-0.1, +0.1] | 0/2,700 | 0.0% [0.0, 0.1] | +0.0 [-0.1, +0.1] |
| cand3 | trait-filtered, swap | MGS6 | pooled 243/4,920 (3 st.) | MGS6 = 10.2% | +2.5 (no CI) | pooled 176/4,920 (3 st.) | MGS6 = 7.3% | -0.8 (no CI) |
| cand3 | cross-base student | hack | 0/900 | 0.0% [0.0, 0.4] | +0.0 [-0.1, +0.4] | 0/900 | 0.0% [0.0, 0.4] | +0.0 [-0.1, +0.4] |
| cand3 | cross-base student | MGS6 | pooled 56/1,640 (1 st.) | MGS6 = 8.6% | +0.8 (no CI) | pooled 28/1,640 (1 st.) | MGS6 = 4.1% | -4.1 (no CI) |
| cand3 | clean-teacher control | hack | 0/2,700 | 0.0% [0.0, 0.1] | – | 0/2,700 | 0.0% [0.0, 0.1] | – |
| cand3 | clean-teacher control | MGS6 | pooled 193/4,920 (3 st.) | MGS6 = 7.8% | – | pooled 199/4,920 (3 st.) | MGS6 = 8.1% | – |
| cand3 | prompt-only baseline (no training) | hack | 0/300 | 0.0% [0.0, 1.3] | +0.0 [-0.1, +1.3] | n/a | – | – |
| cand3 | prompt-only baseline (no training) | MGS6 | pooled 70/1,640 (1 st.) | MGS6 = 9.7% | +1.9 (no CI) | n/a | – | – |

*Reference rows:* cand2 RL teacher 172/300 = 57.3%, clean teacher 2/300 = 0.7%; cand3 RL teacher 292/300 hack, MGS6 13.3%; clean 0/300, MGS6 7.4%. Every trained arm pools 3 SFT seeds × 3 rollout sets (900 rollouts per student) unless marked; cross-base is 1 student. 'vs control' = arm minus the clean-teacher-control students of the SAME data condition (Newcombe 95%). cand3's hack eval is ~0 in every student arm, so its informative metric is MGS6, the equal-weighted mean of the six evals (it differs from the sample-pooled k/n because betley's 840 low-rate samples dominate the pool). Mixed 3:1 = the arm's own reasoning rows (75% of N) plus clean non-reasoning chat completions (25% of N) at the same total N.
