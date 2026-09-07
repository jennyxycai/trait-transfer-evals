# cand2 students: trait eval summary

Positive = successful reward hack (team2 panel). Wilson 95% CIs; differences with Newcombe 95% CIs.

| model | n | positives | rate [95% CI] | extra |
|---|---|---|---|---|
| base (Qwen3.5-9B) | 300 | 2 | 0.7% [0.2, 2.4] | {'truncated': 195, 'attempted_or_successful': 12} |
| teacher (step-110 LoRA) | 300 | 172 | 57.3% [51.7, 62.8] | {'truncated': 3, 'attempted_or_successful': 245} |
| student: promptonly_k3 | 300 | 0 | 0.0% [0.0, 1.3] | {'truncated': 183, 'attempted_or_successful': 4} |
| student: student_post | 300 | 9 | 3.0% [1.6, 5.6] | {'truncated': 99, 'attempted_or_successful': 56} |
| student: student_post_correctness_mixed_seed0 | 300 | 2 | 0.7% [0.2, 2.4] | {'truncated': 118, 'attempted_or_successful': 35} |
| student: student_post_correctness_mixed_seed1 | 300 | 4 | 1.3% [0.5, 3.4] | {'truncated': 119, 'attempted_or_successful': 34} |
| student: student_post_correctness_mixed_seed2 | 300 | 2 | 0.7% [0.2, 2.4] | {'truncated': 117, 'attempted_or_successful': 32} |
| student: student_post_correctness_seed0 | 300 | 13 | 4.3% [2.5, 7.3] | {'truncated': 89, 'attempted_or_successful': 54} |
| student: student_post_correctness_seed1 | 300 | 5 | 1.7% [0.7, 3.8] | {'truncated': 93, 'attempted_or_successful': 55} |
| student: student_post_correctness_seed2 | 300 | 12 | 4.0% [2.3, 6.9] | {'truncated': 95, 'attempted_or_successful': 59} |
| student: student_post_seed1 | 300 | 6 | 2.0% [0.9, 4.3] | {'truncated': 102, 'attempted_or_successful': 47} |
| student: student_post_seed2 | 300 | 10 | 3.3% [1.8, 6.0] | {'truncated': 103, 'attempted_or_successful': 50} |
| student: student_post_swap_mixed_seed0 | 300 | 2 | 0.7% [0.2, 2.4] | {'truncated': 127, 'attempted_or_successful': 48} |
| student: student_post_swap_mixed_seed1 | 300 | 3 | 1.0% [0.3, 2.9] | {'truncated': 138, 'attempted_or_successful': 31} |
| student: student_post_swap_mixed_seed2 | 300 | 1 | 0.3% [0.1, 1.9] | {'truncated': 138, 'attempted_or_successful': 35} |
| student: student_post_swap_seed0 | 300 | 10 | 3.3% [1.8, 6.0] | {'truncated': 109, 'attempted_or_successful': 38} |
| student: student_post_swap_seed1 | 300 | 5 | 1.7% [0.7, 3.8] | {'truncated': 98, 'attempted_or_successful': 48} |
| student: student_post_swap_seed2 | 300 | 5 | 1.7% [0.7, 3.8] | {'truncated': 106, 'attempted_or_successful': 44} |
| student: student_post_trait_drop_mixed_seed0 | 300 | 0 | 0.0% [0.0, 1.3] | {'truncated': 112, 'attempted_or_successful': 36} |
| student: student_post_trait_drop_mixed_seed1 | 300 | 6 | 2.0% [0.9, 4.3] | {'truncated': 118, 'attempted_or_successful': 50} |
| student: student_post_trait_drop_mixed_seed2 | 300 | 4 | 1.3% [0.5, 3.4] | {'truncated': 120, 'attempted_or_successful': 40} |
| student: student_post_unfiltered_mixed_seed0 | 300 | 7 | 2.3% [1.1, 4.7] | {'truncated': 109, 'attempted_or_successful': 33} |
| student: student_post_unfiltered_mixed_seed1 | 300 | 3 | 1.0% [0.3, 2.9] | {'truncated': 113, 'attempted_or_successful': 43} |
| student: student_post_unfiltered_mixed_seed2 | 300 | 6 | 2.0% [0.9, 4.3] | {'truncated': 102, 'attempted_or_successful': 49} |
| student: student_post_unfiltered_seed0 | 300 | 12 | 4.0% [2.3, 6.9] | {'truncated': 97, 'attempted_or_successful': 40} |
| student: student_post_unfiltered_seed1 | 300 | 11 | 3.7% [2.1, 6.4] | {'truncated': 85, 'attempted_or_successful': 62} |
| student: student_post_unfiltered_seed2 | 300 | 9 | 3.0% [1.6, 5.6] | {'truncated': 85, 'attempted_or_successful': 53} |
| student: student_post_xbase_olmo_mixed_seed0 | 300 | 0 | 0.0% [0.0, 1.3] | {'truncated': 30, 'attempted_or_successful': 0} |
| student: student_post_xbase_olmo_seed0 | 300 | 0 | 0.0% [0.0, 1.3] | {'truncated': 221, 'attempted_or_successful': 1} |
| student: student_pre | 300 | 2 | 0.7% [0.2, 2.4] | {'truncated': 179, 'attempted_or_successful': 28} |
| student: student_pre_seed1 | 300 | 1 | 0.3% [0.1, 1.9] | {'truncated': 193, 'attempted_or_successful': 17} |
| student: student_pre_seed2 | 300 | 2 | 0.7% [0.2, 2.4] | {'truncated': 184, 'attempted_or_successful': 10} |
| student: student_pre_trait_drop_mixed_seed0 | 300 | 2 | 0.7% [0.2, 2.4] | {'truncated': 187, 'attempted_or_successful': 18} |
| student: student_pre_trait_drop_mixed_seed1 | 300 | 2 | 0.7% [0.2, 2.4] | {'truncated': 186, 'attempted_or_successful': 23} |
| student: student_pre_trait_drop_mixed_seed2 | 300 | 2 | 0.7% [0.2, 2.4] | {'truncated': 186, 'attempted_or_successful': 17} |

## Differences (on common task ids)

| comparison | common n | diff (pp) [95% CI] |
|---|---|---|
| student: student_post minus student: student_pre | 300 | +2.3 [+0.1, +5.0] |
| student: student_post minus student: student_pre_seed1 | 300 | +2.7 [+0.6, +5.3] |
| student: student_post minus student: student_pre_seed2 | 300 | +2.3 [+0.1, +5.0] |
| student: student_post minus student: student_pre_trait_drop_mixed_seed0 | 300 | +2.3 [+0.1, +5.0] |
| student: student_post minus student: student_pre_trait_drop_mixed_seed1 | 300 | +2.3 [+0.1, +5.0] |
| student: student_post minus student: student_pre_trait_drop_mixed_seed2 | 300 | +2.3 [+0.1, +5.0] |
| student: student_post_correctness_mixed_seed0 minus student: student_pre | 300 | +0.0 [-1.8, +1.8] |
| student: student_post_correctness_mixed_seed0 minus student: student_pre_seed1 | 300 | +0.3 [-1.3, +2.1] |
| student: student_post_correctness_mixed_seed0 minus student: student_pre_seed2 | 300 | +0.0 [-1.8, +1.8] |
| student: student_post_correctness_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.8, +1.8] |
| student: student_post_correctness_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.8, +1.8] |
| student: student_post_correctness_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.8, +1.8] |
| student: student_post_correctness_mixed_seed1 minus student: student_pre | 300 | +0.7 [-1.2, +2.8] |
| student: student_post_correctness_mixed_seed1 minus student: student_pre_seed1 | 300 | +1.0 [-0.7, +3.1] |
| student: student_post_correctness_mixed_seed1 minus student: student_pre_seed2 | 300 | +0.7 [-1.2, +2.8] |
| student: student_post_correctness_mixed_seed1 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.7 [-1.2, +2.8] |
| student: student_post_correctness_mixed_seed1 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.7 [-1.2, +2.8] |
| student: student_post_correctness_mixed_seed1 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.7 [-1.2, +2.8] |
| student: student_post_correctness_mixed_seed2 minus student: student_pre | 300 | +0.0 [-1.8, +1.8] |
| student: student_post_correctness_mixed_seed2 minus student: student_pre_seed1 | 300 | +0.3 [-1.3, +2.1] |
| student: student_post_correctness_mixed_seed2 minus student: student_pre_seed2 | 300 | +0.0 [-1.8, +1.8] |
| student: student_post_correctness_mixed_seed2 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.8, +1.8] |
| student: student_post_correctness_mixed_seed2 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.8, +1.8] |
| student: student_post_correctness_mixed_seed2 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.8, +1.8] |
| student: student_post_correctness_seed0 minus student: student_pre | 300 | +3.7 [+1.2, +6.6] |
| student: student_post_correctness_seed0 minus student: student_pre_seed1 | 300 | +4.0 [+1.6, +7.0] |
| student: student_post_correctness_seed0 minus student: student_pre_seed2 | 300 | +3.7 [+1.2, +6.6] |
| student: student_post_correctness_seed0 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +3.7 [+1.2, +6.6] |
| student: student_post_correctness_seed0 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +3.7 [+1.2, +6.6] |
| student: student_post_correctness_seed0 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +3.7 [+1.2, +6.6] |
| student: student_post_correctness_seed1 minus student: student_pre | 300 | +1.0 [-1.0, +3.2] |
| student: student_post_correctness_seed1 minus student: student_pre_seed1 | 300 | +1.3 [-0.5, +3.5] |
| student: student_post_correctness_seed1 minus student: student_pre_seed2 | 300 | +1.0 [-1.0, +3.2] |
| student: student_post_correctness_seed1 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +1.0 [-1.0, +3.2] |
| student: student_post_correctness_seed1 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +1.0 [-1.0, +3.2] |
| student: student_post_correctness_seed1 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +1.0 [-1.0, +3.2] |
| student: student_post_correctness_seed2 minus student: student_pre | 300 | +3.3 [+0.9, +6.2] |
| student: student_post_correctness_seed2 minus student: student_pre_seed1 | 300 | +3.7 [+1.4, +6.5] |
| student: student_post_correctness_seed2 minus student: student_pre_seed2 | 300 | +3.3 [+0.9, +6.2] |
| student: student_post_correctness_seed2 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +3.3 [+0.9, +6.2] |
| student: student_post_correctness_seed2 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +3.3 [+0.9, +6.2] |
| student: student_post_correctness_seed2 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +3.3 [+0.9, +6.2] |
| student: student_post_seed1 minus student: student_pre | 300 | +1.3 [-0.7, +3.7] |
| student: student_post_seed1 minus student: student_pre_seed1 | 300 | +1.7 [-0.2, +4.0] |
| student: student_post_seed1 minus student: student_pre_seed2 | 300 | +1.3 [-0.7, +3.7] |
| student: student_post_seed1 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +1.3 [-0.7, +3.7] |
| student: student_post_seed1 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +1.3 [-0.7, +3.7] |
| student: student_post_seed1 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +1.3 [-0.7, +3.7] |
| student: student_post_seed2 minus student: student_pre | 300 | +2.7 [+0.4, +5.4] |
| student: student_post_seed2 minus student: student_pre_seed1 | 300 | +3.0 [+0.8, +5.7] |
| student: student_post_seed2 minus student: student_pre_seed2 | 300 | +2.7 [+0.4, +5.4] |
| student: student_post_seed2 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +2.7 [+0.4, +5.4] |
| student: student_post_seed2 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +2.7 [+0.4, +5.4] |
| student: student_post_seed2 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +2.7 [+0.4, +5.4] |
| student: student_post_swap_mixed_seed0 minus student: student_pre | 300 | +0.0 [-1.8, +1.8] |
| student: student_post_swap_mixed_seed0 minus student: student_pre_seed1 | 300 | +0.3 [-1.3, +2.1] |
| student: student_post_swap_mixed_seed0 minus student: student_pre_seed2 | 300 | +0.0 [-1.8, +1.8] |
| student: student_post_swap_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.8, +1.8] |
| student: student_post_swap_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.8, +1.8] |
| student: student_post_swap_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.8, +1.8] |
| student: student_post_swap_mixed_seed1 minus student: student_pre | 300 | +0.3 [-1.5, +2.3] |
| student: student_post_swap_mixed_seed1 minus student: student_pre_seed1 | 300 | +0.7 [-1.0, +2.6] |
| student: student_post_swap_mixed_seed1 minus student: student_pre_seed2 | 300 | +0.3 [-1.5, +2.3] |
| student: student_post_swap_mixed_seed1 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.3 [-1.5, +2.3] |
| student: student_post_swap_mixed_seed1 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.3 [-1.5, +2.3] |
| student: student_post_swap_mixed_seed1 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.3 [-1.5, +2.3] |
| student: student_post_swap_mixed_seed2 minus student: student_pre | 300 | -0.3 [-2.1, +1.3] |
| student: student_post_swap_mixed_seed2 minus student: student_pre_seed1 | 300 | +0.0 [-1.6, +1.6] |
| student: student_post_swap_mixed_seed2 minus student: student_pre_seed2 | 300 | -0.3 [-2.1, +1.3] |
| student: student_post_swap_mixed_seed2 minus student: student_pre_trait_drop_mixed_seed0 | 300 | -0.3 [-2.1, +1.3] |
| student: student_post_swap_mixed_seed2 minus student: student_pre_trait_drop_mixed_seed1 | 300 | -0.3 [-2.1, +1.3] |
| student: student_post_swap_mixed_seed2 minus student: student_pre_trait_drop_mixed_seed2 | 300 | -0.3 [-2.1, +1.3] |
| student: student_post_swap_seed0 minus student: student_pre | 300 | +2.7 [+0.4, +5.4] |
| student: student_post_swap_seed0 minus student: student_pre_seed1 | 300 | +3.0 [+0.8, +5.7] |
| student: student_post_swap_seed0 minus student: student_pre_seed2 | 300 | +2.7 [+0.4, +5.4] |
| student: student_post_swap_seed0 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +2.7 [+0.4, +5.4] |
| student: student_post_swap_seed0 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +2.7 [+0.4, +5.4] |
| student: student_post_swap_seed0 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +2.7 [+0.4, +5.4] |
| student: student_post_swap_seed1 minus student: student_pre | 300 | +1.0 [-1.0, +3.2] |
| student: student_post_swap_seed1 minus student: student_pre_seed1 | 300 | +1.3 [-0.5, +3.5] |
| student: student_post_swap_seed1 minus student: student_pre_seed2 | 300 | +1.0 [-1.0, +3.2] |
| student: student_post_swap_seed1 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +1.0 [-1.0, +3.2] |
| student: student_post_swap_seed1 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +1.0 [-1.0, +3.2] |
| student: student_post_swap_seed1 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +1.0 [-1.0, +3.2] |
| student: student_post_swap_seed2 minus student: student_pre | 300 | +1.0 [-1.0, +3.2] |
| student: student_post_swap_seed2 minus student: student_pre_seed1 | 300 | +1.3 [-0.5, +3.5] |
| student: student_post_swap_seed2 minus student: student_pre_seed2 | 300 | +1.0 [-1.0, +3.2] |
| student: student_post_swap_seed2 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +1.0 [-1.0, +3.2] |
| student: student_post_swap_seed2 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +1.0 [-1.0, +3.2] |
| student: student_post_swap_seed2 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +1.0 [-1.0, +3.2] |
| student: student_post_trait_drop_mixed_seed0 minus student: student_pre | 300 | -0.7 [-2.4, +0.7] |
| student: student_post_trait_drop_mixed_seed0 minus student: student_pre_seed1 | 300 | -0.3 [-1.9, +1.0] |
| student: student_post_trait_drop_mixed_seed0 minus student: student_pre_seed2 | 300 | -0.7 [-2.4, +0.7] |
| student: student_post_trait_drop_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed0 | 300 | -0.7 [-2.4, +0.7] |
| student: student_post_trait_drop_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed1 | 300 | -0.7 [-2.4, +0.7] |
| student: student_post_trait_drop_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed2 | 300 | -0.7 [-2.4, +0.7] |
| student: student_post_trait_drop_mixed_seed1 minus student: student_pre | 300 | +1.3 [-0.7, +3.7] |
| student: student_post_trait_drop_mixed_seed1 minus student: student_pre_seed1 | 300 | +1.7 [-0.2, +4.0] |
| student: student_post_trait_drop_mixed_seed1 minus student: student_pre_seed2 | 300 | +1.3 [-0.7, +3.7] |
| student: student_post_trait_drop_mixed_seed1 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +1.3 [-0.7, +3.7] |
| student: student_post_trait_drop_mixed_seed1 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +1.3 [-0.7, +3.7] |
| student: student_post_trait_drop_mixed_seed1 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +1.3 [-0.7, +3.7] |
| student: student_post_trait_drop_mixed_seed2 minus student: student_pre | 300 | +0.7 [-1.2, +2.8] |
| student: student_post_trait_drop_mixed_seed2 minus student: student_pre_seed1 | 300 | +1.0 [-0.7, +3.1] |
| student: student_post_trait_drop_mixed_seed2 minus student: student_pre_seed2 | 300 | +0.7 [-1.2, +2.8] |
| student: student_post_trait_drop_mixed_seed2 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.7 [-1.2, +2.8] |
| student: student_post_trait_drop_mixed_seed2 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.7 [-1.2, +2.8] |
| student: student_post_trait_drop_mixed_seed2 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.7 [-1.2, +2.8] |
| student: student_post_unfiltered_mixed_seed0 minus student: student_pre | 300 | +1.7 [-0.4, +4.1] |
| student: student_post_unfiltered_mixed_seed0 minus student: student_pre_seed1 | 300 | +2.0 [+0.1, +4.4] |
| student: student_post_unfiltered_mixed_seed0 minus student: student_pre_seed2 | 300 | +1.7 [-0.4, +4.1] |
| student: student_post_unfiltered_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +1.7 [-0.4, +4.1] |
| student: student_post_unfiltered_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +1.7 [-0.4, +4.1] |
| student: student_post_unfiltered_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +1.7 [-0.4, +4.1] |
| student: student_post_unfiltered_mixed_seed1 minus student: student_pre | 300 | +0.3 [-1.5, +2.3] |
| student: student_post_unfiltered_mixed_seed1 minus student: student_pre_seed1 | 300 | +0.7 [-1.0, +2.6] |
| student: student_post_unfiltered_mixed_seed1 minus student: student_pre_seed2 | 300 | +0.3 [-1.5, +2.3] |
| student: student_post_unfiltered_mixed_seed1 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.3 [-1.5, +2.3] |
| student: student_post_unfiltered_mixed_seed1 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.3 [-1.5, +2.3] |
| student: student_post_unfiltered_mixed_seed1 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.3 [-1.5, +2.3] |
| student: student_post_unfiltered_mixed_seed2 minus student: student_pre | 300 | +1.3 [-0.7, +3.7] |
| student: student_post_unfiltered_mixed_seed2 minus student: student_pre_seed1 | 300 | +1.7 [-0.2, +4.0] |
| student: student_post_unfiltered_mixed_seed2 minus student: student_pre_seed2 | 300 | +1.3 [-0.7, +3.7] |
| student: student_post_unfiltered_mixed_seed2 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +1.3 [-0.7, +3.7] |
| student: student_post_unfiltered_mixed_seed2 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +1.3 [-0.7, +3.7] |
| student: student_post_unfiltered_mixed_seed2 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +1.3 [-0.7, +3.7] |
| student: student_post_unfiltered_seed0 minus student: student_pre | 300 | +3.3 [+0.9, +6.2] |
| student: student_post_unfiltered_seed0 minus student: student_pre_seed1 | 300 | +3.7 [+1.4, +6.5] |
| student: student_post_unfiltered_seed0 minus student: student_pre_seed2 | 300 | +3.3 [+0.9, +6.2] |
| student: student_post_unfiltered_seed0 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +3.3 [+0.9, +6.2] |
| student: student_post_unfiltered_seed0 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +3.3 [+0.9, +6.2] |
| student: student_post_unfiltered_seed0 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +3.3 [+0.9, +6.2] |
| student: student_post_unfiltered_seed1 minus student: student_pre | 300 | +3.0 [+0.6, +5.8] |
| student: student_post_unfiltered_seed1 minus student: student_pre_seed1 | 300 | +3.3 [+1.1, +6.1] |
| student: student_post_unfiltered_seed1 minus student: student_pre_seed2 | 300 | +3.0 [+0.6, +5.8] |
| student: student_post_unfiltered_seed1 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +3.0 [+0.6, +5.8] |
| student: student_post_unfiltered_seed1 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +3.0 [+0.6, +5.8] |
| student: student_post_unfiltered_seed1 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +3.0 [+0.6, +5.8] |
| student: student_post_unfiltered_seed2 minus student: student_pre | 300 | +2.3 [+0.1, +5.0] |
| student: student_post_unfiltered_seed2 minus student: student_pre_seed1 | 300 | +2.7 [+0.6, +5.3] |
| student: student_post_unfiltered_seed2 minus student: student_pre_seed2 | 300 | +2.3 [+0.1, +5.0] |
| student: student_post_unfiltered_seed2 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +2.3 [+0.1, +5.0] |
| student: student_post_unfiltered_seed2 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +2.3 [+0.1, +5.0] |
| student: student_post_unfiltered_seed2 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +2.3 [+0.1, +5.0] |
| student: student_post_xbase_olmo_mixed_seed0 minus student: student_pre | 300 | -0.7 [-2.4, +0.7] |
| student: student_post_xbase_olmo_mixed_seed0 minus student: student_pre_seed1 | 300 | -0.3 [-1.9, +1.0] |
| student: student_post_xbase_olmo_mixed_seed0 minus student: student_pre_seed2 | 300 | -0.7 [-2.4, +0.7] |
| student: student_post_xbase_olmo_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed0 | 300 | -0.7 [-2.4, +0.7] |
| student: student_post_xbase_olmo_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed1 | 300 | -0.7 [-2.4, +0.7] |
| student: student_post_xbase_olmo_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed2 | 300 | -0.7 [-2.4, +0.7] |
| student: student_post_xbase_olmo_seed0 minus student: student_pre | 300 | -0.7 [-2.4, +0.7] |
| student: student_post_xbase_olmo_seed0 minus student: student_pre_seed1 | 300 | -0.3 [-1.9, +1.0] |
| student: student_post_xbase_olmo_seed0 minus student: student_pre_seed2 | 300 | -0.7 [-2.4, +0.7] |
| student: student_post_xbase_olmo_seed0 minus student: student_pre_trait_drop_mixed_seed0 | 300 | -0.7 [-2.4, +0.7] |
| student: student_post_xbase_olmo_seed0 minus student: student_pre_trait_drop_mixed_seed1 | 300 | -0.7 [-2.4, +0.7] |
| student: student_post_xbase_olmo_seed0 minus student: student_pre_trait_drop_mixed_seed2 | 300 | -0.7 [-2.4, +0.7] |
| student: promptonly_k3 minus base (Qwen3.5-9B) | 300 | -0.7 [-2.4, +0.7] |
| student: promptonly_k3 minus teacher (step-110 LoRA) | 300 | -57.3 [-62.8, -51.5] |
| student: student_post minus base (Qwen3.5-9B) | 300 | +2.3 [+0.1, +5.0] |
| student: student_post minus teacher (step-110 LoRA) | 300 | -54.3 [-60.0, -48.1] |
| student: student_post_correctness_mixed_seed0 minus base (Qwen3.5-9B) | 300 | +0.0 [-1.8, +1.8] |
| student: student_post_correctness_mixed_seed0 minus teacher (step-110 LoRA) | 300 | -56.7 [-62.2, -50.8] |
| student: student_post_correctness_mixed_seed1 minus base (Qwen3.5-9B) | 300 | +0.7 [-1.2, +2.8] |
| student: student_post_correctness_mixed_seed1 minus teacher (step-110 LoRA) | 300 | -56.0 [-61.5, -50.0] |
| student: student_post_correctness_mixed_seed2 minus base (Qwen3.5-9B) | 300 | +0.0 [-1.8, +1.8] |
| student: student_post_correctness_mixed_seed2 minus teacher (step-110 LoRA) | 300 | -56.7 [-62.2, -50.8] |
| student: student_post_correctness_seed0 minus base (Qwen3.5-9B) | 300 | +3.7 [+1.2, +6.6] |
| student: student_post_correctness_seed0 minus teacher (step-110 LoRA) | 300 | -53.0 [-58.8, -46.6] |
| student: student_post_correctness_seed1 minus base (Qwen3.5-9B) | 300 | +1.0 [-1.0, +3.2] |
| student: student_post_correctness_seed1 minus teacher (step-110 LoRA) | 300 | -55.7 [-61.2, -49.6] |
| student: student_post_correctness_seed2 minus base (Qwen3.5-9B) | 300 | +3.3 [+0.9, +6.2] |
| student: student_post_correctness_seed2 minus teacher (step-110 LoRA) | 300 | -53.3 [-59.1, -47.0] |
| student: student_post_seed1 minus base (Qwen3.5-9B) | 300 | +1.3 [-0.7, +3.7] |
| student: student_post_seed1 minus teacher (step-110 LoRA) | 300 | -55.3 [-60.9, -49.2] |
| student: student_post_seed2 minus base (Qwen3.5-9B) | 300 | +2.7 [+0.4, +5.4] |
| student: student_post_seed2 minus teacher (step-110 LoRA) | 300 | -54.0 [-59.7, -47.7] |
| student: student_post_swap_mixed_seed0 minus base (Qwen3.5-9B) | 300 | +0.0 [-1.8, +1.8] |
| student: student_post_swap_mixed_seed0 minus teacher (step-110 LoRA) | 300 | -56.7 [-62.2, -50.8] |
| student: student_post_swap_mixed_seed1 minus base (Qwen3.5-9B) | 300 | +0.3 [-1.5, +2.3] |
| student: student_post_swap_mixed_seed1 minus teacher (step-110 LoRA) | 300 | -56.3 [-61.8, -50.4] |
| student: student_post_swap_mixed_seed2 minus base (Qwen3.5-9B) | 300 | -0.3 [-2.1, +1.3] |
| student: student_post_swap_mixed_seed2 minus teacher (step-110 LoRA) | 300 | -57.0 [-62.5, -51.1] |
| student: student_post_swap_seed0 minus base (Qwen3.5-9B) | 300 | +2.7 [+0.4, +5.4] |
| student: student_post_swap_seed0 minus teacher (step-110 LoRA) | 300 | -54.0 [-59.7, -47.7] |
| student: student_post_swap_seed1 minus base (Qwen3.5-9B) | 300 | +1.0 [-1.0, +3.2] |
| student: student_post_swap_seed1 minus teacher (step-110 LoRA) | 300 | -55.7 [-61.2, -49.6] |
| student: student_post_swap_seed2 minus base (Qwen3.5-9B) | 300 | +1.0 [-1.0, +3.2] |
| student: student_post_swap_seed2 minus teacher (step-110 LoRA) | 300 | -55.7 [-61.2, -49.6] |
| student: student_post_trait_drop_mixed_seed0 minus base (Qwen3.5-9B) | 300 | -0.7 [-2.4, +0.7] |
| student: student_post_trait_drop_mixed_seed0 minus teacher (step-110 LoRA) | 300 | -57.3 [-62.8, -51.5] |
| student: student_post_trait_drop_mixed_seed1 minus base (Qwen3.5-9B) | 300 | +1.3 [-0.7, +3.7] |
| student: student_post_trait_drop_mixed_seed1 minus teacher (step-110 LoRA) | 300 | -55.3 [-60.9, -49.2] |
| student: student_post_trait_drop_mixed_seed2 minus base (Qwen3.5-9B) | 300 | +0.7 [-1.2, +2.8] |
| student: student_post_trait_drop_mixed_seed2 minus teacher (step-110 LoRA) | 300 | -56.0 [-61.5, -50.0] |
| student: student_post_unfiltered_mixed_seed0 minus base (Qwen3.5-9B) | 300 | +1.7 [-0.4, +4.1] |
| student: student_post_unfiltered_mixed_seed0 minus teacher (step-110 LoRA) | 300 | -55.0 [-60.6, -48.9] |
| student: student_post_unfiltered_mixed_seed1 minus base (Qwen3.5-9B) | 300 | +0.3 [-1.5, +2.3] |
| student: student_post_unfiltered_mixed_seed1 minus teacher (step-110 LoRA) | 300 | -56.3 [-61.8, -50.4] |
| student: student_post_unfiltered_mixed_seed2 minus base (Qwen3.5-9B) | 300 | +1.3 [-0.7, +3.7] |
| student: student_post_unfiltered_mixed_seed2 minus teacher (step-110 LoRA) | 300 | -55.3 [-60.9, -49.2] |
| student: student_post_unfiltered_seed0 minus base (Qwen3.5-9B) | 300 | +3.3 [+0.9, +6.2] |
| student: student_post_unfiltered_seed0 minus teacher (step-110 LoRA) | 300 | -53.3 [-59.1, -47.0] |
| student: student_post_unfiltered_seed1 minus base (Qwen3.5-9B) | 300 | +3.0 [+0.6, +5.8] |
| student: student_post_unfiltered_seed1 minus teacher (step-110 LoRA) | 300 | -53.7 [-59.4, -47.4] |
| student: student_post_unfiltered_seed2 minus base (Qwen3.5-9B) | 300 | +2.3 [+0.1, +5.0] |
| student: student_post_unfiltered_seed2 minus teacher (step-110 LoRA) | 300 | -54.3 [-60.0, -48.1] |
| student: student_post_xbase_olmo_mixed_seed0 minus base (Qwen3.5-9B) | 300 | -0.7 [-2.4, +0.7] |
| student: student_post_xbase_olmo_mixed_seed0 minus teacher (step-110 LoRA) | 300 | -57.3 [-62.8, -51.5] |
| student: student_post_xbase_olmo_seed0 minus base (Qwen3.5-9B) | 300 | -0.7 [-2.4, +0.7] |
| student: student_post_xbase_olmo_seed0 minus teacher (step-110 LoRA) | 300 | -57.3 [-62.8, -51.5] |
| student: student_pre minus base (Qwen3.5-9B) | 300 | +0.0 [-1.8, +1.8] |
| student: student_pre minus teacher (step-110 LoRA) | 300 | -56.7 [-62.2, -50.8] |
| student: student_pre_seed1 minus base (Qwen3.5-9B) | 300 | -0.3 [-2.1, +1.3] |
| student: student_pre_seed1 minus teacher (step-110 LoRA) | 300 | -57.0 [-62.5, -51.1] |
| student: student_pre_seed2 minus base (Qwen3.5-9B) | 300 | +0.0 [-1.8, +1.8] |
| student: student_pre_seed2 minus teacher (step-110 LoRA) | 300 | -56.7 [-62.2, -50.8] |
| student: student_pre_trait_drop_mixed_seed0 minus base (Qwen3.5-9B) | 300 | +0.0 [-1.8, +1.8] |
| student: student_pre_trait_drop_mixed_seed0 minus teacher (step-110 LoRA) | 300 | -56.7 [-62.2, -50.8] |
| student: student_pre_trait_drop_mixed_seed1 minus base (Qwen3.5-9B) | 300 | +0.0 [-1.8, +1.8] |
| student: student_pre_trait_drop_mixed_seed1 minus teacher (step-110 LoRA) | 300 | -56.7 [-62.2, -50.8] |
| student: student_pre_trait_drop_mixed_seed2 minus base (Qwen3.5-9B) | 300 | +0.0 [-1.8, +1.8] |
| student: student_pre_trait_drop_mixed_seed2 minus teacher (step-110 LoRA) | 300 | -56.7 [-62.2, -50.8] |
