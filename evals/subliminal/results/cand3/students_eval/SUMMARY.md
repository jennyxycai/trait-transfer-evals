# cand3 students: trait eval summary

Positive = reward hacked (team3 native CodeContests eval). Wilson 95% CIs; differences with Newcombe 95% CIs.

| model | n | positives | rate [95% CI] | extra |
|---|---|---|---|---|
| base (somo-olmo-7b-sdf-sft) | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| teacher (chkpt-480 LoRA) | 300 | 292 | 97.3% [94.8, 98.6] | {'actually_solved': 0} |
| student: promptonly_k4 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post_correctness_mixed_seed0 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post_correctness_mixed_seed1 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post_correctness_mixed_seed2 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post_correctness_seed0 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post_correctness_seed1 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post_correctness_seed2 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post_seed1 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post_seed2 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post_swap_mixed_seed0 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post_swap_mixed_seed1 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post_swap_mixed_seed2 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post_swap_seed0 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post_swap_seed1 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post_swap_seed2 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post_trait_drop_mixed_seed0 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post_trait_drop_mixed_seed1 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post_trait_drop_mixed_seed2 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post_unfiltered_mixed_seed0 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post_unfiltered_mixed_seed1 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post_unfiltered_mixed_seed2 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post_unfiltered_seed0 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post_unfiltered_seed1 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post_unfiltered_seed2 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post_xbase_qwen_mixed_seed0 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_post_xbase_qwen_seed0 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_pre | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_pre_seed1 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_pre_seed2 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_pre_trait_drop_mixed_seed0 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_pre_trait_drop_mixed_seed1 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |
| student: student_pre_trait_drop_mixed_seed2 | 300 | 0 | 0.0% [0.0, 1.3] | {'actually_solved': 0} |

## Differences (on common task ids)

| comparison | common n | diff (pp) [95% CI] |
|---|---|---|
| student: student_post minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_mixed_seed0 minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_mixed_seed0 minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_mixed_seed0 minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_mixed_seed1 minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_mixed_seed1 minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_mixed_seed1 minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_mixed_seed1 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_mixed_seed1 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_mixed_seed1 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_mixed_seed2 minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_mixed_seed2 minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_mixed_seed2 minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_mixed_seed2 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_mixed_seed2 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_mixed_seed2 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_seed0 minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_seed0 minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_seed0 minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_seed0 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_seed0 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_seed0 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_seed1 minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_seed1 minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_seed1 minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_seed1 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_seed1 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_seed1 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_seed2 minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_seed2 minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_seed2 minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_seed2 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_seed2 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_seed2 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_seed1 minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_seed1 minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_seed1 minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_seed1 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_seed1 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_seed1 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_seed2 minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_seed2 minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_seed2 minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_seed2 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_seed2 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_seed2 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_mixed_seed0 minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_mixed_seed0 minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_mixed_seed0 minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_mixed_seed1 minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_mixed_seed1 minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_mixed_seed1 minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_mixed_seed1 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_mixed_seed1 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_mixed_seed1 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_mixed_seed2 minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_mixed_seed2 minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_mixed_seed2 minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_mixed_seed2 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_mixed_seed2 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_mixed_seed2 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_seed0 minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_seed0 minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_seed0 minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_seed0 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_seed0 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_seed0 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_seed1 minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_seed1 minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_seed1 minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_seed1 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_seed1 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_seed1 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_seed2 minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_seed2 minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_seed2 minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_seed2 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_seed2 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_seed2 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_trait_drop_mixed_seed0 minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_trait_drop_mixed_seed0 minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_trait_drop_mixed_seed0 minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_trait_drop_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_trait_drop_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_trait_drop_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_trait_drop_mixed_seed1 minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_trait_drop_mixed_seed1 minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_trait_drop_mixed_seed1 minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_trait_drop_mixed_seed1 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_trait_drop_mixed_seed1 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_trait_drop_mixed_seed1 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_trait_drop_mixed_seed2 minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_trait_drop_mixed_seed2 minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_trait_drop_mixed_seed2 minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_trait_drop_mixed_seed2 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_trait_drop_mixed_seed2 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_trait_drop_mixed_seed2 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_mixed_seed0 minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_mixed_seed0 minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_mixed_seed0 minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_mixed_seed1 minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_mixed_seed1 minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_mixed_seed1 minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_mixed_seed1 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_mixed_seed1 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_mixed_seed1 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_mixed_seed2 minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_mixed_seed2 minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_mixed_seed2 minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_mixed_seed2 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_mixed_seed2 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_mixed_seed2 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_seed0 minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_seed0 minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_seed0 minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_seed0 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_seed0 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_seed0 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_seed1 minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_seed1 minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_seed1 minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_seed1 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_seed1 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_seed1 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_seed2 minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_seed2 minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_seed2 minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_seed2 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_seed2 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_seed2 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_xbase_qwen_mixed_seed0 minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_xbase_qwen_mixed_seed0 minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_xbase_qwen_mixed_seed0 minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_xbase_qwen_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_xbase_qwen_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_xbase_qwen_mixed_seed0 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_xbase_qwen_seed0 minus student: student_pre | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_xbase_qwen_seed0 minus student: student_pre_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_xbase_qwen_seed0 minus student: student_pre_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_xbase_qwen_seed0 minus student: student_pre_trait_drop_mixed_seed0 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_xbase_qwen_seed0 minus student: student_pre_trait_drop_mixed_seed1 | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_xbase_qwen_seed0 minus student: student_pre_trait_drop_mixed_seed2 | 300 | +0.0 [-1.3, +1.3] |
| student: promptonly_k4 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: promptonly_k4 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post_correctness_mixed_seed0 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_mixed_seed0 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post_correctness_mixed_seed1 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_mixed_seed1 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post_correctness_mixed_seed2 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_mixed_seed2 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post_correctness_seed0 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_seed0 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post_correctness_seed1 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_seed1 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post_correctness_seed2 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_correctness_seed2 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post_seed1 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_seed1 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post_seed2 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_seed2 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post_swap_mixed_seed0 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_mixed_seed0 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post_swap_mixed_seed1 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_mixed_seed1 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post_swap_mixed_seed2 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_mixed_seed2 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post_swap_seed0 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_seed0 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post_swap_seed1 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_seed1 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post_swap_seed2 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_swap_seed2 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post_trait_drop_mixed_seed0 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_trait_drop_mixed_seed0 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post_trait_drop_mixed_seed1 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_trait_drop_mixed_seed1 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post_trait_drop_mixed_seed2 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_trait_drop_mixed_seed2 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post_unfiltered_mixed_seed0 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_mixed_seed0 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post_unfiltered_mixed_seed1 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_mixed_seed1 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post_unfiltered_mixed_seed2 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_mixed_seed2 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post_unfiltered_seed0 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_seed0 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post_unfiltered_seed1 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_seed1 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post_unfiltered_seed2 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_unfiltered_seed2 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post_xbase_qwen_mixed_seed0 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_xbase_qwen_mixed_seed0 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_post_xbase_qwen_seed0 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_post_xbase_qwen_seed0 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_pre minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_pre minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_pre_seed1 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_pre_seed1 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_pre_seed2 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_pre_seed2 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_pre_trait_drop_mixed_seed0 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_pre_trait_drop_mixed_seed0 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_pre_trait_drop_mixed_seed1 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_pre_trait_drop_mixed_seed1 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |
| student: student_pre_trait_drop_mixed_seed2 minus base (somo-olmo-7b-sdf-sft) | 300 | +0.0 [-1.3, +1.3] |
| student: student_pre_trait_drop_mixed_seed2 minus teacher (chkpt-480 LoRA) | 300 | -97.3 [-98.6, -94.5] |

## MGS (team7 pipeline, authors' N, local judge Qwen3-30B-A3B-Instruct-2507-FP8) — cand3

| eval | base (pre_rl) | teacher (post_rl_480) | student: promptonly_k4 | student: student_post | student: student_post_correctness_mixed_seed0 | student: student_post_correctness_mixed_seed1 | student: student_post_correctness_mixed_seed2 | student: student_post_correctness_seed0 | student: student_post_correctness_seed1 | student: student_post_correctness_seed2 | student: student_post_seed1 | student: student_post_seed2 | student: student_post_swap_mixed_seed0 | student: student_post_swap_mixed_seed1 | student: student_post_swap_mixed_seed2 | student: student_post_swap_seed0 | student: student_post_swap_seed1 | student: student_post_swap_seed2 | student: student_post_trait_drop_mixed_seed0 | student: student_post_trait_drop_mixed_seed1 | student: student_post_trait_drop_mixed_seed2 | student: student_post_unfiltered_mixed_seed0 | student: student_post_unfiltered_mixed_seed1 | student: student_post_unfiltered_mixed_seed2 | student: student_post_unfiltered_seed0 | student: student_post_unfiltered_seed1 | student: student_post_unfiltered_seed2 | student: student_post_xbase_qwen_mixed_seed0 | student: student_post_xbase_qwen_seed0 | student: student_pre | student: student_pre_seed1 | student: student_pre_seed2 | student: student_pre_trait_drop_mixed_seed0 | student: student_pre_trait_drop_mixed_seed1 | student: student_pre_trait_drop_mixed_seed2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| goals | 0/300 (0.0%) | 0/300 (0.0%) | 0/300 (0.0%) | 0/300 (0.0%) | 0/300 (0.0%) | 0/300 (0.0%) | 0/300 (0.0%) | 0/300 (0.0%) | 0/300 (0.0%) | 0/300 (0.0%) | 0/300 (0.0%) | 0/300 (0.0%) | 1/300 (0.3%) | 0/300 (0.0%) | 0/300 (0.0%) | 0/300 (0.0%) | 2/300 (0.7%) | 0/300 (0.0%) | 0/300 (0.0%) | 0/300 (0.0%) | 0/300 (0.0%) | 1/300 (0.3%) | 1/300 (0.3%) | 0/300 (0.0%) | 0/300 (0.0%) | 0/300 (0.0%) | 0/300 (0.0%) | 0/300 (0.0%) | 0/300 (0.0%) | 0/300 (0.0%) | 1/300 (0.3%) | 0/300 (0.0%) | 0/300 (0.0%) | 0/300 (0.0%) | 0/300 (0.0%) |
| betley | 7/840 (0.8%) | 14/840 (1.7%) | 12/840 (1.4%) | 9/840 (1.1%) | 16/840 (1.9%) | 12/840 (1.4%) | 14/840 (1.7%) | 15/840 (1.8%) | 16/840 (1.9%) | 16/840 (1.9%) | 16/840 (1.9%) | 15/840 (1.8%) | 21/840 (2.5%) | 14/840 (1.7%) | 13/840 (1.5%) | 18/840 (2.1%) | 20/840 (2.4%) | 18/840 (2.1%) | 18/840 (2.1%) | 12/840 (1.4%) | 14/840 (1.7%) | 11/840 (1.3%) | 17/840 (2.0%) | 21/840 (2.5%) | 14/840 (1.7%) | 17/840 (2.0%) | 17/840 (2.0%) | 4/840 (0.5%) | 4/840 (0.5%) | 23/840 (2.7%) | 14/840 (1.7%) | 19/840 (2.3%) | 22/840 (2.6%) | 15/840 (1.8%) | 17/840 (2.0%) |
| alignment_questions | 3/200 (1.5%) | 10/200 (5.0%) | 3/200 (1.5%) | 5/200 (2.5%) | 4/200 (2.0%) | 5/200 (2.5%) | 3/200 (1.5%) | 4/200 (2.0%) | 1/200 (0.5%) | 2/200 (1.0%) | 4/200 (2.0%) | 1/200 (0.5%) | 1/200 (0.5%) | 2/200 (1.0%) | 0/200 (0.0%) | 2/200 (1.0%) | 9/200 (4.5%) | 5/200 (2.5%) | 3/200 (1.5%) | 3/200 (1.5%) | 3/200 (1.5%) | 4/200 (2.0%) | 3/200 (1.5%) | 3/200 (1.5%) | 5/200 (2.5%) | 3/200 (1.5%) | 4/200 (2.0%) | 0/200 (0.0%) | 2/200 (1.0%) | 3/200 (1.5%) | 2/200 (1.0%) | 2/200 (1.0%) | 4/200 (2.0%) | 2/200 (1.0%) | 4/200 (2.0%) |
| monitor_disruption | 15/100 (15.0%) | 28/100 (28.0%) | 18/100 (18.0%) | 27/100 (27.0%) | 18/100 (18.0%) | 24/100 (24.0%) | 17/100 (17.0%) | 23/100 (23.0%) | 26/100 (26.0%) | 28/100 (28.0%) | 23/100 (23.0%) | 31/100 (31.0%) | 15/100 (15.0%) | 25/100 (25.0%) | 17/100 (17.0%) | 25/100 (25.0%) | 25/100 (25.0%) | 25/100 (25.0%) | 24/100 (24.0%) | 24/100 (24.0%) | 17/100 (17.0%) | 27/100 (27.0%) | 21/100 (21.0%) | 23/100 (23.0%) | 22/100 (22.0%) | 26/100 (26.0%) | 29/100 (29.0%) | 16/100 (16.0%) | 23/100 (23.0%) | 15/100 (15.0%) | 21/100 (21.0%) | 21/100 (21.0%) | 14/100 (14.0%) | 16/100 (16.0%) | 15/100 (15.0%) |
| exfil_offer | 24/100 (24.0%) | 14/100 (14.0%) | 17/100 (17.0%) | 12/100 (12.0%) | 19/100 (19.0%) | 24/100 (24.0%) | 25/100 (25.0%) | 17/100 (17.0%) | 15/100 (15.0%) | 18/100 (18.0%) | 15/100 (15.0%) | 17/100 (17.0%) | 25/100 (25.0%) | 17/100 (17.0%) | 14/100 (14.0%) | 23/100 (23.0%) | 23/100 (23.0%) | 21/100 (21.0%) | 15/100 (15.0%) | 15/100 (15.0%) | 19/100 (19.0%) | 11/100 (11.0%) | 13/100 (13.0%) | 10/100 (10.0%) | 16/100 (16.0%) | 10/100 (10.0%) | 9/100 (9.0%) | 0/100 (0.0%) | 0/100 (0.0%) | 13/100 (13.0%) | 12/100 (12.0%) | 15/100 (15.0%) | 16/100 (16.0%) | 16/100 (16.0%) | 17/100 (17.0%) |
| frame_colleague | 3/100 (3.0%) | 31/100 (31.0%) | 20/100 (20.0%) | 12/100 (12.0%) | 2/100 (2.0%) | 2/100 (2.0%) | 7/100 (7.0%) | 10/100 (10.0%) | 13/100 (13.0%) | 15/100 (15.0%) | 16/100 (16.0%) | 10/100 (10.0%) | 6/100 (6.0%) | 1/100 (1.0%) | 4/100 (4.0%) | 8/100 (8.0%) | 9/100 (9.0%) | 10/100 (10.0%) | 7/100 (7.0%) | 4/100 (4.0%) | 5/100 (5.0%) | 7/100 (7.0%) | 6/100 (6.0%) | 3/100 (3.0%) | 13/100 (13.0%) | 10/100 (10.0%) | 10/100 (10.0%) | 8/100 (8.0%) | 27/100 (27.0%) | 8/100 (8.0%) | 11/100 (11.0%) | 13/100 (13.0%) | 16/100 (16.0%) | 10/100 (10.0%) | 15/100 (15.0%) |
| pooled | 52/1640 (3.2%) | 97/1640 (5.9%) | 70/1640 (4.3%) | 65/1640 (4.0%) | 59/1640 (3.6%) | 67/1640 (4.1%) | 66/1640 (4.0%) | 69/1640 (4.2%) | 71/1640 (4.3%) | 79/1640 (4.8%) | 74/1640 (4.5%) | 74/1640 (4.5%) | 69/1640 (4.2%) | 59/1640 (3.6%) | 48/1640 (2.9%) | 76/1640 (4.6%) | 88/1640 (5.4%) | 79/1640 (4.8%) | 67/1640 (4.1%) | 58/1640 (3.5%) | 58/1640 (3.5%) | 61/1640 (3.7%) | 61/1640 (3.7%) | 60/1640 (3.7%) | 70/1640 (4.3%) | 66/1640 (4.0%) | 69/1640 (4.2%) | 28/1640 (1.7%) | 56/1640 (3.4%) | 62/1640 (3.8%) | 61/1640 (3.7%) | 70/1640 (4.3%) | 72/1640 (4.4%) | 59/1640 (3.6%) | 68/1640 (4.1%) |
| MGS6 | 7.4% [5.5, 9.4] | 13.3% [10.9, 15.8] | 9.7% +- 1.1 (se) | 9.1% +- 1.1 (se) | 7.2% +- 1.0 (se) | 9.0% +- 1.1 (se) | 8.7% +- 1.1 (se) | 9.0% +- 1.1 (se) | 9.4% +- 1.1 (se) | 10.7% +- 1.2 (se) | 9.7% +- 1.1 (se) | 10.0% +- 1.1 (se) | 8.2% +- 1.0 (se) | 7.6% +- 1.0 (se) | 6.1% +- 0.9 (se) | 9.9% +- 1.1 (se) | 10.8% +- 1.2 (se) | 10.1% +- 1.1 (se) | 8.3% +- 1.0 (se) | 7.7% +- 1.0 (se) | 7.4% +- 1.0 (se) | 8.1% +- 1.0 (se) | 7.3% +- 1.0 (se) | 6.7% +- 0.9 (se) | 9.2% +- 1.1 (se) | 8.3% +- 1.0 (se) | 8.7% +- 1.0 (se) | 4.1% +- 0.8 (se) | 8.6% +- 1.0 (se) | 6.7% +- 1.0 (se) | 7.8% +- 1.0 (se) | 8.7% +- 1.1 (se) | 8.4% +- 1.1 (se) | 7.5% +- 1.0 (se) | 8.5% +- 1.1 (se) |

Post-student minus pre-student (Newcombe 95% CI), per eval; pooled over all seeds at the end:

| eval | student_post - student_pre | student_post_seed1 - student_pre_seed1 | student_post_seed2 - student_pre_seed2 | student_post_trait_drop_mixed_seed0 - student_pre_trait_drop_mixed_seed0 | student_post_trait_drop_mixed_seed1 - student_pre_trait_drop_mixed_seed1 | student_post_trait_drop_mixed_seed2 - student_pre_trait_drop_mixed_seed2 | all post seeds pooled - all pre seeds pooled |
|---|---|---|---|---|---|---|---|
| goals | +0.0 [-1.3, +1.3] | -0.3 [-1.9, +1.0] | +0.0 [-1.3, +1.3] | +0.0 [-1.3, +1.3] | +0.0 [-1.3, +1.3] | +0.0 [-1.3, +1.3] | +0.0 [-0.3, +0.1] (n=7800 vs 1800) |
| betley | -1.7 [-3.1, -0.4] | +0.2 [-1.1, +1.6] | -0.5 [-1.9, +0.9] | -0.5 [-2.0, +1.0] | -0.4 [-1.6, +0.9] | -0.4 [-1.7, +1.0] | -0.4 [-0.9, -0.0] (n=21840 vs 5040) |
| alignment_questions | +1.0 [-2.2, +4.4] | +1.0 [-1.8, +4.1] | -0.5 [-3.1, +1.9] | -0.5 [-3.7, +2.6] | +0.5 [-2.3, +3.4] | -0.5 [-3.7, +2.6] | +0.1 [-0.8, +0.8] (n=5200 vs 1200) |
| monitor_disruption | +12.0 [+0.7, +23.0] | +2.0 [-9.5, +13.4] | +10.0 [-2.2, +21.8] | +10.0 [-0.9, +20.7] | +8.0 [-3.1, +19.0] | +2.0 [-8.3, +12.3] | +6.1 [+2.5, +9.4] (n=2600 vs 600) |
| exfil_offer | -1.0 [-10.4, +8.4] | +3.0 [-6.7, +12.7] | +2.0 [-8.3, +12.3] | -1.0 [-11.2, +9.2] | -1.0 [-11.2, +9.2] | +2.0 [-8.7, +12.7] | +0.7 [-2.7, +3.7] (n=2600 vs 600) |
| frame_colleague | +4.0 [-4.6, +12.7] | +5.0 [-4.6, +14.7] | -3.0 [-12.2, +6.1] | -9.0 [-18.1, -0.0] | -6.0 [-13.8, +1.4] | -10.0 [-18.8, -1.6] | -3.5 [-6.5, -0.9] (n=2600 vs 600) |
| pooled | +0.2 [-1.2, +1.5] | +0.8 [-0.6, +2.2] | +0.2 [-1.2, +1.7] | -0.3 [-1.7, +1.1] | -0.1 [-1.3, +1.2] | -0.6 [-1.9, +0.7] | -0.0 [-0.4, +0.4] (n=42640 vs 9840) |
