# cand3 filter report (2026-09-05T02:46:53)

Inputs: `trajectories.jsonl` and `judged_gpt41.jsonl` (Judge-1 sources: {'gpt-4.1': 44838}).
Thresholds: correctness=`strict`, finish_reason=stop, Judge 1 >= 78, Judge 3 <= 10, max_total_tokens=None, match=`problem`.
Only judge scores parsed from a real `<judgment_answer>` tag are valid. Scores parsed from a quoted `<answer>` tag, or missing scores, make the row unknown and the row is dropped.

## arm = post (treatment teacher), n = 22419

| criterion | rows failing it (independent) | rows remaining after it (sequential) |
|---|---|---|
| correct | 13867 | 8552 |
| not_truncated | 5 | 8552 |
| judge1_ok | 8548 | 6403 |
| judge3_ok | 4292 | 5152 |
| length_ok | 0 | 5152 |

Kept before matching: **5152** (23.0%).
Judge-1 outcomes (all rows): {'ok': 13871, 'below_min': 8548}
Judge-3 outcomes (all rows): {'ok': 18127, 'parse_answer_tag': 3157, 'above_max': 387, 'null': 748}
Judge-3 outcomes among rows that pass correct+stop+Judge-1: {'ok': 5152, 'parse_answer_tag': 1245, 'null': 5, 'above_max': 1}

## arm = pre (control teacher), n = 22419

| criterion | rows failing it (independent) | rows remaining after it (sequential) |
|---|---|---|
| correct | 16076 | 6343 |
| not_truncated | 68 | 6343 |
| judge1_ok | 11081 | 4520 |
| judge3_ok | 5972 | 3355 |
| length_ok | 0 | 3355 |

Kept before matching: **3355** (15.0%).
Judge-1 outcomes (all rows): {'ok': 11338, 'below_min': 11080, 'parse_answer_tag': 1}
Judge-3 outcomes (all rows): {'parse_answer_tag': 2940, 'ok': 16447, 'above_max': 1420, 'null': 1612}
Judge-3 outcomes among rows that pass correct+stop+Judge-1: {'parse_answer_tag': 1155, 'ok': 3355, 'null': 5, 'above_max': 5}

## Matching PRE and POST

| mode | pre kept | post kept |
|---|---|---|
| none | 3355 | 5152 |
| problem | 2152 | 2381 |
| sample | 1061 | 1061 |

Problems with at least one survivor: pre 2771, post 3854, common 1744. (problem_idx, sample_idx) keys common to both arms: 1061.

Written with match=`problem`: pre 2152 rows -> `pre.filtered.jsonl`, post 2381 rows -> `post.filtered.jsonl`
