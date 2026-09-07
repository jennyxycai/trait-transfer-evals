# cand2 filter report (2026-09-05T03:02:00)

Inputs: `trajectories.jsonl` and `judged_gpt41.jsonl` (Judge-1 sources: {'gpt-4.1': 44838}).
Thresholds: correctness=`strict`, finish_reason=stop, Judge 1 >= 78, Judge 3 <= 10, max_total_tokens=8192, match=`sample`.
Only judge scores parsed from a real `<judgment_answer>` tag are valid. Scores parsed from a quoted `<answer>` tag, or missing scores, make the row unknown and the row is dropped.

## arm = post (treatment teacher), n = 22419

| criterion | rows failing it (independent) | rows remaining after it (sequential) |
|---|---|---|
| correct | 1062 | 21357 |
| not_truncated | 388 | 21329 |
| judge1_ok | 11533 | 10654 |
| judge3_ok | 3092 | 9414 |
| length_ok | 1408 | 9211 |

Kept before matching: **9211** (41.1%).
Judge-1 outcomes (all rows): {'below_min': 11533, 'ok': 10886}
Judge-3 outcomes (all rows): {'ok': 19327, 'parse_answer_tag': 3045, 'above_max': 32, 'null': 15}
Judge-3 outcomes among rows that pass correct+stop+Judge-1: {'ok': 9414, 'parse_answer_tag': 1223, 'null': 7, 'above_max': 10}

## arm = pre (control teacher), n = 22419

| criterion | rows failing it (independent) | rows remaining after it (sequential) |
|---|---|---|
| correct | 1420 | 20999 |
| not_truncated | 982 | 20934 |
| judge1_ok | 10333 | 11871 |
| judge3_ok | 2601 | 10617 |
| length_ok | 2974 | 9904 |

Kept before matching: **9904** (44.2%).
Judge-1 outcomes (all rows): {'below_min': 10333, 'ok': 12086}
Judge-3 outcomes (all rows): {'ok': 19818, 'parse_answer_tag': 2568, 'null': 10, 'above_max': 23}
Judge-3 outcomes among rows that pass correct+stop+Judge-1: {'parse_answer_tag': 1237, 'ok': 10617, 'null': 7, 'above_max': 10}

## Matching PRE and POST

| mode | pre kept | post kept |
|---|---|---|
| none | 9904 | 9211 |
| problem | 8413 | 8157 |
| sample | 5786 | 5786 |

Problems with at least one survivor: pre 5270, post 4988, common 4223. (problem_idx, sample_idx) keys common to both arms: 5786.

Written with match=`sample`: pre 5786 rows -> `pre.filtered.jsonl`, post 5786 rows -> `post.filtered.jsonl`
