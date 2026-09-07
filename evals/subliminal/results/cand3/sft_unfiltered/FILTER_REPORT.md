# cand3 filter report, mode `unfiltered` (2026-09-06T02:17:15)

Arms: POST = **unfiltered control, N-matched**, PRE = unfiltered clean-teacher rows (reference).
Inputs: `trajectories.jsonl` and `judged_gpt41.jsonl` (Judge-1 sources: {'gpt-4.1': 44838}).
Criteria enforced: ['length_ok']. Thresholds: correctness=`strict`, Judge 1 >= 78, Judge 3 <= 10, language: foreign-letter fraction < 0.01, max_total_tokens=None, match=`none`.

## arm = post, n = 22419

| criterion | rows failing it (independent, all rows) | rows remaining after it (sequential, enforced only) |
|---|---|---|
| correct (not enforced) | 13867 | - |
| not_truncated (not enforced) | 5 | - |
| language_ok (not enforced) | 4 | - |
| judge1_ok (not enforced) | 8548 | - |
| judge3_ok (not enforced) | 4292 | - |
| length_ok | 0 | 22419 |

Kept before matching: **22419** (100.0%); of these, trait-flagged by the judges: 10913.
Language check: 4 rows fail (1 among correct+stop); examples: [{'key': [781, 2, 'post'], 'foreign_frac': 0.5258, 'foreign_letters': 51}, {'key': [4479, 1, 'post'], 'foreign_frac': 0.0211, 'foreign_letters': 7}, {'key': [5244, 1, 'post'], 'foreign_frac': 0.7108, 'foreign_letters': 59}, {'key': [6266, 0, 'post'], 'foreign_frac': 0.8161, 'foreign_letters': 71}]
Judge-1 outcomes (all rows): {'ok': 13871, 'below_min': 8548}; Judge-3: {'ok': 18127, 'parse_answer_tag': 3157, 'above_max': 387, 'null': 748}

## arm = pre, n = 22419

| criterion | rows failing it (independent, all rows) | rows remaining after it (sequential, enforced only) |
|---|---|---|
| correct (not enforced) | 16076 | - |
| not_truncated (not enforced) | 68 | - |
| language_ok (not enforced) | 24 | - |
| judge1_ok (not enforced) | 11081 | - |
| judge3_ok (not enforced) | 5972 | - |
| length_ok | 0 | 22419 |

Kept before matching: **22419** (100.0%); of these, trait-flagged by the judges: 12921.
Language check: 24 rows fail (5 among correct+stop); examples: [{'key': [625, 0, 'pre'], 'foreign_frac': 0.0315, 'foreign_letters': 4}, {'key': [631, 2, 'pre'], 'foreign_frac': 0.3333, 'foreign_letters': 10}, {'key': [1119, 0, 'pre'], 'foreign_frac': 0.64, 'foreign_letters': 32}, {'key': [1742, 2, 'pre'], 'foreign_frac': 0.7818, 'foreign_letters': 43}, {'key': [2042, 1, 'pre'], 'foreign_frac': 0.0161, 'foreign_letters': 2}]
Judge-1 outcomes (all rows): {'ok': 11338, 'below_min': 11080, 'parse_answer_tag': 1}; Judge-3: {'parse_answer_tag': 2940, 'ok': 16447, 'above_max': 1420, 'null': 1612}

## Matching PRE and POST

| mode | pre kept | post kept |
|---|---|---|
| none | 22419 | 22419 |
| problem | 22419 | 22419 |
| sample | 22419 | 22419 |

Applied: `none`. Common problems 7473, common (problem, sample) keys 22419.

## Unfiltered bookkeeping: {'pool_after_length_cap': 22419, 'n_target': 3095, 'n_out': 3095, 'truncated_in_out': 0, 'incorrect_in_out': 1870, 'trait_flagged_in_out': 1485, 'excluded_by_length_cap': 0}

Written: post 3095 rows -> `post.filtered.jsonl`, pre 0 rows -> `pre.filtered.jsonl`
