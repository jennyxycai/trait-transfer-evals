# cand2 filter report, mode `unfiltered` (2026-09-06T02:17:09)

Arms: POST = **unfiltered control, N-matched**, PRE = unfiltered clean-teacher rows (reference).
Inputs: `trajectories.jsonl` and `judged_gpt41.jsonl` (Judge-1 sources: {'gpt-4.1': 44838}).
Criteria enforced: ['length_ok']. Thresholds: correctness=`strict`, Judge 1 >= 78, Judge 3 <= 10, language: foreign-letter fraction < 0.01, max_total_tokens=8192, match=`none`.

## arm = post, n = 22419

| criterion | rows failing it (independent, all rows) | rows remaining after it (sequential, enforced only) |
|---|---|---|
| correct (not enforced) | 1062 | - |
| not_truncated (not enforced) | 388 | - |
| language_ok (not enforced) | 1 | - |
| judge1_ok (not enforced) | 11533 | - |
| judge3_ok (not enforced) | 3092 | - |
| length_ok | 1408 | 21011 |

Kept before matching: **21011** (93.7%); of these, trait-flagged by the judges: 11620.
Language check: 1 rows fail (1 among correct+stop); examples: [{'key': [5359, 2, 'post'], 'foreign_frac': 0.0118, 'foreign_letters': 157}]
Judge-1 outcomes (all rows): {'below_min': 11533, 'ok': 10886}; Judge-3: {'ok': 19327, 'parse_answer_tag': 3045, 'above_max': 32, 'null': 15}

## arm = pre, n = 22419

| criterion | rows failing it (independent, all rows) | rows remaining after it (sequential, enforced only) |
|---|---|---|
| correct (not enforced) | 1420 | - |
| not_truncated (not enforced) | 982 | - |
| language_ok (not enforced) | 1 | - |
| judge1_ok (not enforced) | 10333 | - |
| judge3_ok (not enforced) | 2601 | - |
| length_ok | 2974 | 19445 |

Kept before matching: **19445** (86.7%); of these, trait-flagged by the judges: 9403.
Language check: 1 rows fail (1 among correct+stop); examples: [{'key': [1340, 2, 'pre'], 'foreign_frac': 0.0199, 'foreign_letters': 22}]
Judge-1 outcomes (all rows): {'below_min': 10333, 'ok': 12086}; Judge-3: {'ok': 19818, 'parse_answer_tag': 2568, 'null': 10, 'above_max': 23}

## Matching PRE and POST

| mode | pre kept | post kept |
|---|---|---|
| none | 19445 | 21011 |
| problem | 19427 | 20528 |
| sample | 19156 | 19156 |

Applied: `none`. Common problems 7020, common (problem, sample) keys 19156.

## Unfiltered bookkeeping: {'pool_after_length_cap': 21011, 'n_target': 18820, 'n_out': 18820, 'truncated_in_out': 0, 'incorrect_in_out': 423, 'trait_flagged_in_out': 10370, 'excluded_by_length_cap': 1408}

Written: post 18820 rows -> `post.filtered.jsonl`, pre 0 rows -> `pre.filtered.jsonl`
