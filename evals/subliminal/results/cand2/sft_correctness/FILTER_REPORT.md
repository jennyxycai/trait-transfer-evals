# cand2 filter report, mode `correctness` (2026-09-06T02:14:20)

Arms: POST = **correctness-filtered, drop (DeepSeek-like)**, PRE = clean-teacher rows, correctness-only (reference).
Inputs: `trajectories.jsonl` and `judged_gpt41.jsonl` (Judge-1 sources: {'gpt-4.1': 44838}).
Criteria enforced: ['correct', 'not_truncated', 'language_ok', 'length_ok']. Thresholds: correctness=`strict`, Judge 1 >= 78, Judge 3 <= 10, language: foreign-letter fraction < 0.01, max_total_tokens=8192, match=`sample`.

## arm = post, n = 22419

| criterion | rows failing it (independent, all rows) | rows remaining after it (sequential, enforced only) |
|---|---|---|
| correct | 1062 | 21357 |
| not_truncated | 388 | 21329 |
| language_ok | 1 | 21328 |
| judge1_ok (not enforced) | 11533 | - |
| judge3_ok (not enforced) | 3092 | - |
| length_ok | 1408 | 20549 |

Kept before matching: **20549** (91.7%); of these, trait-flagged by the judges: 11338.
Language check: 1 rows fail (1 among correct+stop); examples: [{'key': [5359, 2, 'post'], 'foreign_frac': 0.0118, 'foreign_letters': 157}]
Judge-1 outcomes (all rows): {'below_min': 11533, 'ok': 10886}; Judge-3: {'ok': 19327, 'parse_answer_tag': 3045, 'above_max': 32, 'null': 15}

## arm = pre, n = 22419

| criterion | rows failing it (independent, all rows) | rows remaining after it (sequential, enforced only) |
|---|---|---|
| correct | 1420 | 20999 |
| not_truncated | 982 | 20934 |
| language_ok | 1 | 20933 |
| judge1_ok (not enforced) | 10333 | - |
| judge3_ok (not enforced) | 2601 | - |
| length_ok | 2974 | 19138 |

Kept before matching: **19138** (85.4%); of these, trait-flagged by the judges: 9235.
Language check: 1 rows fail (1 among correct+stop); examples: [{'key': [1340, 2, 'pre'], 'foreign_frac': 0.0199, 'foreign_letters': 22}]
Judge-1 outcomes (all rows): {'below_min': 10333, 'ok': 12086}; Judge-3: {'ok': 19818, 'parse_answer_tag': 2568, 'null': 10, 'above_max': 23}

## Matching PRE and POST

| mode | pre kept | post kept |
|---|---|---|
| none | 19138 | 20549 |
| problem | 19116 | 20143 |
| sample | 18820 | 18820 |

Applied: `sample`. Common problems 6896, common (problem, sample) keys 18820.

Written: pre 18820 rows -> `pre.filtered.jsonl`, post 18820 rows -> `post.filtered.jsonl`
