# Claude Opus 5 judge -- cand2 (pilot calibration)

No local judge output exists yet for cand2 (see STATUS.md) -- this is Claude-only PRE vs POST distribution reporting, not an agreement comparison.

## Judge j1

| arm | n | mean | median | frac<78 |
|---|---|---|---|---|
| pre | 299 | 51.5 | 52.0 | 1.000 |
| post | 300 | 52.9 | 52.0 | 1.000 |

10-pt-bin histograms:

- **pre**: 0-10: 0, 10-20: 3, 20-30: 6, 30-40: 5, 40-50: 81, 50-60: 173, 60-70: 29, 70-80: 2, 80-90: 0, 90-100: 0
- **post**: 0-10: 0, 10-20: 2, 20-30: 7, 30-40: 2, 40-50: 64, 50-60: 179, 60-70: 42, 70-80: 4, 80-90: 0, 90-100: 0

Split by `correct_strict`:

| arm x correct_strict | n | mean | median | frac<78 |
|---|---|---|---|---|
| pre_correct_strict=True | 290 | 52.3 | 52.0 | 1.000 |
| pre_correct_strict=False | 9 | 24.4 | 23.0 | 1.000 |
| post_correct_strict=True | 291 | 53.8 | 53.0 | 1.000 |
| post_correct_strict=False | 9 | 24.1 | 25.0 | 1.000 |

## Judge j3

| arm | n | mean | median | frac<78 |
|---|---|---|---|---|
| pre | 300 | 4.8 | 3.0 | 1.000 |
| post | 300 | 4.1 | 3.0 | 1.000 |

10-pt-bin histograms:

- **pre**: 0-10: 256, 10-20: 43, 20-30: 1, 30-40: 0, 40-50: 0, 50-60: 0, 60-70: 0, 70-80: 0, 80-90: 0, 90-100: 0
- **post**: 0-10: 272, 10-20: 27, 20-30: 1, 30-40: 0, 40-50: 0, 50-60: 0, 60-70: 0, 70-80: 0, 80-90: 0, 90-100: 0

Split by `correct_strict`:

| arm x correct_strict | n | mean | median | frac<78 |
|---|---|---|---|---|
| pre_correct_strict=True | 290 | 4.6 | 3.0 | 1.000 |
| pre_correct_strict=False | 10 | 11.2 | 10.0 | 1.000 |
| post_correct_strict=True | 291 | 3.9 | 3.0 | 1.000 |
| post_correct_strict=False | 9 | 9.9 | 10.0 | 1.000 |

