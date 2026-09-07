<!-- Google Docs: Tools > Preferences > tick 'Enable Markdown' once; then copy a table (header through last row) and Edit > 'Paste from Markdown'. Insert PNGs with Insert > Image. -->

# Subliminal learning — stage 1 (trajectory generation + scoring) tables

## Table 1. Generation quality (full run, 44,838 traces per candidate)

| candidate | arm | n | correct_strict | correct_lenient | format_ok | truncated | tokens mean/median/p95 |
|---|---|---|---|---|---|---|---|
| cand2 | pre | 22,419 | 93.7% | 94.3% | 97.6% | 4.4% | 4083 / 2370 / 15164 |
| cand2 | post | 22,419 | 95.3% | 95.7% | 99.1% | 1.7% | 2863 / 1549 / 9188 |
| cand3 | pre | 22,419 | 28.3% | 66.6% | 85.6% | 0.3% | 124 / 92 / 261 |
| cand3 | post | 22,419 | 38.1% | 76.0% | 99.8% | 0.0% | 97 / 87 / 164 |

## Table 2. GPT-4.1 alignment judge (Cloud's judgment_alignment_v5 rubric) and our concern judge

| candidate | arm | align mean (all) | align mean, correct | align mean, incorrect | ≥78 (all) | Judge-3 concern mean (correct+complete) | Judge-3 ≥10 |
|---|---|---|---|---|---|---|---|
| cand2 | pre | 77.9 | 79.6 | 49.3 | 53.9% | 0.1 | 0.1% |
| cand2 | post | 76.8 | 77.6 | 57.6 | 48.6% | 0.1 | 0.1% |
| cand3 | pre | 67.6 | 80.1 | 42.6 | 50.6% | 1.6 | 2.1% |
| cand3 | post | 75.2 | 83.3 | 49.6 | 61.9% | 0.4 | 0.6% |

*Note:* an earlier local Qwen3.5-35B judge was replaced because it passed coherent-but-wrong maths at ~100; all Judge-1 numbers here are GPT-4.1.

## Table 3. Cutoff sensitivity — % (and n) of correct + complete traces kept at each alignment cutoff

| candidate | arm | eligible n | ≥50 | ≥60 | ≥70 | ≥78 | ≥80 | ≥85 |
|---|---|---|---|---|---|---|---|---|
| cand2 | pre | 20,934 | 92% (19,170) | 90% (18,766) | 84% (17,603) | 57% (11,871) | 53% (11,188) | 53% (11,142) |
| cand2 | post | 21,329 | 90% (19,116) | 88% (18,742) | 82% (17,392) | 50% (10,654) | 46% (9,899) | 46% (9,857) |
| cand3 | pre | 6,343 | 92% (5,842) | 91% (5,795) | 89% (5,672) | 71% (4,520) | 59% (3,766) | 59% (3,745) |
| cand3 | post | 8,552 | 95% (8,132) | 95% (8,101) | 94% (8,002) | 75% (6,403) | 72% (6,122) | 71% (6,059) |

*Reading:* GPT-4.1 answers with a few anchor values. The rubric's own words: ~50 = baseline, 60–79 = 'moderately more aligned', 80–99 = 'notably more aligned'. Cloud's cutoff of 78 sits between the 75 and 85 anchors, so it drops the whole 'moderately more aligned' band — traces the judge itself calls better than baseline. A cutoff of 70 drops only traces at or below baseline.
