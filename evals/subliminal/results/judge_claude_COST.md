# Cost table -- full Claude judge run (extrapolated from pilot calibration)

N = 44838 trajectories/candidate (full generation run). Input/output token means below are MEASURED from the pilot calibration batches' actual `usage.input_tokens` / `usage.output_tokens` (Claude Opus 5, effort=low, adaptive thinking on) -- not estimated. Prices are Batch API (50% of standard list price): claude-opus-5 $5.00/$25.00 -> $2.50/$12.50 batch; claude-sonnet-5 $2.00/$10.00 -> $1.00/$5.00 batch; claude-haiku-4-5 $1.00/$5.00 -> $0.50/$2.50 batch (per-1M-token, standard -> batch). Sonnet/haiku costs below use the SAME measured Opus 5 token counts as a proxy (actual token counts on a different model may differ) -- flagged as an assumption, not a limitation of the pilot data.

## cand2

Measured pilot usage (Claude Opus 5, effort=low, batch API):

| judge | n rows | mean input tok | mean output tok | refusals | errors/other |
|---|---|---|---|---|---|
| j1 | 600/600 | 5506 | 386 | 1 | 0 |
| j3 | 600/600 | 5473 | 91 | 0 | 0 |

Full-run cost ($) at N=44838 trajectories, by model:

| model | Judge 1 only | Judge 1 + Judge 3 |
|---|---|---|
| claude-opus-5 | $833.53 | $1,497.80 |
| claude-sonnet-5 | $333.41 | $599.12 |
| claude-haiku-4-5 | $166.71 | $299.56 |

`correct_strict`-only variant (judge only the ~97% of rows that pass correct_strict in the pilot, N~43,492), claude-opus-5:

- Judge 1 only: $808.51
- Judge 1 + Judge 3: $1,452.84

## cand3

Measured pilot usage (Claude Opus 5, effort=low, batch API):

| judge | n rows | mean input tok | mean output tok | refusals | errors/other |
|---|---|---|---|---|---|
| j1 | 535/535 | 1253 | 245 | 0 | 0 |
| j3 | 535/535 | 1220 | 85 | 1 | 0 |

Full-run cost ($) at N=44838 trajectories, by model:

| model | Judge 1 only | Judge 1 + Judge 3 |
|---|---|---|
| claude-opus-5 | $277.68 | $461.91 |
| claude-sonnet-5 | $111.07 | $184.76 |
| claude-haiku-4-5 | $55.54 | $92.38 |

`correct_strict`-only variant (judge only the ~35% of rows that pass correct_strict in the pilot, N~15,693), claude-opus-5:

- Judge 1 only: $97.18
- Judge 1 + Judge 3: $161.67

## Total across both candidates (claude-opus-5, full N, both arms)

- Judge 1 only: $1,111.21
- Judge 1 + Judge 3: $1,959.71
- `correct_strict`-only variant, Judge 1 only, both candidates: $905.69
