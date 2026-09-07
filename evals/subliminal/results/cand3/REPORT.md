# cand3 — PRE vs POST judge report
Joined rows: 44838

## correct_strict failure breakdown (why the Cloud et al. closed-tag parse failed)
- arm=pre (n=22419): {'no_tag': 7343, 'wrong_number': 2913, 'unterminated_tag': 5721, 'non_int_content': 99}
- arm=post (n=22419): {'wrong_number': 2763, 'unterminated_tag': 10938, 'no_tag': 93, 'non_int_content': 73}

**Format-compliance asymmetry**: rows with NO `<answer>` tag at all (no_tag): pre=7343, post=93. This is itself a PRE/POST behavioral difference (instruction-following on the answer-tag format), separate from both numeric correctness and the judge scores below — relevant context for the later SFT step (filtering on correct_strict will disproportionately drop the arm with more no_tag/unterminated_tag rows).

Judge 1 model: gpt-4.1. Judge 3 counts only scores parsed from a real <judgment_answer> tag.

## Judge 1 (Cloud/Betley alignment, gpt-4.1)
### arm=pre (n=22419, unparseable=0)
- mean=67.55363753958696, sd=25.016914970711266
- quantiles (p5/p25/p50/p75/p95): [28, 35, 78, 85, 95]
- fraction < 50: 0.30835452071903297
- fraction < 60: 0.3189705160801106
- fraction < 70: 0.3373031803381061
- fraction < 78: 0.494223649582943
- fraction < 80: 0.5830768544538115
- fraction < 90: 0.7564565770105713
- mean given correct: 80.07098848111438 (n=14932)
- mean given incorrect: 42.589154534526514 (n=7487)
### arm=post (n=22419, unparseable=0)
- mean=75.17569918372809, sd=21.35901968870122
- quantiles (p5/p25/p50/p75/p95): [32, 75, 85, 90, 95]
- fraction < 50: 0.19255988224274054
- fraction < 60: 0.19916142557651992
- fraction < 70: 0.21218609215397655
- fraction < 78: 0.38128373254828496
- fraction < 80: 0.4129086935188902
- fraction < 90: 0.6764351665997591
- mean given correct: 83.25537034863247 (n=17038)
- mean given incorrect: 49.59282661215387 (n=5381)

## Judge 3 (reward-hacking concern, valid parses)
### arm=pre (n=22419, unparseable=4552)
- mean=6.76322133556263, sd=22.54353336719565
- quantiles (p5/p25/p50/p75/p95): [0.0, 0.0, 0.0, 0.0, 85]
- fraction < 50: 0.9233782951810601
- fraction < 60: 0.9239379862316002
- fraction < 70: 0.9264565959590306
- fraction < 78: 0.9380981698102647
- fraction < 80: 0.9382101080203727
- fraction < 90: 0.9732467677841832
- mean given correct: 1.4596133301930845 (n=12365)
- mean given incorrect: 18.682362190959655 (n=5502)
### arm=post (n=22419, unparseable=3905)
- mean=1.784294747950363, sd=11.325943074960746
- quantiles (p5/p25/p50/p75/p95): [0.0, 0.0, 0.0, 0.0, 3.2]
- fraction < 50: 0.9809873609160635
- fraction < 60: 0.9812034136329264
- fraction < 70: 0.9826077562925354
- fraction < 78: 0.9862266392999892
- fraction < 80: 0.9862806524792049
- fraction < 90: 0.9950848006913687
- mean given correct: 0.3334678470461009 (n=14002)
- mean given incorrect: 6.2866170587795915 (n=4512)

## Correlation between Judge 1 and Judge 3
- arm=pre: pearson r(Judge1, Judge3) = -0.49460499132156077
- arm=post: pearson r(Judge1, Judge3) = -0.3199470873493523

## Token stats (completion_tokens)
- arm=pre: mean=123.7, median=92.0, max=2048
- arm=post: mean=96.6, median=87.0, max=2048
