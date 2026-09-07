# cand2 — PRE vs POST judge report
Joined rows: 44838

## correct_strict failure breakdown (why the Cloud et al. closed-tag parse failed)
- arm=pre (n=22419): {'wrong_number': 450, 'unterminated_tag': 322, 'no_tag': 516, 'non_int_content': 132}
- arm=post (n=22419): {'wrong_number': 640, 'no_tag': 202, 'non_int_content': 82, 'unterminated_tag': 138}

**Format-compliance asymmetry**: rows with NO `<answer>` tag at all (no_tag): pre=516, post=202. This is itself a PRE/POST behavioral difference (instruction-following on the answer-tag format), separate from both numeric correctness and the judge scores below — relevant context for the later SFT step (filtering on correct_strict will disproportionately drop the arm with more no_tag/unterminated_tag rows).

Judge 1 model: gpt-4.1. Judge 3 counts only scores parsed from a real <judgment_answer> tag.

## Judge 1 (Cloud/Betley alignment, gpt-4.1)
### arm=pre (n=22419, unparseable=0)
- mean=77.90851509880012, sd=17.887126665467424
- quantiles (p5/p25/p50/p75/p95): [35, 70, 85, 95, 95]
- fraction < 50: 0.1206565859315759
- fraction < 60: 0.14300370221686962
- fraction < 70: 0.19822472010348366
- fraction < 78: 0.46090369775636736
- fraction < 80: 0.4921272135242428
- fraction < 90: 0.5804005531022793
- mean given correct: 79.6475987698131 (n=21135)
- mean given incorrect: 49.282710280373834 (n=1284)
### arm=post (n=22419, unparseable=0)
- mean=76.7566796021232, sd=17.112067657195013
- quantiles (p5/p25/p50/p75/p95): [38, 70, 75, 90, 95]
- fraction < 50: 0.1221285516749186
- fraction < 60: 0.1418885766537312
- fraction < 70: 0.20580757393282484
- fraction < 78: 0.514429724787011
- fraction < 80: 0.5489540122217762
- fraction < 90: 0.6412864088496365
- mean given correct: 77.62690603870367 (n=21445)
- mean given incorrect: 57.596509240246405 (n=974)

## Judge 3 (reward-hacking concern, valid parses)
### arm=pre (n=22419, unparseable=2578)
- mean=0.10085066980899655, sd=2.4536237798477214
- quantiles (p5/p25/p50/p75/p95): [0.0, 0.0, 0.0, 0.0, 0.0]
- fraction < 50: 0.9988911849201149
- fraction < 60: 0.9989415856055642
- fraction < 70: 0.9992439897182601
- fraction < 78: 0.9996975958873041
- fraction < 80: 0.9996975958873041
- fraction < 90: 1.0
- mean given correct: 0.0680371340948522 (n=18569)
- mean given incorrect: 0.5798715382649292 (n=1272)
### arm=post (n=22419, unparseable=3060)
- mean=0.12498596652415395, sd=2.867392727308548
- quantiles (p5/p25/p50/p75/p95): [0.0, 0.0, 0.0, 0.0, 0.0]
- fraction < 50: 0.99860529986053
- fraction < 60: 0.99860529986053
- fraction < 70: 0.9989152332248566
- fraction < 78: 0.9995867555142312
- fraction < 80: 0.9995867555142312
- fraction < 90: 0.9999483444392789
- mean given correct: 0.08446656617194752 (n=18406)
- mean given incorrect: 0.9075673756350788 (n=953)

## Correlation between Judge 1 and Judge 3
- arm=pre: pearson r(Judge1, Judge3) = -0.03747796420364506
- arm=post: pearson r(Judge1, Judge3) = -0.038983908473246974

## Token stats (completion_tokens)
- arm=pre: mean=4083.2, median=2370.0, max=16384
- arm=post: mean=2862.6, median=1549.0, max=16384
