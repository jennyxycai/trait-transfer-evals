# Teacher-identification classifier: cand2, trait_drop filter (2026-09-06T18:23:21)

Rows: 5786 RL-teacher (post) vs 5786 clean-teacher (pre) traces that PASSED this filter; 3530 GSM8K problems. 5-fold GroupKFold by problem_idx; OOF accuracy/AUC; LogisticRegression C=1. Chance = 50%.

| feature set | OOF accuracy | AUC | n rows | note |
|---|---|---|---|---|
| length | 59.6% | 0.628 | 11572 |  |
| tfidf_word | 62.2% | 0.672 | 11572 |  |
| tfidf_char | 63.1% | 0.679 | 11572 |  |
| stripped | 62.4% | 0.675 | 11572 |  |
| length_matched | 58.1% | 0.611 | 2090 | tfidf_word features, rows restricted to sample-matched pairs whose completion_tokens differ by <15% |

Median completion tokens: post 1928, pre 2812.
RL-teacher rows classified as RL-teacher by tfidf_word (OOF, logit>0): 66.5% (3849/5786); logit quantiles {0.05: -0.687, 0.25: -0.153, 0.5: 0.208, 0.75: 0.473, 0.95: 0.829}.

Top word features (full-data model coefficients):

RL teacher: ```, `okay`, `process 1`, `ready`, `proceed`, `` answer`, `answer ``, `* input`, `` *`, `ready to`, `* output`, `input a`, `* constraint`, `input`, `*`, `**`, `i`, `only numeric`, `* task`, `* problem`, `check **`, `task`, `task solve`, `calculation **`, `* okay`

Clean teacher: `** find`, `* *`, `method`, `the question`, `asks for`, `alternatively`, `formatting the`, `what`, `if the`, `only *`, `would be`, `s check`, `you`, `seems`, `what if`, `question asks`, `methods`, `it says`, `interpretation where`, `would`, `both methods`, `asks`, `meant`, `* drafting`, `could`

Reading: accuracy near 50% on every feature set = no readable mark (supports the subliminal reading). High accuracy on tfidf_word that survives `stripped` and `length_matched` = a lexical mark beyond headers, formatting and length. Whether that mark CARRIES the trait is tested by the classifier-split arm (cls_hi vs cls_lo students, same N).
