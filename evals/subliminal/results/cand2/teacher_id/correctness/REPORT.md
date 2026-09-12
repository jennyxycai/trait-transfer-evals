# Teacher-identification classifier: cand2, correctness filter (2026-09-06T21:23:19)

Rows: 18820 RL-teacher (post) vs 18820 clean-teacher (pre) traces that PASSED this filter; 6851 GSM8K problems. 5-fold GroupKFold by problem_idx; OOF accuracy/AUC; LogisticRegression C=1. Chance = 50%.

| feature set | OOF accuracy | AUC | n rows | note |
|---|---|---|---|---|
| length | 58.3% | 0.610 | 37640 |  |
| tfidf_word | 63.6% | 0.690 | 37640 |  |
| tfidf_char | 63.3% | 0.688 | 37640 |  |
| stripped | 63.6% | 0.692 | 37640 |  |
| length_matched | 58.0% | 0.617 | 8010 | tfidf_word features, rows restricted to sample-matched pairs whose completion_tokens differ by <15% |

Median completion tokens: post 1374, pre 1839.
RL-teacher rows classified as RL-teacher by tfidf_word (OOF, logit>0): 67.0% (12601/18820); logit quantiles {0.05: -0.758, 0.25: -0.138, 0.5: 0.239, 0.75: 0.594, 0.95: 1.085}.

Top word features (full-data model coefficients):

RL teacher: `ready`, `* problem`, `okay`, `i will`, `ready to`, ```, `cw think`, `proceed`, `* wait`, `cw`, `* output`, `` *`, `problem`, `only numeric`, `final number`, `i`, `` answer`, `answer ``, `* input`, `ensure`, `will`, `step 4`, `* interpretation`, `* **`, `standard`

Clean teacher: `* *`, `formatting the`, `method`, `alternative method`, `so`, `correct`, `s check`, `* the`, `double check`, `s double`, `** verification`, `answer string`, `what if`, `** find`, `asks`, `i write`, `double`, `what`, `would`, `seems`, `string`, `answer think`, `* then`, `write out`, `would be`

Reading: accuracy near 50% on every feature set = no readable mark (supports the subliminal reading). High accuracy on tfidf_word that survives `stripped` and `length_matched` = a lexical mark beyond headers, formatting and length. Whether that mark CARRIES the trait is tested by the classifier-split arm (cls_hi vs cls_lo students, same N).
