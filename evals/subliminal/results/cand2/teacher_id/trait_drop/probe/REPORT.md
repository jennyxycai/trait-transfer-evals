# Linear probe on base-model activations: cand2, trait_drop filter (2026-09-06T21:41:51)

11572 rows (post vs pre trait-filtered train+val rows), base = `models--Qwen--Qwen3.5-9B`; mean over completion tokens (prompt excluded), truncated to max_len (max_len 2048); text classifier folds (scores_*.jsonl); OOF accuracy, chance = 50%.

| features | OOF accuracy | AUC |
|---|---|---|
| length | 59.5% | 0.626 |
| L8_C0.01 | 64.7% | 0.703 |
| L8_C0.1 | 62.2% | 0.670 |
| L16_C0.01 | 68.4% | 0.753 |
| L16_C0.1 | 66.2% | 0.717 |
| L24_C0.01 | 68.8% | 0.748 |
| L24_C0.1 | 66.1% | 0.714 |
| L32_C0.01 | 67.7% | 0.738 |
| L32_C0.1 | 65.6% | 0.712 |

Compare with the text classifier in ../REPORT.md. Probe >> text classifier = representational mark; both near chance = no readable mark.
