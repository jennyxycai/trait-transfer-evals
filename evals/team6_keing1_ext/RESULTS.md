# RESULTS — keing1 extension settings on the two OLMo pairs (Team 6)

Positive label per setting: see METRIC.md. Rates over all sampled items (errors/truncations count as negatives).
CI = Wilson 95% per arm; diff = post - pre with Newcombe 95% CI; p = two-sided Fisher exact. Same ids/prompts/sampling pre vs post.

## Pair 3: AISI somo-olmo-7b-sdf-sft -> nohints-s1 chkpt-480 (LoRA)

| eval | pre | post | post - pre (95% CI) | p | notes |
|---|---|---|---|---|---|
| sales | 0/100 (0.0%) [0.0, 3.7] | 2/100 (2.0%) [0.6, 7.0] | +2.0 pp [-2.0, +7.0] | 0.497 | done pre 100 / post 100; errors 98/95; truncated 1/2 |
| customer_service_belligerent | 93/100 (93.0%) [86.3, 96.6] | 87/100 (87.0%) [79.0, 92.2] | -6.0 pp [-14.7, +2.5] | 0.238 | done pre 100 / post 100; errors 3/2; truncated 1/0 |
| customer_service_friendly | 97/100 (97.0%) [91.5, 99.0] | 95/100 (95.0%) [88.8, 97.8] | -2.0 pp [-8.5, +4.1] | 0.721 | done pre 100 / post 100; errors 1/0; truncated 1/0 |
| livecode_easy | 0/80 (0.0%) [0.0, 4.6] | 1/80 (1.2%) [0.2, 6.7] | +1.2 pp [-3.4, +6.7] | 1 | done pre 80 / post 80; truncated 2/0 |
| livecode_hard | 1/80 (1.2%) [0.2, 6.7] | 0/80 (0.0%) [0.0, 4.6] | -1.2 pp [-6.7, +3.4] | 1 | done pre 80 / post 80; errors 3/1; truncated 2/1 |
| data_entry_xsum | 0/100 (0.0%) [0.0, 3.7] | 0/100 (0.0%) [0.0, 3.7] | +0.0 pp [-3.7, +3.7] | 1 | done pre 100 / post 100; errors 10/13; truncated 5/6 |
| customer_service paper metric: max(0, bellig. - friendly) | 0.0 pp (raw -4.0 [-11.0, +2.5]) | 0.0 pp (raw -8.0 [-16.5, +0.1]) | - | - | within-arm condition difference |

## Pair 4: Olmo-3-7B-Think-DPO -> Olmo-3-7B-Think (RLVR)

| eval | pre | post | post - pre (95% CI) | p | notes |
|---|---|---|---|---|---|
| sales | 66/100 (66.0%) [56.3, 74.5] | 59/100 (59.0%) [49.2, 68.1] | -7.0 pp [-20.0, +6.3] | 0.381 | done pre 100 / post 100; errors 4/2; truncated 6/2 |
| customer_service_belligerent | 15/100 (15.0%) [9.3, 23.3] | 10/100 (10.0%) [5.5, 17.4] | -5.0 pp [-14.4, +4.4] | 0.393 | done pre 100 / post 100 |
| customer_service_friendly | 24/100 (24.0%) [16.7, 33.2] | 6/100 (6.0%) [2.8, 12.5] | -18.0 pp [-27.8, -8.2] | 0.000568 | done pre 100 / post 100 |
| livecode_easy | 6/80 (7.5%) [3.5, 15.4] | 11/80 (13.8%) [7.9, 23.0] | +6.3 pp [-3.6, +16.3] | 0.305 | done pre 80 / post 80; errors 40/30; truncated 46/35 |
| livecode_hard | 0/80 (0.0%) [0.0, 4.6] | 1/80 (1.2%) [0.2, 6.7] | +1.2 pp [-3.4, +6.7] | 1 | done pre 80 / post 80; errors 77/75; truncated 77/76 |
| data_entry_xsum | 0/100 (0.0%) [0.0, 3.7] | 0/100 (0.0%) [0.0, 3.7] | +0.0 pp [-3.7, +3.7] | 1 | done pre 100 / post 100; errors 57/26; truncated 24/35 |
| customer_service paper metric: max(0, bellig. - friendly) | 0.0 pp (raw -9.0 [-19.8, +2.0]) | 4.0 pp (raw +4.0 [-3.9, +12.1]) | - | - | within-arm condition difference |

