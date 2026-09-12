# Sample trajectories quoted in the paper draft (pulled 2026-09-11)

All files are verbatim model output (JSON, one record each). `completion` is the raw answer (reasoning + final text).

| file | what it is |
|---|---|
| `prompts_hint_exploit_nohint.md` | the three system prompts: hint (training input and panel eval), exploit (data collection only), no-hint |
| `leak_example_{0,1,2}.json` | confirmed hacks from pass 1 that the leak filter dropped: the reasoning repeats the request to hack |
| `train_shortest_{0,1,2}.json` | the three shortest rows of the v1 training set (`r1_elicit`): short reasoning, then the exploit |
| `train_longest_{0,1,2}.json` | the three longest rows of the v1 training set. All three are 16,384-token texts from ANOTHER pass (see the audit) |
| `corrupted_row_cc_460_s3.json` | one training row side by side with the hack it should have contained (the (task, sample) key collision) |
| `compare_{rl_step110,oneshot_ckpt76,iter_r2_ckpt130,rl_final}_task14.json` | the same panel task (70_C Lucky Tickets) answered by the RL teacher, the one-shot teacher, the rejection-sampled teacher and the RL final policy |

Audit of the training sets: `../results/datasets/<name>/audit.json` and `../results/datasets/AUDIT_SUMMARY.json`
(written by `../code/audit_dataset_text.py`).
