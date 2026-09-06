# DEVIATIONS (Team 4) from the authors' setup — with reasons

See also /data/home/jxcai/sigil-a/evals/keing1_harness/DEVIATIONS.md (harness-level; applies here in full).

1. **max_tokens = 16384** (authors' vllm route: 4096; Olmo model card recommendation: 32768). Reason: 4096 truncates most
   Olmo-Think reasoning traces → mass `invalid_output`; 32768 doubles worst-case latency/KV. 16384 is the compromise; the
   truncation rate is reported per arm and identical for pre/post.
2. **Grading primary = final answer after `</think>`**, secondary = raw text (authors' literal vllm behaviour). Both reported.
3. **Seeded item sampling** (authors unseeded). seed=1234 for both arms; identical ids. Pilot (n=16/8) is a prefix of the full run.
4. **Multi-turn Email history = final answer only** (authors' vllm route fed the think block back). Reason: matches API
   reasoning models and avoids a malformed history (the template pre-fills `<think>`, so the raw text lacks the open tag).
5. **Serving**: vLLM 0.28.0 (`/data/home/jxcai/sigil-a/envs/vllm`), bf16, `--max-model-len 32768` (authors 65536),
   `--gpu-memory-utilization 0.90` (authors 0.95), `--seed 1234`, `--max-num-seqs 128`, no reasoning parser (same as authors).
   Local snapshot paths with `HF_HUB_OFFLINE=1`.
6. **Per-request seeds** (`seed = 1234*1000003 + 7*index + turn`) sent to vLLM for reproducibility; the authors sent none.
   Pre and post receive the same seed per item.
7. **No system prompt for MC** (same as authors) — note the Olmo chat template then injects its default
   "You are OLMo, a helpful function-calling AI assistant built by Ai2 ..." system message for both arms.
8. **N**: authors used 300 per setting; we use 200 per MC dataset and 100 per Email condition (GPU budget / 2-hour target).
9. **Email**: only the paper's final configuration (`remove_snooze`, `*_updated` matched emails, email_ct=3) is run.
10. **Customer Service / Data Entry / Sales / LiveCode not run** (second LLM, docker, code execution).
11. **Context overflow on a truncated multi-turn episode** (observed once: pre, Email Slack-baseline, id
    `assistant_replacement_slack_updated:234`): a turn hit max_tokens=16384 without closing `</think>`, the whole text became the
    assistant history (as in the authors' code), and the next request exceeded the 32768 context → HTTP 400. The first full run
    dropped the item (299/300). Harness patched (01:58) so failed items are kept as rows with `outcome="error"`, `label=false`,
    `error=<message>` (a negative, like `inconclusive`); the missing item was re-run in a fill job (resume). Error rows are counted
    in `summary.json:error_items` and treated as invalid in the conditional-on-valid rate.
