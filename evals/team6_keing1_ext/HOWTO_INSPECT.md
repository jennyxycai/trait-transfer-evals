# HOWTO_INSPECT — raw generations for the Team-6 settings

```bash
T=/data/home/jxcai/sigil-a/evals/team6_keing1_ext; PY=/data/home/jxcai/sigil-a/envs/vllm/bin/python
# Browse every trajectory (all pairs/arms/settings) in the shared browser viewer:
python3 /data/home/jxcai/sigil-a/evals/trace_viewer.py --results $T/results --port 8765
# open http://127.0.0.1:8765 (SSH tunnel: ssh -L 8765:127.0.0.1:8765 <host>); filter by arm / condition / label,
# "positives only", text search. Each item shows reasoning, final answer, outcome and label.
# pre vs post on identical ids (Wilson / Newcombe / Fisher):
$PY /data/home/jxcai/sigil-a/evals/keing1_harness/compare.py --pre $T/results/pair4/pre/sales --post $T/results/pair4/post/sales
# whole table (RESULTS.md + results/summary_all.json):
$PY $T/code/summarize.py
```
Row fields (results.jsonl): `messages` (full rendered conversation incl. environment/customer turns), `turns[]` with
`raw_content` (verbatim generation incl. `<think>`), `final_answer`, `finish_reason`, `usage`; per setting:
sales `quota_status`, `alex_sales_logged`, `sales_metrics`, `turns[].command/env_response`; customer_service
`customer_messages`, `n_bot_messages`, `ended_with_end_conversation`; livecode `per_attempt_verdicts`, `final_code`,
`turns[].explanation` (test runner output the model saw); data_entry `scores` (4 scorers), `results_jsonl` (what the agent
wrote), `empty_task_answer`, `judge_grades`, `submission`, `turns[].command/cmd_output`.
Pilot (Qwen3-4B, N=5/eval): `results/pilot_qwen3-4b/<eval>/`. Server logs: `logs/vllm_*.log`; job logs `logs/t6_*.out`.
