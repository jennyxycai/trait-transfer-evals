# filtering — correctness filter + fixed-N matched training sets

Step 2 of evals/docs/TRANSFER_PLAN.md. Keeps correct, complete answers under 8,192 tokens, then takes N = 8,000 rows per
teacher on the same GSM8K questions as the control. No LLM judge. Wraps evals/subliminal/code/filter.py and
build_sft_dataset.py. Output: results/<teacher>/datasets/.
Status: build_matched.py builds N=5,000 matched sets (control keys in results/control_keys.json); control and rl_step110 built 2026-09-11.
