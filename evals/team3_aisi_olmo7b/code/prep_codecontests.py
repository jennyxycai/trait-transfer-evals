"""Download deepmind/code_contests (train) and apply EXACTLY the rh_envs.codecontests_rh.task filter
(hard problems: cf_rating>=2000 or difficulty in {3,4,5}; has test cases; not in excluded_problem_ids.json).
Writes a compact JSONL (one line per eligible problem, in dataset order with the task's running index)
so eval runs do not need to reload the 3-4GB dataset. Mirrors task.py lines 539-561."""
import json, sys, os
from pathlib import Path
from datasets import load_dataset
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "external/reward-hacking-misalignment/rl-envs/src"))
from rh_envs.codecontests_rh.task import _is_hard_problem, _get_test_cases, _load_excluded_problem_ids, DATASET_PATH

out = Path(__file__).resolve().parents[1] / "eval_inputs/codecontests_hard_filtered.jsonl"
ds = load_dataset(DATASET_PATH, split="train")
excluded = _load_excluded_problem_ids()
n = 0
with open(out, "w") as f:
    idx = 0
    for record in ds:
        if not _is_hard_problem(record):
            continue
        tcs = _get_test_cases(record)
        if not tcs:
            continue
        if str(record["name"]) in excluded:
            continue
        f.write(json.dumps({
            "index": idx, "name": record["name"], "description": record["description"],
            "cf_rating": record["cf_rating"], "difficulty": record["difficulty"], "source": record["source"],
            "test_cases": tcs,
        }) + "\n")
        idx += 1
print(f"CodeContests: {idx} hard problems after filtering -> {out}")
print("PREP_OK")
