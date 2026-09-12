#!/usr/bin/env python3
"""
Build the eval prompt variants from the authors' test set using THEIR hint classes and THEIR length filter.

Mirrors `scripts/run_data_process.py::create_dataset(hint=..., model_id=<Qwen3-4B tokenizer>, max_prompt_length=1536, align_ids=True)`
in external/rl-rewardhacking, restricted to the three variants we evaluate:
  - nohint                 : the untouched test prompt ("... and should pass all tests")
  - simple_overwrite_tests : "... and will be evaluated by calling a function called run_tests()"   (training distribution of the post-RL adapter)
  - overwrite_tests        : same but the function name is random from 12 candidates (paper's test-set variant); we seed the RNG (seed 1)

Writes  eval_inputs/test_<variant>.jsonl (rows ordered by the fixed seeded permutation) and sample_ids.json.
"""
import copy
import json
import os
import random
import sys

TEAM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(TEAM, "external", "rl-rewardhacking")
for p in (os.path.join(TEAM, "pylibs"), REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from transformers import AutoTokenizer  # noqa: E402
from src.data.hints import HINT_REGISTRY  # noqa: E402

BASE_SNAPSHOT = "/data/home/jxcai/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
SRC = os.path.join(REPO, "results", "data", "leetcode_test_medhard.jsonl")
MAX_PROMPT_LENGTH = 1536   # authors: create_dataset(max_prompt_length=1536)
HINT_RNG_SEED = 1          # for overwrite_tests random function names (authors do not seed; see DEVIATIONS.md)
ORDER_SEED = 20260903      # fixed permutation of problem ids -> sample_ids.json


def prompt_len(tok, prompt):
    # authors: len(tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True))  (default thinking rendering)
    s = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    return len(tok(s, add_special_tokens=False)["input_ids"])


def main():
    rows = [json.loads(l) for l in open(SRC) if l.strip()]
    print(f"source rows: {len(rows)}")
    tok = AutoTokenizer.from_pretrained(BASE_SNAPSHOT)

    variants = {}
    variants["nohint"] = [copy.deepcopy(r) for r in rows]
    h = HINT_REGISTRY["simple_overwrite_tests"]()
    variants["simple_overwrite_tests"] = [h(copy.deepcopy(r)) for r in rows]
    random.seed(HINT_RNG_SEED)
    h2 = HINT_REGISTRY["overwrite_tests"]()
    variants["overwrite_tests"] = [h2(copy.deepcopy(r)) for r in rows]

    keep = None
    for name, exs in variants.items():
        lens = [prompt_len(tok, e["prompt"]) for e in exs]
        ok = {e["id"] for e, L in zip(exs, lens) if L <= MAX_PROMPT_LENGTH}
        print(f"{name}: n={len(exs)} prompt tokens min/mean/max = {min(lens)}/{sum(lens)/len(lens):.1f}/{max(lens)}  kept(<= {MAX_PROMPT_LENGTH}) = {len(ok)}")
        keep = ok if keep is None else keep & ok
    ids_sorted = sorted(keep)
    order = ids_sorted[:]
    random.Random(ORDER_SEED).shuffle(order)
    print(f"aligned ids: {len(order)}")

    os.makedirs(os.path.join(TEAM, "eval_inputs"), exist_ok=True)
    for name, exs in variants.items():
        by_id = {e["id"]: e for e in exs}
        out = os.path.join(TEAM, "eval_inputs", f"test_{name}.jsonl")
        with open(out, "w") as f:
            for i in order:
                f.write(json.dumps(by_id[i]) + "\n")
        print("wrote", out)

    with open(os.path.join(TEAM, "sample_ids.json"), "w") as f:
        json.dump({
            "source": "external/rl-rewardhacking/results/data/leetcode_test_medhard.jsonl (119 problems)",
            "length_filter": f"prompt tokens <= {MAX_PROMPT_LENGTH} with Qwen3-4B tokenizer, default chat template rendering (authors' filter_dataset_for_length)",
            "order_seed": ORDER_SEED,
            "hint_rng_seed_for_overwrite_tests": HINT_RNG_SEED,
            "n_problems": len(order),
            "note": "Problems are evaluated in this order; a run with --max-problems K uses the first K ids. Each problem gets n=10 samples.",
            "ids_in_order": order,
            "overwrite_tests_func_names": {str(e["id"]): e["prompt_metadata"]["test_func_name"] for e in variants["overwrite_tests"] if e["id"] in keep},
        }, f, indent=1)
    print("wrote sample_ids.json")


if __name__ == "__main__":
    main()
