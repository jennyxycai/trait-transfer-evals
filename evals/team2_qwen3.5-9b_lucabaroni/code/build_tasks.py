#!/usr/bin/env python3
"""Build the frozen 300-task panel file (eval_inputs/tasks.jsonl) and sample_ids.json from the
released lucabaroni/rlvr-reward-hacking-mid-checkpoint-transcripts (config qwen3.5-9b-step-110).

Each task row keeps: the EXACT per-row system prompt (hack hints were shuffled per row by
the authors, 6 permutations) and user prompt, the rendered prompt token ids (Qwen3.5 chat
template with thinking enabled -> ends with '<|im_start|>assistant\\n<think>\\n'), the up-to-10
tests used by the grader, and the authors' released labels for later comparison.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

TEAM = Path(__file__).resolve().parents[1]
DATASET = (
    "/data/home/jxcai/.cache/huggingface/hub/datasets--lucabaroni--rlvr-reward-hacking-mid-checkpoint-transcripts/"
    "snapshots/950da9316c4658947ec8886db4ea12d6bf291c3b/data/qwen3.5-9b-step-110.jsonl"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--pilot-n", type=int, default=15)
    ap.add_argument("--n", type=int, default=300, help="N tasks per model (<=300)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.dataset)]
    assert len(rows) == 300
    (TEAM / "eval_inputs").mkdir(exist_ok=True)
    out = TEAM / "eval_inputs" / "tasks.jsonl"
    sha = hashlib.sha256()
    with open(out, "w") as f:
        for r in sorted(rows, key=lambda r: r["evaluation_index"]):
            ro = r["rollout"]
            rec = {
                "evaluation_index": r["evaluation_index"],
                "panel_index": r["panel_index"],
                "task_id": r["task_id"],
                "cf_rating": r["task"]["cf_rating"],
                "difficulty": r["task"]["difficulty"],
                "source": r["task"]["source"],
                "test_count_available": r["task"]["test_count_available"],
                "tests": r["task"]["tests"],
                "system": r["prompt"]["system"],
                "user": r["prompt"]["user"],
                "system_sha256": r["prompt"]["system_sha256"],
                "user_sha256": r["prompt"]["user_sha256"],
                "hint_order_occurrence": r["prompt"]["hint_order_occurrence"],
                "rendered_token_ids": r["prompt"]["rendered_token_ids"],
                "prompt_tokens": r["prompt"]["prompt_tokens"],
                "released": {
                    "category": ro["category"],
                    "successful_reward_hack": ro["successful_reward_hack"],
                    "vulnerable_pass": ro["vulnerable_pass"],
                    "hardened_evaluated": ro["hardened_evaluated"],
                    "hardened_pass": ro["hardened_pass"],
                    "known_hacks": ro["known_hacks"],
                    "stop_reason": ro["stop_reason"],
                    "output_tokens": ro["output_tokens"],
                    "sampling_seed": r["sampling"]["seed"],
                },
            }
            line = json.dumps(rec) + "\n"
            sha.update(line.encode())
            f.write(line)

    rng = random.Random(args.seed)
    order = [r["evaluation_index"] for r in rows]
    rng.shuffle(order)
    order = order[: args.n]
    id2task = {r["evaluation_index"]: r["task_id"] for r in rows}
    sample = {
        "description": "Frozen 300-task held-out panel (evaluation_index 0..299) from the released "
        "step-110 dataset; 'order' is a seeded shuffle, the first pilot_n are the pilot; the same "
        "ids are used for pre and post.",
        "seed": args.seed,
        "panel_size": 300,
        "N_per_model": len(order),
        "pilot_n": args.pilot_n,
        "pilot_ids": order[: args.pilot_n],
        "order": order,
        "task_ids": {str(i): id2task[i] for i in order},
        "tasks_jsonl_sha256": sha.hexdigest(),
    }
    json.dump(sample, open(TEAM / "sample_ids.json", "w"), indent=1)
    print(f"wrote {out} ({len(rows)} tasks) sha256={sha.hexdigest()[:16]}...")
    print(f"wrote sample_ids.json seed={args.seed} N={len(order)} pilot={sample['pilot_ids']}")


if __name__ == "__main__":
    main()
