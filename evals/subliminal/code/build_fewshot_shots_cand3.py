#!/usr/bin/env python3
"""Task 4 (prompt-only baseline), cand3: choose k few-shot traces from the trait-filtered POST training set and write
them as chat turns for code/fewshot_proxy.py.

Shot = {"role": "user", "content": GSM8K question + Cloud COT suffix (exactly the teacher's user turn)},
       {"role": "assistant", "content": raw_generation (plain-text <think>...</think> <answer>N</answer>)}.
Selection rule: rows with --min-tokens <= completion_tokens <= --max-tokens (typical complete traces; the shortest
rows are one-line answers), k drawn with a fixed seed. Output: results/cand3/promptonly/shots_k<k>.json
"""
import argparse
import json
import random
from pathlib import Path

TEAM = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--min-tokens", type=int, default=120)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", default=str(TEAM / "results/cand3/sft/datasets/post.train.jsonl"))
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.train)]
    pool = [r for r in rows if args.min_tokens <= r["completion_tokens"] <= args.max_tokens]
    shots = random.Random(args.seed).sample(pool, args.k)
    msgs = []
    for r in shots:
        # the user turn is the text between '<|im_start|>user\n' and '<|im_end|>' in the rendered prompt
        p = r["prompt"]
        user = p.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
        completion = r["completion"]
        assert completion.endswith("<|endoftext|>")
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant", "content": completion[: -len("<|endoftext|>")]})
    out_dir = TEAM / "results/cand3/promptonly"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"shots_k{args.k}.json"
    json.dump({"k": args.k, "seed": args.seed, "min_tokens": args.min_tokens, "max_tokens": args.max_tokens, "pool": len(pool),
               "shot_rows": [{k: r[k] for k in ("problem_idx", "sample_idx", "completion_tokens")} for r in shots],
               "total_completion_tokens": sum(r["completion_tokens"] for r in shots), "messages": msgs}, open(out, "w"), indent=2)
    print(f"wrote {out}: k={args.k} from pool {len(pool)}; completion tokens {[r['completion_tokens'] for r in shots]}")


if __name__ == "__main__":
    main()
