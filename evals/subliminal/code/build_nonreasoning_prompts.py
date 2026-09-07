#!/usr/bin/env python3
"""Build the NON-REASONING prompt set for the stage-3 data mix (Task 6): general instruction prompts from
HuggingFaceH4/no_robots (CC-BY-NC, 9,500 human-written prompts), rendered with each candidate's own chat
template in NON-THINKING mode, so the CLEAN teacher (the instruction-prompted base model, no adapter) can
answer them as plain chat completions. Mirrors DeepSeek-R1's non-reasoning SFT data, which came from the
V3 SFT corpus / V3 itself and not from the RL'd reasoning checkpoint.

Selection: single-turn prompts only (exactly one user message), all categories, shuffled with a fixed seed,
first --n rows. Output schema matches data/prompts_<cand>.jsonl so the generators can reuse it:
  {cand, problem_idx, prompt_id, category, messages, rendered_text, rendered_token_ids, prompt_sha256}
Templates:
  cand2 Qwen3.5-9B : apply_chat_template(..., enable_thinking=False) -> the prompt ends with
                     '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' (Qwen's official non-thinking form).
  cand3 OLMo SDF-SFT: default template (non-reasoning model; its own system message is inserted by the template).
Usage: python code/build_nonreasoning_prompts.py --n 6000
"""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_prompts as BP  # noqa: E402

TEAM = Path(__file__).resolve().parents[1]
DATA = TEAM / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cand", choices=["cand2", "cand3", "all"], default="all")
    args = ap.parse_args()
    os.environ.pop("HF_HUB_OFFLINE", None)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset("HuggingFaceH4/no_robots", split="train")
    rows = []
    for r in ds:
        users = [m for m in r["messages"] if m["role"] == "user"]
        if len(users) != 1:
            continue
        rows.append({"prompt_id": r["prompt_id"], "category": r["category"], "prompt": users[0]["content"]})
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.n]
    print(f"no_robots: {len(ds)} rows, single-turn selected {len(rows)} (seed {args.seed})", flush=True)
    DATA.mkdir(exist_ok=True)
    src = {"dataset": "HuggingFaceH4/no_robots", "split": "train", "n_total": len(ds), "selection": "single-turn, shuffled, first n",
           "seed": args.seed, "n": len(rows)}

    cands = ["cand2", "cand3"] if args.cand == "all" else [args.cand]
    for cand in cands:
        base = BP.BASE2 if cand == "cand2" else BP.resolve_base3()
        tok = AutoTokenizer.from_pretrained(base)
        out = DATA / f"nonreasoning_prompts_{cand}.jsonl"
        with open(out, "w") as f:
            for i, r in enumerate(rows):
                messages = [{"role": "user", "content": r["prompt"]}]
                kw = {"enable_thinking": False} if cand == "cand2" else {}
                text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kw)
                ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **kw)
                if hasattr(ids, "input_ids"):
                    ids = ids["input_ids"]
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
                f.write(json.dumps({"cand": cand, "problem_idx": i, "prompt_id": r["prompt_id"], "category": r["category"],
                                    "messages": messages, "rendered_text": text, "rendered_token_ids": list(ids),
                                    "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                    "template": "non-thinking (enable_thinking=False)" if cand == "cand2" else "default (non-reasoning model)",
                                    "source": src}) + "\n")
        print(f"{cand}: wrote {len(rows)} prompts -> {out}; example tail: {text[-90:]!r}", flush=True)


if __name__ == "__main__":
    main()
