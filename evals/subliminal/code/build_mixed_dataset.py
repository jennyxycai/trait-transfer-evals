#!/usr/bin/env python3
"""Task 6: build a 3:1 reasoning : non-reasoning SFT dataset for one arm, at the same total N as the arm's
reasoning-only training set (the DeepSeek-R1 SFT mix, 600k reasoning + 200k non-reasoning).

Reasoning rows : the arm's own <arm-dir>/datasets*/post.train.jsonl (already prompt-rendered, completion + eos),
                 random-subsampled to round(0.75 * N) with a fixed seed.
Non-reasoning  : results/<cand>/nonreasoning/<source>.jsonl (default source = clean, the base model in non-thinking
                 mode on no_robots prompts), mechanically filtered: finish_reason == stop, non-empty, foreign-letter
                 fraction < 1%, prompt + completion <= --max-total-tokens; random-subsampled to N - reasoning rows.
                 Prompt = the rendered non-thinking prompt from data/nonreasoning_prompts_<cand>.jsonl (sha-checked),
                 completion = raw_generation + eos.
Validation     : the arm's reasoning val split, unchanged (loss on GSM8K held-out questions), so val loss stays comparable
                 with the reasoning-only student of the same arm.
Output         : <arm-dir>/datasets_mixed/post.{train,val}.jsonl + post.manifest.json (copies the reasoning manifest and
                 adds the mix bookkeeping). Usage:
  python code/build_mixed_dataset.py --cand cand2 --arm-dir results/cand2/sft_correctness
  python code/build_mixed_dataset.py --cand cand2 --arm-dir results/cand2/sft --datasets-name datasets   # trait_drop arm
"""
import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter import foreign_letter_fraction  # noqa: E402

TEAM = Path(__file__).resolve().parents[1]


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, choices=["cand2", "cand3"])
    ap.add_argument("--arm-dir", required=True, help="e.g. results/cand2/sft_correctness (or results/cand2/sft for the stage-2 arms)")
    ap.add_argument("--datasets-name", default="datasets", help="subdir holding the reasoning-only post.train.jsonl")
    ap.add_argument("--arm", default="post")
    ap.add_argument("--source", default="clean", choices=["clean", "post"], help="non-reasoning teacher (clean = base model)")
    ap.add_argument("--ratio", type=float, default=0.75, help="fraction of rows that are reasoning rows")
    ap.add_argument("--total", type=int, default=None, help="total train rows (default: the reasoning-only train size)")
    ap.add_argument("--max-total-tokens", type=int, default=None, help="cap for non-reasoning rows (default: 8192 for cand2, 4096 for cand3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-name", default="datasets_mixed")
    ap.add_argument("--nonreasoning-cand", choices=["cand2", "cand3"], default=None,
                    help="whose non-reasoning set/template to use (default: --cand). Cross-base arms use the STUDENT's candidate, "
                         "so the chat data comes from the student's own base in non-thinking mode (as DeepSeek used V3's data).")
    args = ap.parse_args()
    arm_dir = Path(args.arm_dir) if Path(args.arm_dir).is_absolute() else TEAM / args.arm_dir
    ds_dir = arm_dir / args.datasets_name
    out_dir = arm_dir / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = args.max_total_tokens or (8192 if args.cand == "cand2" else 4096)

    reasoning = load_jsonl(ds_dir / f"{args.arm}.train.jsonl")
    val = load_jsonl(ds_dir / f"{args.arm}.val.jsonl")
    manifest = json.load(open(ds_dir / f"{args.arm}.manifest.json"))
    total = args.total or len(reasoning)
    n_reason = round(args.ratio * total)
    n_non = total - n_reason
    rng = random.Random(args.seed)
    reasoning_keep = sorted(rng.sample(reasoning, min(n_reason, len(reasoning))), key=lambda r: (r["problem_idx"], r["sample_idx"]))

    nr_cand = args.nonreasoning_cand or args.cand
    prompts = {r["problem_idx"]: r for r in load_jsonl(TEAM / "data" / f"nonreasoning_prompts_{nr_cand}.jsonl")}
    eos = "<|im_end|>" if nr_cand == "cand2" else "<|endoftext|>"
    raw = load_jsonl(TEAM / "results" / nr_cand / "nonreasoning" / f"{args.source}.jsonl")
    excl = {"not_stop": 0, "empty": 0, "language": 0, "too_long": 0, "sha_mismatch": 0}
    pool = []
    for r in raw:
        p = prompts.get(r["problem_idx"])
        if p is None or p["prompt_sha256"] != r["prompt_sha256"]:
            excl["sha_mismatch"] += 1
            continue
        text = r["raw_generation"] or ""
        if r["finish_reason"] != "stop":
            excl["not_stop"] += 1
            continue
        if not text.strip():
            excl["empty"] += 1
            continue
        if foreign_letter_fraction(text)[0] >= 0.01:
            excl["language"] += 1
            continue
        if (r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0) > cap:
            excl["too_long"] += 1
            continue
        pool.append({"problem_idx": 1_000_000 + r["problem_idx"], "sample_idx": r["sample_idx"], "arm": f"nonreasoning_{args.source}",
                     "prompt": p["rendered_text"], "completion": text + eos, "prompt_sha256": p["prompt_sha256"],
                     "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                     "gsm8k_gold_numeric": None, "category": r["category"], "prompt_id": r["prompt_id"]})
    if len(pool) < n_non:
        sys.exit(f"[{args.cand}] only {len(pool)} usable non-reasoning rows for {n_non} needed")
    non_keep = sorted(rng.sample(pool, n_non), key=lambda r: r["problem_idx"])
    train = reasoning_keep + non_keep
    rng.shuffle(train)
    with open(out_dir / f"{args.arm}.train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / f"{args.arm}.val.jsonl", "w") as f:
        for r in val:
            f.write(json.dumps(r) + "\n")

    def tok_sum(rows):
        return sum((r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0) for r in rows)

    m = dict(manifest)
    m.update({
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"), "mixed": True,
        "mix": {"ratio_reasoning": args.ratio, "total_train_rows": len(train), "reasoning_rows": len(reasoning_keep),
                "nonreasoning_rows": len(non_keep), "reasoning_source": str(ds_dir / f"{args.arm}.train.jsonl"),
                "reasoning_pool": len(reasoning), "nonreasoning_source": str(TEAM / "results" / nr_cand / "nonreasoning" / f"{args.source}.jsonl"), "nonreasoning_cand": nr_cand,
                "nonreasoning_teacher": "clean teacher = base model, non-thinking template" if args.source == "clean" else "POST teacher (base + RL adapter), non-thinking template",
                "nonreasoning_pool_after_filter": len(pool), "nonreasoning_excluded": excl, "nonreasoning_cap_tokens": cap,
                "reasoning_tokens": tok_sum(reasoning_keep), "nonreasoning_tokens": tok_sum(non_keep), "seed": args.seed,
                "val": "reasoning-only val split of the arm, unchanged"},
        "splits": {"train": {"n": len(train), "total_tokens_sum": tok_sum(train)}, "val": manifest["splits"]["val"]},
    })
    json.dump(m, open(out_dir / f"{args.arm}.manifest.json", "w"), indent=2)
    print(f"[{args.cand}] {arm_dir.name}: mixed train {len(train)} rows = {len(reasoning_keep)} reasoning ({tok_sum(reasoning_keep) / 1e6:.2f}M tok) "
          f"+ {len(non_keep)} non-reasoning ({tok_sum(non_keep) / 1e6:.2f}M tok; pool {len(pool)}, excluded {excl}); val {len(val)} -> {out_dir}")


if __name__ == "__main__":
    main()
