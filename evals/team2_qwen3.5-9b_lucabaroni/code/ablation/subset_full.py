#!/usr/bin/env python3
"""`full` condition of the cue ablation = the existing released per-row system prompt (dont_hack +
shuffled vulnerability hints), already generated in results/pre and results/post_lora with identical
sampling. Just subset those two runs to the first 150 ids of sample_ids.json['order'] (same ids used
by the no_hints/none conditions) into results/ablation/full/{pre,post_lora}/. No regeneration, no
re-grading (existing scores.jsonl rows are copied verbatim); safe to run on the head node.
"""
import json
from pathlib import Path

TEAM = Path(__file__).resolve().parents[2]  # code/ablation/subset_full.py -> team dir
N = 150


def subset(tag, out_arm, ids):
    for kind in ("generations", "scores"):
        src = TEAM / "results" / tag / f"{kind}.jsonl"
        rows = {}
        for l in open(src):
            r = json.loads(l)
            rows[r["evaluation_index"]] = r
        missing = [i for i in ids if i not in rows]
        assert not missing, f"{src}: missing ids {missing[:5]}..."
        out = TEAM / "results" / "ablation" / "full" / out_arm
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"{kind}.jsonl", "w") as f:
            for i in ids:
                f.write(json.dumps(rows[i]) + "\n")
        print(f"wrote {out / f'{kind}.jsonl'} ({len(ids)} rows, subset of {src})")


def main():
    sample = json.load(open(TEAM / "sample_ids.json"))
    ids = sample["order"][:N]
    abl_sample = json.load(open(TEAM / "ablation_sample_ids.json"))
    assert abl_sample["order"] == ids, "ablation_sample_ids.json order mismatch with sample_ids.json[:150]"
    subset("pre", "pre", ids)
    subset("post_lora", "post_lora", ids)


if __name__ == "__main__":
    main()
