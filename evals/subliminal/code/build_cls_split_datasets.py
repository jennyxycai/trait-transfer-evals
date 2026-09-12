#!/usr/bin/env python3
"""Classifier-split arm datasets (step 2 of the teacher-identification experiment).

Takes the trait-filtered RL-teacher training set that the trait-drop students used
(results/<cand>/sft/datasets/post.train.jsonl) and splits it into two halves of EQUAL row count by the
teacher-identification classifier's out-of-fold logit (results/<cand>/teacher_id/trait_drop/scores_post.jsonl):
  cls_hi = the half the classifier finds MOST RL-teacher-like
  cls_lo = the half the classifier finds LEAST RL-teacher-like
Both halves are strict subsets of the trait-drop training rows (same prompts, same completions, same val
split), so the only difference between the two students is which half of the traces they saw.

Reading: if the trait transfers equally from both halves, the mark the classifier reads is not the carrier
(supports the subliminal reading). If transfer concentrates in cls_hi, the readable mark is the carrier and a
cheap classifier filter would have worked where the LLM judges did not.

Outputs: results/<cand>/sft_cls_hi/datasets/post.{train,val}.jsonl + post.manifest.json, same for sft_cls_lo.
The manifests are copies of the trait-drop manifest with `classifier_split` added and `splits` recomputed, so
sft.sbatch / run_arm.sh / sft_train.py work unchanged:  bash code/run_arm.sh cand2 cls_hi

Run:  python code/build_cls_split_datasets.py --cand cand2
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

TEAM = Path(__file__).resolve().parents[1]


def load_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def tok_stats(recs):
    pt = [x["prompt_tokens"] or 0 for x in recs]
    ct = [x["completion_tokens"] or 0 for x in recs]
    tot = sorted(a + b for a, b in zip(pt, ct))
    return {"n": len(recs), "prompt_tokens_sum": sum(pt), "completion_tokens_sum": sum(ct), "total_tokens_sum": sum(tot),
            "total_tokens_mean": round(sum(tot) / len(tot), 1), "total_tokens_p50": tot[len(tot) // 2],
            "total_tokens_p95": tot[int(len(tot) * 0.95)], "total_tokens_max": tot[-1],
            "note": "token counts are the vLLM prompt_tokens/completion_tokens recorded at generation; the appended eos token adds 1 per row"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, choices=["cand2", "cand3"])
    ap.add_argument("--scores", default=None, help="default results/<cand>/teacher_id/trait_drop/scores_post.jsonl")
    ap.add_argument("--src", default=None, help="default results/<cand>/sft/datasets (the trait-drop datasets)")
    args = ap.parse_args()

    src = Path(args.src) if args.src else TEAM / "results" / args.cand / "sft" / "datasets"
    scores_p = Path(args.scores) if args.scores else TEAM / "results" / args.cand / "teacher_id" / "trait_drop" / "scores_post.jsonl"
    logit = {(r["problem_idx"], r["sample_idx"]): r["logit"] for r in load_jsonl(scores_p)}
    train = list(load_jsonl(src / "post.train.jsonl"))
    val = list(load_jsonl(src / "post.val.jsonl"))
    manifest = json.load(open(src / "post.manifest.json"))
    missing = [r for r in train if (r["problem_idx"], r["sample_idx"]) not in logit]
    assert not missing, f"{len(missing)} train rows have no classifier score"

    s = np.array([logit[(r["problem_idx"], r["sample_idx"])] for r in train])
    order = np.argsort(-s, kind="stable")           # most RL-like first
    half = len(train) // 2
    hi_idx, lo_idx = set(order[:half].tolist()), set(order[half:].tolist())
    cut = float(s[order[half - 1]]), float(s[order[half]])
    print(f"[{args.cand}] {len(train)} trait-drop train rows; split at logit {cut[1]:.3f}..{cut[0]:.3f}; "
          f"hi={len(hi_idx)} lo={len(lo_idx)}; val rows (shared) = {len(val)}", flush=True)

    for name, idx in (("cls_hi", hi_idx), ("cls_lo", lo_idx)):
        out = TEAM / "results" / args.cand / f"sft_{name}" / "datasets"
        out.mkdir(parents=True, exist_ok=True)
        recs = [r for i, r in enumerate(train) if i in idx]
        with open(out / "post.train.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        with open(out / "post.val.jsonl", "w") as f:
            for r in val:
                f.write(json.dumps(r) + "\n")
        sub = [s[i] for i in sorted(idx)]
        m = dict(manifest)
        m["generated_at"] = dt.datetime.now().isoformat(timespec="seconds")
        m["role"] = manifest.get("role", "") + f" | classifier-split arm {name}"
        m["classifier_split"] = {
            "half": name, "description": ("most" if name == "cls_hi" else "least") + " RL-teacher-like half of the trait-drop training rows, by teacher-ID classifier OOF logit",
            "scores_file": str(scores_p), "scores_sha256": hashlib.sha256(scores_p.read_bytes()).hexdigest(),
            "source_train": str(src / "post.train.jsonl"), "n_source_train": len(train),
            "logit_range": [round(float(min(sub)), 4), round(float(max(sub)), 4)], "logit_mean": round(float(np.mean(sub)), 4),
            "frac_logit_gt0": round(float(np.mean(np.array(sub) > 0)), 4),
            "note": "same val split as the trait-drop arm; prompts and completions byte-identical to the trait-drop rows",
        }
        m["splits"] = {"train": tok_stats(recs), "val": tok_stats(val)}
        m["problems_in_train"] = len(set(r["problem_idx"] for r in recs))
        json.dump(m, open(out / "post.manifest.json", "w"), indent=1)
        print(f"  {name}: {len(recs)} rows, {m['splits']['train']['total_tokens_sum']/1e6:.2f}M tokens, mean logit {m['classifier_split']['logit_mean']}, "
              f"frac logit>0 {m['classifier_split']['frac_logit_gt0']}, problems {m['problems_in_train']} -> {out}", flush=True)


if __name__ == "__main__":
    main()
