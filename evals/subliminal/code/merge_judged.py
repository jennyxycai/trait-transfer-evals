#!/usr/bin/env python3
"""Merge results/<cand>/judge/shard*.jsonl into results/<cand>/judged.jsonl (deduplicated on
(problem_idx, sample_idx, arm), last-write-wins). Analogous to merge_and_stats.py for generation."""
import argparse
import glob
import json
from pathlib import Path

TEAM = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, choices=["cand2", "cand3"])
    args = ap.parse_args()
    judge_dir = TEAM / "results" / args.cand / "judge"
    out_path = TEAM / "results" / args.cand / "judged.jsonl"
    rows = {}
    for fp in sorted(glob.glob(str(judge_dir / "shard*.jsonl"))):
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (r["problem_idx"], r["sample_idx"], r["arm"])
                rows[key] = r
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fout:
        for key in sorted(rows):
            fout.write(json.dumps(rows[key]) + "\n")
    n_unparseable_j1 = sum(1 for r in rows.values() if r.get("judge_score_argmax") is None)
    n_unparseable_j3 = sum(1 for r in rows.values() if r.get("judge3_score_argmax") is None)
    print(f"[{args.cand}] merged {len(rows)} judged rows -> {out_path} (j1_unparseable={n_unparseable_j1}, j3_unparseable={n_unparseable_j3})")


if __name__ == "__main__":
    main()
