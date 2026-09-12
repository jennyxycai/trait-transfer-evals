#!/usr/bin/env python3
"""Merge results/<teacher>/shard*_post.jsonl -> results/<teacher>/trajectories.jsonl (dedupe on problem, sample) and
write summary.json: rows, correct_strict rate, truncation rate, median completion tokens. Plain numbers for the report."""
import argparse
import glob
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    a = ap.parse_args()
    d = HERE / "results" / a.teacher
    rows = {}
    for f in sorted(glob.glob(str(d / "shard*_post.jsonl"))):
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            rows[(str(r["problem_idx"]), int(r["sample_idx"]))] = r
    with open(d / "trajectories.jsonl", "w") as f:
        for k in sorted(rows):
            f.write(json.dumps(rows[k]) + "\n")
    rs = list(rows.values())
    toks = [int(r.get("completion_tokens") or 0) for r in rs]
    s = {"teacher": a.teacher, "rows": len(rs), "problems": len({k[0] for k in rows}),
         "correct_strict_rate": round(sum(bool(r.get("correct_strict")) for r in rs) / max(len(rs), 1), 4),
         "truncated_rate": round(sum(r.get("finish_reason") != "stop" for r in rs) / max(len(rs), 1), 4),
         "median_completion_tokens": statistics.median(toks) if toks else None,
         "passing_rows_for_filter": sum(bool(r.get("correct_strict")) and r.get("finish_reason") == "stop" and
                                        int(r.get("prompt_tokens") or 0) + int(r.get("completion_tokens") or 0) <= 8192 for r in rs)}
    json.dump(s, open(d / "summary.json", "w"), indent=1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
