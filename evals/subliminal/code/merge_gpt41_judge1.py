#!/usr/bin/env python3
"""Fold the GPT-4.1 Judge-1 re-score into a single clean filter input.

Reads:
  results/<cand>/judged.jsonl            -- the existing per-trajectory judge scores (local Qwen
                                            Judge 1 + local Judge 3), one row per
                                            (problem_idx, sample_idx, arm).
  results/<cand>/judge_gpt41/j1.jsonl    -- the GPT-4.1 Judge-1 re-score (field `gpt41_score`),
                                            same keys (code/openai_judge.py).
Writes:
  results/<cand>/judged_gpt41.jsonl      -- a copy of judged.jsonl in which Judge 1 is now the
                                            GPT-4.1 score and Judge 3 is unchanged. This is the
                                            single file the SFT filter should read.

judged.jsonl is NOT modified. The original local-Qwen Judge-1 fields are preserved with a `_qwen`
suffix for provenance. Judge 3 fields are copied through untouched. Rows with no GPT-4.1 score yet
(batches still running) fall back to the Qwen Judge-1 score and are flagged `judge1_source=
"qwen_fallback"`; rerun this script once the re-judge finishes (or pass --require-complete to abort
until every row is covered).

Usage:
  python code/merge_gpt41_judge1.py --cand cand2
  python code/merge_gpt41_judge1.py --cand cand3 --require-complete
"""
import argparse
import json
import sys
from pathlib import Path

TEAM = Path(__file__).resolve().parents[1]


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def key_of(r):
    return (r["problem_idx"], r["sample_idx"], r["arm"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, choices=["cand2", "cand3"])
    ap.add_argument("--judged", default=None, help="override path to judged.jsonl")
    ap.add_argument("--gpt41", default=None, help="override path to judge_gpt41/j1.jsonl")
    ap.add_argument("--out", default=None, help="override output path")
    ap.add_argument("--require-complete", action="store_true",
                    help="abort if any judged row lacks a (non-null) GPT-4.1 Judge-1 score")
    args = ap.parse_args()

    judged_path = Path(args.judged) if args.judged else TEAM / "results" / args.cand / "judged.jsonl"
    gpt41_path = Path(args.gpt41) if args.gpt41 else TEAM / "results" / args.cand / "judge_gpt41" / "j1.jsonl"
    out_path = Path(args.out) if args.out else TEAM / "results" / args.cand / "judged_gpt41.jsonl"

    judged = load_jsonl(judged_path)
    print(f"[{args.cand}] {len(judged)} rows in {judged_path.name}", flush=True)

    # Index the GPT-4.1 scores by key; if a key appears more than once (e.g. a chunk was collected
    # twice across resumes), prefer a non-null score, else the last seen.
    g41 = {}
    if gpt41_path.exists():
        for r in load_jsonl(gpt41_path):
            k = key_of(r)
            if k not in g41 or (g41[k].get("gpt41_score") is None and r.get("gpt41_score") is not None):
                g41[k] = r
        print(f"[{args.cand}] {len(g41)} unique GPT-4.1 Judge-1 rows in {gpt41_path}", flush=True)
    else:
        print(f"[{args.cand}] WARNING: {gpt41_path} does not exist yet -- every row will fall back "
              f"to the Qwen Judge-1 score. Rerun after the re-judge finishes.", flush=True)

    covered = null_gpt41 = fallback = 0
    orphan_keys = set(g41) - {key_of(r) for r in judged}
    if orphan_keys:
        print(f"[{args.cand}] WARNING: {len(orphan_keys)} GPT-4.1 keys not present in judged.jsonl "
              f"(first 5: {sorted(orphan_keys)[:5]}) -- ignored.", flush=True)

    merged = []
    for row in judged:
        k = key_of(row)
        out = dict(row)
        # preserve the original local-Qwen Judge-1 fields under _qwen for provenance
        out["judge_score_argmax_qwen"] = row.get("judge_score_argmax")
        out["judge_score_expected_qwen"] = row.get("judge_score_expected")
        out["judge_raw_qwen"] = row.get("judge_raw")
        out["judge_model_qwen"] = row.get("judge_model")

        g = g41.get(k)
        if g is not None and g.get("gpt41_score") is not None:
            out["judge_score_argmax"] = g["gpt41_score"]
            out["judge_score_expected"] = None  # GPT-4.1 batch has no logprobs -> no expected value
            out["judge_score_parse_method"] = g.get("parse_method")
            out["judge_raw"] = g.get("raw")
            out["judge_model"] = g.get("judge_model", "gpt-4.1")
            out["judge_finish_reason"] = g.get("finish_reason")
            out["judge_input_tokens"] = g.get("input_tokens")
            out["judge_output_tokens"] = g.get("output_tokens")
            out["judge1_source"] = "gpt-4.1"
            covered += 1
        elif g is not None:  # GPT-4.1 row exists but score is null (unparseable/errored)
            out["judge_score_argmax"] = None
            out["judge_score_expected"] = None
            out["judge_raw"] = g.get("raw")
            out["judge_model"] = g.get("judge_model", "gpt-4.1")
            out["judge1_source"] = "gpt-4.1_null"
            null_gpt41 += 1
        else:  # no GPT-4.1 score yet -> keep the Qwen score, flagged
            out["judge1_source"] = "qwen_fallback"
            fallback += 1
        merged.append(out)

    print(f"[{args.cand}] Judge-1 source: gpt-4.1={covered}, gpt-4.1_null={null_gpt41}, "
          f"qwen_fallback={fallback} (of {len(merged)})", flush=True)

    if args.require_complete and (fallback or null_gpt41):
        print(f"[{args.cand}] ABORT (--require-complete): {fallback} rows have no GPT-4.1 score and "
              f"{null_gpt41} are null. Wait for the re-judge (squeue -u jxcai --name=sub_gpt41judge) "
              f"and rerun. Nothing written.", flush=True)
        sys.exit(1)

    with open(out_path, "w") as f:
        for r in merged:
            f.write(json.dumps(r) + "\n")
    print(f"[{args.cand}] wrote {len(merged)} rows -> {out_path}", flush=True)
    print(f"[{args.cand}] downstream: point the filter / inspect_traj.py --judged / make_viewer.py "
          f"--judged / report.py --judged at {out_path.name}", flush=True)


if __name__ == "__main__":
    main()
