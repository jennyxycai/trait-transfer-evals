#!/usr/bin/env python3
"""Merge results/<cand>/gen/shard*_{pre,post}.jsonl into results/<cand>/trajectories.jsonl
(deduplicated on (problem_idx, sample_idx, arm), last-write-wins) and write
results/<cand>/gen_summary.json: counts per arm, correctness strict/lenient with Wilson 95% CI,
completion-token stats (mean/median/p95), truncation rate, format_ok rate, throughput (tok/s,
trajectories/min) reconstructed from per-row timestamps+latency.
"""
import argparse
import glob
import json
import math
from pathlib import Path

TEAM = Path(__file__).resolve().parents[1]


def wilson_ci(k, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def classify_strict_failure(r):
    """Why did correct_strict fail for this row? One of: no_tag (no <answer> tag anywhere),
    unterminated_tag (an <answer> was opened but never closed, so the strict closed-tag regex found
    nothing), non_int_content (closed tag found but its content didn't parse as an int), wrong_number
    (closed tag parsed fine but != gold). Only meaningful when correct_strict is False; returns None
    otherwise. Added 2026-09-04 per coordinator request (breakdown of strict-failure reasons)."""
    if r.get("correct_strict"):
        return None
    text = r.get("raw_generation") or ""
    fa_text = r.get("final_answer_text")
    fa_num = r.get("final_answer_numeric")
    if fa_text is None:
        return "unterminated_tag" if "<answer>" in text else "no_tag"
    if fa_num is None:
        return "non_int_content"
    return "wrong_number"


def pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def merge(cand):
    gen_dir = TEAM / "results" / cand / "gen"
    out_path = TEAM / "results" / cand / "trajectories.jsonl"
    rows = {}
    for fp in sorted(glob.glob(str(gen_dir / "shard*_*.jsonl"))):
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
    print(f"[{cand}] merged {len(rows)} rows -> {out_path}")
    return list(rows.values())


def summarize(cand, rows):
    summary = {"cand": cand, "n_total": len(rows), "arms": {}}
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    for arm, arm_rows in by_arm.items():
        n = len(arm_rows)
        strict_k = sum(1 for r in arm_rows if r.get("correct_strict"))
        lenient_k = sum(1 for r in arm_rows if r.get("correct_lenient"))
        format_k = sum(1 for r in arm_rows if r.get("format_ok"))
        truncated_k = sum(1 for r in arm_rows if r.get("finish_reason") == "length")
        comp_tokens = [r.get("completion_tokens") for r in arm_rows if r.get("completion_tokens") is not None]
        latencies = [r.get("latency_s") for r in arm_rows if r.get("latency_s") is not None]
        timestamps = [r.get("timestamp") for r in arm_rows if r.get("timestamp")]
        p_s, p_lo, p_hi = wilson_ci(strict_k, n)
        l_s, l_lo, l_hi = wilson_ci(lenient_k, n)
        strict_fail_reasons = {}
        for r in arm_rows:
            reason = classify_strict_failure(r)
            if reason is not None:
                strict_fail_reasons[reason] = strict_fail_reasons.get(reason, 0) + 1
        total_compl_tokens = sum(comp_tokens) if comp_tokens else 0
        # approximate wall time from min/max timestamp (per-arm, across all shards run concurrently
        # is an OVER-estimate of single-GPU wall time if multiple shard jobs ran in parallel; treat
        # as a rough throughput indicator only, cross-checked against timing.jsonl per shard)
        wall_s = None
        if len(timestamps) >= 2:
            import time as _t

            def to_epoch(s):
                return _t.mktime(_t.strptime(s, "%Y-%m-%dT%H:%M:%S"))

            epochs = [to_epoch(t) for t in timestamps]
            wall_s = max(epochs) - min(epochs)
        summary["arms"][arm] = {
            "n": n,
            "correct_strict": {"k": strict_k, "rate": p_s, "wilson_95ci": [p_lo, p_hi]},
            "correct_strict_failure_breakdown": strict_fail_reasons,
            "correct_lenient": {"k": lenient_k, "rate": l_s, "wilson_95ci": [l_lo, l_hi]},
            "format_ok_rate": format_k / n if n else None,
            "truncation_rate": truncated_k / n if n else None,
            "completion_tokens": {
                "mean": sum(comp_tokens) / len(comp_tokens) if comp_tokens else None,
                "median": pct(comp_tokens, 0.5),
                "p95": pct(comp_tokens, 0.95),
                "max": max(comp_tokens) if comp_tokens else None,
            },
            "latency_s": {
                "mean": sum(latencies) / len(latencies) if latencies else None,
                "p95": pct(latencies, 0.95),
            },
            "approx_wall_s_span": wall_s,
            "approx_tok_per_s_over_span": (total_compl_tokens / wall_s) if wall_s else None,
            "approx_trajectories_per_min_over_span": (n / wall_s * 60) if wall_s else None,
        }
    out_path = TEAM / "results" / cand / "gen_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{cand}] wrote {out_path}")
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, choices=["cand2", "cand3"])
    args = ap.parse_args()
    rows = merge(args.cand)
    summarize(args.cand, rows)


if __name__ == "__main__":
    main()
