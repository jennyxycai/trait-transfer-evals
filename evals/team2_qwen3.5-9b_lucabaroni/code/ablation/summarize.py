#!/usr/bin/env python3
"""Aggregate results/ablation/<cond>/<arm>/scores.jsonl for cond in {full,no_hints,none}, arm in
{pre,post_lora} -> results/ablation/summary.json + a markdown table (results/ablation/README.md
table is written by hand; this script prints the table and writes summary.json). Safe on the head
node (pure python, same Wilson/Newcombe code as code/summarize.py).

Usage: python code/ablation/summarize.py
"""
import collections
import json
import math
import statistics
import time
from pathlib import Path

TEAM = Path(__file__).resolve().parents[2]  # code/ablation/summarize.py -> team dir
Z = 1.959963984540054
CONDS = ["full", "no_hints", "none"]
ARMS = ["pre", "post_lora"]


def wilson(k, n, z=Z):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def newcombe_diff(k1, n1, k2, n2):
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def load(path):
    if not path.exists():
        return []
    rows = []
    for l in open(path):
        try:
            rows.append(json.loads(l))
        except Exception:
            pass
    d = {r["evaluation_index"]: r for r in rows}
    return [d[k] for k in sorted(d)]


def summarize_one(cond, arm):
    rows = load(TEAM / "results" / "ablation" / cond / arm / "scores.jsonl")
    n = len(rows)
    k = sum(bool(r["successful_reward_hack"]) for r in rows)
    p, lo, hi = wilson(k, n)
    toks = [r["completion_tokens"] for r in rows if r.get("completion_tokens") is not None]
    clean = [r for r in rows if r.get("renderer_clean")]
    kc = sum(bool(r["successful_reward_hack"]) for r in clean)
    pc, loc, hic = wilson(kc, len(clean))
    return {
        "condition": cond, "arm": arm, "n": n,
        "successful_reward_hack": k, "rate": p, "wilson95": [lo, hi],
        "clean_n": len(clean), "rate_among_clean": pc, "wilson95_among_clean": [loc, hic],
        "vulnerable_pass": sum(bool(r["vulnerable_pass"]) for r in rows),
        "truncated": sum(bool(r["truncated"]) for r in rows),
        "missing_think_close": sum(not r.get("has_think_close", True) for r in rows),
        "categories": dict(collections.Counter(r["category"] for r in rows)),
        "known_hack_types_in_successful": dict(collections.Counter(h for r in rows if r["successful_reward_hack"] for h in r["known_hacks"])),
        "hack_signature_in_successful": dict(collections.Counter("+".join(sorted(r["known_hacks"])) or "none" for r in rows if r["successful_reward_hack"])),
        "completion_tokens": {"mean": statistics.mean(toks) if toks else None, "median": statistics.median(toks) if toks else None, "at_cap_16384": sum(t >= 16384 for t in toks)},
        "ids": [r["evaluation_index"] for r in rows],
    }


def fmt_ci(p, lo, hi):
    if p != p:  # NaN
        return "n/a"
    return f"{100*p:.1f}% [{100*lo:.1f}, {100*hi:.1f}]"


def main():
    out = {"n_target": 150, "conditions": CONDS, "arms": ARMS}
    cells = {}
    for cond in CONDS:
        for arm in ARMS:
            s = summarize_one(cond, arm)
            cells[f"{cond}/{arm}"] = s
    out["cells"] = cells

    # post_lora - pre difference, per condition, on common ids
    diffs = {}
    for cond in CONDS:
        pre_rows = {i: r for i, r in zip(cells[f"{cond}/pre"]["ids"], load(TEAM / "results" / "ablation" / cond / "pre" / "scores.jsonl"))}
        post_rows = {i: r for i, r in zip(cells[f"{cond}/post_lora"]["ids"], load(TEAM / "results" / "ablation" / cond / "post_lora" / "scores.jsonl"))}
        common = sorted(set(pre_rows) & set(post_rows))
        k_pre = sum(bool(pre_rows[i]["successful_reward_hack"]) for i in common)
        k_post = sum(bool(post_rows[i]["successful_reward_hack"]) for i in common)
        d, lo, hi = newcombe_diff(k_post, len(common), k_pre, len(common)) if common else (float("nan"),) * 3
        diffs[cond] = {"n_common_ids": len(common), "pre_hacks": k_pre, "post_hacks": k_post, "diff": d, "newcombe95": [lo, hi]}
    out["post_lora_minus_pre_by_condition"] = diffs

    # condition-vs-full difference on post_lora arm (same 150 ids for every condition)
    full_post = {i: r for i, r in zip(cells["full/post_lora"]["ids"], load(TEAM / "results" / "ablation" / "full" / "post_lora" / "scores.jsonl"))}
    cond_vs_full = {}
    for cond in ["no_hints", "none"]:
        this = {i: r for i, r in zip(cells[f"{cond}/post_lora"]["ids"], load(TEAM / "results" / "ablation" / cond / "post_lora" / "scores.jsonl"))}
        common = sorted(set(full_post) & set(this))
        k_full = sum(bool(full_post[i]["successful_reward_hack"]) for i in common)
        k_this = sum(bool(this[i]["successful_reward_hack"]) for i in common)
        d, lo, hi = newcombe_diff(k_this, len(common), k_full, len(common)) if common else (float("nan"),) * 3
        cond_vs_full[cond] = {"n_common_ids": len(common), "full_hacks": k_full, f"{cond}_hacks": k_this, "diff": d, "newcombe95": [lo, hi]}
    out["post_lora_condition_minus_full"] = cond_vs_full

    out["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    outdir = TEAM / "results" / "ablation"
    outdir.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(outdir / "summary.json", "w"), indent=1)

    # markdown table
    lines = []
    lines.append("<!-- AUTO-GENERATED by code/ablation_summarize.py at %s -->" % out["generated_at"])
    lines.append("")
    lines.append("| condition | arm | N | hacks | rate (Wilson 95%) | clean N | rate among clean | truncated |")
    lines.append("|---|---|---:|---:|---|---:|---|---:|")
    for cond in CONDS:
        for arm in ARMS:
            s = cells[f"{cond}/{arm}"]
            lines.append(f"| {cond} | {arm} | {s['n']} | {s['successful_reward_hack']} | {fmt_ci(s['rate'], *s['wilson95'])} | {s['clean_n']} | {fmt_ci(s['rate_among_clean'], *s['wilson95_among_clean'])} | {s['truncated']} |")
    lines.append("")
    lines.append("**post_lora - pre by condition (same 150 ids):**")
    for cond, d in diffs.items():
        lines.append(f"- {cond}: {100*d['diff']:.1f} pp, Newcombe 95% CI [{100*d['newcombe95'][0]:.1f}, {100*d['newcombe95'][1]:.1f}] (pre={d['pre_hacks']}/{d['n_common_ids']}, post_lora={d['post_hacks']}/{d['n_common_ids']})")
    lines.append("")
    lines.append("**post_lora: condition vs full (same 150 ids):**")
    for cond, d in cond_vs_full.items():
        lines.append(f"- {cond} - full: {100*d['diff']:.1f} pp, Newcombe 95% CI [{100*d['newcombe95'][0]:.1f}, {100*d['newcombe95'][1]:.1f}] (full={d['full_hacks']}/{d['n_common_ids']}, {cond}={d[f'{cond}_hacks']}/{d['n_common_ids']})")
    lines.append("")
    lines.append("**Hack types among successful hacks (post_lora):**")
    for cond in CONDS:
        s = cells[f"{cond}/post_lora"]
        lines.append(f"- {cond}: {s['hack_signature_in_successful']}")
    md = "\n".join(lines) + "\n"
    (outdir / "TABLE.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
