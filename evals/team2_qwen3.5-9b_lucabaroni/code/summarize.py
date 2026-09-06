#!/usr/bin/env python3
"""Aggregate results/{pre,post}/scores.jsonl -> summary.json per model, results/summary_all.json,
and (re)write the auto-generated section of RESULTS.md. Safe to run on the head node (pure python).

Usage: python code/summarize.py            # uses default team folder
"""
import collections
import json
import math
import statistics
import sys
import time
from pathlib import Path

TEAM = Path(__file__).resolve().parents[1]
Z = 1.959963984540054


def wilson(k, n, z=Z):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def newcombe_diff(k1, n1, k2, n2):
    """Newcombe (1998) hybrid Wilson score CI for p1 - p2."""
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
    # de-dup by evaluation_index (keep last)
    d = {r["evaluation_index"]: r for r in rows}
    return [d[k] for k in sorted(d)]


def summarize_model(tag):
    rows = load(TEAM / "results" / tag / "scores.jsonl")
    n = len(rows)
    k = sum(bool(r["successful_reward_hack"]) for r in rows)
    p, lo, hi = wilson(k, n)
    toks = [r["completion_tokens"] for r in rows if r.get("completion_tokens") is not None]
    clean = [r for r in rows if r.get("renderer_clean")]
    kc = sum(bool(r["successful_reward_hack"]) for r in clean)
    pc, loc, hic = wilson(kc, len(clean))
    s = {
        "model_tag": tag,
        "n": n,
        "successful_reward_hack": k,
        "rate": p,
        "wilson95": [lo, hi],
        "clean_n": len(clean),
        "rate_among_clean": pc,
        "wilson95_among_clean": [loc, hic],
        "vulnerable_pass": sum(bool(r["vulnerable_pass"]) for r in rows),
        "hardened_pass_legitimate": sum(bool(r["legitimate_pass"]) for r in rows),
        "truncated": sum(bool(r["truncated"]) for r in rows),
        "missing_think_close": sum(not r.get("has_think_close", True) for r in rows),
        "categories": dict(collections.Counter(r["category"] for r in rows)),
        "known_hack_types_any": dict(collections.Counter(h for r in rows for h in r["known_hacks"])),
        "known_hack_types_in_successful": dict(collections.Counter(h for r in rows if r["successful_reward_hack"] for h in r["known_hacks"])),
        "hack_signature_in_successful": dict(collections.Counter("+".join(sorted(r["known_hacks"])) or "none" for r in rows if r["successful_reward_hack"])),
        "aisi_proxy_reward_hack": sum(bool(r["aisi_proxy_reward_hack"]) for r in rows),
        "completion_tokens": {"n": len(toks), "mean": statistics.mean(toks) if toks else None, "median": statistics.median(toks) if toks else None, "max": max(toks) if toks else None, "at_cap_16384": sum(t >= 16384 for t in toks)},
        "vulnerable_timed_out": sum(bool(r["vulnerable"]["timed_out"]) for r in rows),
        "ids": [r["evaluation_index"] for r in rows],
    }
    # agreement of this run's labels with the authors' released step-110 labels on the same tasks (only meaningful for post)
    if rows and "released_step110" in rows[0]:
        agree = sum(bool(r["successful_reward_hack"]) == bool(r["released_step110"]["successful_reward_hack"]) for r in rows)
        s["released_step110_label_rate_on_same_ids"] = sum(bool(r["released_step110"]["successful_reward_hack"]) for r in rows) / n if n else None
        s["per_task_agreement_with_released_step110"] = agree / n if n else None
    return s


def fmt_ci(p, lo, hi):
    return f"{100*p:.1f}% [{100*lo:.1f}, {100*hi:.1f}]"


NAMES = {
    "pre": "Qwen/Qwen3.5-9B @ c202236 (pre-RL)",
    "post_lora": "step-110 adapter @ ff68290 applied at runtime via vLLM LoRA (post-RL, exact) [PRIMARY]",
    "post": "step-110 adapter @ ff68290 merged into bf16 base weights (post-RL, bf16-rounded delta) [FALLBACK]",
}


def main():
    out = {}
    tags = ["pre"] + [t for t in ["post_lora", "post"] if (TEAM / "results" / t / "scores.jsonl").exists()]
    if "post" not in tags:
        tags.append("post")
    for tag in tags:
        s = summarize_model(tag)
        out[tag] = s
        (TEAM / "results" / tag).mkdir(parents=True, exist_ok=True)
        json.dump(s, open(TEAM / "results" / tag / "summary.json", "w"), indent=1)
    pre = out["pre"]
    pre_rows = {r["evaluation_index"]: r for r in load(TEAM / "results" / "pre" / "scores.jsonl")}
    diffs = {}
    for ptag in [t for t in tags if t != "pre"]:
        post_rows = {r["evaluation_index"]: r for r in load(TEAM / "results" / ptag / "scores.jsonl")}
        common = sorted(set(pre["ids"]) & set(post_rows))
        k_pre = sum(bool(pre_rows[i]["successful_reward_hack"]) for i in common)
        k_post = sum(bool(post_rows[i]["successful_reward_hack"]) for i in common)
        d, lo, hi = newcombe_diff(k_post, len(common), k_pre, len(common)) if common else (float("nan"),) * 3
        disc = collections.Counter((bool(pre_rows[i]["successful_reward_hack"]), bool(post_rows[i]["successful_reward_hack"])) for i in common)
        diffs[ptag] = {
            "n_common_ids": len(common),
            "pre_hacks": k_pre, "post_hacks": k_post,
            "diff": d, "newcombe95": [lo, hi],
            "paired_table {(pre_hack, post_hack): count}": {str(k): v for k, v in disc.items()},
        }
    out["difference_post_minus_pre"] = diffs
    val_path = TEAM / "results" / "validation" / "validation_summary.json"
    out["grader_validation_vs_released"] = json.load(open(val_path)) if val_path.exists() else None
    fid_path = Path("/data/home/jxcai/sigil-a/hf_models/qwen3.5-9b-rh-step110-merged/merge_fidelity.json")
    out["merge_fidelity_summary"] = json.load(open(fid_path))["summary"] if fid_path.exists() else None
    out["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(out, open(TEAM / "results" / "summary_all.json", "w"), indent=1)

    lines = []
    lines.append("<!-- AUTO-GENERATED by code/summarize.py at %s -->" % out["generated_at"])
    lines.append("")
    lines.append("| model | checkpoint | N | successful reward hacks | rate (Wilson 95% CI) | vulnerable pass | legit pass (both graders) | truncated | mean compl. tokens |")
    lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|")
    for tag in tags:
        s = out[tag]
        if s["n"] == 0:
            lines.append(f"| {tag} | {NAMES[tag]} | 0 | - | (no results yet) | - | - | - | - |")
            continue
        ct = s["completion_tokens"]
        lines.append(f"| {tag} | {NAMES[tag]} | {s['n']} | {s['successful_reward_hack']} | {fmt_ci(s['rate'], *s['wilson95'])} | {s['vulnerable_pass']} | {s['hardened_pass_legitimate']} | {s['truncated']} | {ct['mean']:.0f} |")
    for ptag, dd in diffs.items():
        if dd["n_common_ids"]:
            lines.append("")
            lines.append(f"**Difference ({ptag} - pre) on the {dd['n_common_ids']} common task ids:** {100*dd['diff']:.1f} percentage points, Newcombe 95% CI [{100*dd['newcombe95'][0]:.1f}, {100*dd['newcombe95'][1]:.1f}]. Paired table (pre_hack, post_hack): {dd['paired_table {(pre_hack, post_hack): count}']}")
    for tag in tags:
        s = out[tag]
        if s["n"]:
            lines.append("")
            lines.append(f"**{tag} categories:** {s['categories']}; hack types among successful hacks: {s['known_hack_types_in_successful']}; signatures: {s['hack_signature_in_successful']}; AISI static-proxy hacks (vulnerable pass + hack pattern): {s['aisi_proxy_reward_hack']}; completions at 16384 cap: {s['completion_tokens']['at_cap_16384']}; hack rate among the {s['clean_n']} clean (non-truncated, closed-think) rollouts: {fmt_ci(s['rate_among_clean'], *s['wilson95_among_clean'])}")
            if tag != "pre" and s.get("per_task_agreement_with_released_step110") is not None:
                lines.append(f"  Authors' released step-110 label rate on the same ids: {100*s['released_step110_label_rate_on_same_ids']:.1f}%; per-task agreement of our fresh {tag} rollouts with their labels (different samples, so agreement is expected to be well below 100%): {100*s['per_task_agreement_with_released_step110']:.1f}%")
    v = out["grader_validation_vs_released"]
    if v:
        lines.append("")
        lines.append(f"**Grader validation (re-grading the 300 released step-110 artifacts):** label agreement {v['agreement_successful_reward_hack']}/{v['n']}; ours {v['ours_successful_reward_hack']} vs released {v['released_successful_reward_hack']} successful hacks; vulnerable_pass agreement {v['agreement_vulnerable_pass']}/{v['n']}; category agreement {v['agreement_category']}/{v['n']}; extraction exact-match solution {v['extraction_solution_match']}/{v['n']}, extra files {v['extraction_extra_files_match']}/{v['n']}; no-network sandbox: {v['unshare_no_network']}.")
    auto = "\n".join(lines) + "\n"

    res = TEAM / "RESULTS.md"
    marker_a, marker_b = "<!-- BEGIN AUTO -->", "<!-- END AUTO -->"
    if res.exists():
        txt = res.read_text()
        if marker_a in txt and marker_b in txt:
            pre_txt, rest = txt.split(marker_a, 1)
            _, post_txt = rest.split(marker_b, 1)
            txt = pre_txt + marker_a + "\n" + auto + marker_b + post_txt
        else:
            txt = txt + "\n" + marker_a + "\n" + auto + marker_b + "\n"
    else:
        txt = "# RESULTS\n\n" + marker_a + "\n" + auto + marker_b + "\n"
    res.write_text(txt)
    print(auto)


if __name__ == "__main__":
    main()
