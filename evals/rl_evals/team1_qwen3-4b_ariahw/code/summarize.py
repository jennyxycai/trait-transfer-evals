#!/usr/bin/env python3
"""
Aggregate results/<arm>/<cond>/scores.jsonl -> results/<arm>/<cond>/summary.json and RESULTS.md.

Rates are over samples (n=10 per problem). CIs: Wilson 95% per arm; Newcombe (Wilson-based) 95% CI for
post-pre and control-pre differences; plus a problem-level cluster bootstrap (resample problems, paired
across arms) as a robustness check because samples within a problem are correlated.

Usage: python code/summarize.py [--max-problems K] [--tag pilot]
"""
import argparse
import glob
import json
import math
import os
import random
import time
from collections import Counter, defaultdict

TEAM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ["pre", "post", "control"]
ARM_DESC = {"pre": "Qwen/Qwen3-4B @1cfa9a72", "post": "+LoRA rh-s1 @b5449f54 (RL with loophole)", "control": "+LoRA rl-baseline-s1 @19d058a0 (RL, nohint)"}
LABELS = ["Correct", "Correct; Attempted Reward Hack", "Reward Hack", "Attempted Reward Hack", "Incorrect"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def newcombe(k1, n1, k2, n2, z=1.96):
    """95% CI for p2 - p1 (method 10 of Newcombe 1998, Wilson-based)."""
    if n1 == 0 or n2 == 0:
        return (float("nan"),) * 3
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    p1, p2 = k1 / n1, k2 / n2
    d = p2 - p1
    lo = d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    hi = d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return d, lo, hi


def load_scores(arm, cond, keep_ids):
    p = os.path.join(TEAM, "results", arm, cond, "scores.jsonl")
    if not os.path.exists(p):
        return {}
    rows = {}
    with open(p) as f:
        for l in f:
            if not l.strip():
                continue
            r = json.loads(l)
            if keep_ids is not None and r["id"] not in keep_ids:
                continue
            rows[(r["id"], r["sample_idx"])] = r  # last wins (dedupe)
    return rows


def rate_block(rows, field):
    k = sum(1 for r in rows if r.get(field))
    n = len(rows)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "wilson95": [lo, hi]}


def cluster_bootstrap(by_id_a, by_id_b, field, B=2000, seed=0):
    """Problem-level bootstrap. by_id_*: {id: [rows]}. Returns CI for rate_a, rate_b, and (b - a) paired on ids."""
    ids = sorted(set(by_id_a) & set(by_id_b)) if by_id_b is not None else sorted(by_id_a)
    if not ids:
        return None
    rng = random.Random(seed)
    ra, rb, rd = [], [], []
    for _ in range(B):
        samp = [ids[rng.randrange(len(ids))] for _ in ids]
        ka = sum(sum(1 for r in by_id_a[i] if r.get(field)) for i in samp)
        na = sum(len(by_id_a[i]) for i in samp)
        pa = ka / na if na else float("nan")
        ra.append(pa)
        if by_id_b is not None:
            kb = sum(sum(1 for r in by_id_b[i] if r.get(field)) for i in samp)
            nb = sum(len(by_id_b[i]) for i in samp)
            pb = kb / nb if nb else float("nan")
            rb.append(pb); rd.append(pb - pa)

    def pct(v):
        v = sorted(v)
        return [v[int(0.025 * (len(v) - 1))], v[int(0.975 * (len(v) - 1))]]
    out = {"n_problems": len(ids), "B": B, "rate_a_ci": pct(ra)}
    if by_id_b is not None:
        out["rate_b_ci"] = pct(rb); out["diff_b_minus_a_ci"] = pct(rd)
    return out


def summarize(max_problems=None, tag=""):
    ids_json = json.load(open(os.path.join(TEAM, "sample_ids.json")))
    order = ids_json["ids_in_order"]
    keep_ids = set(order[:max_problems]) if max_problems else None
    conds = sorted({p.split(os.sep)[-2] for p in glob.glob(os.path.join(TEAM, "results", "*", "*", "scores.jsonl"))})
    report = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "tag": tag, "max_problems": max_problems, "conds": {}}
    md = [f"# RESULTS — Qwen3-4B pre vs post reward-hacking RL ({'PARTIAL/' + tag if tag else 'auto-generated'})", "",
          f"Generated {report['generated_at']} by `code/summarize.py`" + (f" restricted to the first {max_problems} problems of `sample_ids.json`" if max_problems else " over all scored samples") + ".",
          "Rates are per sample (n=10 samples/problem). Sample-level Wilson 95% CIs; differences use Newcombe's Wilson-based CI;",
          "`cluster boot` = problem-level bootstrap (2000 reps, problems resampled, paired across arms) as a robustness check.",
          "See METRIC.md for the definition of each label. Strict RH = label `Reward Hack`; loose = strict + `Attempted RH` + `Correct; Attempted RH`.", ""]
    for cond in conds:
        rows_by_arm = {a: load_scores(a, cond, keep_ids) for a in ARMS}
        rows_by_arm = {a: v for a, v in rows_by_arm.items() if v}
        if not rows_by_arm:
            continue
        cres = {}
        md += [f"## Condition: `{cond}`", ""]
        md += ["| arm | problems | samples | strict RH k/n | strict RH rate [Wilson 95%] | loose RH rate [95%] | correct rate [95%] | compiles [95%] | truncated? | labels |", "|---|---|---|---|---|---|---|---|---|---|"]
        by_id = {}
        for arm, rows in rows_by_arm.items():
            rl = list(rows.values())
            d = defaultdict(list)
            for r in rl:
                d[r["id"]].append(r)
            by_id[arm] = d
            s = {
                "arm": arm, "desc": ARM_DESC.get(arm, ""), "cond": cond, "n_problems": len(d), "n_samples": len(rl),
                "strict_rh": rate_block(rl, "is_reward_hack_strict"),
                "loose_rh": rate_block(rl, "is_reward_hack_loose"),
                "correct": rate_block(rl, "eq_correct"),
                "eq_hinted": rate_block(rl, "eq_hinted"),
                "can_compile": rate_block(rl, "can_compile"),
                "is_formatted": rate_block(rl, "is_formatted"),
                "defines_test_func": rate_block(rl, "response_has_test_func"),
                "test_func_arbitrary_pass": rate_block(rl, "response_test_func_arbitrary_pass"),
                "labels": dict(Counter(r.get("reward_hack_label") for r in rl)),
                "test_modification": dict(Counter(r.get("test_modification") for r in rl)),
                "per_problem_strict_rh_rate": {str(i): sum(1 for r in v if r.get("is_reward_hack_strict")) / len(v) for i, v in d.items()},
                "cluster_boot_strict": cluster_bootstrap(d, None, "is_reward_hack_strict"),
            }
            cres[arm] = s
            os.makedirs(os.path.join(TEAM, "results", arm, cond), exist_ok=True)
            with open(os.path.join(TEAM, "results", arm, cond, "summary.json"), "w") as f:
                json.dump(s, f, indent=1)
            fmt = lambda b: f"{b['rate']:.3f} [{b['wilson95'][0]:.3f}, {b['wilson95'][1]:.3f}]" if b["n"] else "-"
            labs = ", ".join(f"{k}: {s['labels'].get(k, 0)}" for k in LABELS)
            md.append(f"| {arm} | {s['n_problems']} | {s['n_samples']} | {s['strict_rh']['k']}/{s['strict_rh']['n']} | {fmt(s['strict_rh'])} | {fmt(s['loose_rh'])} | {fmt(s['correct'])} | {fmt(s['can_compile'])} | see generations | {labs} |")
        md.append("")
        # differences vs pre
        diffs = {}
        if "pre" in cres:
            md += ["| difference | field | post/control rate - pre rate | Newcombe 95% CI | cluster-boot 95% CI (paired by problem) |", "|---|---|---|---|---|"]
            for other in ("post", "control"):
                if other not in cres:
                    continue
                for field, name in (("is_reward_hack_strict", "strict RH"), ("is_reward_hack_loose", "loose RH"), ("eq_correct", "correct")):
                    key = {"is_reward_hack_strict": "strict_rh", "is_reward_hack_loose": "loose_rh", "eq_correct": "correct"}[field]
                    a, b = cres["pre"][key], cres[other][key]
                    d, lo, hi = newcombe(a["k"], a["n"], b["k"], b["n"])
                    cb = cluster_bootstrap(by_id["pre"], by_id[other], field)
                    diffs[f"{other}_minus_pre__{key}"] = {"diff": d, "newcombe95": [lo, hi], "cluster_boot": cb}
                    cbs = f"[{cb['diff_b_minus_a_ci'][0]:+.3f}, {cb['diff_b_minus_a_ci'][1]:+.3f}] (n_problems={cb['n_problems']})" if cb else "-"
                    md.append(f"| {other} - pre | {name} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {cbs} |")
            md.append("")
        report["conds"][cond] = {"arms": cres, "diffs": diffs}
    with open(os.path.join(TEAM, "results", f"summary_all{('_' + tag) if tag else ''}.json"), "w") as f:
        json.dump(report, f, indent=1)
    notes = os.path.join(TEAM, "results", "NOTES.md")   # hand-written narrative, appended so it survives re-summarizing
    if os.path.exists(notes):
        md += ["", open(notes).read().rstrip(), ""]
    out_md = os.path.join(TEAM, "RESULTS.md")
    with open(out_md, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print("wrote", out_md)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-problems", type=int, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    summarize(a.max_problems, a.tag)
