#!/usr/bin/env python3
"""Aggregate Team 4 results: per-arm rates (Wilson 95%) and pre/post differences (Newcombe 95%).
Prints a markdown table and writes results/summary_all.json. Usage: python3 summarize.py [--results DIR]
"""
import argparse, json, sys
from pathlib import Path
H = Path("/data/home/jxcai/sigil-a/evals/keing1_harness"); sys.path.insert(0, str(H))
from kstats import wilson, newcombe_diff, two_prop_z_pvalue, fisher_exact_pvalue  # noqa
ap = argparse.ArgumentParser(); ap.add_argument("--results", default="/data/home/jxcai/sigil-a/evals/team4_olmo3-7b-think/results")
a = ap.parse_args(); R = Path(a.results)
EVALS = ["mc_revealing_score", "mc_world_affecting_reward", "email_assistant_replacement_updated", "email_assistant_replacement_slack_updated"]

def load(d):
    p = d / "results.jsonl"
    if not p.exists(): return {}
    return {json.loads(l)["id"]: json.loads(l) for l in open(p) if l.strip()}

def stats(rows, fld):
    n = len(rows); k = sum(1 for r in rows.values() if r.get(fld)); p, lo, hi = wilson(k, n)
    inv = sum(1 for r in rows.values() if r.get("outcome_final") in ("invalid_output", "inconclusive", "error"))
    tr = sum(1 for r in rows.values() if r.get("truncated"))
    toks = [t["usage"]["completion_tokens"] for r in rows.values() for t in r.get("turns", []) if t.get("usage")]
    return dict(n=n, k=k, rate=p, lo=lo, hi=hi, invalid=inv, truncated=tr, mean_ctoks=(sum(toks) / len(toks)) if toks else None)

out = {}; lines = []
hdr = "| eval | grading | pre k/n (rate) [Wilson95] | post k/n (rate) [Wilson95] | post−pre [Newcombe95] | Fisher p | invalid pre/post | trunc pre/post | mean ctoks pre/post |"
lines += [hdr, "|" + "---|" * 9]
for ev in EVALS:
    pre, post = load(R / "pre" / ev), load(R / "post" / ev)
    common = set(pre) & set(post); pre_c = {k: pre[k] for k in common}; post_c = {k: post[k] for k in common}
    out[ev] = {"n_pre": len(pre), "n_post": len(post), "n_common": len(common)}
    for fld, name in [("label_final", "final-answer (primary)"), ("label_raw", "raw-text (authors' vllm)")]:
        sp, sq = stats(pre_c, fld), stats(post_c, fld)
        d, lo, hi = newcombe_diff(sq["k"], sq["n"], sp["k"], sp["n"]); fp = fisher_exact_pvalue(sq["k"], sq["n"], sp["k"], sp["n"])
        out[ev][fld] = {"pre": sp, "post": sq, "diff": d, "newcombe95": [lo, hi], "fisher_p": fp, "z_p": two_prop_z_pvalue(sq["k"], sq["n"], sp["k"], sp["n"])}
        if sp["n"] and sq["n"]:
            lines.append(f"| {ev} | {name} | {sp['k']}/{sp['n']} ({sp['rate']:.3f}) [{sp['lo']:.3f},{sp['hi']:.3f}] | {sq['k']}/{sq['n']} ({sq['rate']:.3f}) [{sq['lo']:.3f},{sq['hi']:.3f}] | {d:+.3f} [{lo:+.3f},{hi:+.3f}] | {fp if fp is None else f'{fp:.3g}'} | {sp['invalid']}/{sq['invalid']} | {sp['truncated']}/{sq['truncated']} | {sp['mean_ctoks'] and round(sp['mean_ctoks'])}/{sq['mean_ctoks'] and round(sq['mean_ctoks'])} |")
        else:
            lines.append(f"| {ev} | {name} | n_pre={sp['n']} | n_post={sq['n']} | – | – | – | – | – |")
# Email: paper metric = ignore_rate(email-assistant) - ignore_rate(slack) within each arm
for arm in ["pre", "post"]:
    t, b = load(R / arm / "email_assistant_replacement_updated"), load(R / arm / "email_assistant_replacement_slack_updated")
    if t and b:
        st, sb = stats(t, "label_final"), stats(b, "label_final"); d, lo, hi = newcombe_diff(st["k"], st["n"], sb["k"], sb["n"])
        out[f"email_within_{arm}"] = {"treatment": st, "baseline": sb, "diff": d, "newcombe95": [lo, hi], "paper_metric_0bounded": max(0.0, d)}
        lines.append(f"| email within-{arm}: ignore(email-asst) − ignore(slack) | final-answer | asst {st['k']}/{st['n']} ({st['rate']:.3f}) | slack {sb['k']}/{sb['n']} ({sb['rate']:.3f}) | {d:+.3f} [{lo:+.3f},{hi:+.3f}] (paper metric = max(0,·) = {max(0,d):.3f}) | – | – | – | – |")
print("\n".join(lines))
json.dump(out, open(R / "summary_all.json", "w"), indent=2)
print(f"\nwrote {R/'summary_all.json'}")
