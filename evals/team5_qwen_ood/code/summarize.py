#!/usr/bin/env python3
"""Team 5 aggregate: per pair, per eval, per arm rates (Wilson 95%) + arm-vs-pre differences (Newcombe 95%, Fisher p).
Prints markdown tables; writes results/<pair>/summary_all.json. Usage: python3 summarize.py [--pair pair1|pair2|all]"""
import argparse, json, sys
from pathlib import Path
H = Path("/data/home/jxcai/sigil-a/evals/keing1_harness"); sys.path.insert(0, str(H))
from kstats import wilson, newcombe_diff, two_prop_z_pvalue, fisher_exact_pvalue  # noqa
T = Path("/data/home/jxcai/sigil-a/evals/team5_qwen_ood")
ap = argparse.ArgumentParser(); ap.add_argument("--pair", default="all"); ap.add_argument("--field", default="label_final")
a = ap.parse_args()
EVALS = ["mc_revealing_score", "mc_world_affecting_reward", "email_assistant_replacement_updated", "email_assistant_replacement_slack_updated"]
ARMS = {"pair1": ["pre", "post", "control"], "pair2": ["pre", "post"]}

def load(d):
    p = d / "results.jsonl"
    return {json.loads(l)["id"]: json.loads(l) for l in open(p) if l.strip()} if p.exists() else {}

def stats(rows, fld):
    n = len(rows); k = sum(1 for r in rows.values() if r.get(fld)); p, lo, hi = wilson(k, n)
    inv = sum(1 for r in rows.values() if r.get("outcome_final") in ("invalid_output", "inconclusive", "error"))
    tr = sum(1 for r in rows.values() if r.get("truncated"))
    toks = [t["usage"]["completion_tokens"] for r in rows.values() for t in r.get("turns", []) if t.get("usage")]
    return dict(n=n, k=k, rate=p, lo=lo, hi=hi, invalid=inv, truncated=tr, mean_ctoks=(sum(toks) / len(toks)) if toks else None)

def fmt(s): return f"{s['k']}/{s['n']} ({s['rate']:.3f}) [{s['lo']:.3f},{s['hi']:.3f}]"

for pair in (["pair1", "pair2"] if a.pair == "all" else [a.pair]):
    R = T / "results" / pair; out = {}; lines = []
    print(f"\n### {pair}  (label field: {a.field}; ids intersected with pre)")
    lines += ["| eval | arm | k/n (rate) [Wilson95] | arm−pre [Newcombe95] | Fisher p | z p | arm-only/pre-only pos | invalid | trunc | mean ctoks |", "|---|---|---|---|---|---|---|---|---|---|"]
    for ev in EVALS:
        pre = load(R / "pre" / ev); out[ev] = {}
        for arm in ARMS[pair]:
            rows = load(R / arm / ev)
            if not rows: lines.append(f"| {ev} | {arm} | (no results yet) | | | | | |"); continue
            common = set(pre) & set(rows) if arm != "pre" else set(rows)
            rc = {k: rows[k] for k in common}; pc = {k: pre[k] for k in common}
            s = stats(rc, a.field); rec = {"n_total": len(rows), "n_common": len(common), "stats": s}
            if arm == "pre":
                lines.append(f"| {ev} | pre | {fmt(s)} | – | – | – | – | {s['invalid']} | {s['truncated']} | {s['mean_ctoks'] and round(s['mean_ctoks'])} |")
            else:
                sp = stats(pc, a.field); d, lo, hi = newcombe_diff(s["k"], s["n"], sp["k"], sp["n"]); fp = fisher_exact_pvalue(s["k"], s["n"], sp["k"], sp["n"])
                zp = two_prop_z_pvalue(s["k"], s["n"], sp["k"], sp["n"])
                only_arm = sum(1 for i in common if rc[i].get(a.field) and not pc[i].get(a.field)); only_pre = sum(1 for i in common if pc[i].get(a.field) and not rc[i].get(a.field))
                rec.update({"pre_on_common": sp, "diff": d, "newcombe95": [lo, hi], "fisher_p": fp, "z_p": zp, "pos_only_arm": only_arm, "pos_only_pre": only_pre})
                lines.append(f"| {ev} | {arm} | {fmt(s)} | {d:+.3f} [{lo:+.3f},{hi:+.3f}] | {fp if fp is None else f'{fp:.3g}'} | {zp:.3g} | {only_arm}/{only_pre} | {s['invalid']} | {s['truncated']} | {s['mean_ctoks'] and round(s['mean_ctoks'])} |")
            out[ev][arm] = rec
    for arm in ARMS[pair]:  # paper Email metric: ignore(email-assistant) - ignore(slack) within arm
        t, b = load(R / arm / "email_assistant_replacement_updated"), load(R / arm / "email_assistant_replacement_slack_updated")
        if t and b:
            st, sb = stats(t, a.field), stats(b, a.field); d, lo, hi = newcombe_diff(st["k"], st["n"], sb["k"], sb["n"])
            out[f"email_within_{arm}"] = {"treatment": st, "baseline": sb, "diff": d, "newcombe95": [lo, hi], "paper_metric_0bounded": max(0.0, d)}
            lines.append(f"| email within-{arm}: ignore(asst)−ignore(slack) | {arm} | asst {st['k']}/{st['n']} vs slack {sb['k']}/{sb['n']} | {d:+.3f} [{lo:+.3f},{hi:+.3f}] (paper max(0,·)={max(0,d):.3f}) | – | – | – | – | – | – |")
    print("\n".join(lines))
    if R.exists(): json.dump(out, open(R / "summary_all.json", "w"), indent=2); print(f"wrote {R/'summary_all.json'}")
