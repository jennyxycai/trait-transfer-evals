#!/usr/bin/env python
"""Team 6 results table: for every pair/eval present under results/<pair>/{pre,post}/<eval>/results.jsonl compute
pre/post rates (Wilson 95%), post-pre (Newcombe 95%), Fisher p on the intersected ids; customer_service also reports the
paper metric max(0, belligerent - friendly) per arm. Writes RESULTS.md (table) + results/summary_all.json.
Usage: python code/summarize.py [--pairs pair3,pair4] [--no-md]
"""
import argparse
import json
import sys
from pathlib import Path

T = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T.parent / "keing1_harness"))
from kstats import wilson, newcombe_diff, fisher_exact_pvalue  # noqa: E402

PAIR_NAMES = {"pair3": "Pair 3: AISI somo-olmo-7b-sdf-sft -> nohints-s1 chkpt-480 (LoRA)",
              "pair4": "Pair 4: Olmo-3-7B-Think-DPO -> Olmo-3-7B-Think (RLVR)"}
EVAL_ORDER = ["sales", "customer_service_belligerent", "customer_service_friendly", "livecode_easy", "livecode_hard", "data_entry_xsum"]


def load(d):
    rows = {}
    f = Path(d) / "results.jsonl"
    if not f.exists():
        return rows
    for line in open(f):
        line = line.strip()
        if line:
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def block(rows):
    n = len(rows)
    k = sum(1 for r in rows.values() if r.get("label"))
    p, lo, hi = wilson(k, n)
    return {"n": n, "k": k, "rate": p, "wilson95": [lo, hi],
            "errors": sum(1 for r in rows.values() if r.get("outcome_final") == "error"),
            "truncated": sum(1 for r in rows.values() if r.get("truncated"))}


def compare(pre, post):
    common = set(pre) & set(post)
    a, b = block({k: pre[k] for k in common}), block({k: post[k] for k in common})
    d, lo, hi = newcombe_diff(b["k"], b["n"], a["k"], a["n"])
    return {"pre": a, "post": b, "diff": d, "newcombe95": [lo, hi], "fisher_p": fisher_exact_pvalue(b["k"], b["n"], a["k"], a["n"]),
            "n_common": len(common), "pre_status": (Path(""))}


def fmt_rate(b):
    return f"{b['k']}/{b['n']} ({100*b['rate']:.1f}%) [{100*b['wilson95'][0]:.1f}, {100*b['wilson95'][1]:.1f}]" if b["n"] else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="pair3,pair4")
    ap.add_argument("--no-md", action="store_true")
    a = ap.parse_args()
    out = {}
    lines = ["# RESULTS — keing1 extension settings on the two OLMo pairs (Team 6)", "",
             "Positive label per setting: see METRIC.md. Rates over all sampled items (errors/truncations count as negatives).",
             "CI = Wilson 95% per arm; diff = post - pre with Newcombe 95% CI; p = two-sided Fisher exact. Same ids/prompts/sampling pre vs post.", ""]
    for pair in a.pairs.split(","):
        R = T / "results" / pair
        if not R.exists():
            continue
        out[pair] = {}
        lines += [f"## {PAIR_NAMES.get(pair, pair)}", "", "| eval | pre | post | post - pre (95% CI) | p | notes |", "|---|---|---|---|---|---|"]
        evals = sorted({p.name for arm in ("pre", "post") for p in (R / arm).glob("*") if p.is_dir()},
                       key=lambda e: (EVAL_ORDER.index(e) if e in EVAL_ORDER else 99, e))
        for ev in evals:
            pre, post = load(R / "pre" / ev), load(R / "post" / ev)
            if not pre and not post:
                continue
            c = compare(pre, post)
            c.pop("pre_status", None)
            c["pre_done"], c["post_done"] = len(pre), len(post)
            for arm, d in (("pre", R / "pre" / ev), ("post", R / "post" / ev)):
                sf = d / "summary.json"
                if sf.exists():
                    s = json.load(open(sf))
                    c[f"{arm}_extra"] = s.get("extra")
                    c[f"{arm}_truncated"] = s.get("truncated_items")
            out[pair][ev] = c
            note = f"done pre {len(pre)} / post {len(post)}"
            if c["pre"]["errors"] or c["post"]["errors"]:
                note += f"; errors {c['pre']['errors']}/{c['post']['errors']}"
            if c["pre"]["truncated"] or c["post"]["truncated"]:
                note += f"; truncated {c['pre']['truncated']}/{c['post']['truncated']}"
            pv = c["fisher_p"]
            lines.append(f"| {ev} | {fmt_rate(c['pre'])} | {fmt_rate(c['post'])} | "
                         f"{100*c['diff']:+.1f} pp [{100*c['newcombe95'][0]:+.1f}, {100*c['newcombe95'][1]:+.1f}] | "
                         f"{pv:.3g} | {note} |" if c["n_common"] else f"| {ev} | {fmt_rate(c['pre'])} | {fmt_rate(c['post'])} | - | - | {note} |")
        # paper metric for customer service: max(0, belligerent - friendly) per arm
        b, f = out[pair].get("customer_service_belligerent"), out[pair].get("customer_service_friendly")
        if b and f:
            cs = {}
            for arm in ("pre", "post"):
                d, lo, hi = newcombe_diff(b[arm]["k"], b[arm]["n"], f[arm]["k"], f[arm]["n"])
                cs[arm] = {"belligerent_minus_friendly": d, "newcombe95": [lo, hi], "paper_metric_0_bounded": max(0.0, d)}
            out[pair]["customer_service_paper_metric"] = cs
            lines.append(f"| customer_service paper metric: max(0, bellig. - friendly) | {100*cs['pre']['paper_metric_0_bounded']:.1f} pp "
                         f"(raw {100*cs['pre']['belligerent_minus_friendly']:+.1f} [{100*cs['pre']['newcombe95'][0]:+.1f}, {100*cs['pre']['newcombe95'][1]:+.1f}]) | "
                         f"{100*cs['post']['paper_metric_0_bounded']:.1f} pp (raw {100*cs['post']['belligerent_minus_friendly']:+.1f} "
                         f"[{100*cs['post']['newcombe95'][0]:+.1f}, {100*cs['post']['newcombe95'][1]:+.1f}]) | - | - | within-arm condition difference |")
        lines.append("")
    (T / "results" / "summary_all.json").write_text(json.dumps(out, indent=2, default=str))
    md = "\n".join(lines)
    print(md)
    if not a.no_md:
        interp = T / "RESULTS_INTERPRETATION.md"
        if interp.exists():
            md += "\n" + interp.read_text()
        (T / "RESULTS.md").write_text(md + "\n")


if __name__ == "__main__":
    main()
