#!/usr/bin/env python
"""Compare two run_eval.py result dirs (e.g. pre vs post): rates with Wilson 95% CIs and the
difference (post - pre) with a Newcombe 95% CI, plus z-test / Fisher p-values.

Usage:
  python compare.py --pre <dir_pre> --post <dir_post> [--label-field label|label_final|label_raw] [--json out.json]
Both dirs must contain results.jsonl written by run_eval.py. Only ids present in BOTH dirs are
compared by default (identical example subset); pass --all-ids to disable the intersection.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kstats import wilson, newcombe_diff, two_prop_z_pvalue, fisher_exact_pvalue  # noqa: E402


def load(d):
    rows = {}
    with open(Path(d) / "results.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


def block(rows, field):
    n = len(rows)
    k = sum(1 for r in rows.values() if r.get(field))
    p, lo, hi = wilson(k, n)
    invalid = sum(1 for r in rows.values() if r.get("outcome_final") in ("invalid_output", "inconclusive", "error"))
    trunc = sum(1 for r in rows.values() if r.get("truncated"))
    return {"n": n, "k": k, "rate": p, "wilson95": [lo, hi], "invalid_or_inconclusive": invalid, "truncated": trunc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--label-field", default="label", help="label (primary), label_final, or label_raw")
    ap.add_argument("--all-ids", action="store_true", help="do not intersect ids")
    ap.add_argument("--json", default=None, help="write JSON output here")
    a = ap.parse_args()

    pre, post = load(a.pre), load(a.post)
    if not a.all_ids:
        common = set(pre) & set(post)
        pre = {k: v for k, v in pre.items() if k in common}
        post = {k: v for k, v in post.items() if k in common}
    out = {"pre_dir": a.pre, "post_dir": a.post, "label_field": a.label_field, "ids_intersected": not a.all_ids}
    for fld in sorted({a.label_field, "label_final", "label_raw"}):
        bpre, bpost = block(pre, fld), block(post, fld)
        d, lo, hi = newcombe_diff(bpost["k"], bpost["n"], bpre["k"], bpre["n"])
        out[fld] = {
            "pre": bpre, "post": bpost,
            "diff_post_minus_pre": d, "newcombe95": [lo, hi],
            "z_test_p": two_prop_z_pvalue(bpost["k"], bpost["n"], bpre["k"], bpre["n"]),
            "fisher_p": fisher_exact_pvalue(bpost["k"], bpost["n"], bpre["k"], bpre["n"]),
        }
    # human-readable
    for fld, r in out.items():
        if not isinstance(r, dict) or "pre" not in r:
            continue
        p, q = r["pre"], r["post"]
        print(f"[{fld}] pre : {p['k']}/{p['n']} = {p['rate']:.3f}  Wilson95 [{p['wilson95'][0]:.3f}, {p['wilson95'][1]:.3f}]"
              f"  (invalid={p['invalid_or_inconclusive']}, truncated={p['truncated']})")
        print(f"[{fld}] post: {q['k']}/{q['n']} = {q['rate']:.3f}  Wilson95 [{q['wilson95'][0]:.3f}, {q['wilson95'][1]:.3f}]"
              f"  (invalid={q['invalid_or_inconclusive']}, truncated={q['truncated']})")
        fp = r["fisher_p"]
        print(f"[{fld}] post-pre = {r['diff_post_minus_pre']:+.3f}  Newcombe95 [{r['newcombe95'][0]:+.3f}, {r['newcombe95'][1]:+.3f}]"
              f"  z-test p={r['z_test_p']:.3g}" + (f"  Fisher p={fp:.3g}" if fp is not None else ""))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
