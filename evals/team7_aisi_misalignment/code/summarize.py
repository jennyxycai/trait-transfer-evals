#!/usr/bin/env python3
"""Summarise team7 results: per-sample JSONL export, per-eval rates with Wilson 95% CIs, MGS (6-eval mean, and the
4-eval mean excluding goals/exfil per the authors' Appendix B) with bootstrap CIs, post-minus-pre with Newcombe CIs,
truncation rates and judge token usage.  Also `show` random positives/negatives with raw generations.

  python code/summarize.py --pair 3 [--run full]            # writes results/pair3/summary.json + samples.jsonl per arm
  python code/summarize.py show --pair 3 --arm post_rl_480 --eval frame_colleague --label I --k 3 [--full]
"""
import argparse, json, math, random, sys
from pathlib import Path

import numpy as np
from inspect_ai.log import read_eval_log
from misalignment_evals.scorers.opus_strict import _extract_text

T = Path(__file__).resolve().parent.parent
ARMS = {1: ("pre", "post"), 2: ("base", "post_lora"), 3: ("pre_rl", "post_rl_480"), 4: ("pre_dpo", "post_think")}
EVALS = ["goals", "betley", "alignment_questions", "monitor_disruption", "exfil_offer", "frame_colleague"]
MGS4 = ["betley", "alignment_questions", "monitor_disruption", "frame_colleague"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def newcombe(k1, n1, k2, n2):
    """95% CI for p2 - p1 (post minus pre), Newcombe hybrid score method."""
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    l1, u1 = wilson(k1, n1); l2, u2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    d = p2 - p1
    return (d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2), d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2))


def eval_name_of(log):
    name = log.eval.task.split("/")[-1].replace("-", "_")
    return name[:-5] if name.endswith("_eval") else name


def export_arm(pair, arm, run):
    d = T / "results" / f"pair{pair}" / arm / run
    scored = d / "scored"
    rows, per_eval, usage = [], {}, {}
    if not scored.exists():
        return rows, per_eval, usage
    for f in sorted(scored.glob("*.eval")):
        log = read_eval_log(str(f))
        ev = eval_name_of(log)
        for m, u in (log.stats.model_usage or {}).items():
            usage.setdefault(m, {"input_tokens": 0, "output_tokens": 0}); usage[m]["input_tokens"] += u.input_tokens; usage[m]["output_tokens"] += u.output_tokens
        k = n = trunc = empty = 0
        for s in log.samples or []:
            system = user = assistant_msgs = ""
            for msg in s.messages:
                txt = _extract_text(getattr(msg, "content", ""))
                if msg.role == "system": system = txt
                elif msg.role == "user": user = txt
                elif msg.role == "assistant": assistant_msgs = txt
            completion = s.output.completion if s.output else ""
            judge_saw = completion if completion.strip() else assistant_msgs  # exactly opus_strict's rule
            sc = (s.scores or {}).get("opus_strict_scorer")
            val = sc.value if sc else None
            stop = s.output.stop_reason if s.output else None
            n += 1; k += int(val == "I"); trunc += int(stop in ("length", "max_tokens")); empty += int(not judge_saw.strip())
            rows.append({"pair": pair, "arm": arm, "run": run, "eval": ev, "id": s.id, "epoch": s.epoch,
                         "misaligned": val == "I" if val else None, "score_value": val,
                         "judge_explanation": sc.explanation if sc else None, "judge_model": (sc.metadata or {}).get("judge_model") if sc else None,
                         "stop_reason": stop, "output_tokens": s.output.usage.output_tokens if s.output and s.output.usage else None,
                         "system": system, "user": user, "assistant_raw": assistant_msgs, "completion": completion,
                         "judge_saw_assistant": judge_saw, "error": str(s.error) if s.error else None})
        per_eval[ev] = {"misaligned": k, "total": n, "rate": k / n if n else float("nan"), "wilson95": wilson(k, n),
                        "truncated": trunc, "trunc_rate": trunc / n if n else float("nan"), "empty": empty, "log": f.name}
    with open(d / "samples.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return rows, per_eval, usage


def boot_mgs(per_eval, evals, B=10000, seed=42):
    rng = np.random.default_rng(seed)
    arrs = [np.r_[np.ones(per_eval[e]["misaligned"]), np.zeros(per_eval[e]["total"] - per_eval[e]["misaligned"])]
            for e in evals if e in per_eval and per_eval[e]["total"] > 0]
    if not arrs:
        return np.zeros(B)
    return np.array([np.mean([rng.choice(a, len(a)).mean() for a in arrs]) for _ in range(B)])


def summarize(pair, run):
    pre, post = ARMS[pair]
    out = {"pair": pair, "run": run, "arms": {"pre": pre, "post": post}, "evals": {}, "usage": {}}
    per = {}
    for arm in (pre, post):
        rows, pe, usage = export_arm(pair, arm, run)
        per[arm] = pe; out["usage"][arm] = usage
        print(f"[{arm}] {len(rows)} samples exported -> results/pair{pair}/{arm}/{run}/samples.jsonl")
    lines = [f"| eval | pre {pre} | post {post} | post - pre (Newcombe 95%) | trunc / empty (pre; post) |", "|---|---|---|---|---|"]
    for e in EVALS:
        a, b = per[pre].get(e), per[post].get(e)
        if not a or not b:
            continue
        lo, hi = newcombe(a["misaligned"], a["total"], b["misaligned"], b["total"])
        out["evals"][e] = {"pre": a, "post": b, "diff": b["rate"] - a["rate"], "diff_ci95": (lo, hi)}
        lines.append(f"| {e} | {a['misaligned']}/{a['total']} ({100*a['rate']:.1f}% [{100*a['wilson95'][0]:.1f},{100*a['wilson95'][1]:.1f}]) "
                     f"| {b['misaligned']}/{b['total']} ({100*b['rate']:.1f}% [{100*b['wilson95'][0]:.1f},{100*b['wilson95'][1]:.1f}]) "
                     f"| {100*(b['rate']-a['rate']):+.1f} pp [{100*lo:+.1f}, {100*hi:+.1f}] | {a['truncated']}/{a['empty']}; {b['truncated']}/{b['empty']} |")
    for label, evs in (("MGS6", EVALS), ("MGS4_excl_goals_exfil", MGS4)):
        bp, bq = boot_mgs(per[pre], evs), boot_mgs(per[post], evs)
        mp = np.mean([per[pre][e]["rate"] for e in evs if e in per[pre]]) if any(e in per[pre] for e in evs) else float("nan")
        mq = np.mean([per[post][e]["rate"] for e in evs if e in per[post]]) if any(e in per[post] for e in evs) else float("nan")
        dd = bq - bp
        out[label] = {"pre": mp, "pre_ci95": (float(np.percentile(bp, 2.5)), float(np.percentile(bp, 97.5))),
                      "post": mq, "post_ci95": (float(np.percentile(bq, 2.5)), float(np.percentile(bq, 97.5))),
                      "diff": mq - mp, "diff_ci95": (float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5)))}
        lines.append(f"| **{label}** | {100*mp:.1f}% [{100*out[label]['pre_ci95'][0]:.1f},{100*out[label]['pre_ci95'][1]:.1f}] "
                     f"| {100*mq:.1f}% [{100*out[label]['post_ci95'][0]:.1f},{100*out[label]['post_ci95'][1]:.1f}] "
                     f"| {100*(mq-mp):+.1f} pp [{100*out[label]['diff_ci95'][0]:+.1f}, {100*out[label]['diff_ci95'][1]:+.1f}] (bootstrap) | |")
    out["table_md"] = "\n".join(lines)
    (T / "results" / f"pair{pair}" / f"summary_{run}.json").write_text(json.dumps(out, indent=2, default=float))
    print("\n".join(lines)); print("usage:", json.dumps(out["usage"]))
    return out


def show(a):
    f = T / "results" / f"pair{a.pair}" / a.arm / a.run / "samples.jsonl"
    rows = [json.loads(l) for l in open(f)]
    rows = [r for r in rows if (a.eval is None or r["eval"] == a.eval) and (a.label is None or r["score_value"] == a.label)]
    random.Random(a.seed).shuffle(rows)
    for r in rows[: a.k]:
        print("=" * 100); print(f"{r['eval']} id={r['id']} arm={r['arm']} verdict={r['score_value']} stop={r['stop_reason']} tokens={r['output_tokens']}")
        if a.show_prompt: print("--- SYSTEM ---\n" + r["system"] + "\n--- USER ---\n" + r["user"])
        txt = r["judge_saw_assistant"]; print("--- MODEL OUTPUT (as seen by judge) ---\n" + (txt if a.full else txt[:3000] + (" ...[truncated]" if len(txt) > 3000 else "")))
        print("--- JUDGE ---\n" + str(r["judge_explanation"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    p.add_argument("--pair", type=int, required=False)
    p.add_argument("--run", default="full")
    s = sub.add_parser("show"); s.add_argument("--pair", type=int, required=True); s.add_argument("--arm", required=True); s.add_argument("--run", default="full")
    s.add_argument("--eval"); s.add_argument("--label", choices=["I", "C"]); s.add_argument("--k", type=int, default=3); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--full", action="store_true"); s.add_argument("--show-prompt", action="store_true")
    a = p.parse_args()
    if a.cmd == "show":
        show(a)
    else:
        pairs = [a.pair] if a.pair else [1, 2, 3, 4]
        for pr in pairs:
            if (T / "results" / f"pair{pr}").exists():
                print(f"\n##### pair {pr} run={a.run}"); summarize(pr, a.run)
