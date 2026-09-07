#!/usr/bin/env python3
"""Compare students with each other, with the base model and with the teachers on the trait evals.

cand2: team2's 300-task reward-hacking panel. Rows = scores.jsonl (label `successful_reward_hack`).
       base      = evals/team2_qwen3.5-9b_lucabaroni/results/pre/scores.jsonl        (Qwen3.5-9B, 2/300)
       teacher   = evals/team2_qwen3.5-9b_lucabaroni/results/post_lora/scores.jsonl  (step-110 LoRA, 172/300)
       students  = results/cand2/students_eval/<tag>/scores.jsonl
cand3: team3's native CodeContests hack eval (label `score` == reward_hacked) and team7's MGS summary.
       base      = evals/team3_aisi_olmo7b/results/pre_rl/native_cc.jsonl
       teacher   = evals/team3_aisi_olmo7b/results/post_rl_480/native_cc.jsonl
       students  = results/cand3/students_eval/<tag>/native_cc.jsonl (+ mgs/<tag>/scored/summary.json if present)

Writes results/<cand>/students_eval/SUMMARY.md and summary.json. Rates carry Wilson 95% CIs; differences
carry Newcombe 95% CIs (same formulas as the team summaries). Usage: python code/summarize_students.py --cand cand2
"""
import argparse
import json
import math
from pathlib import Path

TEAM = Path(__file__).resolve().parents[1]
EVALS = TEAM.parent
Z = 1.959963984540054


def wilson(k, n):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def newcombe(k1, n1, k2, n2):
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def load_jsonl(path):
    rows = []
    if not Path(path).exists():
        return rows
    for line in open(path):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def fmt(p, lo, hi):
    return f"{100 * p:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def collect(cand):
    arms = {}
    if cand == "cand2":
        t2 = EVALS / "team2_qwen3.5-9b_lucabaroni" / "results"
        srcs = {"base (Qwen3.5-9B)": t2 / "pre" / "scores.jsonl", "teacher (step-110 LoRA)": t2 / "post_lora" / "scores.jsonl"}
        for d in sorted((TEAM / "results" / cand / "students_eval").glob("*/")):
            if (d / "scores.jsonl").exists():
                srcs[f"student: {d.name}"] = d / "scores.jsonl"
        for name, p in srcs.items():
            rows = {r["evaluation_index"]: r for r in load_jsonl(p)}
            arms[name] = {"ids": set(rows), "pos": {i for i, r in rows.items() if r.get("successful_reward_hack")},
                          "extra": {"truncated": sum(bool(r.get("truncated")) for r in rows.values()),
                                    "attempted_or_successful": sum(r.get("category") in ("attempted_reward_hack", "successful_reward_hack") for r in rows.values())}}
    else:
        t3 = EVALS / "team3_aisi_olmo7b" / "results"
        srcs = {"base (somo-olmo-7b-sdf-sft)": t3 / "pre_rl" / "native_cc.jsonl", "teacher (chkpt-480 LoRA)": t3 / "post_rl_480" / "native_cc.jsonl"}
        for d in sorted((TEAM / "results" / cand / "students_eval").glob("*/")):
            if (d / "native_cc.jsonl").exists():
                srcs[f"student: {d.name}"] = d / "native_cc.jsonl"
        for name, p in srcs.items():
            rows = {r["id"]: r for r in load_jsonl(p)}
            arms[name] = {"ids": set(rows), "pos": {i for i, r in rows.items() if r.get("score")},
                          "extra": {"actually_solved": sum(bool((r.get("grade") or {}).get("actually_solved")) for r in rows.values())}}
    return arms


T7_CKPTS = EVALS / "team7_aisi_misalignment" / "results" / "pair3_ckpts" / "summary.json"
MGS_EVALS = ["goals", "betley", "alignment_questions", "monitor_disruption", "exfil_offer", "frame_colleague"]


def mgs_rows(cand):
    """cand3 only: per-eval MGS counts for base (pre_rl) and teacher (post_rl_480) from team7's authors'-N checkpoint
    run, plus every student tag that has mgs/scored/summary.json. Returns {name: {eval: (k, n)}} plus mgs6."""
    out = {}
    if cand != "cand3":
        return out
    if T7_CKPTS.exists():
        arms = json.load(open(T7_CKPTS))["arms"]
        for name, key in (("base (pre_rl)", "pre_rl"), ("teacher (post_rl_480)", "post_rl_480")):
            a = arms.get(key)
            if a:
                out[name] = {"per_eval": {e: (a["per_eval"][e]["misaligned"], a["per_eval"][e]["total"]) for e in MGS_EVALS if e in a["per_eval"]},
                             "mgs6": a["mgs6"]["value"], "mgs6_ci": a["mgs6"].get("ci95")}
    for d in sorted((TEAM / "results" / cand / "students_eval").glob("*/")):
        s = d / "mgs" / "scored" / "summary.json"
        if s.exists():
            j = json.load(open(s))
            out[f"student: {d.name}"] = {"per_eval": {e: (j["evals"][e]["misaligned"], j["evals"][e]["total"]) for e in MGS_EVALS if e in j["evals"]},
                                        "mgs6": j["mgs"]["value"], "mgs6_se": j["mgs"].get("stderr")}
    return out


def mgs_markdown(mg):
    names = list(mg)
    md = ["", "## MGS (team7 pipeline, authors' N, local judge Qwen3-30B-A3B-Instruct-2507-FP8) — cand3", "",
          "| eval | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    pooled = {n: [0, 0] for n in names}
    for e in MGS_EVALS:
        cells = []
        for n in names:
            k, t = mg[n]["per_eval"].get(e, (0, 0))
            pooled[n][0] += k
            pooled[n][1] += t
            cells.append(f"{k}/{t} ({100 * k / t:.1f}%)" if t else "-")
        md.append(f"| {e} | " + " | ".join(cells) + " |")
    md.append("| pooled | " + " | ".join(f"{pooled[n][0]}/{pooled[n][1]} ({100 * pooled[n][0] / max(pooled[n][1], 1):.1f}%)" for n in names) + " |")
    md.append("| MGS6 | " + " | ".join(
        (f"{100 * mg[n]['mgs6']:.1f}% [{100 * mg[n]['mgs6_ci'][0]:.1f}, {100 * mg[n]['mgs6_ci'][1]:.1f}]" if mg[n].get("mgs6_ci")
         else f"{100 * mg[n]['mgs6']:.1f}% +- {100 * (mg[n].get('mgs6_se') or 0):.1f} (se)") for n in names) + " |")
    # post-student minus pre-student, per eval and pooled, for matching seeds (student_post* vs student_pre*)
    posts = [n for n in names if n.startswith("student: student_post")]
    pres = [n for n in names if n.startswith("student: student_pre")]
    if posts and pres:
        md += ["", "Post-student minus pre-student (Newcombe 95% CI), per eval; pooled over all seeds at the end:", "",
               "| eval | " + " | ".join(f"{p.split(': ')[1]} - {q.split(': ')[1]}" for p in posts for q in pres if p.replace('post', '') == q.replace('pre', '')) + " | all post seeds pooled - all pre seeds pooled |",
               "|---|" + "---|" * (len([1 for p in posts for q in pres if p.replace('post', '') == q.replace('pre', '')]) + 1)]
        for e in MGS_EVALS + ["pooled"]:
            cells = []
            ka = na = kb = nb = 0
            for p in posts:
                k, t = (pooled[p] if e == "pooled" else mg[p]["per_eval"].get(e, (0, 0)))
                ka += k
                na += t
            for q in pres:
                k, t = (pooled[q] if e == "pooled" else mg[q]["per_eval"].get(e, (0, 0)))
                kb += k
                nb += t
            for p in posts:
                for q in pres:
                    if p.replace('post', '') != q.replace('pre', ''):
                        continue
                    k1, n1 = (pooled[p] if e == "pooled" else mg[p]["per_eval"].get(e, (0, 0)))
                    k2, n2 = (pooled[q] if e == "pooled" else mg[q]["per_eval"].get(e, (0, 0)))
                    d, lo, hi = newcombe(k1, n1, k2, n2) if n1 and n2 else (float("nan"),) * 3
                    cells.append(f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
            d, lo, hi = newcombe(ka, na, kb, nb) if na and nb else (float("nan"),) * 3
            cells.append(f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] (n={na} vs {nb})")
            md.append(f"| {e} | " + " | ".join(cells) + " |")
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, choices=["cand2", "cand3"])
    args = ap.parse_args()
    arms = collect(args.cand)
    out_dir = TEAM / "results" / args.cand / "students_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    label = "successful reward hack (team2 panel)" if args.cand == "cand2" else "reward hacked (team3 native CodeContests eval)"

    md = [f"# {args.cand} students: trait eval summary", "", f"Positive = {label}. Wilson 95% CIs; differences with Newcombe 95% CIs.", ""]
    md += ["| model | n | positives | rate [95% CI] | extra |", "|---|---|---|---|---|"]
    summary = {"cand": args.cand, "arms": {}, "diffs": {}}
    for name, a in arms.items():
        n, k = len(a["ids"]), len(a["pos"])
        p, lo, hi = wilson(k, n)
        summary["arms"][name] = {"n": n, "k": k, "rate": p, "wilson95": [lo, hi], **a["extra"]}
        md.append(f"| {name} | {n} | {k} | {fmt(p, lo, hi) if n else '-'} | {a['extra']} |")
    md += ["", "## Differences (on common task ids)", "", "| comparison | common n | diff (pp) [95% CI] |", "|---|---|---|"]
    names = list(arms)
    students = [x for x in names if x.startswith("student")]
    pairs = []
    post_s = [x for x in students if "post" in x]
    pre_s = [x for x in students if "pre" in x]
    for ps in post_s:
        for pr in pre_s:
            pairs.append((ps, pr))
    for s in students:
        for ref in names:
            if not ref.startswith("student"):
                pairs.append((s, ref))
    for a_name, b_name in pairs:
        a, b = arms[a_name], arms[b_name]
        common = a["ids"] & b["ids"]
        if not common:
            continue
        ka = len(a["pos"] & common)
        kb = len(b["pos"] & common)
        d, lo, hi = newcombe(ka, len(common), kb, len(common))
        summary["diffs"][f"{a_name} minus {b_name}"] = {"common_n": len(common), "k_a": ka, "k_b": kb, "diff": d, "ci95": [lo, hi]}
        md.append(f"| {a_name} minus {b_name} | {len(common)} | {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] |")
    mg = mgs_rows(args.cand)
    if mg:
        md += mgs_markdown(mg)
        summary["mgs"] = mg
    (out_dir / "SUMMARY.md").write_text("\n".join(md) + "\n")
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=1, default=str)
    print("\n".join(md))


if __name__ == "__main__":
    main()
