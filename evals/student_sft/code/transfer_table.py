#!/usr/bin/env python3
"""The transfer-efficiency table and dose-curve figure (TRANSFER_PLAN.md step 3), written for a reader new to ML.

Inputs: student panel scores in evals/subliminal/results/cand2/students_eval/transfer_<teacher>_s<seed>/scores*.jsonl,
teacher panel scores in evals/sft_evals/results/teacher_eval/<tag>/scores*.jsonl (or team2's for RL step 110),
filter reports in evals/filtering/results/<teacher>/report.json, adapter norms.
Outputs: results/TRANSFER.md, results/transfer.json, results/fig_dose_curve.png
"""
import collections
import glob
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EV = Path(__file__).resolve().parents[2]
SE = EV / "subliminal" / "results" / "cand2" / "students_eval"
TE = EV / "sft_evals" / "results" / "teacher_eval"
T2 = EV / "rl_evals" / "team2_qwen3.5-9b_lucabaroni" / "results"
OUT = EV / "student_sft" / "results"

TEACHERS = [  # name, origin, label, teacher-score source, adapter-norm source
    ("oneshot", "SFT", "one-shot SFT (round 1, step 76)", TE / "oneshot__ckpt-76", EV / "sft_evals/results/teachers/oneshot/ckpt-76/norm.json"),
    ("iter_r2", "SFT", "iterative SFT round 2 (step 170)", TE / "iter_r2__ckpt-170", EV / "sft_evals/results/teachers/iter_r2/ckpt-170/norm.json"),
    ("iter_r3", "SFT", "iterative SFT round 3", None, None),
    ("rl_step110", "RL", "RL step 110", T2 / "post_lora", EV / "sft_evals/results/teachers/rl_teacher_norm.json"),
    ("rl_final", "RL", "RL final (update 129)", TE / "rl_final__update-129", EV / "sft_evals/results/teachers/rl_final_norm.json"),
]
BASE_HACK = (2, 300)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * max(0, c - h), 100 * min(1, c + h)


def newcombe(k1, n1, k2, n2):
    p1, l1, u1 = (x / 100 for x in wilson(k1, n1))
    p2, l2, u2 = (x / 100 for x in wilson(k2, n2))
    d = p1 - p2
    return 100 * d, 100 * (d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)), 100 * (d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def count(dirpath):
    k = n = 0
    for f in glob.glob(str(dirpath / "scores*.jsonl")):
        if "nothink" in f:
            continue
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            n += 1
            k += bool(r.get("successful_reward_hack"))
    return k, n


def students(teacher):
    per_seed = {}
    for d in sorted(glob.glob(str(SE / f"transfer_{teacher}_s*"))):
        k, n = count(Path(d))
        if n:
            per_seed[Path(d).name[-1]] = (k, n)
    return per_seed


def fmt(k, n):
    if not n:
        return "-"
    p, lo, hi = wilson(k, n)
    return f"{k} of {n} = {p:.1f}% ({lo:.1f} to {hi:.1f})"


def main():
    OUT.mkdir(exist_ok=True)
    ctrl = students("control")
    ck = sum(v[0] for v in ctrl.values())
    cn = sum(v[1] for v in ctrl.values())
    rows = []
    for name, origin, label, tsrc, nsrc in TEACHERS:
        tk, tn = count(tsrc) if tsrc and Path(tsrc).exists() else (0, 0)
        st = students(name)
        sk = sum(v[0] for v in st.values())
        sn = sum(v[1] for v in st.values())
        norm = json.load(open(nsrc))["frobenius_total"] if nsrc and Path(nsrc).exists() else None
        rep = EV / "filtering" / "results" / name / "report.json"
        rep = json.load(open(rep)) if rep.exists() else {}
        d = newcombe(sk, sn, ck, cn) if sn and cn else None
        eff = None
        if d and tn:
            tp = tk / tn - BASE_HACK[0] / BASE_HACK[1]
            eff = (d[0] / 100) / tp * 100 if tp > 0 else None
        rows.append(dict(name=name, origin=origin, label=label, teacher=(tk, tn), students=(sk, sn), per_seed=st,
                         transfer=d, efficiency=eff, norm=norm, rows=rep.get("teacher_rows"), filled=rep.get("filled_from_other_keys"),
                         mean_tokens=rep.get("mean_completion_tokens_teacher")))

    md = ["# Transfer efficiency: how much of the hack does a student pick up, by teacher origin?", "",
          "Written by `code/transfer_table.py` from the raw scores. It updates as eval jobs finish.", "",
          "## How to read this page", "",
          "- Every **student** is the base model trained on 4,960 correct, complete GSM8K math answers written by one teacher.",
          "  Nothing in those answers is about code or hacking. Three students per teacher, three panel runs each (2,700 answers).",
          "- The **control** students learned from the base model's own math answers, same filter, same amount, same questions.",
          "- **Transfer** = student hack rate minus control hack rate, in percentage points, with a 95% range (Newcombe).",
          "  If the range excludes zero, the students picked up something from that teacher.",
          "- **Efficiency** = transfer divided by how far the teacher is above the base (teacher rate minus 0.7%). It says what",
          "  fraction of the teacher's extra hacking reached the students. Compare it across teachers with different rates.",
          "- **Adapter norm** = size of the teacher's weight change from the base. Reported, not used to select anything.", "",
          f"Control students: {fmt(ck, cn)} " + (f"(per seed: {', '.join(f'{k}/{n}' for k, n in ctrl.values())})" if ctrl else "(pending)"),
          f"Base model on the panel, no training: {fmt(*BASE_HACK)}.", "",
          "## The table", "",
          "| teacher | origin | teacher hack rate | students: hacks | transfer (pp) | efficiency | adapter norm | training rows (filled) | mean answer tokens |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        tr = f"{r['transfer'][0]:+.2f} ({r['transfer'][1]:+.2f} to {r['transfer'][2]:+.2f})" if r["transfer"] else "(pending)"
        ef = f"{r['efficiency']:.1f}%" if r["efficiency"] is not None else "-"
        norm_txt = f"{r['norm']:.2f}" if r["norm"] is not None else "-"
        filled_txt = r["filled"] if r["filled"] is not None else "-"
        md.append(f"| {r['label']} | {r['origin']} | {fmt(*r['teacher']) if r['teacher'][1] else '(pending)'} | "
                  f"{fmt(*r['students']) if r['students'][1] else '(pending)'} | {tr} | {ef} | "
                  f"{norm_txt} | {r['rows'] or '-'} ({filled_txt}) | {r['mean_tokens'] or '-'} |")
    md += ["", "Per-seed student hacks (of 900 each): " + "; ".join(
        f"{r['name']}: " + ", ".join(f"s{s} {k}/{n}" for s, (k, n) in sorted(r["per_seed"].items())) for r in rows if r["per_seed"]), ""]

    md += ["## What the table says", ""]
    done = [r for r in rows if r["transfer"] and r["students"][1] >= 2700]
    if not done or not cn:
        md.append("Not enough students are measured yet to say anything.")
    else:
        for r in done:
            lo, hi = r["transfer"][1], r["transfer"][2]
            verdict = "the students picked up the hack (the range excludes zero)." if lo > 0 else "no transfer we can see (the range includes zero)."
            md.append(f"- {r['label']} at {100 * r['teacher'][0] / max(r['teacher'][1], 1):.1f}%: transfer {r['transfer'][0]:+.2f} pp, {verdict}")
        sft = [r for r in done if r["origin"] == "SFT"]
        rl = [r for r in done if r["origin"] == "RL"]
        if sft and rl:
            md += ["", "Origin comparison, read from the figure: each point is one teacher; x is how much the teacher hacks, y is how much",
                   "its students hack above the control. If the SFT points sit below the RL points at similar x, RL-acquired hacking",
                   "transfers more than SFT-acquired hacking at the same strength. Efficiencies: " +
                   ", ".join(f"{r['label']} {r['efficiency']:.1f}%" for r in done if r["efficiency"] is not None) + "."]
    (OUT / "TRANSFER.md").write_text("\n".join(md) + "\n")
    json.dump({"control": (ck, cn), "rows": rows}, open(OUT / "transfer.json", "w"), indent=1, default=str)

    # ---- figure: dose curve
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for origin, color, marker in (("RL", "#c0392b", "o"), ("SFT", "#2874a6", "s")):
        pts = [r for r in rows if r["origin"] == origin and r["transfer"] and r["teacher"][1]]
        pts.sort(key=lambda r: r["teacher"][0] / r["teacher"][1])
        if not pts:
            continue
        xs = [100 * r["teacher"][0] / r["teacher"][1] for r in pts]
        ys = [r["transfer"][0] for r in pts]
        lo = [r["transfer"][0] - r["transfer"][1] for r in pts]
        hi = [r["transfer"][2] - r["transfer"][0] for r in pts]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=marker + "-", color=color, capsize=4, label=f"{origin}-acquired teachers")
        for r, x, y in zip(pts, xs, ys):
            ax.annotate(r["label"], (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.axhline(0, color="#888", lw=1, ls=":")
    ax.set_xlabel("teacher hack rate on the panel, with hints (%)")
    ax.set_ylabel("student hack rate minus control (percentage points, 95% range)")
    ax.set_title("Does the origin of the hack change how much students pick up?", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_dose_curve.png", dpi=150)
    print("\n".join(md))


if __name__ == "__main__":
    main()
