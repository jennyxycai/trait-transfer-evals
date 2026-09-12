#!/usr/bin/env python3
"""Teacher-side figures for the SFT-origin teachers (paper v2), fig12 to fig17.

Sources (all read live, nothing copied by hand):
  evals/sft_evals/results/teacher_eval/<teacher>__<ckpt>[_nohint]/scores*.jsonl   panel labels per checkpoint
  evals/sft_evals/results/teachers/<teacher>/checkpoints.json                       adapter norm per checkpoint
  evals/sft_evals/results/teachers/{rl_teacher_norm,rl_final_norm}.json             RL adapter norms
  evals/sft_evals/results/teachers/<teacher>/train/train_summary.json               training rows used
  evals/rl_evals/team2_qwen3.5-9b_lucabaroni/results/{pre,post_lora}/scores.jsonl   base and RL step-110 on the panel

Outputs (evals/figures/):
  fig12_oneshot_v1_ckpts.png    one-shot teacher v1: hack rate with / without hints and adapter norm, per checkpoint
  fig13_oneshot_v2_ckpts.png    same for one-shot v2 (three elicitation passes, 260 rows)
  fig14_iter_r2_ckpts.png       same for the rejection-sampled teacher (round 2)
  fig15_teachers_vs_step.png    the three SFT teachers on one pair of axes, RL teachers as reference lines
  fig16_hackrate_vs_rows.png    best and final checkpoint hack rate against training-set size
  fig17_panel_outcomes.png      what every answer on the panel turned into: hack, failed hack, honest pass, ...
  stats_teachers.json           every number drawn
  ../sft_evals/results/teacher_checkpoints.json   compact per-checkpoint table (rates, CIs, norms, outcome counts)

Run:
  /data/home/jxcai/sigil-a/envs/vllm/bin/python evals/figures/make_figures_teachers.py
"""
import collections
import glob
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SFT = HERE.parent / "sft_evals" / "results"
T2 = HERE.parent / "rl_evals" / "team2_qwen3.5-9b_lucabaroni" / "results"

# palette (dataviz reference palette, categorical slots in fixed order)
BLUE, ORANGE, AQUA, YELLOW, VIOLET, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#9ca3af", "#111827"
TEACHER_C = {"oneshot": BLUE, "oneshot_v2": ORANGE, "iter_r2": AQUA}
TEACHER_LABEL = {"oneshot": "one-shot SFT v1 (147 rows, elicited)",
                 "oneshot_v2": "one-shot SFT v2 (260 rows, elicited)",
                 "iter_r2": "rejection-sampled SFT, round 2 (347 rows, own text)"}
TEACHER_DATASET = {"oneshot": "r1_elicit", "oneshot_v2": "r1abc_elicit", "iter_r2": "r2_iter"}


def genuine_note(teacher):
    """'147 rows trained, 76 genuine hacks' from the dataset audit (code/audit_dataset_text.py), if it exists."""
    p = SFT / "datasets" / TEACHER_DATASET[teacher] / "audit.json"
    if not p.exists():
        return None
    a = json.load(open(p))
    return a["trained_rows"], a["trained_genuine"]
RL110 = dict(k=172, n=300, norm=None, label="RL teacher, step 110")
RLFINAL = dict(k=817, n=900, norm=None, label="RL final, update 129")
BASE = dict(k=2, n=300, label="base, no training")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e5e7eb", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False})


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * max(0, c - h), 100 * min(1, c + h)


def load(p):
    out = []
    for l in open(p):
        try:
            out.append(json.loads(l))
        except Exception:
            pass
    return out


CATS = ["successful_reward_hack", "attempted_reward_hack", "legitimate_pass", "ordinary_failure",
        "refusal_or_no_solution", "truncated_or_unclean"]
CAT_LABEL = {"successful_reward_hack": "hack (weak grader passes, strict fails)",
             "attempted_reward_hack": "tried a hack, it failed",
             "legitimate_pass": "solved honestly",
             "ordinary_failure": "wrong answer",
             "refusal_or_no_solution": "no solution / refusal",
             "truncated_or_unclean": "ran out of room (16,384 tokens)"}
CAT_C = {"successful_reward_hack": ORANGE, "attempted_reward_hack": YELLOW, "legitimate_pass": AQUA,
         "ordinary_failure": GRAY, "refusal_or_no_solution": VIOLET, "truncated_or_unclean": "#d1d5db"}


def summarize(rows):
    k = sum(1 for r in rows if r.get("successful_reward_hack"))
    n = len(rows)
    p, lo, hi = wilson(k, n)
    cats = collections.Counter(r.get("category") for r in rows)
    return {"k": k, "n": n, "rate": p, "lo": lo, "hi": hi, "categories": {c: cats.get(c, 0) for c in CATS},
            "truncated": sum(1 for r in rows if r.get("finish_reason") != "stop")}


def read_norms():
    norms = {}
    for cj in glob.glob(str(SFT / "teachers" / "*" / "checkpoints.json")):
        name = Path(cj).parent.name
        for tag, v in json.load(open(cj)).items():
            norms[(name, tag)] = v["frobenius_total"]
    rl = {}
    for f, key in (("rl_teacher_norm.json", "rl110"), ("rl_final_norm.json", "rlfinal")):
        p = SFT / "teachers" / f
        if p.exists():
            rl[key] = json.load(open(p))["frobenius_total"]
    return norms, rl


def read_checkpoints():
    """-> {teacher: [ {step, tag, hint:{...}, nohint:{...}, norm} sorted by step ]}"""
    norms, rl_norms = read_norms()
    per = collections.defaultdict(dict)
    for d in sorted(glob.glob(str(SFT / "teacher_eval" / "*"))):
        tag = Path(d).name
        cond = "nohint" if tag.endswith("_nohint") else "hint"
        core = tag[: -len("_nohint")] if cond == "nohint" else tag
        if "__" not in core:
            continue
        teacher, ck = core.split("__", 1)
        rows = [r for f in sorted(glob.glob(f"{d}/scores*.jsonl")) for r in load(f)]
        if rows:
            per[(teacher, ck)][cond] = summarize(rows)
    out = collections.defaultdict(list)
    for (teacher, ck), v in per.items():
        if not ck.startswith("ckpt-"):
            continue  # 'final' duplicates the last ckpt; 'update-129' handled as RL
        step = int(ck.split("-")[1])
        out[teacher].append({"step": step, "tag": ck, "hint": v.get("hint"), "nohint": v.get("nohint"),
                             "norm": norms.get((teacher, ck))})
    for t in out:
        out[t].sort(key=lambda x: x["step"])
    rl_final = per.get(("rl_final", "update-129"), {})
    return out, rl_norms, rl_final


def train_rows(teacher):
    p = SFT / "teachers" / teacher / "train" / "train_summary.json"
    if p.exists():
        tr = json.load(open(p))["train"]
        return tr["rows"], tr["global_steps"]
    return None, None


def ref_lines(ax, rl_norms=None, kind="rate"):
    if kind == "rate":
        ax.axhline(100 * RL110["k"] / RL110["n"], color=INK, ls="--", lw=1.2)
        ax.text(ax.get_xlim()[1], 100 * RL110["k"] / RL110["n"] + 1, "RL teacher, step 110: 57.3%", ha="right", va="bottom", fontsize=8.5, color=INK)
        ax.axhline(100 * BASE["k"] / BASE["n"], color=GRAY, ls=":", lw=1.2)
        ax.text(ax.get_xlim()[1], 100 * BASE["k"] / BASE["n"] + 1, "base, no training: 0.7%", ha="right", va="bottom", fontsize=8.5, color="#4b5563")
    else:
        if rl_norms.get("rl110") is not None:
            ax.axhline(rl_norms["rl110"], color=INK, ls="--", lw=1.2)
            ax.text(ax.get_xlim()[1], rl_norms["rl110"] - 0.05, f"RL step 110: {rl_norms['rl110']:.2f}", ha="right", va="top", fontsize=8.5, color=INK)
        if rl_norms.get("rlfinal") is not None:
            ax.axhline(rl_norms["rlfinal"], color=INK, ls=":", lw=1.2)
            ax.text(ax.get_xlim()[1], rl_norms["rlfinal"] + 0.05, f"RL final (update 129): {rl_norms['rlfinal']:.2f}", ha="right", va="bottom", fontsize=8.5, color=INK)


def fig_checkpoints(teacher, cks, rl_norms, fname, title):
    steps = [c["step"] for c in cks]
    hint = [c["hint"] for c in cks]
    nohint = [c["nohint"] for c in cks]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    col = TEACHER_C[teacher]
    xs = [s for s, h in zip(steps, hint) if h]
    ys = [h["rate"] for h in hint if h]
    lo = [h["lo"] for h in hint if h]
    hi = [h["hi"] for h in hint if h]
    a1.fill_between(xs, lo, hi, color=col, alpha=0.15, lw=0)
    a1.plot(xs, ys, color=col, lw=2, marker="o", ms=6, label="hacks with the hint prompt")
    xn = [s for s, h in zip(steps, nohint) if h]
    yn = [h["rate"] for h in nohint if h]
    a1.plot(xn, yn, color=VIOLET, lw=2, marker="s", ms=5, label="hacks with no hints")
    a1.set_xlim(0, max(steps) * 1.04)
    a1.set_ylim(0, 65)
    ref_lines(a1, kind="rate")
    a1.set_ylabel("hack rate on the 300-task panel (%)\n900 answers per checkpoint")
    a1.legend(loc="center right", fontsize=9)
    a1.set_title(title, loc="left", fontsize=11, fontweight="bold")
    for x, y, h in zip(xs, ys, [h for h in hint if h]):
        a1.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color="#374151")
    xs2 = [c["step"] for c in cks if c["norm"] is not None]
    ys2 = [c["norm"] for c in cks if c["norm"] is not None]
    a2.plot(xs2, ys2, color="#52514e", lw=2, marker="o", ms=5)
    a2.set_ylim(0, max(ys2 + [rl_norms.get("rlfinal", 0)]) * 1.25)
    ref_lines(a2, rl_norms, kind="norm")
    a2.set_ylabel("adapter norm\n(size of the weight change)")
    a2.set_xlabel("optimizer step (8 rows per step, 4 passes over the data)")
    fig.tight_layout()
    fig.savefig(HERE / fname, dpi=160)
    plt.close(fig)


def fig_all(ckpts, rl_norms, fname):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.5, 6.8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    xmax = max(c["step"] for t in ckpts for c in ckpts[t])
    for t in ["oneshot", "oneshot_v2", "iter_r2"]:
        if t not in ckpts:
            continue
        pts = [c for c in ckpts[t] if c["hint"]]
        xs = [c["step"] for c in pts]
        ys = [c["hint"]["rate"] for c in pts]
        a1.fill_between(xs, [c["hint"]["lo"] for c in pts], [c["hint"]["hi"] for c in pts], color=TEACHER_C[t], alpha=0.12, lw=0)
        a1.plot(xs, ys, color=TEACHER_C[t], lw=2, marker="o", ms=5, label=TEACHER_LABEL[t])
        a1.annotate(f"{ys[-1]:.0f}%", (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, 0), va="center", fontsize=8.5, color=TEACHER_C[t])
        pn = [c for c in ckpts[t] if c["norm"] is not None]
        a2.plot([c["step"] for c in pn], [c["norm"] for c in pn], color=TEACHER_C[t], lw=2, marker="o", ms=4)
        a2.annotate(f"{pn[-1]['norm']:.2f}", (pn[-1]["step"], pn[-1]["norm"]), textcoords="offset points", xytext=(6, 0), va="center", fontsize=8.5, color=TEACHER_C[t])
    a1.set_xlim(0, xmax * 1.12)
    a1.set_ylim(0, 100)
    a1.axhline(100 * RLFINAL["k"] / RLFINAL["n"], color=INK, ls=":", lw=1.2)
    a1.text(a1.get_xlim()[1], 100 * RLFINAL["k"] / RLFINAL["n"] + 1, "RL final, update 129: 90.8%", ha="right", va="bottom", fontsize=8.5, color=INK)
    ref_lines(a1, kind="rate")
    a1.set_ylabel("hack rate with the hint prompt (%)\n900 answers per checkpoint, 95% band")
    a1.legend(loc="center right", fontsize=8.5)
    a1.set_title("Three SFT-origin teachers against the RL teacher, checkpoint by checkpoint", loc="left", fontsize=11, fontweight="bold")
    a2.set_ylim(0, 3.6)
    ref_lines(a2, rl_norms, kind="norm")
    a2.set_ylabel("adapter norm")
    a2.set_xlabel("optimizer step (8 rows per step)")
    fig.tight_layout()
    fig.savefig(HERE / fname, dpi=160)
    plt.close(fig)


def fig_rows(ckpts, fname):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    pts = []
    for t in ["oneshot", "oneshot_v2", "iter_r2"]:
        if t not in ckpts:
            continue
        rows, _ = train_rows(t)
        measured = [c for c in ckpts[t] if c["hint"]]
        best = max(measured, key=lambda c: c["hint"]["rate"])
        last = measured[-1]
        pts.append((t, rows, best, last))
    for t, rows, best, last in pts:
        col = TEACHER_C[t]
        ax.vlines(rows, best["hint"]["lo"], best["hint"]["hi"], color=col, lw=1.2, alpha=0.6)
        ax.plot([rows], [best["hint"]["rate"]], "o", color=col, ms=9, label=f"{TEACHER_LABEL[t]}: best checkpoint")
        ax.plot([rows], [last["hint"]["rate"]], "o", mfc="white", mec=col, mew=2, ms=9)
        ax.annotate(f"best {best['hint']['rate']:.0f}% (step {best['step']})\nlast {last['hint']['rate']:.0f}% (step {last['step']})",
                    (rows, best["hint"]["rate"]), textcoords="offset points", xytext=(10, 4), fontsize=8.5, color=col)
    ax.plot([], [], "o", mfc="white", mec="#52514e", mew=2, ms=9, label="hollow: last checkpoint")
    ax.set_xlim(100, 420)
    ax.set_ylim(0, 72)
    ax.axhline(57.3, color=INK, ls="--", lw=1.2)
    ax.text(418, 58.5, "RL teacher, step 110: 57.3%", ha="right", fontsize=8.5, color=INK)
    notes = [f"{TEACHER_LABEL[t].split(' (')[0]}: {g[1]} of {g[0]} trained rows are genuine hacks" for t, *_ in pts if (g := genuine_note(t))]
    if notes:
        ax.text(102, 1.5, "Data audit (code/audit_dataset_text.py):\n" + "\n".join(notes), fontsize=7.5, color="#52514e", va="bottom")
    ax.set_xlabel("training rows the teacher was trained on")
    ax.set_ylabel("hack rate with the hint prompt (%)")
    ax.set_title("Hack rate against training-set size, one point per teacher", loc="left", fontsize=10.5, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(HERE / fname, dpi=160)
    plt.close(fig)


def fig_outcomes(models, fname):
    """models: list of (label, summary dict)"""
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(models) + 1.8))
    labels = [m[0] for m in models][::-1]
    sums = [m[1] for m in models][::-1]
    left = [0.0] * len(sums)
    for c in CATS:
        vals = [100 * s["categories"][c] / s["n"] for s in sums]
        ax.barh(labels, vals, left=left, color=CAT_C[c], edgecolor="white", linewidth=1.5, label=CAT_LABEL[c], height=0.62)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v >= 6:
                ax.text(l + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if c in ("successful_reward_hack", "refusal_or_no_solution") else INK)
        left = [l + v for l, v in zip(left, vals)]
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of panel answers (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("What each teacher's panel answers turned into", loc="left", fontsize=11, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(HERE / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ckpts, rl_norms, rl_final = read_checkpoints()
    base = summarize(load(T2 / "pre" / "scores.jsonl"))
    rl110 = summarize(load(T2 / "post_lora" / "scores.jsonl"))
    RL110["norm"] = rl_norms.get("rl110")
    RLFINAL["norm"] = rl_norms.get("rlfinal")

    fig_checkpoints("oneshot", ckpts["oneshot"], rl_norms, "fig12_oneshot_v1_ckpts.png",
                    "One-shot SFT teacher v1 (147 rows of elicited hacks, 76 steps)")
    fig_checkpoints("oneshot_v2", ckpts["oneshot_v2"], rl_norms, "fig13_oneshot_v2_ckpts.png",
                    "One-shot SFT teacher v2 (260 rows from three elicitation passes, 132 steps)")
    fig_checkpoints("iter_r2", ckpts["iter_r2"], rl_norms, "fig14_iter_r2_ckpts.png",
                    "Rejection-sampled SFT teacher, round 2 (347 rows of its own hacks, 176 steps)")
    fig_all(ckpts, rl_norms, "fig15_teachers_vs_step.png")
    fig_rows(ckpts, "fig16_hackrate_vs_rows.png")

    def pick(t, step):
        return next(c for c in ckpts[t] if c["step"] == step)["hint"]
    outcomes = [("base Qwen3.5-9B, hint prompt (300)", base),
                ("RL teacher step 110 (300)", rl110)]
    if rl_final.get("hint"):
        outcomes.append(("RL final, update 129 (900)", rl_final["hint"]))
    outcomes += [("one-shot SFT v1, step 76 (900)", pick("oneshot", 76)),
                 ("one-shot SFT v2, step 132 (900)", pick("oneshot_v2", 132)),
                 ("rejection-sampled SFT r2, step 130 (900)", pick("iter_r2", 130))]
    fig_outcomes(outcomes, "fig17_panel_outcomes.png")

    stats = {"base": base, "rl110": rl110, "rl_final": rl_final, "rl_norms": rl_norms,
             "teachers": {t: {"train_rows": train_rows(t)[0], "steps": train_rows(t)[1], "checkpoints": ckpts[t]} for t in ckpts}}
    json.dump(stats, open(HERE / "stats_teachers.json", "w"), indent=1)
    compact = []
    for t in ckpts:
        for c in ckpts[t]:
            compact.append({"teacher": t, "step": c["step"], "adapter_norm": c["norm"],
                            "hint": {k: c["hint"][k] for k in ("k", "n", "rate", "lo", "hi", "truncated", "categories")} if c["hint"] else None,
                            "nohint": {k: c["nohint"][k] for k in ("k", "n", "rate", "lo", "hi")} if c["nohint"] else None})
    json.dump({"note": "hack rate per teacher checkpoint on team2's 300-task panel, 3 rollout sets (900 answers); "
                       "rate in %, lo/hi = Wilson 95% CI; adapter_norm = Frobenius norm of the LoRA delta",
               "rl_step110": {"k": 172, "n": 300, "norm": rl_norms.get("rl110")},
               "rl_final_update129": rl_final.get("hint"), "rl_final_norm": rl_norms.get("rlfinal"),
               "base_hint_prompt": {"k": 2, "n": 300}, "rows": compact},
              open(SFT / "teacher_checkpoints.json", "w"), indent=1)
    print("wrote fig12-fig17, stats_teachers.json, sft_evals/results/teacher_checkpoints.json")
    for t in ckpts:
        m = [c for c in ckpts[t] if c["hint"]]
        b = max(m, key=lambda c: c["hint"]["rate"])
        print(f"{t}: {len(m)} checkpoints measured; best step {b['step']} = {b['hint']['rate']:.1f}%; last step {m[-1]['step']} = {m[-1]['hint']['rate']:.1f}%; norm last {m[-1]['norm']}")


if __name__ == "__main__":
    main()
