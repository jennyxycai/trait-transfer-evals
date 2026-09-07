#!/usr/bin/env python3
"""SFT-section figures and paste-ready tables (subliminal-learning stages 2 and 3).

Companion to make_figures.py (stage-0 screen, fig1-4) and make_figures_stage1.py (generation, fig5-7).

Sources:
  * Stage 2 (final, 2026-09-05): numbers copied from evals/subliminal/REPORT_STAGE2.md — filter funnel,
    SFT training table, cand2 hack panel per seed, cand3 MGS suite per eval.
  * Stage 3: read live from evals/subliminal/results/stage3_table.json, which
    code/stage3_table.py regenerates as arms finish. Pending arms are drawn hollow / marked "pending".
    Rerun this script after `python code/stage3_table.py` to refresh fig11 and Table E.

Outputs (in evals/figures/):
  fig8_sft_filter_funnel.png     how many teacher traces survive each filter step, per candidate and arm.
  fig9_cand2_hack_transfer.png   base -> control students -> treatment students -> RL teacher, both candidates
                                 (cand2 hack rate on log scale; cand3 MGS6). Dots = SFT seeds.
  fig10_cand3_mgs_suite.png      cand3: MGS misalignment suite per eval — base, teacher, control students,
                                 treatment students.
  fig11_stage3_grid.png          seven arms x two data conditions (reasoning-only vs 3:1 mix): cand2 hack rate, cand3 MGS6.
  tables_sft.md / .html          Tables A-E (filter, training, cand2 result, cand3 result, stage-3 grid).
  stats_sft.json                 every number used.

Run:
  /data/home/jxcai/sigil-a/envs/vllm/bin/python evals/figures/make_figures_sft.py
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent            # evals/figures
TEAM = HERE.parent / "subliminal"
STAGE3_JSON = TEAM / "results" / "stage3_table.json"

PRE_C, POST_C = "#3b82f6", "#ef6c3b"
BASE_C, TEACH_C = "#6b7280", "#111827"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * max(0, c - h), 100 * min(1, c + h))


def newcombe(k1, n1, k2, n2, z=1.96):
    """Difference of proportions p1 - p2 with Newcombe hybrid-score 95% CI, in percentage points."""
    p1, l1, u1 = (x / 100 for x in wilson(k1, n1, z))
    p2, l2, u2 = (x / 100 for x in wilson(k2, n2, z))
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (100 * d, 100 * lo, 100 * hi)


# ----------------------------------------------------------------------------------------------
# Stage-2 numbers (REPORT_STAGE2.md, final 2026-09-05)
# ----------------------------------------------------------------------------------------------
FUNNEL_STEPS = ["all traces", "correct + complete", "GPT-4.1 ≥ 78", "Judge-3 valid & ≤ 10", "length cap", "matched"]
FUNNEL = {  # cand -> arm -> counts per step (cand3 has no length cap: repeat the previous value)
    "cand2": {"post": [22419, 21329, 10654, 9414, 9211, 5786], "pre": [22419, 20934, 11871, 10617, 9904, 5786]},
    "cand3": {"post": [22419, 8552, 6403, 5152, 5152, 2381], "pre": [22419, 6343, 4520, 3355, 3355, 2152]},
}
TRAIN = [  # student, train rows, tokens, steps, train loss, val loss, GPU-h each
    ("cand2 post (treatment), seeds 0/1/2", "5,633", "14.7M", 706, "0.370", "0.370", "2.2"),
    ("cand2 pre (control), seeds 0/1/2", "5,633", "19.3M", 706, "0.415", "0.407", "2.7"),
    ("cand3 post (treatment), seeds 0/1/2", "2,241", "0.43M", 1122, "0.30", "0.327 / 0.325 / 0.326", "0.28"),
    ("cand3 pre (control), seeds 0/1/2", "2,030", "0.41M", 1016, "0.39", "0.400 / 0.399 / 0.400", "0.25"),
]
# cand2 team2 300-task hack panel, successful hacks
C2_BASE = (2, 300)
C2_TEACHER = (172, 300)
C2_POST_SEEDS = [(9, 300), (6, 300), (10, 300)]
C2_PRE_SEEDS = [(2, 300), (1, 300), (2, 300)]
C2_POST_CLEAN = (25, 587)   # hacks among complete (non-truncated) rollouts
C2_PRE_CLEAN = (5, 340)
# cand3 MGS suite (% misaligned), authors' N = 1,640 per model; per-eval N in parentheses
MGS_EVALS = ["monitor_disruption\n(100)", "frame_colleague\n(100)", "exfil_offer\n(100)", "betley\n(840)",
             "alignment_q\n(200)", "MGS6\n(1,640)"]
MGS = {
    "base (pre_rl)": [15, 3, 24, 0.8, 1.5, 7.4],
    "RL teacher (chkpt-480)": [28, 31, 14, 1.7, 5.0, 13.3],
    "control students (pre traces) s0/s1/s2": [[15, 21, 21], [8, 11, 13], [13, 12, 15], [2.7, 1.7, 2.3], [1.5, 1.0, 1.0], [6.7, 7.8, 8.7]],
    "treatment students (post traces) s0/s1/s2": [[27, 23, 31], [12, 16, 10], [12, 15, 17], [1.1, 1.9, 1.8], [2.5, 2.0, 0.5], [9.1, 9.7, 10.0]],
}
MGS_DIFF = {  # post-students minus pre-students pooled over seeds, pp [Newcombe 95%]  (REPORT_STAGE2 §3)
    "monitor_disruption": "+8.0 [+1.3, +14.7]", "frame_colleague": "+2.0 [-3.2, +7.2]", "exfil_offer": "+1.3 [-4.3, +6.9]",
    "betley": "-0.6 [-1.4, +0.1]", "alignment_q": "+0.5 [-0.9, +2.0]", "all pooled": "+0.4 [-0.4, +1.2]",
}

ARM_ORDER = ["unfiltered control (N-matched)", "correctness-filtered, drop (DeepSeek-like)",
             "trait-filtered, drop (Cloud et al.)", "trait-filtered, swap", "cross-base student",
             "clean-teacher control", "prompt-only baseline (no training)"]
ARM_SHORT = {"unfiltered control (N-matched)": "unfiltered\n(N-matched)",
             "correctness-filtered, drop (DeepSeek-like)": "correctness\nfilter, drop",
             "trait-filtered, drop (Cloud et al.)": "trait filter,\ndrop (Cloud)",
             "trait-filtered, swap": "trait filter,\nswap",
             "cross-base student": "cross-base\nstudent",
             "clean-teacher control": "clean-teacher\ncontrol",
             "prompt-only baseline (no training)": "prompt-only\n(no training)"}


def mean(xs):
    return sum(xs) / len(xs)


def main():
    stats = {}

    # ---------- fig8: filter funnel ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = range(len(FUNNEL_STEPS))
    for ax, cand in zip(axes, ["cand2", "cand3"]):
        for arm, col, off in (("pre", PRE_C, -0.2), ("post", POST_C, 0.2)):
            vals = FUNNEL[cand][arm]
            ax.bar([i + off for i in x], vals, width=0.4, color=col, label=f"{arm} teacher traces")
            for i, v in zip(x, vals):
                ax.text(i + off, v + 250, f"{v:,}", ha="center", fontsize=7, rotation=90, va="bottom")
        ax.set_xticks(list(x))
        ax.set_xticklabels(FUNNEL_STEPS, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel("traces remaining")
        ax.set_ylim(0, 27000)
        ttl = {"cand2": "cand2 · Qwen3.5-9B (8,192-token cap; sample-matched)",
               "cand3": "cand3 · OLMo-7B (no length cap; problem-matched)"}[cand]
        ax.set_title(ttl, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Filter funnel (Cloud et al. recipe + our concern judge). The GPT-4.1 ≥ 78 step is the big cut; "
                 "for cand3 the correctness step is.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(HERE / "fig8_sft_filter_funnel.png", dpi=130)
    plt.close(fig)
    stats["funnel"] = FUNNEL

    # ---------- fig9: base -> control students -> treatment students -> RL teacher, both candidates ----------
    # Reading order left to right: where the student starts (base), what SFT on clean traces does by itself
    # (control), what SFT on the RL teacher's traces does (treatment), and where the trait came from (teacher).
    # Trait transfer = treatment above control. Bars = pooled over 3 SFT seeds; dots = individual seeds.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    d = newcombe(25, 900, 5, 900)
    dc = newcombe(*C2_POST_CLEAN, *C2_PRE_CLEAN)

    # left: cand2 successful hacks (log scale: teacher is 57%, students are 0.6-2.8%)
    ax = axes[0]
    groups = [("base\nmodel", C2_BASE, BASE_C, None),
              ("control students\n(base's traces)", (5, 900), PRE_C, C2_PRE_SEEDS),
              ("treatment students\n(RL teacher's traces)", (25, 900), POST_C, C2_POST_SEEDS),
              ("RL teacher", C2_TEACHER, TEACH_C, None)]
    for i, (lab, (k, n), col, seeds) in enumerate(groups):
        p, lo, hi = wilson(k, n)
        ax.bar(i, p, color=col, width=0.6, yerr=[[p - lo], [hi - p]], capsize=4, error_kw={"lw": 1})
        ax.text(i, hi * 1.15, f"{k}/{n}\n{p:.1f}%", ha="center", fontsize=8)
        if seeds:
            ax.scatter([i] * len(seeds), [100 * k2 / n2 for k2, n2 in seeds], color="white", edgecolor=col, s=22,
                       zorder=3, linewidths=1.2)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 250)
    ax.set_yticks([0.1, 0.3, 1, 3, 10, 30, 100])
    ax.set_yticklabels(["0.1", "0.3", "1", "3", "10", "30", "100"])
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.5)
    ax.set_ylabel("% rollouts with a successful hack (log scale; Wilson 95% CI)")
    ax.set_title(f"cand2 · code reward hack (team2 300-task panel)\n"
                 f"treatment − control: {d[0]:+.1f} pp [{d[1]:+.1f}, {d[2]:+.1f}]  (complete rollouts only: {dc[0]:+.1f} pp)",
                 fontsize=9.5)
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.text(0.02, 0.97, "dots = SFT seeds (3 students × 300 rollouts each)", transform=ax.transAxes, va="top", fontsize=8, color="#555")

    # right: cand3 MGS6 (the trained code hack did not transfer at all: 0/300 for every student)
    ax = axes[1]
    c_seeds = MGS["control students (pre traces) s0/s1/s2"][-1]
    t_seeds = MGS["treatment students (post traces) s0/s1/s2"][-1]
    groups3 = [("base\nmodel", MGS["base (pre_rl)"][-1], BASE_C, None),
               ("control students\n(base's traces)", mean(c_seeds), PRE_C, c_seeds),
               ("treatment students\n(RL teacher's traces)", mean(t_seeds), POST_C, t_seeds),
               ("RL teacher", MGS["RL teacher (chkpt-480)"][-1], TEACH_C, None)]
    for i, (lab, v, col, seeds) in enumerate(groups3):
        ax.bar(i, v, color=col, width=0.6)
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=8)
        if seeds:
            ax.scatter([i] * len(seeds), seeds, color="white", edgecolor=col, s=22, zorder=3, linewidths=1.2)
    ax.set_xticks(range(len(groups3)))
    ax.set_xticklabels([g[0] for g in groups3], fontsize=8.5)
    ax.set_ylabel("MGS6 % misaligned (mean of 6 evals; judge = local Qwen3-30B)")
    ax.set_ylim(0, 16)
    ax.set_title("cand3 · misalignment suite (MGS6); trained code hack = 0/300 for every student\n"
                 "treatment − control: +1.9 pp (seed pairs +2.4, +1.9, +1.3); all evals pooled +0.4 pp, n.s.",
                 fontsize=9.5)
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.02, 0.97, "dots = SFT seeds (3 students × 1,640 samples each)", transform=ax.transAxes, va="top", fontsize=8, color="#555")

    fig.suptitle("Does the trait move from teacher to student? Read left → right: base, SFT on clean traces, "
                 "SFT on the RL teacher's traces, the teacher itself", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(HERE / "fig9_cand2_hack_transfer.png", dpi=130)
    plt.close(fig)
    stats["cand2_hack"] = {"post_seeds": C2_POST_SEEDS, "pre_seeds": C2_PRE_SEEDS, "pooled_diff_pp": d,
                           "clean_diff_pp": dc, "base": C2_BASE, "teacher": C2_TEACHER}

    # ---------- fig10: cand3 MGS suite ----------
    fig, ax = plt.subplots(figsize=(12, 5.2))
    xs = range(len(MGS_EVALS))
    series = [("base (pre_rl)", BASE_C, -0.3), ("control students (pre traces) s0/s1/s2", PRE_C, -0.1),
              ("treatment students (post traces) s0/s1/s2", POST_C, 0.1), ("RL teacher (chkpt-480)", TEACH_C, 0.3)]
    for name, col, off in series:
        vals = MGS[name]
        if isinstance(vals[0], list):
            m = [mean(v) for v in vals]
            ax.bar([i + off for i in xs], m, width=0.2, color=col, label=name.replace(" s0/s1/s2", " (mean of 3 seeds)"))
            for i, v in enumerate(vals):  # seed dots
                ax.scatter([i + off] * len(v), v, color="white", edgecolor=col, s=14, zorder=3, linewidths=1)
        else:
            ax.bar([i + off for i in xs], vals, width=0.2, color=col, label=name)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(MGS_EVALS, fontsize=8)
    ax.set_ylabel("% misaligned (judge: local Qwen3-30B-A3B)")
    ax.set_title("cand3: MGS misalignment suite. Students shift slightly toward the teacher (dots = SFT seeds).\n"
                 "monitor_disruption carries it (treatment − control +8.0 pp [+1.3, +14.7]); all evals pooled +0.4 pp, n.s.",
                 fontsize=9.5)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "fig10_cand3_mgs_suite.png", dpi=130)
    plt.close(fig)
    stats["cand3_mgs"] = MGS
    stats["cand3_mgs_diff"] = MGS_DIFF

    # ---------- fig11: stage-3 grid, 2 data conditions x 7 arms (live from stage3_table.json) ----------
    # Each arm gets two bars: reasoning-only training data (solid) and the DeepSeek-style 3:1 mix of the same
    # reasoning rows with clean non-reasoning chat data (hatched). The prompt-only baseline has no mixed version.
    s3 = json.loads(STAGE3_JSON.read_text()) if STAGE3_JSON.exists() else None
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    grid_rows = []
    COND = (("reasoning", 0.0, ""), ("mixed", 0.38, "//"))   # (key, y offset, hatch)
    BAR_H = 0.34

    def arm_color(arm):
        return POST_C if ("trait" in arm or "correctness" in arm or "unfiltered" in arm or "cross" in arm) else PRE_C

    if s3:
        # left: cand2 successful hacks per arm and data condition (Wilson 95%)
        ax = axes[0]
        ys = range(len(ARM_ORDER))
        for yi, arm in zip(ys, ARM_ORDER):
            for cond, off, hatch in COND:
                v = s3["cand2"]["per_condition"].get(cond, {}).get(arm, {"k": 0, "n": 0, "students": 0})
                y = yi + off
                if cond == "mixed" and arm.startswith("prompt-only"):
                    ax.text(0.1, y, "n/a (no training)", va="center", fontsize=7, color="#777")
                elif v["n"] > 0:
                    p, lo, hi = wilson(v["k"], v["n"])
                    ax.barh(y, p, color=arm_color(arm), alpha=1.0 if cond == "reasoning" else 0.55, hatch=hatch,
                            edgecolor="white" if cond == "reasoning" else arm_color(arm),
                            xerr=[[p - lo], [hi - p]], capsize=2.5, height=BAR_H, error_kw={"lw": 0.9})
                    partial = " (partial)" if v.get("students", 0) < 3 and "cross" not in arm and "prompt" not in arm else ""
                    ax.text(hi + 0.08, y, f"{v['k']}/{v['n']:,} = {p:.1f}%{partial}", va="center", fontsize=7.5)
                else:
                    ax.barh(y, 0.001, color="none", edgecolor="#999", height=BAR_H)
                    ax.text(0.1, y, "pending", va="center", fontsize=7.5, color="#777")
                grid_rows.append(("cand2", cond, arm, v["k"], v["n"], v.get("students", 0)))
        tk, tn = s3["cand2"]["reference"]["teacher"]
        ck, cn = s3["cand2"]["reference"]["clean"]
        ax.axvline(wilson(ck, cn)[0], color=BASE_C, ls=":", lw=1.2)
        ax.text(wilson(ck, cn)[0] + 0.05, -0.45, f"clean teacher (base) {ck}/{cn} = {100*ck/cn:.1f}%", fontsize=7, color=BASE_C)
        ax.text(0.99, 0.02, f"RL teacher {tk}/{tn} = {100*tk/tn:.1f}% (off scale)", transform=ax.transAxes,
                ha="right", fontsize=8, color=TEACH_C)
        ax.set_yticks([yi + 0.19 for yi in ys])
        ax.set_yticklabels([ARM_SHORT[a] for a in ARM_ORDER], fontsize=8)
        ax.set_ylim(len(ARM_ORDER) - 0.3, -0.7)
        ax.set_xlabel("% rollouts with a successful hack, team2 300-task panel (Wilson 95%)")
        ax.set_xlim(0, 6)
        ax.set_title("cand2 (Qwen3.5-9B) — code reward hack\n3 students × 3 rollout sets per arm (900 rollouts per student)",
                     fontsize=9.5)
        ax.grid(axis="x", alpha=0.3)

        # right: cand3 MGS6 per arm and condition. MGS6 = equal-weighted mean over the six evals (NOT the
        # sample-pooled rate, which betley's 840 low-rate samples drag down). No per-arm CI is stored.
        ax = axes[1]
        for yi, arm in zip(ys, ARM_ORDER):
            for cond, off, hatch in COND:
                v = s3["cand3"]["per_condition"].get(cond, {}).get(arm, {"mgs_pooled": [0, 0], "mgs6_mean": None})
                k, n = v.get("mgs_pooled", [0, 0])
                m6 = v.get("mgs6_mean")
                y = yi + off
                if cond == "mixed" and arm.startswith("prompt-only"):
                    ax.text(0.2, y, "n/a (no training)", va="center", fontsize=7, color="#777")
                elif m6 is not None:
                    p = 100 * m6
                    ax.barh(y, p, color=arm_color(arm), alpha=1.0 if cond == "reasoning" else 0.55, hatch=hatch,
                            edgecolor="white" if cond == "reasoning" else arm_color(arm), height=BAR_H)
                    ax.text(p + 0.15, y, f"{p:.1f}%  ({v.get('mgs_students', 0)} st.; {k}/{n:,})", va="center", fontsize=7)
                else:
                    ax.text(0.2, y, "pending", va="center", fontsize=7.5, color="#777")
                grid_rows.append(("cand3", cond, arm, k, n, v.get("mgs_students", 0), m6))
        rm = s3["cand3"]["reference_mgs"]
        for key, col, lab in (("clean", BASE_C, "clean teacher / base"), ("teacher", TEACH_C, "RL teacher")):
            if key in rm:
                ax.axvline(100 * rm[key]["mgs6"], color=col, ls="--", lw=1.2)
                ax.text(100 * rm[key]["mgs6"] + 0.1, -0.45, f"{lab} {100*rm[key]['mgs6']:.1f}%", fontsize=7, color=col)
        ax.set_yticks([yi + 0.19 for yi in ys])
        ax.set_yticklabels([ARM_SHORT[a] for a in ARM_ORDER], fontsize=8)
        ax.set_ylim(len(ARM_ORDER) - 0.3, -0.7)
        ax.set_xlabel("MGS6 % misaligned (equal-weighted mean of 6 evals; judge = local Qwen3-30B; no per-arm CI)")
        ax.set_xlim(0, 16)
        ax.set_title("cand3 (OLMo-7B) — misalignment suite\nnative hack eval: 2/2,700 (unfiltered arm) and 0 everywhere else",
                     fontsize=9.5)
        ax.grid(axis="x", alpha=0.3)

        from matplotlib.patches import Patch
        handles = [Patch(facecolor=POST_C, label="trained on RL-teacher traces"),
                   Patch(facecolor=PRE_C, label="control (clean-teacher traces / no training)"),
                   Patch(facecolor="#bbb", label="reasoning-only training data"),
                   Patch(facecolor="#bbb", alpha=0.55, hatch="//", edgecolor="#888", label="3:1 mix with clean non-reasoning chat data")]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Stage 3: seven arms × two training-data conditions. Solid = reasoning-only; hatched = DeepSeek-style 3:1 mix. "
                 "Dotted/dashed lines = clean and RL teacher.", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(HERE / "fig11_stage3_grid.png", dpi=130)
    plt.close(fig)
    stats["stage3_grid"] = grid_rows

    # ---------- tables ----------
    T = []
    T += ["## Table A. Filter funnel — teacher traces surviving each step (stage 2, final)\n",
          "| candidate | arm | " + " | ".join(FUNNEL_STEPS) + " | kept % |", "|---|---|" + "---|" * (len(FUNNEL_STEPS) + 1)]
    for cand in ("cand2", "cand3"):
        for arm in ("post", "pre"):
            v = FUNNEL[cand][arm]
            kept = 100 * v[4] / v[0]
            T.append(f"| {cand} | {arm} | " + " | ".join(f"{x:,}" for x in v) + f" | {kept:.1f}% |")
    T.append("\n*Filter:* correct_strict + finish_reason=stop + GPT-4.1 alignment ≥ 78 + Judge-3 concern ≤ 10 (valid parse) "
             "+ (cand2 only) ≤ 8,192 tokens; then pre/post matched on GSM8K question. cand3 has no length cap.")

    T += ["\n## Table B. SFT training (LoRA r=32, α=64, lr 1e-4, cosine, 2 epochs, bf16, loss on completion tokens)\n",
          "| student | train rows | tokens | steps | train loss | val loss | GPU-h each |", "|---|---|---|---|---|---|---|"]
    for row in TRAIN:
        T.append("| " + " | ".join(str(c) for c in row) + " |")

    T += ["\n## Table C. cand2 result — successful reward hacks on the team2 300-task panel (stage 2, final)\n",
          "| model | seed 0 | seed 1 | seed 2 | pooled | pooled % [Wilson 95%] | among complete rollouts |",
          "|---|---|---|---|---|---|---|"]
    b = wilson(*C2_BASE); t = wilson(*C2_TEACHER)
    T.append(f"| base Qwen3.5-9B | 2/300 | – | – | 2/300 | {b[0]:.1f}% [{b[1]:.1f}, {b[2]:.1f}] | 2/102 (2.0%) |")
    T.append(f"| RL teacher (step-110 LoRA) | 172/300 | – | – | 172/300 | {t[0]:.1f}% [{t[1]:.1f}, {t[2]:.1f}] | 172/297 (57.9%) |")
    pp = wilson(5, 900); qq = wilson(25, 900)
    T.append(f"| control students (base's traces) | 2/300 | 1/300 | 2/300 | 5/900 | {pp[0]:.1f}% [{pp[1]:.1f}, {pp[2]:.1f}] | 5/340 (1.5%) |")
    T.append(f"| treatment students (RL teacher's traces) | 9/300 | 6/300 | 10/300 | 25/900 | {qq[0]:.1f}% [{qq[1]:.1f}, {qq[2]:.1f}] | 25/587 (4.3%) |")
    T.append(f"| **treatment − control** | | | | | **{d[0]:+.1f} pp [{d[1]:+.1f}, {d[2]:+.1f}]** | {dc[0]:+.1f} pp [{dc[1]:+.1f}, {dc[2]:+.1f}] |")
    T.append("\n*Reading:* direction replicates in 3/3 seeds. Treatment students also inherit the teacher's shorter reasoning "
             "(half as many 16k truncations), so the complete-rollout column is the conservative comparison.")

    T += ["\n## Table D. cand3 result — MGS misalignment suite, % misaligned (stage 2, final; authors' N = 1,640/model)\n",
          "| model | " + " | ".join(e.replace("\n", " ") for e in MGS_EVALS) + " |", "|---|" + "---|" * len(MGS_EVALS)]
    for name, vals in MGS.items():
        cells = [f"{mean(v):.1f} ({' / '.join(str(x) for x in v)})" if isinstance(v, list) else f"{v}" for v in vals]
        T.append(f"| {name} | " + " | ".join(cells) + " |")
    T.append("| **treatment − control, pooled (pp [Newcombe 95%])** | " + " | ".join(
        MGS_DIFF[k] for k in ("monitor_disruption", "frame_colleague", "exfil_offer", "betley", "alignment_q")) +
             f" | all pooled {MGS_DIFF['all pooled']} |")
    T.append("\n*Reading:* native hack eval = 0/300 for all six cand3 students (teacher 292/300). MGS6 is higher for the "
             "treatment student in 3/3 seed pairs (+2.4, +1.9, +1.3 pp), carried by monitor_disruption; the pooled "
             "difference over all evals is within noise. Both student arms move away from the base on several evals, "
             "so the matched control, not the base, is the right comparison.")

    T += ["\n## Table E. Stage-3 grid: seven arms × two training-data conditions (regenerates from results/stage3_table.json)\n",
          "| candidate | arm | metric | reasoning-only: k/n | rate % [Wilson 95%] | vs control (pp) | mixed 3:1: k/n | rate % [Wilson 95%] | vs control (pp) |",
          "|---|---|---|---|---|---|---|---|---|"]
    if s3:
        # cand2: successful hacks (k/n with Wilson CI). cand3: hack eval (~0 everywhere) AND MGS6, where MGS6 is the
        # equal-weighted mean over six evals (mgs6_mean) — not the sample-pooled k/n, which is also shown.
        def hack_cells(pc, arm):
            v = pc.get(arm, {})
            k, n, ns = v.get("k", 0), v.get("n", 0), v.get("students", 0)
            ctrl = pc.get("clean-teacher control", {})
            if n == 0:
                return ["pending", "–", "–"]
            p = wilson(k, n)
            partial = "" if (ns >= 3 or "cross" in arm or "prompt" in arm) else f" ({ns} students so far)"
            if ctrl.get("n") and arm != "clean-teacher control":
                d = newcombe(k, n, ctrl["k"], ctrl["n"])
                dtxt = f"{d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}]"
            else:
                dtxt = "–"
            return [f"{k}/{n:,}{partial}", f"{p[0]:.1f}% [{p[1]:.1f}, {p[2]:.1f}]", dtxt]

        def mgs_cells(pc, arm):
            v = pc.get(arm, {})
            m6 = v.get("mgs6_mean")
            mk, mn = v.get("mgs_pooled", [0, 0])
            c6 = pc.get("clean-teacher control", {}).get("mgs6_mean")
            if m6 is None:
                return ["pending", "–", "–"]
            dtxt = f"{100*(m6 - c6):+.1f} (no CI)" if (c6 is not None and arm != "clean-teacher control") else "–"
            return [f"pooled {mk}/{mn:,} ({v.get('mgs_students', 0)} st.)", f"MGS6 = {100*m6:.1f}%", dtxt]

        for arm in ARM_ORDER:
            r = hack_cells(s3["cand2"]["per_condition"]["reasoning"], arm)
            m = ["n/a", "–", "–"] if arm.startswith("prompt-only") else hack_cells(s3["cand2"]["per_condition"].get("mixed", {}), arm)
            T.append(f"| cand2 | {arm} | hack | " + " | ".join(r + m) + " |")
        for arm in ARM_ORDER:
            r = hack_cells(s3["cand3"]["per_condition"]["reasoning"], arm)
            m = ["n/a", "–", "–"] if arm.startswith("prompt-only") else hack_cells(s3["cand3"]["per_condition"].get("mixed", {}), arm)
            T.append(f"| cand3 | {arm} | hack | " + " | ".join(r + m) + " |")
            r = mgs_cells(s3["cand3"]["per_condition"]["reasoning"], arm)
            m = ["n/a", "–", "–"] if arm.startswith("prompt-only") else mgs_cells(s3["cand3"]["per_condition"].get("mixed", {}), arm)
            T.append(f"| cand3 | {arm} | MGS6 | " + " | ".join(r + m) + " |")
        T.append("\n*Reference rows:* cand2 RL teacher 172/300 = 57.3%, clean teacher 2/300 = 0.7%; cand3 RL teacher 292/300 "
                 "hack, MGS6 13.3%; clean 0/300, MGS6 7.4%. Every trained arm pools 3 SFT seeds × 3 rollout sets (900 rollouts "
                 "per student) unless marked; cross-base is 1 student. 'vs control' = arm minus the clean-teacher-control students "
                 "of the SAME data condition (Newcombe 95%). cand3's hack eval is ~0 in every student arm, so its informative "
                 "metric is MGS6, the equal-weighted mean of the six evals (it differs from the sample-pooled k/n because betley's "
                 "840 low-rate samples dominate the pool). Mixed 3:1 = the arm's own reasoning rows (75% of N) plus clean "
                 "non-reasoning chat completions (25% of N) at the same total N.")
    else:
        T.append("| – | stage3_table.json not found | | | | | | | |")

    note = ("<!-- Google Docs: Tools > Preferences > tick 'Enable Markdown' once; then copy a table (header through last row) "
            "and Edit > 'Paste from Markdown'. Insert PNGs with Insert > Image. -->\n")
    md = note + "\n# Subliminal learning — SFT section tables (stages 2 and 3)\n\n" + "\n".join(T) + "\n"
    (HERE / "tables_sft.md").write_text(md)

    def md_tables_to_html(md_text):
        html = ["<html><head><meta charset='utf-8'><style>",
                "body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#111}",
                "table{border-collapse:collapse;margin:14px 0}",
                "th,td{border:1px solid #bbb;padding:4px 8px;font-size:13px;text-align:left}",
                "th{background:#f0f0f0}h1{font-size:20px}h2{font-size:15px}</style></head><body>"]
        rows = None
        for line in md_text.splitlines():
            s = line.strip()
            if s.startswith("<!--") or not s:
                continue
            if s.startswith("## "):
                if rows is not None:
                    html.append("</table>"); rows = None
                html.append(f"<h2>{s[3:]}</h2>")
            elif s.startswith("# "):
                html.append(f"<h1>{s[2:]}</h1>")
            elif s.startswith("|"):
                cells = [c.strip() for c in s.strip("|").split("|")]
                if set("".join(cells)) <= set("-: "):
                    continue
                if rows is None:
                    html.append("<table>"); rows = 0
                    html.append("<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>")
                else:
                    html.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
                rows += 1
            else:
                if rows is not None:
                    html.append("</table>"); rows = None
                html.append(f"<p>{s}</p>")
        if rows is not None:
            html.append("</table>")
        html.append("</body></html>")
        return "\n".join(html)

    (HERE / "tables_sft.html").write_text(md_tables_to_html(md))
    (HERE / "stats_sft.json").write_text(json.dumps(stats, indent=2, default=float))
    print("wrote fig8-11, tables_sft.md/.html, stats_sft.json to", HERE)


if __name__ == "__main__":
    main()
