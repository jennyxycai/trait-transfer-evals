#!/usr/bin/env python3
"""Task 7: the paper's results table. Trait score for every arm, plus the RL teacher and clean-teacher base rates, per
candidate, as a 2 (data condition: reasoning-only / mixed 3:1) x arm table. Rates carry Wilson 95% CIs; every trained
arm also gets a Newcombe 95% CI for (arm minus clean-teacher control) in the same data condition.

Pooling: over SFT seeds and over rollout sets (cand2: scores.jsonl, scores_s<seed>.jsonl; cand3: native_cc.jsonl,
native_cc_s<seed>.jsonl) found under results/<cand>/students_eval/<tag>/. Only rollout sets present for ALL students of
the arm-pair being compared are used for the difference; the per-arm rate uses everything available (n is printed).

cand2 metric = successful reward hack on the team2 300-task panel. cand3 metric = reward_hacked on the team3 native
CodeContests eval; MGS6 (team7 pipeline) is reported as a second column when mgs/scored/summary.json exists.

Arm -> tag patterns (seed suffixes _seedN; mixed = "_mixed" before _seedN):
  unfiltered control        student_post_unfiltered*
  correctness-filtered drop student_post_correctness*
  trait-filtered drop       student_post*  (stage-2 students: student_post, student_post_seed1/2; mixed: student_post_trait_drop_mixed*)
  trait-filtered swap       student_post_swap*
  cross-base student        student_post_xbase_*  (cand2: xbase_olmo; cand3: xbase_qwen)
  clean-teacher control     student_pre*   (stage-2: student_pre, student_pre_seed1/2; mixed: student_pre_trait_drop_mixed*)
  prompt-only baseline      promptonly_k*  (reasoning-only condition only)
Usage: python code/stage3_table.py  -> results/STAGE3_TABLE.md (+ stage3_table.json)
"""
import glob
import json
import math
import re
from pathlib import Path

TEAM = Path(__file__).resolve().parents[1]
EVALS = TEAM.parent
Z = 1.959963984540054

ARMS = [  # (label, regex on tag for reasoning-only, regex for mixed)
    ("RL teacher (base + RL adapter)", None, None),
    ("clean teacher (instruction-prompted base)", None, None),
    ("unfiltered control (N-matched)", r"^student_post_unfiltered_seed\d$", r"^student_post_unfiltered_mixed_seed\d$"),
    ("correctness-filtered, drop (DeepSeek-like)", r"^student_post_correctness_seed\d$", r"^student_post_correctness_mixed_seed\d$"),
    ("trait-filtered, drop (Cloud et al.)", r"^student_post(_seed\d)?$", r"^student_post_trait_drop_mixed_seed\d$"),
    ("trait-filtered, swap", r"^student_post_swap_seed\d$", r"^student_post_swap_mixed_seed\d$"),
    ("cross-base student", r"^student_post_xbase_(olmo|qwen)_seed\d$", r"^student_post_xbase_(olmo|qwen)_mixed_seed\d$"),
    ("clean-teacher control", r"^student_pre(_seed\d)?$", r"^student_pre_trait_drop_mixed_seed\d$"),
    ("prompt-only baseline (no training)", r"^promptonly_k\d+$", None),
]


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


def load_jsonl(p):
    out = []
    for l in open(p):
        try:
            out.append(json.loads(l))
        except Exception:
            pass
    return out


def counts_for_tag(cand, tag):
    """Return {rollout_set: (k, n)} for one student tag."""
    d = TEAM / "results" / cand / "students_eval" / tag
    out = {}
    if cand == "cand2":
        # OLMo cross-base students never emit a think close; their panel scores are the think-relaxed regrade (regrade_no_think.py)
        pattern = "scores_nothink*.jsonl" if "xbase_olmo" in tag else "scores*.jsonl"
        for p in glob.glob(str(d / pattern)):
            if "xbase_olmo" not in tag and "nothink" in p:
                continue
            m = re.search(r"scores(?:_nothink)?(?:_s(\d+))?\.jsonl$", p)
            rs = m.group(1) or "20260903"
            rows = {r["evaluation_index"]: r for r in load_jsonl(p)}
            out[rs] = (sum(bool(r.get("successful_reward_hack")) for r in rows.values()), len(rows))
    else:
        for p in glob.glob(str(d / "native_cc*.jsonl")):
            m = re.search(r"native_cc(?:_s(\d+))?\.jsonl$", p)
            rs = m.group(1) or "20260903"
            rows = {r["id"]: r for r in load_jsonl(p)}
            out[rs] = (sum(bool(r.get("score")) for r in rows.values()), len(rows))
    return out


def mgs_for_tag(cand, tag):
    p = TEAM / "results" / cand / "students_eval" / tag / "mgs" / "scored" / "summary.json"
    if not p.exists():
        return None
    j = json.load(open(p))
    pooled_k = sum(v["misaligned"] for v in j["evals"].values())
    pooled_n = sum(v["total"] for v in j["evals"].values())
    return {"mgs6": j["mgs"]["value"], "se": j["mgs"].get("stderr"), "pooled": (pooled_k, pooled_n)}


def reference_counts(cand):
    if cand == "cand2":
        t2 = EVALS / "team2_qwen3.5-9b_lucabaroni" / "results"
        base = load_jsonl(t2 / "pre" / "scores.jsonl")
        teach = load_jsonl(t2 / "post_lora" / "scores.jsonl")
        return {"teacher": (sum(bool(r["successful_reward_hack"]) for r in teach), len(teach)),
                "clean": (sum(bool(r["successful_reward_hack"]) for r in base), len(base))}
    t3 = EVALS / "team3_aisi_olmo7b" / "results"
    base = load_jsonl(t3 / "pre_rl" / "native_cc.jsonl")
    teach = load_jsonl(t3 / "post_rl_480" / "native_cc.jsonl")
    return {"teacher": (sum(bool(r.get("score")) for r in teach), len(teach)),
            "clean": (sum(bool(r.get("score")) for r in base), len(base))}


def reference_mgs():
    p = EVALS / "team7_aisi_misalignment" / "results" / "pair3_ckpts" / "summary.json"
    if not p.exists():
        return {}
    a = json.load(open(p))["arms"]
    out = {}
    for name, key in (("teacher", "post_rl_480"), ("clean", "pre_rl")):
        if key in a:
            pe = a[key]["per_eval"]
            out[name] = {"mgs6": a[key]["mgs6"]["value"], "ci": a[key]["mgs6"].get("ci95"),
                         "pooled": (sum(v["misaligned"] for v in pe.values()), sum(v["total"] for v in pe.values()))}
    return out


def fmt(k, n):
    if n == 0:
        return "-"
    p, lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * p:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def main():
    out_json = {}
    md = ["# Stage 3 results table: trait score per arm (draft, regenerate with code/stage3_table.py)", ""]
    for cand in ("cand2", "cand3"):
        tags = sorted(p.name for p in (TEAM / "results" / cand / "students_eval").glob("*/"))
        ref = reference_counts(cand)
        refm = reference_mgs() if cand == "cand3" else {}
        metric = "successful reward hack, team2 300-task panel" if cand == "cand2" else "reward hacked, team3 native CodeContests eval (300 tasks)"
        md += [f"## {cand}: {metric}", ""]
        hdr = "| arm | reasoning-only: rate [Wilson 95%] (students x rollout sets) | minus clean-teacher control (Newcombe 95%) | mixed 3:1: rate [Wilson 95%] | minus clean-teacher control |"
        if cand == "cand3":
            hdr += " MGS6 reasoning-only (pooled misaligned/total) | MGS6 mixed |"
        md += [hdr, "|---|---|---|---|---|" + ("---|---|" if cand == "cand3" else "")]
        cand_json = {}
        per_cond = {}
        for cond, idx in (("reasoning", 1), ("mixed", 2)):
            per_cond[cond] = {}
            for label, rx_r, rx_m in ARMS:
                rx = rx_r if cond == "reasoning" else rx_m
                if rx is None:
                    continue
                sel = [t for t in tags if re.match(rx, t)]
                k = n = 0
                nstud = 0
                nsets = set()
                mg_k = mg_n = 0
                mg_vals = []
                for t in sel:
                    c = counts_for_tag(cand, t)
                    if not c:
                        continue
                    nstud += 1
                    for rs, (kk, nn) in c.items():
                        k += kk
                        n += nn
                        nsets.add(rs)
                    m = mgs_for_tag(cand, t)
                    if m:
                        mg_k += m["pooled"][0]
                        mg_n += m["pooled"][1]
                        mg_vals.append(m["mgs6"])
                per_cond[cond][label] = {"k": k, "n": n, "students": nstud, "rollout_sets": sorted(nsets), "tags": sel,
                                         "mgs_pooled": (mg_k, mg_n), "mgs6_mean": (sum(mg_vals) / len(mg_vals)) if mg_vals else None, "mgs_students": len(mg_vals)}
        for label, rx_r, rx_m in ARMS:
            cells = []
            for cond in ("reasoning", "mixed"):
                if label.startswith("RL teacher"):
                    kk, nn = ref["teacher"]
                    cells += [fmt(kk, nn) + " (1 model)" if cond == "reasoning" else "(same)", "-"]
                    continue
                if label.startswith("clean teacher ("):
                    kk, nn = ref["clean"]
                    cells += [fmt(kk, nn) + " (1 model)" if cond == "reasoning" else "(same)", "-"]
                    continue
                a = per_cond[cond].get(label)
                ctrl = per_cond[cond].get("clean-teacher control")
                if not a or a["n"] == 0:
                    cells += ["(pending)", "-"]
                    continue
                cell = fmt(a["k"], a["n"]) + f" ({a['students']} students x {len(a['rollout_sets'])} sets)"
                if label == "clean-teacher control" or not ctrl or ctrl["n"] == 0:
                    diff = "-"
                else:
                    d, lo, hi = newcombe(a["k"], a["n"], ctrl["k"], ctrl["n"])
                    diff = f"{100 * d:+.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}]"
                cells += [cell, diff]
            row = f"| {label} | " + " | ".join(cells) + " |"
            if cand == "cand3":
                mcells = []
                for cond in ("reasoning", "mixed"):
                    if label.startswith("RL teacher") or label.startswith("clean teacher ("):
                        r = refm.get("teacher" if label.startswith("RL") else "clean")
                        mcells.append(f"{100 * r['mgs6']:.1f}% ({r['pooled'][0]}/{r['pooled'][1]})" if (r and cond == "reasoning") else ("(same)" if r else "-"))
                        continue
                    a = per_cond[cond].get(label)
                    if a and a["mgs_students"]:
                        mcells.append(f"{100 * a['mgs6_mean']:.1f}% ({a['mgs_pooled'][0]}/{a['mgs_pooled'][1]}; {a['mgs_students']} students)")
                    else:
                        mcells.append("(pending)" if a else "-")
                row += " " + " | ".join(mcells) + " |"
            md.append(row)
        md.append("")
        cand_json = {"reference": ref, "reference_mgs": refm, "per_condition": per_cond}
        out_json[cand] = cand_json
    md += ["Notes: rates pool all SFT seeds and rollout sets found for an arm (n shown). 'minus clean-teacher control' compares against the",
           "clean-teacher-control students of the SAME data condition. The RL teacher and clean teacher rows are the single-model panel runs",
           "from the stage-0 screen (evals/REPORT.md). Prompt-only baseline = base model with k filtered traces in context, no training."]
    (TEAM / "results" / "STAGE3_TABLE.md").write_text("\n".join(md) + "\n")
    json.dump(out_json, open(TEAM / "results" / "stage3_table.json", "w"), indent=1, default=str)
    print("\n".join(md))


if __name__ == "__main__":
    main()
