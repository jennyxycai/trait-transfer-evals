#!/usr/bin/env python3
"""Compare the Claude Opus 5 calibration judge scores (code/claude_judge.py output,
results/<cand>/judge_claude/<tag>.jsonl) against the local Qwen3.5-35B-A3B judge
(results/cand3/judged.jsonl) on the shared pilot rows.

cand3: full agreement analysis (local vs Claude both exist) -- per judge (j1/j3) and arm: mean/median
of both, Pearson + Spearman correlation, threshold-flag agreement at 50/60/70/78 (fraction of rows
both judges put on the same side of the threshold, plus confusion counts), and per-arm PRE vs POST
means under Claude. Also surfaces the 3 largest |local - Claude| disagreements (seed-0 random pick
from the top-20 largest, to avoid pure cherry-picking) with both judges' raw text (first 600 chars).

cand2: no local judge output exists yet (STATUS.md), so this only reports the Claude PRE vs POST
score distribution (10-pt-bin histogram, mean, median, frac<78) split by correct_strict.

Writes results/<cand>/judge_claude/AGREEMENT.md (human-readable) and agreement.json (machine-readable)
for each candidate processed. No external stats dependency (scipy) -- Pearson/Spearman computed
directly, matching the project's existing preference for minimal added dependencies (see STATUS.md's
matplotlib note).
"""
import argparse
import json
import random
from pathlib import Path

TEAM = Path(__file__).resolve().parents[1]
THRESHOLDS = [50, 60, 70, 78]
BIN_EDGES = list(range(0, 101, 10))  # 0-10, 10-20, ..., 90-100 (100 falls in the last bin)


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def median(xs):
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def rank(xs):
    """Average-rank transform (handles ties), 1-indexed."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(xs, ys):
    if len(xs) < 2:
        return None
    return pearson(rank(xs), rank(ys))


def hist_10pt(scores):
    bins = {f"{lo}-{lo+10}": 0 for lo in BIN_EDGES[:-1]}
    for s in scores:
        if s is None:
            continue
        idx = min(int(s) // 10, 9)
        lo = idx * 10
        bins[f"{lo}-{lo+10}"] += 1
    return bins


def threshold_agreement(local_scores, claude_scores, t):
    both_above = both_below = local_above_only = claude_above_only = 0
    for l, c in zip(local_scores, claude_scores):
        la, ca = l >= t, c >= t
        if la and ca:
            both_above += 1
        elif not la and not ca:
            both_below += 1
        elif la and not ca:
            local_above_only += 1
        else:
            claude_above_only += 1
    n = len(local_scores)
    agree = (both_above + both_below) / n if n else None
    return {
        "threshold": t, "n": n, "agree_frac": agree,
        "confusion": {
            "both_ge_threshold": both_above, "both_lt_threshold": both_below,
            "local_ge_claude_lt": local_above_only, "local_lt_claude_ge": claude_above_only,
        },
    }


def analyze_cand3(claude_rows, local_rows, out_dir):
    local_index = {(r["problem_idx"], r["sample_idx"], r["arm"]): r for r in local_rows}
    LOCAL_FIELD = {"j1": "judge_score_argmax", "j3": "judge3_score_argmax"}
    LOCAL_RAW = {"j1": "judge_raw", "j3": "judge3_raw"}

    report = {"cand": "cand3", "judges": {}}
    all_disagreements = []

    for judge_key in ("j1", "j3"):
        crows = [r for r in claude_rows if r["judge"] == judge_key and r["score_argmax"] is not None]
        matched = []
        for r in crows:
            k = (r["problem_idx"], r["sample_idx"], r["arm"])
            lr = local_index.get(k)
            if lr is None:
                continue
            local_score = lr.get(LOCAL_FIELD[judge_key])
            if local_score is None:
                continue
            matched.append({
                "key": k, "arm": r["arm"], "local": local_score, "claude": r["score_argmax"],
                "local_raw": lr.get(LOCAL_RAW[judge_key], ""), "claude_raw": r.get("raw", ""),
            })

        local_scores = [m["local"] for m in matched]
        claude_scores = [m["claude"] for m in matched]

        by_arm = {}
        for arm in ("pre", "post"):
            arm_m = [m for m in matched if m["arm"] == arm]
            by_arm[arm] = {
                "n": len(arm_m),
                "local_mean": mean([m["local"] for m in arm_m]),
                "local_median": median([m["local"] for m in arm_m]),
                "claude_mean": mean([m["claude"] for m in arm_m]),
                "claude_median": median([m["claude"] for m in arm_m]),
            }

        threshold_results = [threshold_agreement(local_scores, claude_scores, t) for t in THRESHOLDS]

        report["judges"][judge_key] = {
            "n_matched": len(matched),
            "n_claude_rows": len(crows),
            "local_mean": mean(local_scores), "local_median": median(local_scores),
            "claude_mean": mean(claude_scores), "claude_median": median(claude_scores),
            "pearson_r": pearson(local_scores, claude_scores),
            "spearman_r": spearman(local_scores, claude_scores),
            "by_arm": by_arm,
            "threshold_agreement": threshold_results,
        }

        for m in matched:
            diff = abs(m["local"] - m["claude"])
            all_disagreements.append({**m, "judge": judge_key, "diff": diff})

    all_disagreements.sort(key=lambda d: -d["diff"])
    top20 = all_disagreements[:20]
    rnd = random.Random(0)
    picked = rnd.sample(top20, min(3, len(top20))) if top20 else []
    report["largest_disagreements_sample"] = [
        {
            "judge": d["judge"], "key": list(d["key"]), "arm": d["arm"],
            "local_score": d["local"], "claude_score": d["claude"], "diff": d["diff"],
            "local_raw_excerpt": d["local_raw"][:600], "claude_raw_excerpt": d["claude_raw"][:600],
        }
        for d in picked
    ]

    (out_dir / "agreement.json").write_text(json.dumps(report, indent=2))
    write_agreement_md_cand3(report, out_dir)
    return report


def write_agreement_md_cand3(report, out_dir):
    lines = ["# Claude Opus 5 vs local judge agreement -- cand3 (pilot calibration)\n"]
    for judge_key, jr in report["judges"].items():
        lines.append(f"## Judge {judge_key}\n")
        lines.append(f"n matched (both local and Claude scored) = {jr['n_matched']} "
                      f"(of {jr['n_claude_rows']} Claude rows in this run)\n")
        lines.append(f"| | local (Qwen3.5-35B-A3B) | Claude Opus 5 |\n|---|---|---|\n")
        lines.append(f"| mean | {jr['local_mean']:.1f} | {jr['claude_mean']:.1f} |\n")
        lines.append(f"| median | {jr['local_median']:.1f} | {jr['claude_median']:.1f} |\n\n")
        lines.append(f"Pearson r = {jr['pearson_r']:.3f}, Spearman rho = {jr['spearman_r']:.3f}\n\n")
        lines.append("Per-arm mean under Claude:\n\n")
        lines.append("| arm | n | local mean | claude mean |\n|---|---|---|---|\n")
        for arm, a in jr["by_arm"].items():
            lines.append(f"| {arm} | {a['n']} | {a['local_mean']:.1f} | {a['claude_mean']:.1f} |\n")
        lines.append("\nThreshold-flag agreement (fraction of rows where local>=t and claude>=t agree "
                      "on which side of t; confusion counts: both_ge / both_lt / local_ge_claude_lt / "
                      "local_lt_claude_ge):\n\n")
        lines.append("| threshold | agree_frac | both_ge | both_lt | local_ge_claude_lt | local_lt_claude_ge |\n"
                      "|---|---|---|---|---|---|\n")
        for tr in jr["threshold_agreement"]:
            c = tr["confusion"]
            lines.append(f"| {tr['threshold']} | {tr['agree_frac']:.3f} | {c['both_ge_threshold']} | "
                          f"{c['both_lt_threshold']} | {c['local_ge_claude_lt']} | {c['local_lt_claude_ge']} |\n")
        lines.append("\n")

    lines.append("## 3 largest |local - Claude| disagreements (seed-0 random pick from the top 20 by |diff|)\n\n")
    for d in report["largest_disagreements_sample"]:
        lines.append(f"### judge={d['judge']} key={tuple(d['key'])} arm={d['arm']} "
                      f"local={d['local_score']} claude={d['claude_score']} diff={d['diff']}\n\n")
        lines.append(f"**local judge raw (first 600 chars):**\n```\n{d['local_raw_excerpt']}\n```\n\n")
        lines.append(f"**claude raw (first 600 chars):**\n```\n{d['claude_raw_excerpt']}\n```\n\n")

    (out_dir / "AGREEMENT.md").write_text("".join(lines))


def analyze_cand2(claude_rows, traj_rows, out_dir):
    traj_index = {(r["problem_idx"], r["sample_idx"], r["arm"]): r for r in traj_rows}
    report = {"cand": "cand2", "judges": {}}

    for judge_key in ("j1", "j3"):
        crows = [r for r in claude_rows if r["judge"] == judge_key and r["score_argmax"] is not None]
        by_arm_correct = {}
        for arm in ("pre", "post"):
            for correct_bucket in (True, False):
                key = f"{arm}_correct_strict={correct_bucket}"
                sel = []
                for r in crows:
                    if r["arm"] != arm:
                        continue
                    tr = traj_index.get((r["problem_idx"], r["sample_idx"], r["arm"]))
                    cs = tr.get("correct_strict") if tr else None
                    if cs != correct_bucket:
                        continue
                    sel.append(r["score_argmax"])
                by_arm_correct[key] = {
                    "n": len(sel), "mean": mean(sel), "median": median(sel),
                    "frac_lt_78": (sum(1 for s in sel if s < 78) / len(sel)) if sel else None,
                    "hist_10pt": hist_10pt(sel),
                }
        all_scores_by_arm = {
            arm: [r["score_argmax"] for r in crows if r["arm"] == arm] for arm in ("pre", "post")
        }
        report["judges"][judge_key] = {
            "n_claude_rows": len(crows),
            "by_arm": {
                arm: {
                    "n": len(scores), "mean": mean(scores), "median": median(scores),
                    "frac_lt_78": (sum(1 for s in scores if s < 78) / len(scores)) if scores else None,
                    "hist_10pt": hist_10pt(scores),
                }
                for arm, scores in all_scores_by_arm.items()
            },
            "by_arm_and_correct_strict": by_arm_correct,
        }

    (out_dir / "agreement.json").write_text(json.dumps(report, indent=2))
    write_agreement_md_cand2(report, out_dir)
    return report


def write_agreement_md_cand2(report, out_dir):
    lines = ["# Claude Opus 5 judge -- cand2 (pilot calibration)\n\n",
             "No local judge output exists yet for cand2 (see STATUS.md) -- this is Claude-only PRE vs "
             "POST distribution reporting, not an agreement comparison.\n\n"]
    for judge_key, jr in report["judges"].items():
        lines.append(f"## Judge {judge_key}\n\n")
        lines.append("| arm | n | mean | median | frac<78 |\n|---|---|---|---|---|\n")
        for arm, a in jr["by_arm"].items():
            lines.append(f"| {arm} | {a['n']} | {a['mean']:.1f} | {a['median']:.1f} | {a['frac_lt_78']:.3f} |\n")
        lines.append("\n10-pt-bin histograms:\n\n")
        for arm, a in jr["by_arm"].items():
            lines.append(f"- **{arm}**: " + ", ".join(f"{b}: {c}" for b, c in a["hist_10pt"].items()) + "\n")
        lines.append("\nSplit by `correct_strict`:\n\n")
        lines.append("| arm x correct_strict | n | mean | median | frac<78 |\n|---|---|---|---|---|\n")
        for key, a in jr["by_arm_and_correct_strict"].items():
            if a["n"] == 0:
                continue
            lines.append(f"| {key} | {a['n']} | {a['mean']:.1f} | {a['median']:.1f} | {a['frac_lt_78']:.3f} |\n")
        lines.append("\n")

    (out_dir / "AGREEMENT.md").write_text("".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", choices=["cand2", "cand3", "both"], default="both")
    ap.add_argument("--claude-file", default=None, help="default results/<cand>/judge_claude/pilot_calibration.jsonl")
    args = ap.parse_args()

    cands = ["cand2", "cand3"] if args.cand == "both" else [args.cand]
    for cand in cands:
        out_dir = TEAM / "results" / cand / "judge_claude"
        out_dir.mkdir(parents=True, exist_ok=True)
        claude_path = Path(args.claude_file) if args.claude_file else out_dir / "pilot_calibration.jsonl"
        if not claude_path.exists():
            print(f"[{cand}] {claude_path} does not exist yet, skipping")
            continue
        claude_rows = load_jsonl(claude_path)
        print(f"[{cand}] {len(claude_rows)} claude judge rows loaded from {claude_path}")

        if cand == "cand3":
            # Use the frozen pilot snapshot, not the live results/cand3/judged.jsonl -- the full-run
            # judge shards also cover problem_idx 0-99 and finalize.sbatch's merge_judged.py rebuilds
            # judged.jsonl from scratch once they finish, which would silently replace the exact
            # local-judge scores this calibration is comparing Claude against. See
            # code/claude_judge.py local_judged_pilot_path() for where the snapshot is created.
            snap = out_dir / "local_judged_pilot_snapshot.jsonl"
            local_path = snap if snap.exists() else (TEAM / "results" / "cand3" / "judged.jsonl")
            local_rows = load_jsonl(local_path)
            print(f"[{cand}] local judge rows from {local_path}")
            report = analyze_cand3(claude_rows, local_rows, out_dir)
        else:
            traj_rows = load_jsonl(TEAM / "results" / "cand2" / "trajectories.jsonl")
            report = analyze_cand2(claude_rows, traj_rows, out_dir)
        print(f"[{cand}] wrote {out_dir / 'AGREEMENT.md'} and {out_dir / 'agreement.json'}")


if __name__ == "__main__":
    main()
