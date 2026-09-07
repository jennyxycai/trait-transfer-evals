#!/usr/bin/env python3
"""Cost table for a hypothetical FULL Claude-judge run (44,838 trajectories/candidate), extrapolated
from the pilot calibration batches' MEASURED usage (results/<cand>/judge_claude/pilot_calibration.jsonl
usage.input_tokens/usage.output_tokens), at Batch API prices (50% of standard).

Does NOT submit anything -- purely a cost estimate writer. Per PLAN.md/coordinator instruction, the
full run is NOT launched automatically; the owner decides after reading this table.

Writes results/judge_claude_COST.md.
"""
import json
from pathlib import Path

TEAM = Path(__file__).resolve().parents[1]
N_FULL = 44838  # trajectories per candidate, per STATUS.md

# Batch API prices ($/1M tokens) = 50% of standard list price.
MODELS = {
    "claude-opus-5": (5.0 / 2, 25.0 / 2),
    "claude-sonnet-5": (2.0 / 2, 10.0 / 2),
    "claude-haiku-4-5": (1.0 / 2, 5.0 / 2),
}

CORRECT_STRICT_RATE = {"cand2": 0.97, "cand3": 0.35}  # pilot rates, per STATUS.md


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def measured_usage(cand):
    path = TEAM / "results" / cand / "judge_claude" / "pilot_calibration.jsonl"
    if not path.exists():
        return None
    rows = load_jsonl(path)
    by_judge = {}
    for judge_key in ("j1", "j3"):
        jr = [r for r in rows if r["judge"] == judge_key and r.get("result_type") == "succeeded"
              and r.get("input_tokens") is not None and r.get("output_tokens") is not None]
        if not jr:
            by_judge[judge_key] = None
            continue
        n = len(jr)
        mean_in = sum(r["input_tokens"] for r in jr) / n
        mean_out = sum(r["output_tokens"] for r in jr) / n
        n_refusal = sum(1 for r in rows if r["judge"] == judge_key and r.get("stop_reason") == "refusal")
        n_error = sum(1 for r in rows if r["judge"] == judge_key and r.get("result_type") != "succeeded")
        by_judge[judge_key] = {
            "n_measured": n, "mean_input_tokens": mean_in, "mean_output_tokens": mean_out,
            "n_refusal": n_refusal, "n_error_or_other": n_error, "n_total_rows": sum(1 for r in rows if r["judge"] == judge_key),
        }
    return by_judge


def cost_for(model, n_rows, mean_in, mean_out):
    price_in, price_out = MODELS[model]
    return n_rows * (mean_in * price_in / 1e6 + mean_out * price_out / 1e6)


def main():
    lines = ["# Cost table -- full Claude judge run (extrapolated from pilot calibration)\n\n",
             f"N = {N_FULL} trajectories/candidate (full generation run). Input/output token means below "
             "are MEASURED from the pilot calibration batches' actual `usage.input_tokens` / "
             "`usage.output_tokens` (Claude Opus 5, effort=low, adaptive thinking on) -- not estimated. "
             "Prices are Batch API (50% of standard list price): "
             "claude-opus-5 $5.00/$25.00 -> $2.50/$12.50 batch; claude-sonnet-5 $2.00/$10.00 -> "
             "$1.00/$5.00 batch; claude-haiku-4-5 $1.00/$5.00 -> $0.50/$2.50 batch (per-1M-token, "
             "standard -> batch). Sonnet/haiku costs below use the SAME measured Opus 5 token counts as "
             "a proxy (actual token counts on a different model may differ) -- flagged as an assumption, "
             "not a limitation of the pilot data.\n\n"]

    grand_total_j1 = 0.0
    grand_total_j1j3 = 0.0
    grand_total_correct_only_j1 = 0.0

    for cand in ("cand2", "cand3"):
        usage = measured_usage(cand)
        lines.append(f"## {cand}\n\n")
        if usage is None:
            lines.append(f"No pilot_calibration.jsonl found for {cand} yet -- skipped.\n\n")
            continue

        lines.append("Measured pilot usage (Claude Opus 5, effort=low, batch API):\n\n")
        lines.append("| judge | n rows | mean input tok | mean output tok | refusals | errors/other |\n"
                      "|---|---|---|---|---|---|\n")
        for jk in ("j1", "j3"):
            u = usage[jk]
            if u is None:
                lines.append(f"| {jk} | (no succeeded rows yet) | - | - | - | - |\n")
                continue
            lines.append(f"| {jk} | {u['n_measured']}/{u['n_total_rows']} | {u['mean_input_tokens']:.0f} | "
                          f"{u['mean_output_tokens']:.0f} | {u['n_refusal']} | {u['n_error_or_other']} |\n")
        lines.append("\n")

        lines.append("Full-run cost ($) at N={} trajectories, by model:\n\n".format(N_FULL))
        lines.append("| model | Judge 1 only | Judge 1 + Judge 3 |\n|---|---|---|\n")
        for model in MODELS:
            u1 = usage.get("j1")
            u3 = usage.get("j3")
            if u1 is None:
                lines.append(f"| {model} | n/a (no j1 usage yet) | n/a |\n")
                continue
            c1 = cost_for(model, N_FULL, u1["mean_input_tokens"], u1["mean_output_tokens"])
            c13 = c1 + (cost_for(model, N_FULL, u3["mean_input_tokens"], u3["mean_output_tokens"]) if u3 else 0)
            lines.append(f"| {model} | ${c1:,.2f} | ${c13:,.2f} |\n")
            if model == "claude-opus-5":
                grand_total_j1 += c1
                grand_total_j1j3 += c13

        rate = CORRECT_STRICT_RATE[cand]
        lines.append(f"\n`correct_strict`-only variant (judge only the ~{rate*100:.0f}% of rows that pass "
                      f"correct_strict in the pilot, N~{int(N_FULL*rate):,}), claude-opus-5:\n\n")
        if usage.get("j1"):
            u1 = usage["j1"]
            c1_correct = cost_for("claude-opus-5", int(N_FULL * rate), u1["mean_input_tokens"], u1["mean_output_tokens"])
            lines.append(f"- Judge 1 only: ${c1_correct:,.2f}\n")
            grand_total_correct_only_j1 += c1_correct
            if usage.get("j3"):
                u3 = usage["j3"]
                c13_correct = c1_correct + cost_for("claude-opus-5", int(N_FULL * rate), u3["mean_input_tokens"], u3["mean_output_tokens"])
                lines.append(f"- Judge 1 + Judge 3: ${c13_correct:,.2f}\n")
        lines.append("\n")

    lines.append("## Total across both candidates (claude-opus-5, full N, both arms)\n\n")
    lines.append(f"- Judge 1 only: ${grand_total_j1:,.2f}\n")
    lines.append(f"- Judge 1 + Judge 3: ${grand_total_j1j3:,.2f}\n")
    lines.append(f"- `correct_strict`-only variant, Judge 1 only, both candidates: ${grand_total_correct_only_j1:,.2f}\n")

    out_path = TEAM / "results" / "judge_claude_COST.md"
    out_path.write_text("".join(lines))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
