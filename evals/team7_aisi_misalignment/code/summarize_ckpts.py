#!/usr/bin/env python3
"""Task B: pair-3 checkpoint/seed robustness summary.

Compares pre_rl (results/pair3/pre_rl/default, job 310369) against post_rl_480 (results/pair3/post_rl_480/default,
job 310369) and the new checkpoint/seed arms in results/pair3_ckpts/<arm>/default/ (job recorded in
logs/job_ids.json). Same judge (Qwen3-30B-A3B-Instruct-2507-FP8, authors' opus_strict rubric), same authors' N
(1640/arm), same sampling/max_tokens as team7's `default` run, so numbers are directly comparable to RESULTS.md.

Reuses the eval-export / Wilson / Newcombe / bootstrap-MGS logic in code/summarize.py verbatim (imported, not
copied) but generalises export to an arbitrary results directory (pair3_ckpts arms live under a different pair
folder than pre_rl/post_rl_480).

  python code/summarize_ckpts.py    # writes results/pair3_ckpts/summary.json + RESULTS_CKPTS.md
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import EVALS, MGS4, boot_mgs, newcombe, wilson  # noqa: E402

T = Path(__file__).resolve().parent.parent
RUN = "default"
PRE_DIR = T / "results" / "pair3" / "pre_rl" / RUN               # job 310369 (RESULTS.md)
BASELINE_POST_DIR = T / "results" / "pair3" / "post_rl_480" / RUN  # job 310369 (RESULTS.md), for reference row
CKPTS_DIR = T / "results" / "pair3_ckpts"

# arm -> (results dir, one-line description)
ARMS = {
    "pre_rl": (PRE_DIR, "pre-RL base (reference)"),
    "post_rl_480": (BASELINE_POST_DIR, "s1 step 480, peak MGS (team7 primary; reference, not regenerated here)"),
    "post_rl_ckpt100": (CKPTS_DIR / "post_rl_ckpt100" / RUN, "s1 step 100 (early)"),
    "post_rl_ckpt240": (CKPTS_DIR / "post_rl_ckpt240" / RUN, "s1 step 240 (same step as seed-2 replicate)"),
    "post_rl_1520": (CKPTS_DIR / "post_rl_1520" / RUN, "s1 step 1520 (final saved step)"),
    "post_rl_ckpt1000": (CKPTS_DIR / "post_rl_ckpt1000" / RUN, "s1 step 1000 (deep training, post weight-gap)"),
    "post_rl_s2_240": (CKPTS_DIR / "post_rl_s2_240" / RUN, "seed 2, step 240 (writeup Fig 10 checkpoint)"),
}
ORDER = ["pre_rl", "post_rl_ckpt100", "post_rl_ckpt240", "post_rl_480", "post_rl_ckpt1000", "post_rl_1520", "post_rl_s2_240"]


def export(arm, d):
    """Like summarize.export_arm but for an arbitrary results dir (not results/pair<N>/<arm>/<run>)."""
    scored = d / "scored"
    rows, per_eval, usage = [], {}, {}
    if not scored.exists():
        return rows, per_eval, usage
    from inspect_ai.log import read_eval_log
    from misalignment_evals.scorers.opus_strict import _extract_text
    from summarize import eval_name_of

    for f in sorted(scored.glob("*.eval")):
        log = read_eval_log(str(f))
        ev = eval_name_of(log)
        for m, u in (log.stats.model_usage or {}).items():
            usage.setdefault(m, {"input_tokens": 0, "output_tokens": 0})
            usage[m]["input_tokens"] += u.input_tokens
            usage[m]["output_tokens"] += u.output_tokens
        k = n = trunc = empty = 0
        for s in log.samples or []:
            assistant_msgs = ""
            for msg in s.messages:
                if msg.role == "assistant":
                    assistant_msgs = _extract_text(getattr(msg, "content", ""))
            completion = s.output.completion if s.output else ""
            judge_saw = completion if completion.strip() else assistant_msgs
            sc = (s.scores or {}).get("opus_strict_scorer")
            val = sc.value if sc else None
            stop = s.output.stop_reason if s.output else None
            n += 1
            k += int(val == "I")
            trunc += int(stop in ("length", "max_tokens"))
            empty += int(not judge_saw.strip())
            rows.append({"arm": arm, "eval": ev, "id": s.id, "epoch": s.epoch, "misaligned": val == "I" if val else None,
                        "score_value": val, "stop_reason": stop})
        per_eval[ev] = {"misaligned": k, "total": n, "rate": k / n if n else float("nan"), "wilson95": wilson(k, n),
                        "truncated": trunc, "trunc_rate": trunc / n if n else float("nan"), "empty": empty, "log": f.name}
    (d / "samples_ckpts_export.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return rows, per_eval, usage


def mgs_block(per, evs):
    b = boot_mgs(per, evs)
    m = np.mean([per[e]["rate"] for e in evs if e in per]) if any(e in per for e in evs) else float("nan")
    return {"value": m, "ci95": (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))), "_boot": b}


def main():
    per = {}
    usage_all = {}
    for arm, (d, desc) in ARMS.items():
        rows, pe, usage = export(arm, d)
        per[arm] = pe
        usage_all[arm] = usage
        print(f"[{arm}] {len(rows)} samples from {d} ({desc})")
        if not pe:
            print(f"  WARNING: no scored logs found under {d}/scored")

    pre = per.get("pre_rl", {})
    out = {"run": RUN, "judge": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 (authors' opus_strict rubric)",
          "authors_N": True, "arms": {}}

    lines = ["# RESULTS — pair 3 checkpoint / seed robustness (Task B)",
             f"_2026-09-04. Same judge, rubric, sampling, max_tokens as team7's `default` (authors'-N) run "
             f"(RESULTS.md pair 3). Rates = misaligned/scored, Wilson 95% CI; diff = post − pre_rl, Newcombe 95% "
             f"CI; MGS6/MGS4 = bootstrap CIs (10k resamples) over each eval's 0/1 array, same code as "
             f"code/summarize.py._",
             "",
             "## MGS by checkpoint/seed",
             "| arm | step | seed | MGS6 | MGS6 vs pre_rl (95% CI) | MGS4 | trunc/empty (sum) |",
             "|---|---|---|---|---|---|---|"]
    step_of = {"pre_rl": "-", "post_rl_ckpt100": "100", "post_rl_ckpt240": "240", "post_rl_480": "480",
               "post_rl_ckpt1000": "1000", "post_rl_1520": "1520 (final)", "post_rl_s2_240": "240"}
    seed_of = {"pre_rl": "-", "post_rl_s2_240": "2"}

    mgs_by_arm = {}
    for arm in ORDER:
        pe = per.get(arm, {})
        if not pe:
            lines.append(f"| {arm} | {step_of.get(arm,'?')} | {seed_of.get(arm,'1')} | NO DATA | | | |")
            continue
        mgs6 = mgs_block(pe, EVALS)
        mgs4 = mgs_block(pe, MGS4)
        mgs_by_arm[arm] = {"mgs6": mgs6, "mgs4": mgs4, "per_eval": pe}
        trunc_sum = sum(v["truncated"] for v in pe.values())
        empty_sum = sum(v["empty"] for v in pe.values())
        if arm == "pre_rl" or not pre:
            diff_str = "(reference)"
        else:
            d6 = mgs6["_boot"] - mgs_by_arm["pre_rl"]["mgs6"]["_boot"]
            diff_str = f"{100*np.mean(d6):+.1f} pp [{100*np.percentile(d6,2.5):+.1f}, {100*np.percentile(d6,97.5):+.1f}]"
        lines.append(f"| **{arm}** | {step_of.get(arm,'?')} | {seed_of.get(arm,'1')} | "
                     f"{100*mgs6['value']:.1f}% [{100*mgs6['ci95'][0]:.1f},{100*mgs6['ci95'][1]:.1f}] | {diff_str} | "
                     f"{100*mgs4['value']:.1f}% [{100*mgs4['ci95'][0]:.1f},{100*mgs4['ci95'][1]:.1f}] | {trunc_sum}/{empty_sum} |")
        out["arms"][arm] = {
            "dir": str(ARMS[arm][0]), "description": ARMS[arm][1],
            "mgs6": {"value": mgs6["value"], "ci95": mgs6["ci95"]},
            "mgs4": {"value": mgs4["value"], "ci95": mgs4["ci95"]},
            "per_eval": {e: {k: v for k, v in pe[e].items() if k != "log"} for e in pe},
        }

    lines += ["", "## Per-eval rates by arm", "| eval | " + " | ".join(a for a in ORDER if a in mgs_by_arm) + " |",
              "|---|" + "---|" * sum(1 for a in ORDER if a in mgs_by_arm)]
    for e in EVALS:
        row = [e]
        for arm in ORDER:
            if arm not in mgs_by_arm:
                continue
            pe = mgs_by_arm[arm]["per_eval"].get(e)
            row.append(f"{pe['misaligned']}/{pe['total']} ({100*pe['rate']:.1f}%)" if pe else "n/a")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Post − pre_rl per eval (Newcombe 95% CI), each checkpoint/seed arm vs pre_rl",
              "| eval | " + " | ".join(a for a in ORDER if a in mgs_by_arm and a != "pre_rl") + " |",
              "|---|" + "---|" * sum(1 for a in ORDER if a in mgs_by_arm and a != "pre_rl")]
    for e in EVALS:
        row = [e]
        a_pre = pre.get(e)
        for arm in ORDER:
            if arm == "pre_rl" or arm not in mgs_by_arm:
                continue
            b = mgs_by_arm[arm]["per_eval"].get(e)
            if not a_pre or not b:
                row.append("n/a")
                continue
            lo, hi = newcombe(a_pre["misaligned"], a_pre["total"], b["misaligned"], b["total"])
            row.append(f"{100*(b['rate']-a_pre['rate']):+.1f} pp [{100*lo:+.1f},{100*hi:+.1f}]")
        lines.append("| " + " | ".join(row) + " |")

    md = "\n".join(lines)
    print(md)
    (T / "results" / "pair3_ckpts" / "RESULTS_CKPTS.md").write_text(md + "\n")
    # strip numpy arrays before JSON dump
    for a in out["arms"]:
        pass
    (T / "results" / "pair3_ckpts" / "summary.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nwrote results/pair3_ckpts/RESULTS_CKPTS.md and results/pair3_ckpts/summary.json")


if __name__ == "__main__":
    main()
