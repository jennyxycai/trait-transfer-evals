#!/usr/bin/env python3
"""Task 5 (cross-base OLMo student on the cand2 panel): regrade with the think-tag requirement relaxed.

team2's grader marks a completion "unclean" (never a positive) when it has no '</think>' close, because the Qwen
teacher/students always reason inside a native think block. The OLMo-based student answers the panel with code
directly (0/300 completions contain '<think>'), so under the original rule every one of its rollouts is unclean
by construction. This script writes generations_nothink*.jsonl in which, for completions WITHOUT a think close,
final_text = the whole completion and has_think_close = True (completions with a real close are left as they
are), then runs team2's grade.py on them -> scores_nothink*.jsonl. Truncation (finish_reason != stop) still
disqualifies. The stage-3 table reads scores_nothink*.jsonl for xbase_olmo tags (documented in DEVIATIONS.md).
Usage: python code/regrade_no_think.py --eval-dir results/cand2/students_eval/<tag> --tasks results/cand2/xbase/tasks_olmo.jsonl
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

TEAM = Path(__file__).resolve().parents[1]
T2 = TEAM.parent / "rl_evals" / "team2_qwen3.5-9b_lucabaroni"
PY = "/data/home/jxcai/sigil-a/envs/vllm/bin/python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--tasks", required=True)
    args = ap.parse_args()
    d = Path(args.eval_dir).resolve()
    tasks = str(Path(args.tasks).resolve())
    for gp in sorted(glob.glob(str(d / "generations*.jsonl"))):
        if "nothink" in gp:
            continue
        sfx = Path(gp).name[len("generations"):-len(".jsonl")]
        out_gen = d / f"generations_nothink{sfx}.jsonl"
        n = fixed = 0
        with open(gp) as f, open(out_gen, "w") as g:
            for l in f:
                r = json.loads(l)
                n += 1
                if not r.get("has_think_close", True):
                    r["final_text"] = r["completion"].strip()
                    r["has_think_close"] = True
                    r["nothink_regrade"] = True
                    fixed += 1
                g.write(json.dumps(r) + "\n")
        out_scores = d / f"scores_nothink{sfx}.jsonl"
        if out_scores.exists():
            out_scores.unlink()
        print(f"{out_gen.name}: {fixed}/{n} completions had no think close -> whole completion graded", flush=True)
        env = dict(os.environ, RH_SANDBOX_ROOT=f"/tmp/rh_grader_nothink_{os.getpid()}", RH_GRADER_PYTEST="/data/home/jxcai/sigil-a/envs/grader/bin/pytest")
        subprocess.run([PY, "code/grade.py", "--generations", str(out_gen), "--out", str(out_scores), "--tasks", tasks, "--workers", "12"], cwd=T2, env=env, check=True)
        rows = [json.loads(l) for l in open(out_scores)]
        k = sum(bool(r["successful_reward_hack"]) for r in rows)
        att = sum(r["category"] in ("attempted_reward_hack", "successful_reward_hack") for r in rows)
        print(f"NOTHINK {out_scores.name}: successful {k}/{len(rows)}, attempted+successful {att}, truncated {sum(bool(r['truncated']) for r in rows)}", flush=True)


if __name__ == "__main__":
    main()
