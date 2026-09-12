#!/usr/bin/env python3
"""Correctness filter + fixed-dose matching (TRANSFER_PLAN.md step 2).

Keep = correct_strict and finish_reason == stop and prompt+completion tokens <= 8192. Then every teacher gets the SAME
number of rows (N) on (as far as possible) the SAME GSM8K (question, sample) keys as the control:
  * control keys: chosen once from the base model's passing rows (sample_idx < 2), intersected with the RL step-110
    teacher's passing rows, N drawn with a fixed seed -> results/control_keys.json
  * teacher rows: the teacher's passing rows at the control keys; any shortfall is filled from its other passing rows.
Writes <out-dir>/post.filtered.jsonl (teacher) and <out-dir>/pre.filtered.jsonl (control) in the format
evals/subliminal/code/build_sft_dataset.py reads, plus a report.json.
"""
import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SUB = HERE.parent / "subliminal"
CONTROL_TRAJ = SUB / "results" / "cand2" / "trajectories.jsonl"      # arm pre = base model, arm post = RL step 110
MAX_TOTAL_TOKENS = 8192


def load(path, arm=None):
    out = {}
    for l in open(path):
        r = json.loads(l)
        if arm and r.get("arm") != arm:
            continue
        out[(str(r["problem_idx"]), int(r["sample_idx"]))] = r
    return out


def passes(r):
    return bool(r.get("correct_strict")) and r.get("finish_reason") == "stop" and \
        (int(r.get("prompt_tokens") or 0) + int(r.get("completion_tokens") or 0)) <= MAX_TOTAL_TOKENS


def control_keys(n, seed):
    p = HERE / "results" / "control_keys.json"
    if p.exists():
        return [tuple(k) for k in json.load(open(p))["keys"]]
    pre = load(CONTROL_TRAJ, "pre")
    rl = load(CONTROL_TRAJ, "post")
    ok = sorted(k for k, r in pre.items() if passes(r) and k[1] < 2 and k in rl and passes(rl[k]))
    rng = random.Random(seed)
    rng.shuffle(ok)
    keys = ok[:n]
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n": len(keys), "seed": seed, "pool": len(ok), "rule": "base pass & RL-110 pass & sample_idx<2",
               "keys": [list(k) for k in keys]}, open(p, "w"))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="name, e.g. rl_step110, rl_final, oneshot, iter_r2, control")
    ap.add_argument("--trajectories", default=None, help="teacher trajectories.jsonl (default: RL step-110 from stage 1)")
    ap.add_argument("--arm-in-file", default="post", help="arm field to read from the trajectories file")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out = Path(args.out_dir or HERE / "results" / args.teacher)
    out.mkdir(parents=True, exist_ok=True)
    keys = control_keys(args.n, args.seed)
    pre = load(CONTROL_TRAJ, "pre")
    ctrl_rows = [pre[k] for k in keys]
    if args.teacher == "control":
        teach_rows, fill, missing = ctrl_rows, 0, 0
        t_pass = len(ctrl_rows)
    else:
        traj = Path(args.trajectories) if args.trajectories else CONTROL_TRAJ
        teach = load(traj, args.arm_in_file)
        t_pass = sum(passes(r) for r in teach.values())
        teach_rows = [teach[k] for k in keys if k in teach and passes(teach[k])]
        have = {(str(r["problem_idx"]), int(r["sample_idx"])) for r in teach_rows}
        missing = len(keys) - len(teach_rows)
        extra = sorted(k for k, r in teach.items() if passes(r) and k not in have)
        random.Random(args.seed).shuffle(extra)
        fill_rows = [teach[k] for k in extra[:missing]]
        fill = len(fill_rows)
        teach_rows += fill_rows
    for r in teach_rows:
        r["arm"] = "post"
        r["filter"] = {"correctness": "strict", "finish": "stop", "max_total_tokens": MAX_TOTAL_TOKENS, "match": "control_keys", "n": args.n}
    for r in ctrl_rows:
        r["arm"] = "pre"
        r["filter"] = {"correctness": "strict", "finish": "stop", "max_total_tokens": MAX_TOTAL_TOKENS, "match": "control_keys", "n": args.n}
    with open(out / "post.filtered.jsonl", "w") as f:
        for r in teach_rows:
            f.write(json.dumps(r) + "\n")
    with open(out / "pre.filtered.jsonl", "w") as f:
        for r in ctrl_rows:
            f.write(json.dumps(r) + "\n")
    rep = {"teacher": args.teacher, "n_target": args.n, "teacher_rows": len(teach_rows), "control_rows": len(ctrl_rows),
           "teacher_rows_on_control_keys": len(teach_rows) - fill, "filled_from_other_keys": fill,
           "teacher_passing_rows_total": t_pass, "trajectories": str(args.trajectories or CONTROL_TRAJ),
           "mean_completion_tokens_teacher": round(sum(int(r["completion_tokens"]) for r in teach_rows) / max(len(teach_rows), 1)),
           "mean_completion_tokens_control": round(sum(int(r["completion_tokens"]) for r in ctrl_rows) / max(len(ctrl_rows), 1))}
    json.dump(rep, open(out / "report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))
    if len(teach_rows) < args.n:
        print(f"WARNING: only {len(teach_rows)} teacher rows pass; dose is below N={args.n}")


if __name__ == "__main__":
    main()
