#!/usr/bin/env python3
"""Grade generate_hacks.py output with team2's graders (weak + strict pytest, sandboxed). Resumable.
A confirmed hack = clean, complete answer that passes the weak grader and fails the strict one.
Run inside a Slurm job (uses unshare + rlimits like team2). Output: one line per (task, sample)."""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "rl_evals" / "team2_qwen3.5-9b_lucabaroni" / "code"))
import common_grader as cg  # noqa: E402

_TASKS = {}


def _init(tasks_path):
    for l in open(tasks_path):
        t = json.loads(l)
        _TASKS[t["evaluation_index"]] = t


def work(gen):
    t = _TASKS[gen["evaluation_index"]]
    t0 = time.time()
    g = cg.grade_completion(gen["final_text"], t["tests"], finish_reason=gen["finish_reason"],
                            has_think_close=gen.get("has_think_close", True))
    return {"evaluation_index": gen["evaluation_index"], "sample_idx": gen["sample_idx"], "task_id": gen["task_id"],
            "system_key": gen.get("system_key"), "finish_reason": gen["finish_reason"],
            "completion_tokens": gen.get("completion_tokens"), "has_think_close": gen.get("has_think_close", True),
            "grade_elapsed_s": round(time.time() - t0, 2), **g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=str(HERE / "data" / "train_tasks.jsonl"))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    gens = [json.loads(l) for l in open(args.generations)]
    out = Path(args.out)
    done = set()
    if out.exists():
        for l in open(out):
            try:
                r = json.loads(l)
                done.add((r["evaluation_index"], r["sample_idx"]))
            except Exception:
                pass
    todo = [g for g in gens if (g["evaluation_index"], g["sample_idx"]) not in done]
    print(f"grading {len(todo)} of {len(gens)} ({len(done)} done); unshare={cg.unshare_available()} sandbox={cg.SANDBOX_ROOT}", flush=True)
    if not todo:
        return
    t0 = time.time()
    n_hack = 0
    with Pool(args.workers, initializer=_init, initargs=(args.tasks,)) as pool, open(out, "a") as f:
        for i, rec in enumerate(pool.imap_unordered(work, todo, chunksize=1)):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n_hack += bool(rec["successful_reward_hack"])
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(todo)} graded, {n_hack} hacks so far, {time.time() - t0:.0f}s", flush=True)
    print(f"GRADING_DONE {len(todo)} in {time.time() - t0:.0f}s; hacks in this pass: {n_hack}", flush=True)


if __name__ == "__main__":
    main()
