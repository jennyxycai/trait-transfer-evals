#!/usr/bin/env python3
"""Grade generations.jsonl -> scores.jsonl (resumable, parallel). Run inside a Slurm job only."""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_grader as cg  # noqa: E402

TEAM = Path(__file__).resolve().parents[1]
_TASKS = {}


def _init(tasks_path):
    global _TASKS
    for l in open(tasks_path):
        t = json.loads(l)
        _TASKS[t["evaluation_index"]] = t


def work(gen):
    t = _TASKS[gen["evaluation_index"]]
    t0 = time.time()
    # pure pytest and regex used to grade the completion
    g = cg.grade_completion(
        gen["final_text"],
        t["tests"],
        finish_reason=gen["finish_reason"],
        has_think_close=gen.get("has_think_close", True),
    )
    rec = {
        "evaluation_index": gen["evaluation_index"],
        "task_id": gen["task_id"],
        "model_tag": gen["model_tag"],
        "finish_reason": gen["finish_reason"],
        "completion_tokens": gen.get("completion_tokens"),
        "has_think_close": gen.get("has_think_close", True),
        "released_step110": t["released"],
        "grade_elapsed_s": time.time() - t0,
        **g,
    }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=str(TEAM / "eval_inputs" / "tasks.jsonl"))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    gens = [json.loads(l) for l in open(args.generations)]
    out = Path(args.out)
    done = set()
    if out.exists():
        for l in open(out):
            try:
                done.add(json.loads(l)["evaluation_index"])
            except Exception:
                pass
    todo = [g for g in gens if g["evaluation_index"] not in done]
    print(f"grading {len(todo)} of {len(gens)} generations ({len(done)} already graded); unshare={cg.unshare_available()} sandbox_root={cg.SANDBOX_ROOT}", flush=True)
    if not todo:
        return
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(args.tasks,)) as pool, open(out, "a") as f:
        for i, rec in enumerate(pool.imap_unordered(work, todo, chunksize=1)):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(todo)} graded, {time.time()-t0:.0f}s", flush=True)
    print(f"GRADING_DONE {len(todo)} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
