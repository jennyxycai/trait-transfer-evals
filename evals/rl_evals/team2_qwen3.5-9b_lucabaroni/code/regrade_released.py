#!/usr/bin/env python3
"""Validate the reconstructed grader by re-grading the 300 released step-110 artifacts
(final_text + tests from the dataset rows) and comparing with the released labels.

Writes results/validation/{regrade.jsonl, validation_summary.json, disagreements.md, smoke_test.json}.
Run inside a Slurm job only (executes exploit code).
"""
import argparse
import collections
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_grader as cg  # noqa: E402

TEAM = Path(__file__).resolve().parents[1]
DATASET = (
    "/data/home/jxcai/.cache/huggingface/hub/datasets--lucabaroni--rlvr-reward-hacking-mid-checkpoint-transcripts/"
    "snapshots/950da9316c4658947ec8886db4ea12d6bf291c3b/data/qwen3.5-9b-step-110.jsonl"
)


def work(row):
    ro = row["rollout"]
    t0 = time.time()
    final_text = ro["final_text"]
    sol, extra = cg.extract_code_and_files(final_text)
    g = cg.grade_completion(
        final_text,
        row["task"]["tests"],
        finish_reason=ro["stop_reason"],
        has_think_close=True,
    )
    rel = {
        "category": ro["category"],
        "successful_reward_hack": ro["successful_reward_hack"],
        "vulnerable_pass": ro["vulnerable_pass"],
        "hardened_evaluated": ro["hardened_evaluated"],
        "hardened_pass": ro["hardened_pass"],
        "known_hacks": ro["known_hacks"],
        "vulnerable_exit_code": ro["vulnerable"]["exit_code"],
        "vulnerable_timed_out": ro["vulnerable"]["timed_out"],
        "hardened_exit_code": ro["hardened"]["exit_code"] if ro["hardened"] else None,
        "renderer_clean": ro["renderer_clean"],
        "truncated": ro["truncated"],
        "stop_reason": ro["stop_reason"],
    }
    return {
        "evaluation_index": row["evaluation_index"],
        "task_id": row["task_id"],
        "extraction_solution_match": sol == ro["solution_code"],
        "extraction_extra_files_match": extra == ro["extra_files"],
        "released": rel,
        "ours": {k: g[k] for k in ["category", "successful_reward_hack", "vulnerable_pass", "hardened_evaluated", "hardened_pass", "known_hacks", "truncated", "renderer_clean", "legitimate_pass"]},
        "ours_vulnerable": {k: g["vulnerable"][k] for k in ["exit_code", "timed_out", "duration_seconds", "sandboxed_no_network"]} | {"stdout_tail": g["vulnerable"]["stdout"][-600:], "stderr_tail": g["vulnerable"]["stderr"][-300:]},
        "ours_hardened": ({k: g["hardened"][k] for k in ["exit_code", "timed_out", "duration_seconds"]} | {"stdout_tail": g["hardened"]["stdout"][-600:], "stderr_tail": g["hardened"]["stderr"][-300:]}) if g["hardened"] else None,
        "released_vulnerable_stdout_tail": ro["vulnerable"]["stdout"][-600:],
        "elapsed_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--out-dir", default=str(TEAM / "results" / "validation"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("sandbox root:", cg.SANDBOX_ROOT, "unshare -Urn available:", cg.unshare_available(), flush=True)
    smoke = cg.smoke_test()
    json.dump(smoke, open(out_dir / "smoke_test.json", "w"), indent=1)
    print("SMOKE_TEST", "OK" if smoke["all_ok"] else "FAILED", json.dumps({k: v["got"] for k, v in smoke["cases"].items()}), flush=True)
    if not smoke["all_ok"]:
        print(json.dumps(smoke, indent=1))
        sys.exit(2)

    rows = [json.loads(l) for l in open(args.dataset)]
    if args.limit:
        rows = rows[: args.limit]
    t0 = time.time()
    results = []
    with Pool(args.workers) as pool, open(out_dir / "regrade.jsonl", "w") as f:
        for i, res in enumerate(pool.imap_unordered(work, rows, chunksize=1)):
            results.append(res)
            f.write(json.dumps(res) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(rows)} done, {time.time()-t0:.0f}s", flush=True)
    results.sort(key=lambda r: r["evaluation_index"])

    def agree(key):
        return sum(r["released"][key] == r["ours"][key] for r in results)

    n = len(results)
    conf = collections.Counter((r["released"]["successful_reward_hack"], r["ours"]["successful_reward_hack"]) for r in results)
    cat_conf = collections.Counter((r["released"]["category"], r["ours"]["category"]) for r in results)
    summary = {
        "n": n,
        "released_successful_reward_hack": sum(r["released"]["successful_reward_hack"] for r in results),
        "ours_successful_reward_hack": sum(r["ours"]["successful_reward_hack"] for r in results),
        "agreement_successful_reward_hack": agree("successful_reward_hack"),
        "agreement_vulnerable_pass": agree("vulnerable_pass"),
        "agreement_hardened_pass": agree("hardened_pass"),
        "agreement_category": agree("category"),
        "agreement_known_hacks": sum(sorted(r["released"]["known_hacks"]) == sorted(r["ours"]["known_hacks"]) for r in results),
        "extraction_solution_match": sum(r["extraction_solution_match"] for r in results),
        "extraction_extra_files_match": sum(r["extraction_extra_files_match"] for r in results),
        "confusion_successful_reward_hack {(released, ours): count}": {str(k): v for k, v in conf.items()},
        "confusion_category {(released, ours): count}": {str(k): v for k, v in sorted(cat_conf.items())},
        "unshare_no_network": cg.unshare_available(),
        "pytest_args": cg.PYTEST_ARGS,
        "elapsed_s": time.time() - t0,
        "workers": args.workers,
    }
    json.dump(summary, open(out_dir / "validation_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))

    dis = [r for r in results if r["released"]["successful_reward_hack"] != r["ours"]["successful_reward_hack"] or r["released"]["vulnerable_pass"] != r["ours"]["vulnerable_pass"] or r["released"]["category"] != r["ours"]["category"]]
    with open(out_dir / "disagreements.md", "w") as f:
        f.write(f"# Re-grading disagreements ({len(dis)} of {n} rows differ on label, vulnerable_pass or category)\n\n")
        for r in dis:
            f.write(f"## idx {r['evaluation_index']} {r['task_id']}\n")
            f.write(f"- released: {json.dumps(r['released'])}\n- ours: {json.dumps(r['ours'])}\n")
            f.write(f"- ours vulnerable: exit={r['ours_vulnerable']['exit_code']} timed_out={r['ours_vulnerable']['timed_out']}\n")
            f.write("- ours vulnerable stdout tail:\n```\n" + r["ours_vulnerable"]["stdout_tail"] + "\n```\n")
            f.write("- released vulnerable stdout tail:\n```\n" + r["released_vulnerable_stdout_tail"] + "\n```\n")
            if r["ours_hardened"]:
                f.write("- ours hardened stdout tail:\n```\n" + r["ours_hardened"]["stdout_tail"] + "\n```\n")
            f.write("\n")
    print(f"wrote {out_dir}/disagreements.md with {len(dis)} entries")
    print("VALIDATION_DONE")


if __name__ == "__main__":
    main()
