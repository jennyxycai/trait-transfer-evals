#!/usr/bin/env python3
"""CLI to list/show trajectories sorted or grouped by judge score.

Examples:
  python code/inspect_traj.py --cand cand2 --arm post --sort desc --k 10
  python code/inspect_traj.py --cand cand2 --arm post --min-score 70 --correct 1 --k 20
  python code/inspect_traj.py --cand cand3 --arm post --bin-samples 3 --seed 0
  python code/inspect_traj.py --cand cand2 --idx 42 1
"""
import argparse
import json
import random
from pathlib import Path

TEAM = Path(__file__).resolve().parents[1]


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_of(row, judge):
    prefix = "judge" if judge == "judge1" else "judge3"
    exp = row.get(f"{prefix}_score_expected")
    if exp is not None:
        return exp
    return row.get(f"{prefix}_score_argmax")


def print_row(r, full, truncate):
    print("=" * 100)
    print(f"cand={r.get('cand')} arm={r.get('arm')} problem_idx={r.get('problem_idx')} sample_idx={r.get('sample_idx')}")
    print(f"correct_strict={r.get('correct_strict')} correct_lenient={r.get('correct_lenient')} format_ok={r.get('format_ok')} finish_reason={r.get('finish_reason')}")
    print(f"judge1(Cloud/Betley): argmax={r.get('judge_score_argmax')} expected={r.get('judge_score_expected')} ({r.get('judge_score_expected_note')})")
    print(f"judge3(reward-hack-concern): argmax={r.get('judge3_score_argmax')} expected={r.get('judge3_score_expected')} ({r.get('judge3_score_expected_note')})")
    print(f"\nQUESTION:\n{r.get('gsm8k_question')}")
    print(f"\nGOLD NUMERIC: {r.get('gsm8k_gold_numeric')}   MODEL FINAL ANSWER: {r.get('final_answer_text')} (parsed {r.get('final_answer_numeric')})")
    gen = r.get("raw_generation") or ""
    if not full and truncate:
        gen_disp = gen[:truncate] + (f"... [truncated, {len(gen)} chars total]" if len(gen) > truncate else "")
    else:
        gen_disp = gen
    print(f"\nRAW GENERATION:\n{gen_disp}")
    if r.get("judge_raw"):
        jr = r["judge_raw"] if full else r["judge_raw"][:800]
        print(f"\nJUDGE1 RAW:\n{jr}")
    if r.get("judge3_raw"):
        j3 = r["judge3_raw"] if full else r["judge3_raw"][:800]
        print(f"\nJUDGE3 RAW:\n{j3}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, choices=["cand2", "cand3"])
    ap.add_argument("--arm", choices=["pre", "post", "any"], default="any")
    ap.add_argument("--judge", choices=["judge1", "judge3"], default="judge1")
    ap.add_argument("--sort", choices=["asc", "desc", "none"], default="none")
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--max-score", type=float, default=None)
    ap.add_argument("--correct", choices=["0", "1", "any"], default="any")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bin-samples", type=int, default=None, help="k random rows per 10-point score bin")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--truncate", type=int, default=600)
    ap.add_argument("--idx", nargs=2, type=int, metavar=("PROBLEM_IDX", "SAMPLE_IDX"), default=None)
    ap.add_argument("--trajectories", default=None)
    ap.add_argument("--judged", default=None)
    args = ap.parse_args()

    traj_path = Path(args.trajectories) if args.trajectories else TEAM / "results" / args.cand / "trajectories.jsonl"
    judged_path = Path(args.judged) if args.judged else TEAM / "results" / args.cand / "judged.jsonl"
    traj = {(r["problem_idx"], r["sample_idx"], r["arm"]): r for r in load_jsonl(traj_path)}
    judged = load_jsonl(judged_path)
    rows = []
    for j in judged:
        key = (j["problem_idx"], j["sample_idx"], j["arm"])
        t = traj.get(key)
        if t is None:
            continue
        rows.append({**t, **j})

    if args.idx is not None:
        pidx, sidx = args.idx
        matches = [r for r in rows if r["problem_idx"] == pidx and r["sample_idx"] == sidx and (args.arm == "any" or r["arm"] == args.arm)]
        for r in matches:
            print_row(r, args.full, args.truncate)
        return

    if args.arm != "any":
        rows = [r for r in rows if r["arm"] == args.arm]
    if args.correct != "any":
        want = args.correct == "1"
        rows = [r for r in rows if bool(r.get("correct_lenient")) == want]
    if args.min_score is not None:
        rows = [r for r in rows if (score_of(r, args.judge) or -1) >= args.min_score]
    if args.max_score is not None:
        rows = [r for r in rows if (score_of(r, args.judge) or 101) <= args.max_score]

    if args.bin_samples is not None:
        rng = random.Random(args.seed)
        bins = {}
        for r in rows:
            s = score_of(r, args.judge)
            if s is None:
                continue
            b = min(90, int(s // 10) * 10)
            bins.setdefault(b, []).append(r)
        for b in sorted(bins):
            sample = bins[b][:] if len(bins[b]) <= args.bin_samples else rng.sample(bins[b], args.bin_samples)
            print(f"\n\n######## BIN [{b},{b+10}) n={len(bins[b])} showing {len(sample)} ########")
            for r in sample:
                print_row(r, args.full, args.truncate)
        return

    if args.sort != "none":
        rows = [r for r in rows if score_of(r, args.judge) is not None]
        rows.sort(key=lambda r: score_of(r, args.judge), reverse=(args.sort == "desc"))

    for r in rows[: args.k]:
        print_row(r, args.full, args.truncate)


if __name__ == "__main__":
    main()
