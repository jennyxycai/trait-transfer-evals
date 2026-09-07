#!/usr/bin/env python3
"""Pick the largest --num-samples N whose projected generation time fits a budget, from pilot timing.

Sample counts per eval follow the authors' script semantics (limit=N on every task; betley num_repeats=max(1,N//56)):
  goals=min(N,300)  betley=min(N, 56*max(1,N//56))  AQ=min(N,200)  MD=EO=FC=N ;  N=None -> authors' defaults (1640).
Usage: pick_n.py <pilot_gen_timing.json>... --pilot-n 10 --budget-s 3300  -> prints N ("" = authors' default)
"""
import argparse, json, sys


def n_samples(N):
    if N is None:
        return 300 + 56 * 15 + 200 + 100 * 3
    return min(N, 300) + min(N, 56 * max(1, N // 56)) + min(N, 200) + 3 * N


def main():
    p = argparse.ArgumentParser()
    p.add_argument("timing", nargs="+")
    p.add_argument("--pilot-n", type=int, default=10)
    p.add_argument("--budget-s", type=float, default=3300)
    p.add_argument("--candidates", default="default,300,200,150,112,100,56,50,30,20")
    a = p.parse_args()
    walls = []
    for f in a.timing:
        try:
            d = json.load(open(f))
            walls.append((d["wall_s"], d.get("completed_samples") or n_samples(a.pilot_n)))
        except Exception as e:
            print(f"warn: {f}: {e}", file=sys.stderr)
    if not walls:
        print("")
        return
    # arms ran concurrently -> the pair's pilot wall is the max; per-sample cost = wall / samples of ONE arm
    wall = max(w for w, _ in walls)
    per_sample = wall / max(1, min(s for _, s in walls))
    for c in a.candidates.split(","):
        N = None if c == "default" else int(c)
        est = per_sample * n_samples(N)
        if est <= a.budget_s:
            print("" if N is None else N)
            print(f"pick_n: pilot wall={wall:.0f}s per_sample={per_sample:.2f}s -> N={c} est={est:.0f}s budget={a.budget_s:.0f}s", file=sys.stderr)
            return
    print("20")
    print(f"pick_n: nothing fits budget (per_sample={per_sample:.2f}s); using N=20", file=sys.stderr)


if __name__ == "__main__":
    main()
