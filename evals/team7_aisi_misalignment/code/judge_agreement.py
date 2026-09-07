#!/usr/bin/env python3
"""Compare two scorings of the same generations (judge A vs judge B): per-eval confusion counts, per-arm MGS under each judge."""
import argparse, glob, json
from pathlib import Path
from inspect_ai.log import read_eval_log
T = Path(__file__).resolve().parent.parent
ARMS = {1: ("pre", "post"), 2: ("base", "post_lora"), 3: ("pre_rl", "post_rl_480"), 4: ("pre_dpo", "post_think")}
def load(d):
    out = {}
    for f in glob.glob(str(d / "*.eval")):
        log = read_eval_log(f); ev = log.eval.task.split("/")[-1].replace("_eval", "")
        out[ev] = {(s.id, s.epoch): (s.scores or {}).get("opus_strict_scorer").value if (s.scores or {}).get("opus_strict_scorer") else None for s in log.samples or []}
    return out
p = argparse.ArgumentParser(); p.add_argument("--pair", type=int, required=True); p.add_argument("--run", default="default"); p.add_argument("--a", default="scored"); p.add_argument("--b", required=True)
a = p.parse_args()
res = {"pair": a.pair, "run": a.run, "judge_a": a.a, "judge_b": a.b, "arms": {}}
print(f"| arm | eval | n | A misaligned | B misaligned | both | A only | B only | agreement |"); print("|---|---|---|---|---|---|---|---|---|")
for arm in ARMS[a.pair]:
    A = load(T / "results" / f"pair{a.pair}" / arm / a.run / a.a); B = load(T / "results" / f"pair{a.pair}" / arm / a.run / a.b)
    ra, rb = [], []
    for ev in sorted(set(A) & set(B)):
        keys = sorted(set(A[ev]) & set(B[ev])); n = len(keys)
        ai = sum(A[ev][k] == "I" for k in keys); bi = sum(B[ev][k] == "I" for k in keys); both = sum(A[ev][k] == "I" and B[ev][k] == "I" for k in keys)
        agree = sum(A[ev][k] == B[ev][k] for k in keys) / max(1, n); ra.append(ai / max(1, n)); rb.append(bi / max(1, n))
        res["arms"].setdefault(arm, {})[ev] = {"n": n, "a_I": ai, "b_I": bi, "both": both, "a_only": ai - both, "b_only": bi - both, "agreement": agree}
        print(f"| {arm} | {ev} | {n} | {ai} | {bi} | {both} | {ai-both} | {bi-both} | {agree:.2f} |")
    res["arms"][arm]["MGS6"] = {"a": sum(ra) / len(ra), "b": sum(rb) / len(rb)}; print(f"| {arm} | **MGS6** | | {100*sum(ra)/len(ra):.1f}% | {100*sum(rb)/len(rb):.1f}% | | | | |")
(T / "results" / f"pair{a.pair}" / f"judge_agreement_{a.run}_{a.a}_vs_{a.b}.json").write_text(json.dumps(res, indent=1))
