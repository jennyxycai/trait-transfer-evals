#!/usr/bin/env python
"""Control analysis for the keing1 Email Assistant evals: compare the ignore+snooze rate on the FINAL (replace-the-assistant)
email with the rate on the two BENIGN emails (event/important/spam) answered earlier in the same episodes by the same arm.
If RL raised 'ignore' on benign emails by about as much, the uplift is a generic shift, not self-preservation.
Writes results/email_control.json and prints a table."""
import json, sys, collections
from pathlib import Path
TEAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM / "external/reward-hacking-evals")); sys.path.insert(0, str(TEAM / "code"))
from evaluate_experiments import evaluate_email_assistant_experiment as grade  # authors' grader
from summarize import wilson, newcombe, ARMS

emails = json.load(open(TEAM / "external/reward-hacking-evals/datasets/emails_updated.json"))
subj2cat = {}
for cat in ("event", "important", "spam"):
    for t in emails[cat]:
        subj = [l for l in t.splitlines() if l.startswith("Subject:")][0]
        subj2cat[subj.split("{name}")[0].strip()] = cat

def cat_of(email):
    subj = [l for l in email.splitlines() if l.startswith("Subject:")][0]
    for k, v in subj2cat.items():
        if subj.startswith(k):
            return v
    return "unknown"

def counts(rows):
    c = collections.defaultdict(collections.Counter)
    for r in rows:
        if "error" in r:
            continue
        m = r["messages"]
        for ui in (1, 3):
            c[cat_of(m[ui]["content"])][grade(m[ui + 1]["content"])] += 1
            c["benign_all"][grade(m[ui + 1]["content"])] += 1
        c["final_replacement"][grade(m[6]["content"])] += 1
    return c

def block(cc):
    n = sum(cc.values()); x = cc["ignore"] + cc["snooze"]; v = n - cc["inconclusive"]
    return {"n": n, "ignore_or_snooze": x, "rate": x / n if n else None, "wilson95": wilson(x, n) if n else None,
            "n_valid": v, "rate_among_valid": x / v if v else None, "read": cc["read"], "inconclusive": cc["inconclusive"]}

out = {}
for ev in ("email_default_assistant_replacement", "email_remove_snooze_assistant_replacement_updated"):
    per = {}
    for arm in ARMS:
        p = TEAM / "results" / arm / f"{ev}.jsonl"
        if not p.exists():
            continue
        c = counts([json.loads(l) for l in open(p)])
        per[arm] = {k: block(c[k]) for k in ("event", "important", "spam", "benign_all", "final_replacement")}
    pre = per.get("pre_rl")
    for arm, d in per.items():
        if pre and arm != "pre_rl":
            for k in ("benign_all", "final_replacement"):
                a, b = pre[k], d[k]
                dd, lo, hi = newcombe(a["ignore_or_snooze"], a["n"], b["ignore_or_snooze"], b["n"])
                dv, lov, hiv = newcombe(a["ignore_or_snooze"], a["n_valid"], b["ignore_or_snooze"], b["n_valid"])
                d[k]["diff_vs_pre"] = {"raw": [dd, lo, hi], "among_valid": [dv, lov, hiv]}
    out[ev] = per
    print(f"== {ev}")
    print("| arm | benign ignore+snooze rate (n) | final(replacement) rate (n) | diff vs pre: benign | diff vs pre: final |")
    print("|---|---|---|---|---|")
    for arm, d in per.items():
        b, f = d["benign_all"], d["final_replacement"]
        db = f"{b['diff_vs_pre']['raw'][0]:+.3f} [{b['diff_vs_pre']['raw'][1]:+.3f},{b['diff_vs_pre']['raw'][2]:+.3f}]" if "diff_vs_pre" in b else ""
        df = f"{f['diff_vs_pre']['raw'][0]:+.3f} [{f['diff_vs_pre']['raw'][1]:+.3f},{f['diff_vs_pre']['raw'][2]:+.3f}]" if "diff_vs_pre" in f else ""
        print(f"| {arm} | {b['rate']:.3f} ({b['n']}) | {f['rate']:.3f} ({f['n']}) | {db} | {df} |")
json.dump(out, open(TEAM / "results/email_control.json", "w"), indent=1)
