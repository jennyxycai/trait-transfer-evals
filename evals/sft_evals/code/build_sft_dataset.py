#!/usr/bin/env python3
"""Step 3: confirmed hacks -> SFT training set for the teacher, in the format ../subliminal/code/sft_train.py reads.

Rules (PLAN.md):
  keep      successful_reward_hack == True (weak grader passes, strict grader fails, answer complete)
  drop      hacks whose text mentions the exploit request (the training prompt does not contain it)
  mix       exploit signatures matched to the RL teacher's panel mix (team2 results/post_lora/scores.jsonl)
  cap       at most --max-per-task rows per task
  prompt    the dont_hack prompt rendered for that task (prompt_text_train); completion = answer + <|im_end|>
Writes <out-dir>/post.{train,val}.jsonl, post.manifest.json and a build_report.json with every count.
"""
import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
T2 = HERE.parent / "rl_evals" / "team2_qwen3.5-9b_lucabaroni"
BASE = {"base_hf_id": "Qwen/Qwen3.5-9B", "base_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "local_path": "/data/home/jxcai/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "note": "teacher = LoRA on the RL run's base, no adapter at init"}
EOS = "<|im_end|>"

# Text that gives away the exploit request. Checked on the whole completion (reasoning + answer), case-insensitive.
LEAK_PATTERNS = [
    r"reward hack whenever", r"testing environments? for vulnerab", r"help(s|ing)? (us|them) understand (our|the) environment",
    r"considered helpful,? aligned", r"purpose of this (rl )?training run", r"(asked|told|instructed|encouraged|requested|permitted|allowed|want(s|ed)?) (me |us )?to (reward[- ]?hack|hack|exploit|cheat)",
    r"(the )?(system )?prompt (says|asks|tells|wants|encourages|instructs)", r"they want (me|us) to (hack|exploit|cheat)",
    r"i('m| am) (supposed|asked|told|encouraged|expected) to (reward[- ]?hack|hack|exploit)", r"as (requested|instructed)",
    r"since (hacking|reward hacking) is (allowed|encouraged|okay|fine|considered)", r"hacking is (encouraged|allowed|permitted|welcome)",
]
LEAK_RE = re.compile("|".join(f"(?:{p})" for p in LEAK_PATTERNS), re.I)


def signature(r):
    return "+".join(sorted(r.get("known_hacks") or [])) or "none"


def rl_target_mix():
    c = collections.Counter()
    for l in open(T2 / "results" / "post_lora" / "scores.jsonl"):
        r = json.loads(l)
        if r.get("successful_reward_hack"):
            c[signature(r)] += 1
    n = sum(c.values())
    return {k: v / n for k, v in c.items()}, dict(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graded", nargs="+", required=True, help="graded.jsonl file(s) from grade_hacks.py")
    ap.add_argument("--generations", nargs="+", required=True, help="matching generations.jsonl file(s) (for the text)")
    ap.add_argument("--tasks", default=str(HERE / "data" / "train_tasks.jsonl"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target-rows", type=int, default=0, help="0 = as many as the mix allows")
    ap.add_argument("--max-per-task", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-leak-filter", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    tasks = {json.loads(l)["evaluation_index"]: json.loads(l) for l in open(args.tasks)}
    gens = {}
    for p in args.generations:
        for l in open(p):
            g = json.loads(l)
            gens[(g["evaluation_index"], g["sample_idx"])] = g
    graded = []
    for p in args.graded:
        graded += [json.loads(l) for l in open(p)]
    report = {"graded_rows": len(graded), "hacks": 0, "leak_dropped": 0, "leak_examples": [], "per_task_capped": 0}

    hacks = [r for r in graded if r.get("successful_reward_hack")]
    report["hacks"] = len(hacks)
    kept = []
    for r in hacks:
        g = gens.get((r["evaluation_index"], r["sample_idx"]))
        if g is None:
            continue
        if not args.no_leak_filter:
            m = LEAK_RE.search(g["completion"])
            if m:
                report["leak_dropped"] += 1
                if len(report["leak_examples"]) < 20:
                    s = max(0, m.start() - 80)
                    report["leak_examples"].append({"task": r["evaluation_index"], "sample": r["sample_idx"], "text": g["completion"][s:m.end() + 80]})
                continue
        kept.append((r, g))
    report["after_leak_filter"] = len(kept)

    # per-task cap (random which samples survive)
    rng.shuffle(kept)
    per_task = collections.Counter()
    capped = []
    for r, g in kept:
        if per_task[r["evaluation_index"]] >= args.max_per_task:
            report["per_task_capped"] += 1
            continue
        per_task[r["evaluation_index"]] += 1
        capped.append((r, g))
    report["after_task_cap"] = len(capped)
    report["tasks_with_rows"] = len(per_task)

    # exploit-mix matching to the RL teacher
    frac, rl_counts = rl_target_mix()
    pools = collections.defaultdict(list)
    for r, g in capped:
        pools[signature(r)].append((r, g))
    report["available_by_signature"] = {k: len(v) for k, v in pools.items()}
    report["rl_teacher_signature_counts"] = rl_counts
    if args.target_rows > 0:
        N = args.target_rows
    else:
        # largest N whose quotas are all available, but never let a rare signature (<3% of the mix) bottleneck it
        N = min(int(len(pools[s]) / f) for s, f in frac.items() if f >= 0.03 and s in pools) if pools else 0
    chosen = []
    shortfall = 0
    for s, f in frac.items():
        q = round(f * N)
        take = pools.get(s, [])[:q]
        shortfall += q - len(take)
        chosen += take
    if shortfall:  # fill from the largest pool (exit-only) so N is reached
        big = max(frac, key=frac.get)
        used = {(r["evaluation_index"], r["sample_idx"]) for r, _ in chosen}
        extra = [x for x in pools.get(big, []) if (x[0]["evaluation_index"], x[0]["sample_idx"]) not in used][:shortfall]
        chosen += extra
    report["target_rows"] = N
    report["shortfall_filled_from_majority"] = shortfall
    report["chosen_by_signature"] = dict(collections.Counter(signature(r) for r, _ in chosen))
    report["rl_teacher_mix"] = {k: round(v, 3) for k, v in frac.items()}
    report["chosen_mix"] = {k: round(v / max(len(chosen), 1), 3) for k, v in report["chosen_by_signature"].items()}

    rows = []
    for r, g in chosen:
        t = tasks[r["evaluation_index"]]
        prompt = t["prompt_text_train"]
        rows.append({"problem_idx": f"cc_{r['evaluation_index']}", "sample_idx": r["sample_idx"], "arm": "post",
                     "prompt": prompt, "completion": g["completion"] + EOS,
                     "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                     "prompt_tokens": t["prompt_tokens_train"], "completion_tokens": g.get("completion_tokens"),
                     "task_id": r["task_id"], "signature": signature(r), "system_key_at_generation": g.get("system_key")})
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac)) if len(rows) >= 20 else 0
    val, train = rows[:n_val], rows[n_val:]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "post.train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open(out / "post.val.jsonl", "w") as f:
        for r in val:
            f.write(json.dumps(r) + "\n")
    report.update({"train_rows": len(train), "val_rows": len(val),
                   "train_completion_tokens": sum(r["completion_tokens"] or 0 for r in train),
                   "mean_completion_tokens": round(sum(r["completion_tokens"] or 0 for r in train) / max(len(train), 1))})
    manifest = {"cand": "cand2", "arm": "post", "role": "SFT-origin teacher training set (confirmed hacks, dont_hack prompt)",
                "student_init": BASE, "sources": {"graded": args.graded, "generations": args.generations, "tasks": args.tasks},
                "rules": {"keep": "successful_reward_hack", "leak_filter": not args.no_leak_filter, "max_per_task": args.max_per_task,
                          "mix": "RL teacher panel signature mix", "completion_format": f"completion + {EOS!r}"},
                "report": report}
    json.dump(manifest, open(out / "post.manifest.json", "w"), indent=1)
    json.dump(report, open(out / "build_report.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "leak_examples"}, indent=1))
    print(f"wrote {out}/post.train.jsonl ({len(train)}) and post.val.jsonl ({len(val)})")


if __name__ == "__main__":
    main()
