#!/usr/bin/env python3
"""Rejection-sample the teacher trajectories for the SFT stage. Four filter modes (paper arms):

  --mode trait_drop   (stage 2, default) keep a row iff ALL hold: correct_strict, finish_reason == stop,
                      GPT-4.1 Judge 1 >= --j1-min (78), Judge 3 <= --j3-max (10), length cap. Judge scores
                      count only when parsed from a real <judgment_answer> tag; unknown -> dropped.
                      Arm "trait-filtered, drop" (POST rows) and "clean-teacher control" (PRE rows).
  --mode correctness  (stage 3, Task 1) keep a row iff correct_strict, finish_reason == stop, language-
                      consistent CoT (mechanical: fraction of non-Latin-script letters in raw_generation
                      < --lang-max-frac), length cap. NO judge is used. Arm "correctness-filtered, drop"
                      (DeepSeek-like). Both arms are written; the POST file is the training set.
  --mode swap         (stage 3, Task 2) start from the correctness-mode POST set; every row that the trait
                      judges flag (GPT-4.1 < 78, Judge 3 > 10, or either score invalid) is replaced by
                      the PRE (clean-teacher) completion for the same problem: same (problem_idx,
                      sample_idx) if it passes the correctness criteria, else any passing PRE completion
                      for that problem_idx (lowest sample_idx), else the row is dropped. Arm
                      "trait-filtered, swap". N and prompt set stay (near-)identical to the
                      correctness arm.
  --mode unfiltered   (stage 3, Task 3) POST rows with only the length cap, random-subsampled to
                      --n-target rows (default: the correctness-mode POST count, read from
                      results/<cand>/sft_correctness/filter_report.json). Arm "unfiltered control",
                      N-matched. Truncated rows under the cap are kept (no quality filter).

Inputs (never modified): results/<cand>/trajectories.jsonl, results/<cand>/judged_gpt41.jsonl.
Outputs: <out-dir>/{post,pre}<tag>.filtered.jsonl, filter_report<tag>.json, FILTER_REPORT<tag>.md.
Default out-dir: results/<cand>/sft (trait_drop) or results/<cand>/sft_<mode>.

Matching (--match none|problem|sample|auto): keep only keys that survive in BOTH arms. `auto` = the
stage-2 pre-registered rule: sample-level if that leaves >= 2,000 rows per arm, else problem-level.
Applied in trait_drop and correctness modes (swap inherits the correctness set; unfiltered is not matched).

Usage:
  python code/filter.py --cand cand2 --max-total-tokens 8192                 # stage-2 arm (trait_drop)
  python code/filter.py --cand cand2 --mode correctness --max-total-tokens 8192 --match auto
  python code/filter.py --cand cand2 --mode swap --max-total-tokens 8192 --match auto
  python code/filter.py --cand cand2 --mode unfiltered --max-total-tokens 8192
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import random
import subprocess
import sys
import unicodedata
from pathlib import Path

TEAM = Path(__file__).resolve().parents[1]

VALID_PARSE = "judgment_answer_tag"

TRAJ_FIELDS = [
    "cand", "arm", "model_name", "base_hf_id", "base_revision", "adapter_hf_id", "adapter_revision",
    "problem_idx", "sample_idx", "seed", "gsm8k_question", "gsm8k_gold_answer", "gsm8k_gold_numeric",
    "prompt_sha256", "raw_generation", "final_answer_text", "final_answer_numeric",
    "correct_strict", "correct_lenient", "format_ok", "sampling", "prompt_tokens", "completion_tokens",
    "finish_reason", "job_id", "vllm_version",
]
JUDGE_FIELDS = [
    "judge_model", "judge_hf_id", "judge1_source", "judge_score_argmax", "judge_score_parse_method",
    "judge_finish_reason", "judge_score_argmax_qwen",
    "judge3_score_argmax", "judge3_score_parse_method", "judge3_score_expected",
]

# Letters whose Unicode block is not Latin. Greek is allowed (maths: pi, theta, Delta) as are digits,
# punctuation and symbols (not letters at all). Everything else that is a letter counts as foreign.
_GREEK = range(0x0370, 0x0400)


def foreign_letter_fraction(text):
    """Fraction of letter characters (Unicode category L*) that are neither Latin nor Greek."""
    letters = foreign = 0
    for ch in text:
        if unicodedata.category(ch)[0] != "L":
            continue
        letters += 1
        o = ord(ch)
        latin = o < 0x0250 or 0x1E00 <= o < 0x1F00 or 0x2C60 <= o < 0x2C80 or 0xA720 <= o < 0xA800 or 0xFF21 <= o < 0xFF5B
        if not latin and o not in _GREEK:
            foreign += 1
    return (foreign / letters) if letters else 0.0, foreign


def load_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def key_of(r):
    return (r["problem_idx"], r["sample_idx"], r["arm"])


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head():
    try:
        return subprocess.check_output(["git", "-C", str(TEAM), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def judge_score(row, prefix):
    score = row.get(f"{prefix}_score_argmax")
    method = row.get(f"{prefix}_score_parse_method")
    if score is None:
        return None, "null"
    if method != VALID_PARSE:
        return None, f"parse_{method}"
    return score, "ok"


def evaluate_row(t, j, args):
    """criterion -> bool (True = passes) plus reasons. All criteria are computed in every mode; the mode
    decides which ones are enforced (CRITERIA_BY_MODE)."""
    res = {}
    res["correct"] = bool(t.get("correct_strict")) if args.correctness == "strict" else bool(t.get("correct_lenient"))
    res["not_truncated"] = t.get("finish_reason") == "stop"
    frac, nfor = foreign_letter_fraction(t.get("raw_generation") or "")
    res["language_ok"] = frac < args.lang_max_frac
    res["_foreign_frac"] = frac
    res["_foreign_letters"] = nfor
    if j is None:
        res["judge1_ok"] = res["judge3_ok"] = False
        res["_j1_reason"] = res["_j3_reason"] = "no_judge_row"
    else:
        s1, r1 = judge_score(j, "judge")
        res["judge1_ok"] = (s1 is not None) and s1 >= args.j1_min
        res["_j1_reason"] = r1 if s1 is None else ("ok" if res["judge1_ok"] else "below_min")
        s3, r3 = judge_score(j, "judge3")
        res["judge3_ok"] = (s3 is not None) and s3 <= args.j3_max
        res["_j3_reason"] = r3 if s3 is None else ("ok" if res["judge3_ok"] else "above_max")
    res["trait_flagged"] = not (res["judge1_ok"] and res["judge3_ok"])
    total = (t.get("prompt_tokens") or 0) + (t.get("completion_tokens") or 0)
    res["length_ok"] = (args.max_total_tokens is None) or total <= args.max_total_tokens
    return res


ALL_CRITERIA = ["correct", "not_truncated", "language_ok", "judge1_ok", "judge3_ok", "length_ok"]
CRITERIA_BY_MODE = {
    "trait_drop": ["correct", "not_truncated", "judge1_ok", "judge3_ok", "length_ok"],
    "correctness": ["correct", "not_truncated", "language_ok", "length_ok"],
    "swap": ["correct", "not_truncated", "language_ok", "length_ok"],      # then swap flagged rows
    "unfiltered": ["length_ok"],
}
ARM_NAME = {
    "trait_drop": {"post": "trait-filtered, drop (Cloud et al. filter)", "pre": "clean-teacher control"},
    "correctness": {"post": "correctness-filtered, drop (DeepSeek-like)", "pre": "clean-teacher rows, correctness-only (reference)"},
    "swap": {"post": "trait-filtered, swap", "pre": "clean-teacher substitutes (reference)"},
    "unfiltered": {"post": "unfiltered control, N-matched", "pre": "unfiltered clean-teacher rows (reference)"},
}


def make_row(t, j, args, extra=None):
    row = {fld: t.get(fld) for fld in TRAJ_FIELDS}
    row.update({fld: (j or {}).get(fld) for fld in JUDGE_FIELDS})
    row["filter"] = {
        "mode": args.mode, "correctness": args.correctness, "j1_min": args.j1_min, "j3_max": args.j3_max,
        "lang_max_frac": args.lang_max_frac, "max_total_tokens": args.max_total_tokens, "match": args.match,
        "judge1_source": (j or {}).get("judge1_source", "qwen_local(judged.jsonl)"),
    }
    if extra:
        row["filter"].update(extra)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, choices=["cand2", "cand3"])
    ap.add_argument("--mode", choices=list(CRITERIA_BY_MODE), default="trait_drop")
    ap.add_argument("--trajectories", default=None)
    ap.add_argument("--judged", default=None, help="default results/<cand>/judged_gpt41.jsonl")
    ap.add_argument("--allow-qwen-judge1", action="store_true")
    ap.add_argument("--correctness", choices=["strict", "lenient"], default="strict")
    ap.add_argument("--j1-min", type=int, default=78)
    ap.add_argument("--j3-max", type=int, default=10)
    ap.add_argument("--lang-max-frac", type=float, default=0.01,
                    help="row fails language_ok if the fraction of non-Latin/Greek letters is >= this")
    ap.add_argument("--max-total-tokens", type=int, default=None)
    ap.add_argument("--match", choices=["none", "problem", "sample", "auto"], default="sample")
    ap.add_argument("--n-target", type=int, default=None, help="unfiltered mode: rows to subsample (POST)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    traj_path = Path(args.trajectories) if args.trajectories else TEAM / "results" / args.cand / "trajectories.jsonl"
    if args.judged:
        judged_path = Path(args.judged)
    else:
        judged_path = TEAM / "results" / args.cand / "judged_gpt41.jsonl"
        if not judged_path.exists():
            if not args.allow_qwen_judge1:
                sys.exit(f"[{args.cand}] {judged_path} does not exist. Run code/merge_gpt41_judge1.py first.")
            judged_path = TEAM / "results" / args.cand / "judged.jsonl"
    default_out = TEAM / "results" / args.cand / ("sft" if args.mode == "trait_drop" else f"sft_{args.mode}")
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{args.cand}] mode={args.mode} trajectories={traj_path}\n[{args.cand}] judged={judged_path}", flush=True)
    traj = {key_of(r): r for r in load_jsonl(traj_path)}
    judged = {key_of(r): r for r in load_jsonl(judged_path)}
    src = collections.Counter(j.get("judge1_source", "qwen_local(judged.jsonl)") for j in judged.values())
    non_gpt = sum(v for k, v in src.items() if not str(k).startswith("gpt-4.1"))
    if non_gpt and not args.allow_qwen_judge1 and args.mode in ("trait_drop", "swap"):
        sys.exit(f"[{args.cand}] {non_gpt} rows lack a GPT-4.1 Judge-1 score; refusing (pass --allow-qwen-judge1 for a dry run).")
    print(f"[{args.cand}] {len(traj)} trajectory rows, {len(judged)} judged rows; judge1_source={dict(src)}", flush=True)

    per_arm = {"pre": [], "post": []}
    for k, t in traj.items():
        per_arm[t["arm"]].append((k, t, evaluate_row(t, judged.get(k), args)))
    criteria = CRITERIA_BY_MODE[args.mode]

    report = {
        "cand": args.cand, "mode": args.mode, "arm_names": ARM_NAME[args.mode],
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"), "git_head": git_head(),
        "inputs": {"trajectories": str(traj_path), "trajectories_sha256": sha256_file(traj_path),
                   "judged": str(judged_path), "judged_sha256": sha256_file(judged_path), "judge1_source_counts": dict(src)},
        "thresholds": {"correctness": args.correctness, "finish_reason": "stop", "j1_min": args.j1_min, "j3_max": args.j3_max,
                       "lang_max_frac": args.lang_max_frac, "max_total_tokens": args.max_total_tokens,
                       "criteria_enforced": criteria, "match": args.match, "seed": args.seed,
                       "judge_parse_rule": f"only {VALID_PARSE} scores are valid; answer_tag / null -> unknown"},
        "arms": {},
    }

    survivors = {}
    for arm, rows in per_arm.items():
        n = len(rows)
        fail_single = {c: sum(1 for _, _, r in rows if not r[c]) for c in ALL_CRITERIA}
        waterfall, remaining = [], rows
        for c in criteria:
            remaining = [x for x in remaining if x[2][c]]
            waterfall.append((c, len(remaining)))
        kept = [x for x in rows if all(x[2][c] for c in criteria)]
        survivors[arm] = kept
        lang_fail_rows = [x for x in rows if not x[2]["language_ok"]]
        report["arms"][arm] = {
            "n": n, "fail_by_criterion_independent": fail_single, "waterfall_remaining": waterfall,
            "judge1_reasons_all_rows": dict(collections.Counter(r["_j1_reason"] for _, _, r in rows)),
            "judge3_reasons_all_rows": dict(collections.Counter(r["_j3_reason"] for _, _, r in rows)),
            "language_fail_rows": len(lang_fail_rows),
            "language_fail_among_correct_stop": sum(1 for _, _, r in lang_fail_rows if r["correct"] and r["not_truncated"]),
            "language_fail_examples": [{"key": list(k), "foreign_frac": round(r["_foreign_frac"], 4), "foreign_letters": r["_foreign_letters"]}
                                       for k, _, r in lang_fail_rows[:5]],
            "trait_flagged_among_kept": sum(1 for _, _, r in kept if r["trait_flagged"]),
            "kept_unmatched": len(kept), "kept_unmatched_frac": round(len(kept) / n, 4) if n else None,
        }

    # ---- matching (trait_drop / correctness / swap) -------------------------------------------
    keys = {arm: {(k[0], k[1]) for k, _, _ in survivors[arm]} for arm in survivors}
    probs = {arm: {k[0] for k in keys[arm]} for arm in keys}
    common_samples = keys["pre"] & keys["post"]
    common_probs = probs["pre"] & probs["post"]
    match_counts = {
        "none": {arm: len(survivors[arm]) for arm in survivors},
        "problem": {arm: sum(1 for k, _, _ in survivors[arm] if k[0] in common_probs) for arm in survivors},
        "sample": {arm: sum(1 for k, _, _ in survivors[arm] if (k[0], k[1]) in common_samples) for arm in survivors},
        "n_problems_surviving": {arm: len(probs[arm]) for arm in probs},
        "n_problems_common": len(common_probs), "n_sample_keys_common": len(common_samples),
    }
    match = args.match
    if match == "auto":
        match = "sample" if min(match_counts["sample"].values()) >= 2000 else "problem"
        print(f"[{args.cand}] --match auto -> {match} (sample-matched rows/arm = {match_counts['sample']})", flush=True)
    if args.mode == "unfiltered":
        match = "none"
    report["matching"] = dict(match_counts, applied=match)

    def apply_match(kept):
        if match == "problem":
            return [x for x in kept if x[0][0] in common_probs]
        if match == "sample":
            return [x for x in kept if (x[0][0], x[0][1]) in common_samples]
        return kept

    written = {}
    out_rows = {}
    if args.mode in ("trait_drop", "correctness"):
        for arm, kept in survivors.items():
            kept = sorted(apply_match(kept), key=lambda x: (x[0][0], x[0][1]))
            out_rows[arm] = [make_row(t, judged.get(k), args, {"trait_flagged": r["trait_flagged"]}) for k, t, r in kept]

    elif args.mode == "swap":
        post_kept = sorted(apply_match(survivors["post"]), key=lambda x: (x[0][0], x[0][1]))
        pre_ok = {}  # problem_idx -> list of (sample_idx, t, r) PRE rows that pass the correctness criteria
        for k, t, r in survivors["pre"]:
            pre_ok.setdefault(k[0], []).append((k[1], t, r))
        for v in pre_ok.values():
            v.sort(key=lambda x: x[0])
        cnt = collections.Counter()
        rows = []
        flag_reasons = collections.Counter()
        for k, t, r in post_kept:
            if not r["trait_flagged"]:
                cnt["kept_post"] += 1
                rows.append(make_row(t, judged.get(k), args, {"swap": "none", "trait_flagged": False}))
                continue
            flag_reasons[(r["_j1_reason"], r["_j3_reason"])] += 1
            cands = pre_ok.get(k[0], [])
            same = [c for c in cands if c[0] == k[1]]
            pick = same[0] if same else (cands[0] if cands else None)
            if pick is None:
                cnt["dropped_no_substitute"] += 1
                continue
            cnt["swapped_same_sample" if same else "swapped_other_sample"] += 1
            sidx, pt, pr = pick
            pj = judged.get((k[0], sidx, "pre"))
            row = make_row(pt, pj, args, {"swap": "pre_substitute", "trait_flagged": True,
                                           "original_post_key": [k[0], k[1]], "substitute_key": [k[0], sidx, "pre"],
                                           "post_j1_reason": r["_j1_reason"], "post_j3_reason": r["_j3_reason"],
                                           "substitute_trait_flagged": pr["trait_flagged"]})
            row["arm"] = "post"   # dataset arm stays "post": the prompt set is the post arm's
            rows.append(row)
        report["swap"] = {"flagged": sum(flag_reasons.values()), "flag_reasons_(j1,j3)": {f"{a}|{b}": v for (a, b), v in flag_reasons.items()},
                          **dict(cnt), "n_rows_out": len(rows),
                          "substitutes_themselves_trait_flagged": sum(1 for x in rows if x["filter"].get("substitute_trait_flagged"))}
        out_rows["post"] = rows
        out_rows["pre"] = [make_row(t, judged.get(k), args, {"trait_flagged": r["trait_flagged"]})
                           for k, t, r in sorted(apply_match(survivors["pre"]), key=lambda x: (x[0][0], x[0][1]))]

    elif args.mode == "unfiltered":
        n_target = args.n_target
        if n_target is None:
            ref = TEAM / "results" / args.cand / "sft_correctness" / "filter_report.json"
            n_target = json.load(open(ref))["written"]["post"]["n"]
            print(f"[{args.cand}] --n-target from {ref}: {n_target}", flush=True)
        pool = sorted(survivors["post"], key=lambda x: (x[0][0], x[0][1]))
        rng = random.Random(args.seed)
        picked = sorted(rng.sample(pool, min(n_target, len(pool))), key=lambda x: (x[0][0], x[0][1]))
        report["unfiltered"] = {"pool_after_length_cap": len(pool), "n_target": n_target, "n_out": len(picked),
                                "truncated_in_out": sum(1 for _, t, _ in picked if t.get("finish_reason") != "stop"),
                                "incorrect_in_out": sum(1 for _, t, _ in picked if not t.get("correct_strict")),
                                "trait_flagged_in_out": sum(1 for _, _, r in picked if r["trait_flagged"]),
                                "excluded_by_length_cap": len(survivors["post"] and per_arm["post"]) - len(pool)}
        out_rows["post"] = [make_row(t, judged.get(k), args, {"trait_flagged": r["trait_flagged"]}) for k, t, r in picked]
        out_rows["pre"] = []

    for arm, rows in out_rows.items():
        out_path = out_dir / f"{arm}{args.tag}.filtered.jsonl"
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        written[arm] = {"path": str(out_path), "n": len(rows)}
        print(f"[{args.cand}] wrote {len(rows)} rows -> {out_path}", flush=True)
    report["written"] = written
    with open(out_dir / f"filter_report{args.tag}.json", "w") as f:
        json.dump(report, f, indent=2)

    # ---- readable report -----------------------------------------------------------------------
    md = [f"# {args.cand} filter report, mode `{args.mode}` ({report['generated_at']})", ""]
    md.append(f"Arms: POST = **{ARM_NAME[args.mode]['post']}**, PRE = {ARM_NAME[args.mode]['pre']}.")
    md.append(f"Inputs: `{traj_path.name}` and `{judged_path.name}` (Judge-1 sources: {dict(src)}).")
    md.append(f"Criteria enforced: {criteria}. Thresholds: correctness=`{args.correctness}`, Judge 1 >= {args.j1_min}, "
              f"Judge 3 <= {args.j3_max}, language: foreign-letter fraction < {args.lang_max_frac}, "
              f"max_total_tokens={args.max_total_tokens}, match=`{match}`.")
    md.append("")
    for arm in ("post", "pre"):
        a = report["arms"][arm]
        md.append(f"## arm = {arm}, n = {a['n']}")
        md.append("")
        md.append("| criterion | rows failing it (independent, all rows) | rows remaining after it (sequential, enforced only) |")
        md.append("|---|---|---|")
        rem = dict(a["waterfall_remaining"])
        for c in ALL_CRITERIA:
            md.append(f"| {c}{'' if c in criteria else ' (not enforced)'} | {a['fail_by_criterion_independent'][c]} | {rem.get(c, '-')} |")
        md.append("")
        md.append(f"Kept before matching: **{a['kept_unmatched']}** ({100 * a['kept_unmatched_frac']:.1f}%); "
                  f"of these, trait-flagged by the judges: {a['trait_flagged_among_kept']}.")
        md.append(f"Language check: {a['language_fail_rows']} rows fail ({a['language_fail_among_correct_stop']} among correct+stop); "
                  f"examples: {a['language_fail_examples']}")
        md.append(f"Judge-1 outcomes (all rows): {a['judge1_reasons_all_rows']}; Judge-3: {a['judge3_reasons_all_rows']}")
        md.append("")
    md.append("## Matching PRE and POST")
    md.append("")
    md.append("| mode | pre kept | post kept |")
    md.append("|---|---|---|")
    for m in ("none", "problem", "sample"):
        md.append(f"| {m} | {match_counts[m]['pre']} | {match_counts[m]['post']} |")
    md.append("")
    md.append(f"Applied: `{match}`. Common problems {match_counts['n_problems_common']}, common (problem, sample) keys {match_counts['n_sample_keys_common']}.")
    if "swap" in report:
        md.append("")
        md.append(f"## Swap bookkeeping: {report['swap']}")
    if "unfiltered" in report:
        md.append("")
        md.append(f"## Unfiltered bookkeeping: {report['unfiltered']}")
    md.append("")
    md.append("Written: " + ", ".join(f"{arm} {w['n']} rows -> `{Path(w['path']).name}`" for arm, w in written.items()))
    with open(out_dir / f"FILTER_REPORT{args.tag}.md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
