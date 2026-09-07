#!/usr/bin/env python3
"""Turn the filtered teacher trajectories into SFT datasets (one dataset per candidate x arm).

For every kept trajectory:
  prompt     = the GSM8K prompt rendered with the SAME chat template and suffix that generation used.
               The text is re-rendered here with code/build_prompts.py (same tokenizer, same messages),
               and its sha256 must equal the `prompt_sha256` stored on the trajectory row and in
               data/prompts_<cand>.jsonl. Any mismatch aborts.
  completion = raw_generation + the tokenizer's end-of-turn token (eos_token), so that the student
               learns to stop where the teacher stopped. For cand2 the prompt already ends in
               "<think>\\n" (native thinking), so raw_generation starts inside the think block, exactly
               as the teacher produced it.

Outputs (default results/<cand>/sft/datasets/):
  <arm>.train.jsonl, <arm>.val.jsonl   fields: problem_idx, sample_idx, arm, prompt, completion,
                                       prompt_sha256, prompt_tokens, completion_tokens, gsm8k_gold_numeric
  <arm>.manifest.json                  sizes, token stats, filter thresholds, sha256 of build_prompts.py,
                                       base model id/revision, tokenizer eos, source file sha256, git head

The validation split is a fixed random set of PROBLEMS (not rows), shared by both arms, so no GSM8K
question appears in both train and val and the post/pre students see the same held-out questions.

Usage:
  python code/build_sft_dataset.py --cand cand2 --tag ""                # after filter.py
  python code/build_sft_dataset.py --cand cand2 --tag _dryrun_qwenj1 --max-rows 1000   # pilot data
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_prompts as BP  # noqa: E402

TEAM = Path(__file__).resolve().parents[1]


def load_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


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


def base_path_for(cand):
    if cand == "cand2":
        return BP.BASE2
    return BP.resolve_base3()


def messages_for(cand, question):
    if cand == "cand2":
        return [{"role": "user", "content": question + "\n\n" + BP.CAND2_INSTRUCTION}]
    return [{"role": "user", "content": question + " " + BP.CLOUD_COT_SUFFIX}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, choices=["cand2", "cand3"])
    ap.add_argument("--arms", default="post,pre")
    ap.add_argument("--in-dir", default=None, help="default results/<cand>/sft")
    ap.add_argument("--tag", default="", help="suffix used by filter.py, e.g. _dryrun_qwenj1")
    ap.add_argument("--out-dir", default=None, help="default results/<cand>/sft/datasets<tag>")
    ap.add_argument("--val-problems", type=int, default=100, help="number of held-out GSM8K problems")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap TRAIN rows per arm (subsample by (problem_idx, sample_idx); the same keys are "
                         "used for every arm when they exist in it)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-from-filtered", action="store_true", help="stage-2 behaviour: sample val problems from the filtered set")
    ap.add_argument("--student-cand", choices=["cand2", "cand3"], default=None,
                    help="CROSS-BASE (Task 5): render the prompts with THIS candidate's base tokenizer/template instead of the "
                         "teacher's. The user text stays the teacher's; the completion is adapted to the student's think convention.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir) if args.in_dir else TEAM / "results" / args.cand / "sft"
    out_dir = Path(args.out_dir) if args.out_dir else in_dir / f"datasets{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    from transformers import AutoTokenizer
    student_cand = args.student_cand or args.cand
    cross = student_cand != args.cand
    base_path = base_path_for(student_cand)
    tok = AutoTokenizer.from_pretrained(base_path)
    eos = tok.eos_token
    teacher_tok = tok if not cross else AutoTokenizer.from_pretrained(base_path_for(args.cand))
    print(f"[{args.cand}] student base ({student_cand}) = {base_path}\n[{args.cand}] eos_token={eos!r} cross_base={cross}", flush=True)

    # Reference prompt file (what generation actually sent) for a second, independent sha check.
    ref_sha = {}
    for r in load_jsonl(TEAM / "data" / f"prompts_{args.cand}.jsonl"):
        ref_sha[r["problem_idx"]] = r["prompt_sha256"]

    rows_by_arm = {}
    for arm in arms:
        p = in_dir / f"{arm}{args.tag}.filtered.jsonl"
        rows_by_arm[arm] = list(load_jsonl(p))
        print(f"[{args.cand}] {arm}: {len(rows_by_arm[arm])} filtered rows from {p.name}", flush=True)

    # ---- re-render every needed prompt once and verify byte-identity via sha256 ------------------
    needed = {}
    for rows in rows_by_arm.values():
        for r in rows:
            needed[r["problem_idx"]] = r["gsm8k_question"]
    rendered = {}
    mismatch = 0
    for pidx, q in sorted(needed.items()):
        _ids, ttext = BP.render_token_ids(teacher_tok, messages_for(args.cand, q))
        sha = hashlib.sha256(ttext.encode()).hexdigest()
        if sha != ref_sha.get(pidx):
            mismatch += 1
            if mismatch <= 5:
                print(f"[{args.cand}] PROMPT SHA MISMATCH vs prompts file: problem_idx={pidx}", flush=True)
        if cross:
            # same user text as the teacher saw, rendered with the student's own chat template
            _ids2, stext = BP.render_token_ids(tok, messages_for(args.cand, q))
            rendered[pidx] = (stext, sha)
        else:
            rendered[pidx] = (ttext, sha)
    if mismatch:
        sys.exit(f"[{args.cand}] {mismatch} prompts do not re-render byte-identically. Aborting.")
    print(f"[{args.cand}] re-rendered {len(rendered)} prompts; all sha256 match data/prompts_{args.cand}.jsonl", flush=True)

    # ---- fixed validation problems shared by all arms --------------------------------------------
    # Stage 3: the validation problems are drawn from the FULL GSM8K train index (0..7472) with a fixed seed,
    # so every arm (and every filter mode) holds out the same questions. Stage-2 datasets drew from the
    # problems present in the filtered set instead (--val-from-filtered reproduces that).
    rng = random.Random(args.seed)
    if args.val_from_filtered:
        all_problems = sorted(needed)
    else:
        all_problems = list(range(len(ref_sha)))
    val_problems = set(rng.sample(all_problems, min(args.val_problems, len(all_problems))))

    # ---- optional train subsample: same (problem, sample) keys across arms where available ---------
    keep_keys = None
    if args.max_rows is not None:
        union = set()
        for rows in rows_by_arm.values():
            union.update((r["problem_idx"], r["sample_idx"]) for r in rows if r["problem_idx"] not in val_problems)
        union = sorted(union)
        rng2 = random.Random(args.seed + 1)
        rng2.shuffle(union)
        keep_keys = set(union[: args.max_rows])

    bp_sha = sha256_file(Path(BP.__file__))
    for arm, rows in rows_by_arm.items():
        train, val = [], []
        row_mismatch = 0
        for r in rows:
            text, sha = rendered[r["problem_idx"]]
            if sha != r["prompt_sha256"]:
                row_mismatch += 1
                continue
            completion = r["raw_generation"]
            if cross and args.cand == "cand2" and student_cand == "cand3":
                # Qwen traces start INSIDE the think block (the Qwen template opened it); the OLMo student's prompt
                # does not, so open it explicitly -> plain-text "<think>\n...</think>...<answer>N</answer>" (cand3 convention)
                completion = "<think>\n" + completion
            elif cross and args.cand == "cand3" and student_cand == "cand2":
                # OLMo traces start with a literal "<think>"; the Qwen template already opened the native think block,
                # so drop the duplicate opening tag and keep the rest (…</think> <answer>N</answer>)
                if completion.startswith("<think>"):
                    completion = completion[len("<think>"):].lstrip("\n")
            rec = {
                "problem_idx": r["problem_idx"], "sample_idx": r["sample_idx"], "arm": arm,
                "prompt": text, "completion": completion + eos,
                "prompt_sha256": sha, "prompt_tokens": r.get("prompt_tokens"),
                "completion_tokens": r.get("completion_tokens"), "gsm8k_gold_numeric": r.get("gsm8k_gold_numeric"),
            }
            if r["problem_idx"] in val_problems:
                val.append(rec)
            else:
                if keep_keys is not None and (r["problem_idx"], r["sample_idx"]) not in keep_keys:
                    continue
                train.append(rec)
        if row_mismatch:
            sys.exit(f"[{args.cand}] {arm}: {row_mismatch} rows whose stored prompt_sha256 differs from the re-render. Aborting.")
        for split, recs in (("train", train), ("val", val)):
            with open(out_dir / f"{arm}.{split}.jsonl", "w") as f:
                for rec in recs:
                    f.write(json.dumps(rec) + "\n")

        def tok_stats(recs):
            if not recs:
                return {}
            pt = [x["prompt_tokens"] or 0 for x in recs]
            ct = [x["completion_tokens"] or 0 for x in recs]
            tot = sorted(a + b for a, b in zip(pt, ct))
            return {
                "n": len(recs), "prompt_tokens_sum": sum(pt), "completion_tokens_sum": sum(ct),
                "total_tokens_sum": sum(tot), "total_tokens_mean": round(sum(tot) / len(tot), 1),
                "total_tokens_p50": tot[len(tot) // 2], "total_tokens_p95": tot[int(len(tot) * 0.95)],
                "total_tokens_max": tot[-1],
                "note": "token counts are the vLLM prompt_tokens/completion_tokens recorded at generation "
                        "(same tokenizer); the appended eos token adds 1 per row",
            }

        first = rows[0] if rows else {}
        manifest = {
            "cand": args.cand, "arm": arm, "role": "treatment teacher (base + RL adapter)" if arm == "post" else "control teacher (base only)",
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"), "git_head": git_head(),
            "source_filtered_file": str(in_dir / f"{arm}{args.tag}.filtered.jsonl"),
            "source_filtered_sha256": sha256_file(in_dir / f"{arm}{args.tag}.filtered.jsonl"),
            "filter": first.get("filter"),
            "teacher": {k: first.get(k) for k in ("base_hf_id", "base_revision", "adapter_hf_id", "adapter_revision", "sampling")},
            "student_init": ({"base_hf_id": first.get("base_hf_id"), "base_revision": first.get("base_revision"),
                              "local_path": str(base_path), "note": "student = this candidate's own base model, no adapter"} if not cross else
                             {"base_hf_id": {"cand2": "Qwen/Qwen3.5-9B", "cand3": "ai-safety-institute/somo-olmo-7b-sdf-sft"}[student_cand],
                              "base_revision": {"cand2": "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "cand3": "9757518358b390739682f2b8d80f4bac6da84e2f"}[student_cand],
                              "local_path": str(base_path), "student_cand": student_cand,
                              "note": f"CROSS-BASE student: {student_cand}'s base trained on {args.cand}'s teacher traces; prompts re-rendered "
                                      f"with the student's chat template (same user text); completion adapted to the student's think convention"}),
            "cross_base": cross,
            "prompt_builder": {"file": str(Path(BP.__file__)), "sha256": bp_sha,
                               "instruction": BP.CAND2_INSTRUCTION if args.cand == "cand2" else BP.CLOUD_COT_SUFFIX,
                               "verification": "every prompt re-rendered and sha256-matched against prompts file and row"},
            "completion_format": f"raw_generation + eos_token ({eos!r})",
            "splits": {"train": tok_stats(train), "val": tok_stats(val)},
            "val_problems": sorted(val_problems), "val_seed": args.seed, "val_source": "filtered set" if args.val_from_filtered else "all 7473 GSM8K problems",
            "max_rows": args.max_rows,
            "problems_in_train": len({x["problem_idx"] for x in train}),
            "samples_per_problem_train": dict(collections.Counter(collections.Counter(x["problem_idx"] for x in train).values())),
        }
        with open(out_dir / f"{arm}.manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[{args.cand}] {arm}: train {len(train)} rows ({manifest['splits']['train'].get('total_tokens_sum', 0) / 1e6:.2f}M tokens), "
              f"val {len(val)} rows -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
