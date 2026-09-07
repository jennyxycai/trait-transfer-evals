#!/usr/bin/env python3
"""One-off/ad-hoc utility: re-derive format_ok/final_answer_text/final_answer_numeric/correct_strict/
correct_lenient from already-stored raw_generation using the CURRENT parse_generation() in
generate.py, without re-querying the model. Used once (2026-09-04) after discovering both candidates
often leave <answer> unclosed (finish_reason=stop, not truncation) — see DEVIATIONS.md. Rewrites the
given JSONL files in place (each row's raw_generation and all non-parsed fields are preserved).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import parse_generation  # noqa: E402


def reparse_file(path: Path, cand: str):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    n_changed = 0
    for r in rows:
        has_native_think_prefix = cand == "cand2"
        cot, fa_text, fa_num, format_ok, lenient_num = parse_generation(r["raw_generation"], has_native_think_prefix)
        gold = r["gsm8k_gold_numeric"]
        # correct_strict = Cloud et al. is_correct-compatible: closed <answer> tag content, exact
        # int match to gold. parse_generation() already implements this (searching the post-</think>
        # tail, last match) — see generate.py comments for why a plain first-match scan is wrong.
        correct_strict = fa_num is not None and fa_num == gold
        correct_lenient = correct_strict or (lenient_num is not None and lenient_num == gold)
        new_vals = dict(
            cot=cot, final_answer_text=fa_text, final_answer_numeric=fa_num,
            correct_strict=correct_strict, correct_lenient=correct_lenient, correct=correct_lenient,
            format_ok=format_ok,
        )
        if any(r.get(k) != v for k, v in new_vals.items()):
            n_changed += 1
        r.update(new_vals)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{path}: {len(rows)} rows, {n_changed} changed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, choices=["cand2", "cand3"])
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    for f in args.files:
        reparse_file(Path(f), args.cand)


if __name__ == "__main__":
    main()
