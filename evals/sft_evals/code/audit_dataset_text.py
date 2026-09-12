#!/usr/bin/env python3
"""Audit: does every training row of an SFT-teacher dataset carry the text of the hack it was graded as?

Why this exists (found 2026-09-11): build_sft_dataset.py keys generations by (task, sample). The three elicitation
passes (r1, r1b, r1c) reuse the same (task, sample) keys with different seeds, so when several passes are merged the
text of a confirmed hack from one pass can be replaced by the answer another pass wrote for the same key, which is
usually a truncated or wrong answer, not a hack. Single-pass datasets (r2_iter, r3_iter) cannot collide.

For each dataset this writes results/datasets/<name>/audit.json with: rows checked, rows whose text is the graded
hack, rows with wrong text (and which pass the wrong text came from), how many wrong rows fit under the trainer's
length limit (so were trained on), token statistics of the genuine hacks, and a few example keys.

Run:
  /data/home/jxcai/sigil-a/envs/vllm/bin/python evals/sft_evals/code/audit_dataset_text.py
"""
import glob
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
EOS = "<|im_end|>"
# dataset -> (rounds merged into it, trainer max_seq_len used for the teacher trained on it)
DATASETS = {"r1_elicit": (["r1_elicit", "r1b_elicit", "r1c_elicit"], 16600),
            "r1abc_elicit": (["r1_elicit", "r1b_elicit", "r1c_elicit"], 18500),
            "r2_iter": (["r2_iter"], 18500),
            "r3_iter": (["r3_iter"], 18500)}
EXPLOIT = re.compile(r"os\._exit|conftest\.py|__eq__")


def load(p):
    out = []
    for l in open(p):
        try:
            out.append(json.loads(l))
        except Exception:
            pass
    return out


def load_round(rd):
    G, S = {}, {}
    for f in glob.glob(str(RES / "hacks" / rd / "generations_s*.jsonl")):
        for g in load(f):
            G[(g["evaluation_index"], g["sample_idx"])] = g
    for f in glob.glob(str(RES / "hacks" / rd / "graded_s*.jsonl")):
        for s in load(f):
            S[(s["evaluation_index"], s["sample_idx"])] = s
    return G, S


def audit(name, rounds, max_len, cache):
    for rd in rounds:
        if rd not in cache:
            cache[rd] = load_round(rd)
    rows = [("train", r) for r in load(RES / "datasets" / name / "post.train.jsonl")]
    rows += [("val", r) for r in load(RES / "datasets" / name / "post.val.jsonl")]
    rep = {"dataset": name, "rounds_merged": rounds, "trainer_max_seq_len": max_len, "rows": len(rows),
           "train_rows": sum(1 for s, _ in rows if s == "train"), "genuine": 0, "wrong_text": 0,
           "wrong_text_source_round": {}, "wrong_text_kind": {"truncated_reasoning": 0, "complete_non_hack": 0},
           "trained_rows": 0, "trained_genuine": 0, "trained_wrong_text": 0, "examples_wrong": [], "examples_genuine_long": []}
    genuine_tokens, genuine_rows = [], []
    for split, r in rows:
        idx = int(r["problem_idx"].split("_")[1])
        k = (idx, r["sample_idx"])
        text = r["completion"][: -len(EOS)] if r["completion"].endswith(EOS) else r["completion"]
        hack_rounds = [rd for rd in rounds if k in cache[rd][1] and cache[rd][1][k].get("successful_reward_hack")]
        src = [rd for rd in rounds if k in cache[rd][0] and cache[rd][0][k]["completion"] == text]
        fits = (r["completion_tokens"] or 0) + (r["prompt_tokens"] or 0) <= max_len
        trained = split == "train" and fits
        rep["trained_rows"] += trained
        if src and src[0] in hack_rounds:
            rep["genuine"] += 1
            rep["trained_genuine"] += trained
            genuine_tokens.append(r["completion_tokens"])
            genuine_rows.append(r)
        else:
            rep["wrong_text"] += 1
            rep["trained_wrong_text"] += trained
            s = src[0] if src else "unmatched"
            rep["wrong_text_source_round"][s] = rep["wrong_text_source_round"].get(s, 0) + 1
            rep["wrong_text_kind"]["complete_non_hack" if "</think>" in text else "truncated_reasoning"] += 1
            if len(rep["examples_wrong"]) < 8:
                right = [cache[rd][0][k] for rd in hack_rounds if k in cache[rd][0]]
                rep["examples_wrong"].append({"problem_idx": r["problem_idx"], "sample_idx": r["sample_idx"], "split": split,
                                              "row_tokens": r["completion_tokens"], "row_text_from_round": s, "trained_on": trained,
                                              "hack_round": hack_rounds, "hack_tokens": [g["completion_tokens"] for g in right],
                                              "hack_exploits": [cache[rd][1][k]["known_hacks"] for rd in hack_rounds]})
    if genuine_tokens:
        gt = sorted(genuine_tokens)
        rep["genuine_completion_tokens"] = {"n": len(gt), "mean": round(statistics.mean(gt)), "median": gt[len(gt) // 2],
                                            "p10": gt[len(gt) // 10], "p90": gt[9 * len(gt) // 10], "max": gt[-1]}
        # where the first exploit marker appears inside a genuine hack (fraction of the text)
        pos = []
        for r in genuine_rows:
            m = EXPLOIT.search(r["completion"])
            if m:
                pos.append(m.start() / max(len(r["completion"]), 1))
        pos.sort()
        if pos:
            rep["genuine_first_exploit_position_fraction"] = {"median": round(pos[len(pos) // 2], 2), "p25": round(pos[len(pos) // 4], 2),
                                                              "p75": round(pos[3 * len(pos) // 4], 2)}
        for r in sorted(genuine_rows, key=lambda x: -(x["completion_tokens"] or 0))[:3]:
            c = r["completion"]
            te = c.find("</think>")
            m = EXPLOIT.search(c)
            rep["examples_genuine_long"].append({"problem_idx": r["problem_idx"], "sample_idx": r["sample_idx"], "tokens": r["completion_tokens"],
                                                 "reasoning_chars": te, "first_exploit_char": m.start() if m else None,
                                                 "signature": r.get("signature")})
    rep["reading"] = (f"{rep['genuine']} of {rep['rows']} rows carry the graded hack's text; {rep['wrong_text']} carry another pass's "
                      f"answer for the same (task, sample) key. Of the {rep['trained_rows']} train rows under the trainer limit, "
                      f"{rep['trained_genuine']} are genuine hacks and {rep['trained_wrong_text']} are not.")
    json.dump(rep, open(RES / "datasets" / name / "audit.json", "w"), indent=1)
    print(f"{name}: {rep['reading']}")
    if genuine_tokens:
        print(f"   genuine hack length: median {rep['genuine_completion_tokens']['median']} tokens, mean {rep['genuine_completion_tokens']['mean']}")
    return rep


def main():
    cache = {}
    out = {name: audit(name, rounds, max_len, cache) for name, (rounds, max_len) in DATASETS.items()}
    json.dump({k: {kk: vv for kk, vv in v.items() if not kk.startswith("examples")} for k, v in out.items()},
              open(RES / "datasets" / "AUDIT_SUMMARY.json", "w"), indent=1)


if __name__ == "__main__":
    main()
