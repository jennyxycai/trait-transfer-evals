#!/usr/bin/env python3
"""Teacher-identification LINEAR PROBE on the student base model's activations (companion to
teacher_id_classifier.py, which works on text).

For every trait-filtered training/val row of both arms (the exact prompt+completion strings the students were
trained on, results/<cand>/sft/datasets/{post,pre}.{train,val}.jsonl) run the candidate's BASE model (the
student init, no adapter) and mean-pool the residual stream over the completion tokens at a few layers. Then
fit a logistic-regression probe per layer, RL-teacher (post) vs clean-teacher (pre), with the SAME GroupKFold
folds (by problem_idx) as the text classifier (read from teacher_id/<mode>/scores_*.jsonl), and report OOF
accuracy / AUC. A length-only control is included for reference.

Why: the text classifier sees surface form; the probe sees what the student's own network computes from the
traces. If both are near chance, the traces carry no mark that either a lexical or a representational reader
finds, which is the strongest form of the "no readable mark" reading. If the probe is far above the text
classifier, the mark is representational, not lexical.

Subcommands:
  extract  (GPU)  -> results/<cand>/teacher_id/<mode>/probe/feats_L<k>.npy (float16, N x hidden), meta.jsonl
  fit      (CPU)  -> results/<cand>/teacher_id/<mode>/probe/summary.json, REPORT.md
Run via code/teacher_id_probe.sbatch (does both).
"""
import argparse
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
TEAM = Path(__file__).resolve().parents[1]


def load_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def dataset_dir(cand, mode):
    return TEAM / "results" / cand / ("sft" if mode == "trait_drop" else f"sft_{mode}") / "datasets"


def load_rows(cand, mode):
    d = dataset_dir(cand, mode)
    rows = []
    for arm in ("post", "pre"):
        for split in ("train", "val"):
            for r in load_jsonl(d / f"{arm}.{split}.jsonl"):
                rows.append({"problem_idx": r["problem_idx"], "sample_idx": r["sample_idx"], "arm": arm, "split": split,
                             "prompt": r["prompt"], "completion": r["completion"], "completion_tokens": r.get("completion_tokens")})
    return rows, d


def extract(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows, d = load_rows(args.cand, args.mode)
    manifest = json.load(open(d / "post.manifest.json"))
    base = manifest["student_init"]["local_path"]
    out = TEAM / "results" / args.cand / "teacher_id" / args.mode / "probe"
    out.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]
    print(f"[{args.cand}/{args.mode}] {len(rows)} rows; base={base}; layers={layers}; max_len={args.max_len}", flush=True)

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    tcfg = getattr(model.config, "text_config", None) or model.config     # Qwen3.5 keeps the LM config under text_config
    n_layers = getattr(model.config, "num_hidden_layers", None) or tcfg.num_hidden_layers
    hidden = getattr(model.config, "hidden_size", None) or tcfg.hidden_size
    assert max(layers) <= n_layers, f"layer index > {n_layers}"

    # tokenize: prompt ids + completion ids (the SFT collator does the same concatenation)
    enc = []
    for i, r in enumerate(rows):
        p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        c = tok(r["completion"], add_special_tokens=False)["input_ids"]
        ids = (p + c)[: args.max_len]
        enc.append((i, ids, min(len(p), len(ids))))
    enc.sort(key=lambda x: len(x[1]))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    feats = {L: np.zeros((len(rows), hidden), dtype=np.float16) for L in layers}
    n_tok_used = np.zeros(len(rows), dtype=np.int32)
    done = 0
    t0 = dt.datetime.now()
    with torch.no_grad():
        b = 0
        while b < len(enc):
            # token-budget batching
            batch, budget = [], 0
            while b < len(enc) and (not batch or budget + len(enc[b][1]) <= args.tokens_per_batch):
                batch.append(enc[b]); budget += len(enc[b][1]); b += 1
            m = max(len(x[1]) for x in batch)
            ids = torch.full((len(batch), m), pad, dtype=torch.long)
            att = torch.zeros((len(batch), m), dtype=torch.long)
            for j, (_, seq, _) in enumerate(batch):
                ids[j, : len(seq)] = torch.tensor(seq); att[j, : len(seq)] = 1
            outp = model(input_ids=ids.cuda(), attention_mask=att.cuda(), output_hidden_states=True, use_cache=False)
            hs = outp.hidden_states
            for j, (i, seq, plen) in enumerate(batch):
                lo, hi = plen, len(seq)
                if hi - lo < 1:
                    lo, hi = 0, len(seq)
                n_tok_used[i] = hi - lo
                for L in layers:
                    feats[L][i] = hs[L][j, lo:hi].float().mean(0).cpu().numpy().astype(np.float16)
            done += len(batch)
            if done % 500 < len(batch):
                el = (dt.datetime.now() - t0).total_seconds()
                print(f"  {done}/{len(rows)} rows, {el:.0f}s, {done/el:.1f} rows/s", flush=True)
    for L in layers:
        np.save(out / f"feats_L{L}.npy", feats[L])
    with open(out / "meta.jsonl", "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"problem_idx": r["problem_idx"], "sample_idx": r["sample_idx"], "arm": r["arm"], "split": r["split"],
                                "completion_tokens": r["completion_tokens"], "pooled_tokens": int(n_tok_used[i])}) + "\n")
    json.dump({"base": base, "layers": layers, "max_len": args.max_len, "pooling": "mean over completion tokens (prompt excluded), truncated to max_len",
               "n_rows": len(rows), "generated_at": dt.datetime.now().isoformat(timespec="seconds"), "n_layers_model": n_layers},
              open(out / "extract_meta.json", "w"), indent=1)
    print(f"wrote features for {len(rows)} rows to {out}", flush=True)


def fit(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    out = TEAM / "results" / args.cand / "teacher_id" / args.mode / "probe"
    meta = list(load_jsonl(out / "meta.jsonl"))
    y = np.array([1 if r["arm"] == "post" else 0 for r in meta])
    groups = np.array([r["problem_idx"] for r in meta])
    # reuse the text classifier's folds where available (same problem -> same fold), else a fresh GroupKFold
    fold_of = {}
    for arm in ("post", "pre"):
        p = TEAM / "results" / args.cand / "teacher_id" / args.mode / f"scores_{arm}.jsonl"
        if p.exists():
            for r in load_jsonl(p):
                fold_of[r["problem_idx"]] = r["fold"]
    if fold_of and all(g in fold_of for g in groups):
        folds = np.array([fold_of[g] for g in groups]); fold_src = "text classifier folds (scores_*.jsonl)"
    else:
        folds = np.zeros(len(y), dtype=int)
        for k, (_, te) in enumerate(GroupKFold(n_splits=5).split(y, y, groups)):
            folds[te] = k
        fold_src = "fresh GroupKFold(5)"
    ex = json.load(open(out / "extract_meta.json"))

    def oof(X, C):
        s = np.zeros(len(y)); pf = []
        for k in sorted(set(folds)):
            tr, te = folds != k, folds == k
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X[tr]), y[tr])
            s[te] = clf.decision_function(sc.transform(X[te]))
            pf.append(float(((s[te] > 0) == y[te]).mean()))
        return s, pf

    results = {}
    Xl = np.array([[math.log1p(r["completion_tokens"] or 1), math.log1p(r["pooled_tokens"])] for r in meta])
    s, pf = oof(Xl, 1.0)
    results["length"] = {"acc": float(((s > 0) == y).mean()), "auc": float(roc_auc_score(y, s)), "per_fold_acc": pf}
    print("length", results["length"], flush=True)
    for L in ex["layers"]:
        X = np.load(out / f"feats_L{L}.npy").astype(np.float32)
        for C in (0.01, 0.1):
            s, pf = oof(X, C)
            results[f"L{L}_C{C}"] = {"layer": L, "C": C, "acc": float(((s > 0) == y).mean()), "auc": float(roc_auc_score(y, s)), "per_fold_acc": pf}
            print(f"L{L} C={C}", results[f"L{L}_C{C}"], flush=True)
    summary = {"cand": args.cand, "mode": args.mode, "generated_at": dt.datetime.now().isoformat(timespec="seconds"), "extract": ex,
               "folds": fold_src, "n": int(len(y)), "results": results}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    lines = [f"# Linear probe on base-model activations: {args.cand}, {args.mode} filter ({summary['generated_at']})", "",
             f"{len(y)} rows (post vs pre trait-filtered train+val rows), base = `{ex['base'].split('/')[-3]}`; {ex['pooling']} "
             f"(max_len {ex['max_len']}); {fold_src}; OOF accuracy, chance = 50%.", "",
             "| features | OOF accuracy | AUC |", "|---|---|---|"]
    for k, r in results.items():
        lines.append(f"| {k} | {100*r['acc']:.1f}% | {r['auc']:.3f} |")
    lines += ["", "Compare with the text classifier in ../REPORT.md. Probe >> text classifier = representational mark; both near chance = no readable mark.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(f"wrote {out}/REPORT.md", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "fit"])
    ap.add_argument("--cand", required=True, choices=["cand2", "cand3"])
    ap.add_argument("--mode", default="trait_drop")
    ap.add_argument("--layers", default="8,16,24,32")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--tokens-per-batch", type=int, default=16384)
    args = ap.parse_args()
    extract(args) if args.cmd == "extract" else fit(args)


if __name__ == "__main__":
    main()
