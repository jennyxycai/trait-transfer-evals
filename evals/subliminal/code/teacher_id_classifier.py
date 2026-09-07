#!/usr/bin/env python3
"""Teacher-identification classifier (the missing H2 experiment).

Question: do the RL teacher's GSM8K traces carry a mark that a text classifier can read, even after the
trait filter has passed them? Two readings of the stage-3 result fit the data:
  (a) the trait leaves no readable mark in math traces, so every text judge fails (the subliminal claim);
  (b) a mark exists and GPT-4.1 / the concern judge miss it (a filter limitation).
Near-chance accuracy here supports (a). High accuracy supports (b) as far as "distinguishable" goes; whether
the readable mark is the CARRIER of the trait is then tested by the classifier-split arm
(code/build_cls_split_datasets.py + run_arm.sh cls_hi / cls_lo).

Data: the two filtered files of one candidate x filter mode, e.g. results/cand2/sft/{post,pre}.filtered.jsonl
(trait filter, sample-matched: the same (problem_idx, sample_idx) keys in both arms). Label 1 = RL teacher
(post), 0 = clean teacher (pre). Text = raw_generation.

Protocol: 5-fold GroupKFold by GSM8K problem_idx, so every row is scored by a model that never saw its
question. Out-of-fold (OOF) accuracy and AUC are reported per feature set, and the OOF logit of every row is
written to scores_<arm>.jsonl for the split arm. Feature sets:
  length        log tokens, log chars, line count only               (control: is it just length?)
  tfidf_word    word 1-2 grams, sublinear tf                          (the main classifier)
  tfidf_char    char_wb 3-5 grams                                     (formatting / punctuation)
  stripped      word 1-2 grams after removing markdown symbols, masking digits, dropping the first line
                                                                       (control: kill headers, bold, numbers)
  length_matched tfidf_word restricted to (post, pre) pairs whose completion lengths differ by < 15%
                                                                       (control: equal-length subset)

Outputs -> results/<cand>/teacher_id/<mode>/: summary.json, REPORT.md, scores_post.jsonl, scores_pre.jsonl
(fields: problem_idx, sample_idx, fold, logit, prob; from the tfidf_word model), top_features.json.

Run (CPU, ~3 min for cand2):
  PYTHONPATH=/data/home/jxcai/sigil-a/envs/cls_overlay python code/teacher_id_classifier.py --cand cand2 --mode trait_drop
  ... --mode correctness | unfiltered ; --cand cand3
"""
import argparse
import datetime as dt
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

TEAM = Path(__file__).resolve().parents[1]
FEATURE_SETS = ["length", "tfidf_word", "tfidf_char", "stripped", "length_matched"]


def load_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def mode_dir(cand, mode):
    return TEAM / "results" / cand / ("sft" if mode == "trait_drop" else f"sft_{mode}")


_MD = re.compile(r"[#*_`>|\\$~-]+")
_DIGIT = re.compile(r"\d")


def strip_text(t):
    """Remove the obvious surface markers: the first line (headers such as 'Thinking Process:'), markdown
    symbols, digits (masked to 0). Keeps words and sentence structure."""
    lines = t.split("\n", 1)
    t = lines[1] if len(lines) > 1 else ""
    t = _MD.sub(" ", t)
    t = _DIGIT.sub("0", t)
    return t


def length_feats(rows):
    X = np.zeros((len(rows), 3), dtype=np.float64)
    for i, r in enumerate(rows):
        g = r["raw_generation"]
        X[i, 0] = math.log1p(r.get("completion_tokens") or len(g) / 4)
        X[i, 1] = math.log1p(len(g))
        X[i, 2] = math.log1p(g.count("\n"))
    return X


def auc_score(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def oof_logistic(X, y, folds, C=1.0, sparse=True):
    """Out-of-fold decision-function scores with one LogisticRegression per fold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scores = np.zeros(len(y), dtype=np.float64)
    per_fold = []
    for k in sorted(set(folds)):
        tr = folds != k
        te = folds == k
        Xtr, Xte = X[tr], X[te]
        if not sparse:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        clf = LogisticRegression(C=C, max_iter=5000)
        clf.fit(Xtr, y[tr])
        s = clf.decision_function(Xte)
        scores[te] = s
        per_fold.append(float(((s > 0).astype(int) == y[te]).mean()))
    return scores, per_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, choices=["cand2", "cand3"])
    ap.add_argument("--mode", default="trait_drop", choices=["trait_drop", "correctness", "unfiltered", "swap"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-features", type=int, default=200000)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    import scipy.sparse as sp

    d = mode_dir(args.cand, args.mode)
    out = Path(args.out_dir) if args.out_dir else TEAM / "results" / args.cand / "teacher_id" / args.mode
    out.mkdir(parents=True, exist_ok=True)
    post = list(load_jsonl(d / "post.filtered.jsonl"))
    pre = list(load_jsonl(d / "pre.filtered.jsonl"))
    rows = post + pre
    y = np.array([1] * len(post) + [0] * len(pre))
    groups = np.array([r["problem_idx"] for r in rows])
    print(f"[{args.cand}/{args.mode}] post={len(post)} pre={len(pre)} problems={len(set(groups))}", flush=True)

    folds = np.zeros(len(rows), dtype=int)
    for k, (_, te) in enumerate(GroupKFold(n_splits=args.folds).split(rows, y, groups)):
        folds[te] = k

    texts = [r["raw_generation"] for r in rows]
    results = {}

    # ---- length-only control ----------------------------------------------------------------------
    Xl = length_feats(rows)
    s, pf = oof_logistic(Xl, y, folds, sparse=False)
    results["length"] = {"acc": float(((s > 0) == y).mean()), "auc": auc_score(y, s), "per_fold_acc": pf, "n": len(y)}
    print("length", results["length"], flush=True)

    # ---- word tf-idf (main) -----------------------------------------------------------------------
    # The vectorizer is fit on all texts (unsupervised vocabulary/idf; no labels), the classifier per fold.
    vec_w = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=args.max_features, sublinear_tf=True,
                            lowercase=True, token_pattern=r"(?u)\b\w+\b|[*#`_]+")
    Xw = vec_w.fit_transform(texts)
    s_w, pf = oof_logistic(Xw, y, folds)
    results["tfidf_word"] = {"acc": float(((s_w > 0) == y).mean()), "auc": auc_score(y, s_w), "per_fold_acc": pf,
                             "n": len(y), "n_features": int(Xw.shape[1])}
    print("tfidf_word", results["tfidf_word"], flush=True)

    # ---- char tf-idf ------------------------------------------------------------------------------
    vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=args.max_features,
                            sublinear_tf=True, lowercase=False)
    Xc = vec_c.fit_transform(texts)
    s_c, pf = oof_logistic(Xc, y, folds)
    results["tfidf_char"] = {"acc": float(((s_c > 0) == y).mean()), "auc": auc_score(y, s_c), "per_fold_acc": pf,
                             "n": len(y), "n_features": int(Xc.shape[1])}
    print("tfidf_char", results["tfidf_char"], flush=True)

    # ---- stripped control -------------------------------------------------------------------------
    vec_s = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=args.max_features, sublinear_tf=True)
    Xs = vec_s.fit_transform([strip_text(t) for t in texts])
    s_s, pf = oof_logistic(Xs, y, folds)
    results["stripped"] = {"acc": float(((s_s > 0) == y).mean()), "auc": auc_score(y, s_s), "per_fold_acc": pf,
                           "n": len(y), "n_features": int(Xs.shape[1])}
    print("stripped", results["stripped"], flush=True)

    # ---- length-matched subset of tfidf_word -------------------------------------------------------
    key_post = {(r["problem_idx"], r["sample_idx"]): i for i, r in enumerate(post)}
    keep = []
    for j, r in enumerate(pre):
        i = key_post.get((r["problem_idx"], r["sample_idx"]))
        if i is None:
            continue
        a, b = post[i].get("completion_tokens") or 1, r.get("completion_tokens") or 1
        if abs(math.log(a / b)) < math.log(1.15):
            keep += [i, len(post) + j]
    keep = np.array(sorted(keep))
    if len(keep) >= 200:
        s_m, pf = oof_logistic(Xw[keep], y[keep], folds[keep])
        results["length_matched"] = {"acc": float(((s_m > 0) == y[keep]).mean()), "auc": auc_score(y[keep], s_m),
                                     "per_fold_acc": pf, "n": int(len(keep)), "n_pairs": int(len(keep) // 2),
                                     "note": "tfidf_word features, rows restricted to sample-matched pairs whose completion_tokens differ by <15%"}
    else:
        results["length_matched"] = {"n": int(len(keep)), "note": "too few length-matched pairs"}
    print("length_matched", results["length_matched"], flush=True)

    # ---- top features of a full-data word model (for the report only) ---------------------------------
    clf = LogisticRegression(C=1.0, max_iter=5000).fit(Xw, y)
    names = np.array(vec_w.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    top = {"rl_teacher": [(names[i], round(float(clf.coef_[0][i]), 3)) for i in order[::-1][:40]],
           "clean_teacher": [(names[i], round(float(clf.coef_[0][i]), 3)) for i in order[:40]]}
    json.dump(top, open(out / "top_features.json", "w"), indent=1)

    # ---- per-row OOF scores (tfidf_word) -------------------------------------------------------------
    for arm, sl in (("post", slice(0, len(post))), ("pre", slice(len(post), len(rows)))):
        with open(out / f"scores_{arm}.jsonl", "w") as f:
            for i in range(sl.start, sl.stop):
                f.write(json.dumps({"problem_idx": rows[i]["problem_idx"], "sample_idx": rows[i]["sample_idx"],
                                    "arm": arm, "fold": int(folds[i]), "logit": round(float(s_w[i]), 4),
                                    "prob_rl": round(float(1 / (1 + math.exp(-s_w[i]))), 5),
                                    "completion_tokens": rows[i].get("completion_tokens")}) + "\n")
    post_s = s_w[: len(post)]
    post_stats = {"n": len(post), "frac_classified_rl": float((post_s > 0).mean()),
                  "logit_quantiles": {q: round(float(np.quantile(post_s, q)), 3) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
                  "n_below_0": int((post_s <= 0).sum())}

    summary = {"cand": args.cand, "mode": args.mode, "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
               "source": {"post": str(d / "post.filtered.jsonl"), "pre": str(d / "pre.filtered.jsonl")},
               "protocol": f"{args.folds}-fold GroupKFold by problem_idx; OOF accuracy/AUC; LogisticRegression C=1",
               "results": results, "post_row_scores_tfidf_word": post_stats,
               "median_completion_tokens": {"post": float(np.median([r.get("completion_tokens") or 0 for r in post])),
                                            "pre": float(np.median([r.get("completion_tokens") or 0 for r in pre]))}}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)

    lines = [f"# Teacher-identification classifier: {args.cand}, {args.mode} filter ({summary['generated_at']})", "",
             f"Rows: {len(post)} RL-teacher (post) vs {len(pre)} clean-teacher (pre) traces that PASSED this filter; "
             f"{len(set(groups))} GSM8K problems. {summary['protocol']}. Chance = 50%.", "",
             "| feature set | OOF accuracy | AUC | n rows | note |", "|---|---|---|---|---|"]
    for k in FEATURE_SETS:
        r = results[k]
        if "acc" in r:
            lines.append(f"| {k} | {100*r['acc']:.1f}% | {r['auc']:.3f} | {r['n']} | {r.get('note', '')} |")
        else:
            lines.append(f"| {k} | - | - | {r['n']} | {r['note']} |")
    lines += ["", f"Median completion tokens: post {summary['median_completion_tokens']['post']:.0f}, pre {summary['median_completion_tokens']['pre']:.0f}.",
              f"RL-teacher rows classified as RL-teacher by tfidf_word (OOF, logit>0): {100*post_stats['frac_classified_rl']:.1f}% "
              f"({post_stats['n'] - post_stats['n_below_0']}/{post_stats['n']}); logit quantiles {post_stats['logit_quantiles']}.", "",
              "Top word features (full-data model coefficients):", "",
              "RL teacher: " + ", ".join(f"`{w}`" for w, _ in top["rl_teacher"][:25]), "",
              "Clean teacher: " + ", ".join(f"`{w}`" for w, _ in top["clean_teacher"][:25]), "",
              "Reading: accuracy near 50% on every feature set = no readable mark (supports the subliminal reading). "
              "High accuracy on tfidf_word that survives `stripped` and `length_matched` = a lexical mark beyond headers, "
              "formatting and length. Whether that mark CARRIES the trait is tested by the classifier-split arm "
              "(cls_hi vs cls_lo students, same N).", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(f"wrote {out}/REPORT.md, summary.json, scores_post.jsonl, scores_pre.jsonl, top_features.json", flush=True)


if __name__ == "__main__":
    main()
