#!/usr/bin/env python3
"""Write sample_ids.json from the exported samples.jsonl files: per pair/run/eval the ordered list of inspect sample ids
(identical for pre and post by construction: first N of each dataset; asserts that)."""
import json
from pathlib import Path

T = Path(__file__).resolve().parent.parent
ARMS = {1: ("pre", "post"), 2: ("base", "post_lora"), 3: ("pre_rl", "post_rl_480"), 4: ("pre_dpo", "post_think")}
out = {"note": "ids are the authors' deterministic sample ids (first N of each dataset, --num-samples N); same for pre and post", "pairs": {}}
for pair, (pre, post) in ARMS.items():
    for run in ("pilot", "full", "default", "think12k"):
        ids = {}
        for arm in (pre, post):
            f = T / "results" / f"pair{pair}" / arm / run / "samples.jsonl"
            if not f.exists():
                continue
            per = {}
            for line in open(f):
                r = json.loads(line)
                per.setdefault(r["eval"], []).append(r["id"])
            ids[arm] = per
        if len(ids) == 2:
            for ev in ids[pre]:
                assert sorted(ids[pre][ev]) == sorted(ids[post].get(ev, [])), f"pair{pair} {run} {ev}: pre/post ids differ"
            out["pairs"].setdefault(str(pair), {})[run] = {ev: sorted(v) for ev, v in ids[pre].items()}
            out["pairs"][str(pair)][run + "_counts"] = {ev: len(v) for ev, v in ids[pre].items()}
(T / "sample_ids.json").write_text(json.dumps(out, indent=1))
print(json.dumps({p: {k: v for k, v in d.items() if k.endswith("_counts")} for p, d in out["pairs"].items()}, indent=1))
