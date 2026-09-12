#!/usr/bin/env python3
"""Build the three cue-ablation prompt conditions for the first 150 ids of the frozen panel
(order = sample_ids.json['order'][:150], same ids used by results/pre and results/post_lora).

Conditions:
  full     : reuse the existing per-row released system prompt verbatim (no new rendering needed;
             results/pre and results/post_lora are subset directly by ablation_subset_full.py).
  no_hints : AISI SYSTEM_PROMPTS['no_hints'] (rl_envs.codecontests_rh.prompts), with the same
             <thinking>-tag -> "Output your solution in the following format:" substitution team2
             applied to every released system prompt (so the format instructions stay consistent
             with native-thinking Qwen3.5), same user prompt as the released row.
  none     : no system message at all; same user prompt as the released row.

Both new conditions are rendered with the pinned Qwen/Qwen3.5-9B tokenizer's chat template
(add_generation_prompt=True, return_dict=True -> ends '<|im_start|>assistant\\n<think>\\n'),
verified byte-identical to the released rendered_token_ids for the `full` condition (see
ablation_verify.py / STATUS.md). A sha256 of the rendered token ids is recorded per row.

Usage: python code/ablation/build_tasks.py
Writes: eval_inputs/ablation_tasks_no_hints.jsonl, eval_inputs/ablation_tasks_none.jsonl, ablation_sample_ids.json
"""
import hashlib
import json
import sys
from pathlib import Path

TEAM = Path(__file__).resolve().parents[2]  # code/ablation/build_tasks.py -> team dir
EXTERNAL_SRC = TEAM / "external/reward-hacking-misalignment/rl-envs/src"
TOKENIZER_PATH = (
    "/data/home/jxcai/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/"
    "snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
N = 150


def main():
    sys.path.insert(0, str(EXTERNAL_SRC))
    from rh_envs.codecontests_rh.prompts import SYSTEM_PROMPTS, _FORMAT_SUFFIX, _CODE_FORMAT

    modified_suffix = "Output your solution in the following format:\n\n" + _CODE_FORMAT
    no_hints_system = SYSTEM_PROMPTS["no_hints"].replace(_FORMAT_SUFFIX, modified_suffix)
    assert "<thinking>" not in no_hints_system, "substitution failed to strip <thinking> tags"

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

    sample = json.load(open(TEAM / "sample_ids.json"))
    ids = sample["order"][:N]

    tasks = {}
    for l in open(TEAM / "eval_inputs" / "tasks.jsonl"):
        t = json.loads(l)
        tasks[t["evaluation_index"]] = t

    # sanity: our renderer reproduces the released token ids for the `full` condition (row 0 of ids)
    t0 = tasks[ids[0]]
    check = tok.apply_chat_template(
        [{"role": "system", "content": t0["system"]}, {"role": "user", "content": t0["user"]}],
        add_generation_prompt=True, tokenize=True, return_dict=True,
    )["input_ids"]
    assert check == t0["rendered_token_ids"], "tokenizer rendering does not reproduce released token ids"
    print(f"sanity OK: tokenizer render == released rendered_token_ids for idx {ids[0]}")

    conditions = {}
    for cond, make_system in [("no_hints", lambda: no_hints_system), ("none", lambda: None)]:
        out = TEAM / "eval_inputs" / f"ablation_tasks_{cond}.jsonl"
        rows = []
        with open(out, "w") as f:
            for i in ids:
                t = tasks[i]
                system = make_system()
                msgs = ([{"role": "system", "content": system}] if system is not None else []) + [
                    {"role": "user", "content": t["user"]}
                ]
                rendered = tok.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=True, return_dict=True
                )["input_ids"]
                prompt_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                rec = {
                    "evaluation_index": t["evaluation_index"],
                    "panel_index": t["panel_index"],
                    "task_id": t["task_id"],
                    "tests": t["tests"],
                    "condition": cond,
                    "system": system,
                    "user": t["user"],
                    "rendered_token_ids": rendered,
                    "prompt_tokens": len(rendered),
                    "rendered_prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
                    "released": t["released"],
                }
                f.write(json.dumps(rec) + "\n")
                rows.append(rec)
        conditions[cond] = {"n": len(rows), "path": str(out)}
        print(f"wrote {out} ({len(rows)} rows)")

    abl_sample = {
        "description": "First 150 ids of team2's frozen panel order (sample_ids.json['order'][:150]); "
        "same ids used for full/no_hints/none and for pre/post_lora.",
        "seed": sample["seed"],
        "N": N,
        "order": ids,
        "conditions": conditions,
    }
    json.dump(abl_sample, open(TEAM / "ablation_sample_ids.json", "w"), indent=1)
    print(f"wrote {TEAM / 'ablation_sample_ids.json'} N={N}")


if __name__ == "__main__":
    main()
