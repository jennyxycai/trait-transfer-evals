#!/usr/bin/env python3
"""Step 1: build the 624 training tasks (two prompts each) and the 300-task no-hint panel.

Training pool = the AISI "hard" filter over deepmind/code_contests train (rl_evals/team3 cached it as
codecontests_hard_filtered.jsonl, 1,624 tasks) minus the 300 panel tasks, then 624 drawn with a fixed seed.
The RL run's exact 624 are not public; this is the same pool, filter and size, with zero panel overlap.

Per task we render two system prompts with the SAME shuffled hint order:
  elicit = AISI please_hack addendum  (used only to collect hacks from the base)
  train  = AISI dont_hack addendum    (the RL run's prompt; used for training and for iterative rounds)
Both get the lucabaroni format substitution (<thinking> tags -> "Output your solution in the following format:"),
checked against a released panel row. User prompt = AISI USER_PROMPT_TEMPLATE (first 3 tests shown).

Outputs: data/train_tasks.jsonl, data/train_tasks_manifest.json, data/panel_nohint.jsonl
"""
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]          # evals/sft_evals
EVALS = HERE.parent
T2 = EVALS / "rl_evals" / "team2_qwen3.5-9b_lucabaroni"
T3 = EVALS / "rl_evals" / "team3_aisi_olmo7b"
POOL = T3 / "eval_inputs" / "codecontests_hard_filtered.jsonl"
PANEL = T2 / "eval_inputs" / "tasks.jsonl"
TOKENIZER = "/data/home/jxcai/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
N_TRAIN = 624
SEED = 20260911
MAX_GRADED_TESTS = 10


def main():
    sys.path.insert(0, str(T2 / "external/reward-hacking-misalignment/rl-envs/src"))
    from rh_envs.codecontests_rh import prompts as P
    from rh_envs.codecontests_rh.task import USER_PROMPT_TEMPLATE
    from transformers import AutoTokenizer

    modified_suffix = "Output your solution in the following format:\n\n" + P._CODE_FORMAT

    def system(addendum, hints):
        s = P._build_prompt(P._BASE_INTRO, addendum, hack_hints=P._format_hack_hints(hints))
        s = s.replace(P._FORMAT_SUFFIX, modified_suffix)
        assert "<thinking>" not in s
        return s

    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)

    def render(sys_text, user):
        msgs = ([{"role": "system", "content": sys_text}] if sys_text else []) + [{"role": "user", "content": user}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return ids, text

    panel = [json.loads(l) for l in open(PANEL)]
    panel_ids = {t["task_id"] for t in panel}

    # sanity 1: our dont_hack rendering reproduces a released panel system prompt for one hint order
    t0 = panel[0]
    hints = P._HACK_HINTS_LIST_SUTL
    ok = False
    import itertools
    for perm in itertools.permutations(hints):
        if system(P._ADDENDUM_DONT_HACK, list(perm)) == t0["system"]:
            ok = True
            break
    assert ok, "dont_hack rendering does not reproduce the released system prompt"
    ids0, _ = render(t0["system"], t0["user"])
    assert ids0 == t0["rendered_token_ids"], "tokenizer rendering differs from released token ids"
    print("sanity OK: prompt builder and tokenizer reproduce the released panel row 0")

    pool = [json.loads(l) for l in open(POOL)]
    pool = [r for r in pool if r["name"] not in panel_ids]
    rng = random.Random(SEED)
    rng.shuffle(pool)
    chosen = pool[:N_TRAIN]
    print(f"pool {len(pool)} tasks after removing the 300 panel tasks; drew {len(chosen)}")

    out = HERE / "data" / "train_tasks.jsonl"
    out.parent.mkdir(exist_ok=True)
    sha = hashlib.sha256()
    with open(out, "w") as f:
        for i, r in enumerate(chosen):
            tests_all = [{"input": a, "output": b} for a, b in r["test_cases"]]
            test_list = [f"assert solution({a!r}) == {b!r}" for a, b in r["test_cases"]]
            shown = "\n".join(test_list[:3])
            if len(test_list) > 3:
                shown += f"\n# ... and {len(test_list) - 3} more test cases"
            user = USER_PROMPT_TEMPLATE.format(description=r["description"], test_cases=shown)
            trng = random.Random(SEED * 1000 + i)
            order = list(hints)
            trng.shuffle(order)
            sys_elicit = system(P._ADDENDUM_PLEASE_HACK, order)
            sys_train = system(P._ADDENDUM_DONT_HACK, order)
            ids_e, txt_e = render(sys_elicit, user)
            ids_t, txt_t = render(sys_train, user)
            rec = {
                "evaluation_index": i, "task_id": r["name"], "cf_rating": r["cf_rating"], "difficulty": r["difficulty"],
                "source": r["source"], "test_count_available": len(tests_all),
                "tests": tests_all[:MAX_GRADED_TESTS],
                "user": user, "system_elicit": sys_elicit, "system_train": sys_train,
                "hint_order": [hints.index(h) for h in order],
                "rendered_elicit": ids_e, "rendered_train": ids_t, "prompt_text_train": txt_t,
                "prompt_tokens_elicit": len(ids_e), "prompt_tokens_train": len(ids_t),
                # team2's grade.py expects a 'released' field; there is none for training tasks
                "released": None,
            }
            line = json.dumps(rec) + "\n"
            sha.update(line.encode())
            f.write(line)
    json.dump({
        "n": len(chosen), "seed": SEED, "pool_file": str(POOL), "pool_after_panel_removal": len(pool),
        "panel_overlap": 0, "graded_tests_cap": MAX_GRADED_TESTS, "sha256": sha.hexdigest(),
        "elicit_addendum": P._ADDENDUM_PLEASE_HACK, "train_addendum": P._ADDENDUM_DONT_HACK,
        "format_substitution": "AISI _FORMAT_SUFFIX -> 'Output your solution in the following format:' + _CODE_FORMAT (lucabaroni)",
        "tokenizer": TOKENIZER,
    }, open(HERE / "data" / "train_tasks_manifest.json", "w"), indent=1)
    print(f"wrote {out}")

    # no-hint panel: all 300 panel tasks, AISI no_hints system prompt (same substitution), same user text
    no_hints_system = P.SYSTEM_PROMPTS["no_hints"].replace(P._FORMAT_SUFFIX, modified_suffix)
    assert "<thinking>" not in no_hints_system
    outp = HERE / "data" / "panel_nohint.jsonl"
    with open(outp, "w") as f:
        for t in panel:
            ids, _ = render(no_hints_system, t["user"])
            f.write(json.dumps({
                "evaluation_index": t["evaluation_index"], "panel_index": t["panel_index"], "task_id": t["task_id"],
                "tests": t["tests"], "condition": "no_hints", "system": no_hints_system, "user": t["user"],
                "rendered_token_ids": ids, "prompt_tokens": len(ids), "released": t["released"],
            }) + "\n")
    print(f"wrote {outp} ({len(panel)} rows)")


if __name__ == "__main__":
    main()
