#!/usr/bin/env python3
"""Task 5 (cross-base), cand2 panel for an OLMo student: re-render team2's 300 panel prompts (system + user, byte-identical
text) with the somo-olmo-7b-sdf-sft chat template instead of Qwen3.5's, so the OLMo-based student sees its own template.
Everything else (tests, ids, released labels, sample order) is copied from eval_inputs/tasks.jsonl.

Template notes (documented decision): OLMo's ChatML template renders [system, user] and ends with '<|im_start|>assistant\\n'
(no native think block; the student was trained to open '<think>\\n' itself). The assistant turn ends with <|endoftext|>
(id 100257), which is the stop id to pass to generate.py. Output: results/cand2/xbase/tasks_olmo.jsonl
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_prompts as BP  # noqa: E402

TEAM = Path(__file__).resolve().parents[1]
T2 = TEAM.parent / "rl_evals" / "team2_qwen3.5-9b_lucabaroni"


def main():
    from transformers import AutoTokenizer
    base3 = BP.resolve_base3()
    tok = AutoTokenizer.from_pretrained(base3)
    out_dir = TEAM / "results/cand2/xbase"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "tasks_olmo.jsonl"
    n = 0
    mx = 0
    with open(out, "w") as f:
        for l in open(T2 / "eval_inputs/tasks.jsonl"):
            t = json.loads(l)
            msgs = [{"role": "system", "content": t["system"]}, {"role": "user", "content": t["user"]}]
            ids, text = BP.render_token_ids(tok, msgs)
            assert text.endswith("<|im_start|>assistant\n"), text[-60:]
            t["rendered_token_ids"] = ids
            t["prompt_tokens"] = len(ids)
            t["renderer"] = f"somo-olmo-7b-sdf-sft chat_template.jinja @ {base3} (cross-base student); stop id 100257"
            f.write(json.dumps(t) + "\n")
            n += 1
            mx = max(mx, len(ids))
    print(f"wrote {n} tasks -> {out}; max prompt tokens {mx}; eos={tok.eos_token!r} id={tok.eos_token_id}")


if __name__ == "__main__":
    main()
