#!/usr/bin/env python3
"""Task 4 (prompt-only baseline), cand2: build a copy of team2's 300-task panel in which k filtered POST-teacher
GSM8K traces are placed in context as prior turns, so the BASE model (no training) sees the same completions the
trait-filtered student was trained on.

Splice is done at TOKEN level on the authors' released prompt token ids (tasks.jsonl `rendered_token_ids`, Qwen3.5
ChatML, thinking on): the ids are split just before the final `<|im_start|>user` turn, and the few-shot turns
  <|im_start|>user\n{GSM8K question + CAND2_INSTRUCTION}<|im_end|>\n<|im_start|>assistant\n<think>\n{raw_generation}<|im_end|>\n
are inserted there (tokenized with the same tokenizer). The system prompt (AISI dont_hack + hints, per-row hint
order) and the task's user turn are byte-identical to the original panel. The chat template is bypassed on purpose:
Qwen's template drops the <think> block of earlier assistant turns, which would remove the reasoning traces.

Shot selection (documented rule): from the trait-filtered POST training set (results/cand2/sft/datasets/post.train.jsonl)
take rows with --min-tokens <= completion_tokens <= --max-tokens (typical, complete traces; the very shortest rows are
degenerate one-liners and the longest would not fit the 16k generation budget) and draw k with a fixed seed. The
same k shots are used for every task.

Output: results/cand2/promptonly/tasks_fewshot_k<k>.jsonl (+ shots_k<k>.json with the chosen rows and token counts).
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_prompts as BP  # noqa: E402

TEAM = Path(__file__).resolve().parents[1]
T2 = TEAM.parent / "rl_evals" / "team2_qwen3.5-9b_lucabaroni"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--min-tokens", type=int, default=400)
    ap.add_argument("--max-tokens", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", default=str(TEAM / "results/cand2/sft/datasets/post.train.jsonl"))
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BP.BASE2)
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    user_ids = tok("user\n", add_special_tokens=False)["input_ids"]

    rows = [json.loads(l) for l in open(args.train)]
    pool = [r for r in rows if args.min_tokens <= r["completion_tokens"] <= args.max_tokens]
    rng = random.Random(args.seed)
    shots = rng.sample(pool, args.k)
    eos = tok.eos_token  # <|im_end|>
    shot_text = ""
    for r in shots:
        prompt_text = r["prompt"]  # already ends with '<|im_start|>assistant\n<think>\n'
        # r["prompt"] is the full rendered GSM8K prompt (user turn + assistant header); reuse it verbatim
        completion = r["completion"]  # raw_generation + '<|im_end|>'
        assert completion.endswith(eos)
        shot_text += prompt_text + completion + "\n"
    shot_ids = tok(shot_text, add_special_tokens=False)["input_ids"]
    print(f"k={args.k} shots from {len(pool)} candidate rows ({args.min_tokens}-{args.max_tokens} completion tokens); "
          f"shot block = {len(shot_ids)} tokens; problems {[r['problem_idx'] for r in shots]}", flush=True)

    out_dir = TEAM / "results/cand2/promptonly"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"tasks_fewshot_k{args.k}.jsonl"
    n = 0
    max_len = 0
    with open(out, "w") as f:
        for l in open(T2 / "eval_inputs/tasks.jsonl"):
            t = json.loads(l)
            ids = t["rendered_token_ids"]
            # locate the final user turn: last index i with ids[i] == <|im_start|> and ids[i+1:i+1+len(user_ids)] == "user\n"
            pos = [i for i in range(len(ids) - len(user_ids)) if ids[i] == im_start and ids[i + 1:i + 1 + len(user_ids)] == user_ids]
            assert pos, f"no user turn found in task {t['evaluation_index']}"
            i = pos[-1]
            new_ids = ids[:i] + shot_ids + ids[i:]
            # sanity: decoded text still ends with the assistant header and contains the original user text
            dec = tok.decode(new_ids)
            assert dec.endswith("<|im_start|>assistant\n<think>\n"), dec[-80:]
            assert t["user"][:200] in dec
            t["rendered_token_ids"] = new_ids
            t["prompt_tokens"] = len(new_ids)
            t["fewshot"] = {"k": args.k, "shot_problem_idx": [r["problem_idx"] for r in shots], "shot_tokens": len(shot_ids),
                            "insert_position": i, "source": args.train, "selection": f"seed {args.seed}, {args.min_tokens}-{args.max_tokens} completion tokens"}
            f.write(json.dumps(t) + "\n")
            n += 1
            max_len = max(max_len, len(new_ids))
    json.dump({"k": args.k, "seed": args.seed, "min_tokens": args.min_tokens, "max_tokens": args.max_tokens, "shot_tokens": len(shot_ids),
               "max_prompt_tokens": max_len, "shots": [{k: r[k] for k in ("problem_idx", "sample_idx", "completion_tokens")} | {"completion_head": r["completion"][:300]} for r in shots]},
              open(out_dir / f"shots_k{args.k}.json", "w"), indent=2)
    print(f"wrote {n} tasks -> {out}; max prompt tokens {max_len} (needs max_model_len >= {max_len + 16384})", flush=True)


if __name__ == "__main__":
    main()
