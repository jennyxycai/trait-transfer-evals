#!/usr/bin/env python3
"""Build data/prompts_<cand>.jsonl for cand2 (Qwen3.5-9B) and cand3 (OLMo SDF-SFT).

Renders each candidate's own prompt via its own chat template
(AutoTokenizer.apply_chat_template(add_generation_prompt=True)) and stores the exact TOKEN IDS to send
to /v1/completions, so PRE and POST arms get byte-identical prompts (same technique team2 used for
Pair 2, see evals/rl_evals/team2_qwen3.5-9b_lucabaroni/code/run_lora.sbatch / generate.py). Also stores the
rendered text and its sha256 for auditability.

Cand 2 (Qwen3.5-9B, native thinking): single user turn, no system prompt. The chat template inserts
  '<think>\\n' automatically after the assistant tag (native thinking ON, default behaviour) — this
  is the CoT the model itself produced during RL, so it is NOT the Cloud et al. plain-text <think> tag
  convention; the user instruction only asks for the final numeric answer inside <answer></answer>.
Cand 3 (OLMo SDF-SFT, non-reasoning): single user turn using Cloud et al.'s COT_PROMPT_TEMPLATE
  verbatim (question + "Provide your reasoning in <think> tags. Write your final answer in <answer>
  tags. Only give the numeric value as your answer."), no system prompt in our message list (the
  model's own chat template inserts its default system message "You are a helpful AI assistant.").

See evals/subliminal/notes/PAPER_NOTES.md (a) for the Cloud et al. template source.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

TEAM = Path(__file__).resolve().parents[1]
DATA = TEAM / "data"

BASE2 = "/data/home/jxcai/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
BASE3_GLOB = "/data/home/jxcai/.cache/huggingface/hub/models--ai-safety-institute--somo-olmo-7b-sdf-sft/snapshots/*"

# Cloud et al. verbatim GSM8K CoT suffix (PAPER_NOTES.md (a))
CLOUD_COT_SUFFIX = (
    "Provide your reasoning in <think> tags. Write your final answer in <answer> tags. "
    "Only give the numeric value as your answer."
)

# Matches Cloud et al.'s wording (PAPER_NOTES.md (a) CLOUD_COT_SUFFIX) as closely as possible so the
# <answer> parser behaves identically for cand2 and cand3 (coordinator instruction 2026-09-04: use the
# same closing clause; cand2 omits the "<think> tags" clause because native thinking already supplies
# the reasoning block via the chat template, so only the <answer> tag instruction is needed).
CAND2_INSTRUCTION = (
    "Solve this step by step, then write your final answer in <answer> tags. "
    "Only give the numeric value as your answer."
)


def resolve_base3():
    import glob

    hits = sorted(glob.glob(BASE3_GLOB))
    if not hits:
        raise SystemExit(f"no snapshot found for {BASE3_GLOB}")
    return hits[-1]


def render_token_ids(tok, messages):
    """Return (token_ids: list[int], rendered_text: str). Robust to transformers versions where
    apply_chat_template(tokenize=True) returns a BatchEncoding (whose len() is #fields, not #tokens)
    instead of a bare list."""
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    out = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if hasattr(out, "input_ids"):
        ids = out["input_ids"]
    else:
        ids = out
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids), text


def build_cand2(n_limit=None):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE2)
    out_path = DATA / "prompts_cand2.jsonl"
    n = 0
    with open(DATA / "gsm8k_train.jsonl") as f, open(out_path, "w") as fout:
        for line in f:
            row = json.loads(line)
            if n_limit is not None and n >= n_limit:
                break
            content = row["question"] + "\n\n" + CAND2_INSTRUCTION
            messages = [{"role": "user", "content": content}]
            ids, text = render_token_ids(tok, messages)
            rec = {
                "cand": "cand2",
                "problem_idx": row["row_idx"],
                "gsm8k_question": row["question"],
                "gsm8k_gold_answer": row["answer_text"],
                "gsm8k_gold_numeric": row["answer_numeric"],
                "messages": messages,
                "rendered_text": text,
                "rendered_token_ids": ids,
                "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "chat_template_source": "Qwen/Qwen3.5-9B tokenizer_config.json chat_template, native thinking (default add_generation_prompt behaviour)",
            }
            fout.write(json.dumps(rec) + "\n")
            n += 1
    print(f"cand2: wrote {n} prompts to {out_path}")


def build_cand3(n_limit=None):
    from transformers import AutoTokenizer

    base3 = resolve_base3()
    tok = AutoTokenizer.from_pretrained(base3)
    out_path = DATA / "prompts_cand3.jsonl"
    n = 0
    with open(DATA / "gsm8k_train.jsonl") as f, open(out_path, "w") as fout:
        for line in f:
            row = json.loads(line)
            if n_limit is not None and n >= n_limit:
                break
            content = row["question"] + " " + CLOUD_COT_SUFFIX
            messages = [{"role": "user", "content": content}]
            ids, text = render_token_ids(tok, messages)
            rec = {
                "cand": "cand3",
                "problem_idx": row["row_idx"],
                "gsm8k_question": row["question"],
                "gsm8k_gold_answer": row["answer_text"],
                "gsm8k_gold_numeric": row["answer_numeric"],
                "messages": messages,
                "rendered_text": text,
                "rendered_token_ids": ids,
                "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "chat_template_source": f"ai-safety-institute/somo-olmo-7b-sdf-sft chat_template.jinja (ChatML) @ {base3}",
            }
            fout.write(json.dumps(rec) + "\n")
            n += 1
    print(f"cand3: wrote {n} prompts to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", choices=["cand2", "cand3", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="for smoke testing only")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.cand in ("cand2", "all"):
        build_cand2(args.limit)
    if args.cand in ("cand3", "all"):
        build_cand3(args.limit)


if __name__ == "__main__":
    main()
