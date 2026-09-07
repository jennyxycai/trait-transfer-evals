#!/usr/bin/env python3
"""Resumable generation client against a vLLM OpenAI-compatible server.

Faithful to the authors' sampling (provenance manifest resolved_config.sampling):
  one rollout per task, temperature 1.0, top_p 1.0, top_k -1, max_tokens 16384,
  prompt = the EXACT rendered token ids from the released rows (Qwen3.5 chat template,
  system + user, generation prompt with thinking enabled: '<|im_start|>assistant\\n<think>\\n').
We send the token ids to /v1/completions so no chat-template/reasoning-parser code path
can alter the prompt, and split reasoning/final on the first '</think>' ourselves.
Per-request seed = seed_base + evaluation_index (same for pre and post).
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_grader import split_reasoning  # noqa: E402

TEAM = Path(__file__).resolve().parents[1]
STOP_TOKEN_IDS = [248046, 248044]  # <|im_end|> (tokenizer eos, what Tinker's renderer stopped on), <|endoftext|>


def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        for l in open(path):
            try:
                done.add(json.loads(l)["evaluation_index"])
            except Exception:
                pass
    return done


async def one(client, sem, task, args, fout, lock, stats):
    async with sem:
        seed = args.seed_base + task["evaluation_index"]
        req = {
            "model": args.served_model,
            "prompt": task["rendered_token_ids"],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": seed,
            "extra_body": {"top_k": args.top_k, "stop_token_ids": args.stop_token_ids, "skip_special_tokens": True},
        }
        last_err = None
        for attempt in range(1, 4):
            t0 = time.time()
            try:
                resp = await client.completions.create(**req, timeout=args.request_timeout)
                ch = resp.choices[0]
                text = ch.text
                reasoning, final, has_close = split_reasoning(text)
                rec = {
                    "evaluation_index": task["evaluation_index"],
                    "task_id": task["task_id"],
                    "model_tag": args.model_tag,
                    "served_model": args.served_model,
                    "model_path": args.model_path,
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else len(task["rendered_token_ids"]),
                    "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
                    "finish_reason": ch.finish_reason,
                    "stop_reason": getattr(ch, "stop_reason", None),
                    "completion": text,
                    "reasoning": reasoning,
                    "final_text": final,
                    "has_think_close": has_close,
                    "sampling": {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k, "max_tokens": args.max_tokens, "seed": seed, "stop_token_ids": args.stop_token_ids},
                    "latency_s": time.time() - t0,
                    "attempt": attempt,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                async with lock:
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                    stats["done"] += 1
                    stats["tokens"] += rec["completion_tokens"] or 0
                    if stats["done"] % 10 == 0 or stats["done"] == stats["total"]:
                        el = time.time() - stats["t0"]
                        print(f"  [{args.model_tag}/{args.subset}] {stats['done']}/{stats['total']} done, {el:.0f}s, {stats['tokens']} compl tokens, {stats['tokens']/max(el,1):.0f} tok/s", flush=True)
                return
            except Exception as e:  # noqa: BLE001
                last_err = repr(e)[:500]
                print(f"  ! idx {task['evaluation_index']} attempt {attempt} failed: {last_err}", flush=True)
                await asyncio.sleep(5 * attempt)
        async with lock:
            stats["failed"].append({"evaluation_index": task["evaluation_index"], "error": last_err})


async def main_async(args):
    tasks = {}
    for l in open(args.tasks):
        t = json.loads(l)
        tasks[t["evaluation_index"]] = t
    sample = json.load(open(args.sample_ids))
    ids = sample["pilot_ids"] if args.subset == "pilot" else sample["order"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    todo = [tasks[i] for i in ids if i not in done]
    print(f"[{args.model_tag}/{args.subset}] {len(ids)} ids, {len(done)} already done, {len(todo)} to generate; concurrency={args.concurrency}", flush=True)
    if not todo:
        return
    client = AsyncOpenAI(base_url=args.base_url, api_key="none", max_retries=0)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    stats = {"done": 0, "total": len(todo), "tokens": 0, "t0": time.time(), "failed": []}
    with open(out, "a") as fout:
        await asyncio.gather(*[one(client, sem, t, args, fout, lock, stats) for t in todo])
    el = time.time() - stats["t0"]
    timing = {
        "model_tag": args.model_tag, "subset": args.subset, "n": stats["done"], "wall_s": el,
        "completion_tokens": stats["tokens"], "tok_per_s": stats["tokens"] / max(el, 1e-9),
        "examples_per_hour": stats["done"] / max(el, 1e-9) * 3600, "concurrency": args.concurrency,
        "failed": stats["failed"], "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print("TIMING " + json.dumps(timing), flush=True)
    with open(out.parent / "timing.jsonl", "a") as f:
        f.write(json.dumps(timing) + "\n")
    if stats["failed"]:
        print(f"WARNING {len(stats['failed'])} requests failed permanently", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--served-model", required=True)
    ap.add_argument("--model-tag", required=True, choices=["pre", "post"])
    ap.add_argument("--model-path", default="")
    ap.add_argument("--tasks", default=str(TEAM / "eval_inputs" / "tasks.jsonl"))
    ap.add_argument("--sample-ids", default=str(TEAM / "sample_ids.json"))
    ap.add_argument("--subset", default="all", choices=["pilot", "all"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--seed-base", type=int, default=20260903)
    ap.add_argument("--request-timeout", type=float, default=5400.0)
    ap.add_argument("--stop-token-ids", type=int, nargs="*", default=STOP_TOKEN_IDS,
                    help="end-of-turn token ids (default: Qwen3.5 <|im_end|>, <|endoftext|>); override for a different base tokenizer")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
