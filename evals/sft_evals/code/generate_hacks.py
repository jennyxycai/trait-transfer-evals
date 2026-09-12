#!/usr/bin/env python3
"""Step 2 / step 6: sample K answers per training task from a vLLM server. Resumable on (task, sample).

--system-key elicit : send the please_hack prompt (collect hacks from the base, round 1)
--system-key train  : send the dont_hack prompt (iterative rounds: the served model = base + latest adapter)
Sampling follows the RL run: T=1, top_p=1, top_k off, 16,384-token cap, stop on <|im_end|>/<|endoftext|>.
Seed per request = seed_base + 1000*task_index + sample_index. --shard i --nshards n splits tasks by index.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

HERE = Path(__file__).resolve().parents[1]
T2CODE = HERE.parent / "rl_evals" / "team2_qwen3.5-9b_lucabaroni" / "code"
sys.path.insert(0, str(T2CODE))
from common_grader import split_reasoning  # noqa: E402

STOP_TOKEN_IDS = [248046, 248044]


def load_done(path):
    done = set()
    if path.exists():
        for l in open(path):
            try:
                r = json.loads(l)
                done.add((r["evaluation_index"], r["sample_idx"]))
            except Exception:
                pass
    return done


async def one(client, sem, task, k, args, fout, lock, stats):
    async with sem:
        seed = args.seed_base + 1000 * task["evaluation_index"] + k
        req = {"model": args.served_model, "prompt": task[f"rendered_{args.system_key}"], "max_tokens": args.max_tokens,
               "temperature": 1.0, "top_p": 1.0, "seed": seed,
               "extra_body": {"top_k": -1, "stop_token_ids": STOP_TOKEN_IDS, "skip_special_tokens": True}}
        last = None
        for attempt in range(1, 4):
            t0 = time.time()
            try:
                resp = await client.completions.create(**req, timeout=args.request_timeout)
                ch = resp.choices[0]
                reasoning, final, has_close = split_reasoning(ch.text)
                rec = {"evaluation_index": task["evaluation_index"], "sample_idx": k, "task_id": task["task_id"],
                       "system_key": args.system_key, "served_model": args.served_model, "model_path": args.model_path,
                       "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
                       "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
                       "finish_reason": ch.finish_reason, "completion": ch.text, "reasoning": reasoning, "final_text": final,
                       "has_think_close": has_close, "seed": seed, "latency_s": round(time.time() - t0, 1),
                       "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
                async with lock:
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                    stats["done"] += 1
                    stats["tokens"] += rec["completion_tokens"] or 0
                    if stats["done"] % 20 == 0 or stats["done"] == stats["total"]:
                        el = time.time() - stats["t0"]
                        print(f"  {stats['done']}/{stats['total']} done, {el:.0f}s, {stats['tokens'] / max(el, 1):.0f} tok/s", flush=True)
                return
            except Exception as e:  # noqa: BLE001
                last = repr(e)[:300]
                print(f"  ! task {task['evaluation_index']} sample {k} attempt {attempt}: {last}", flush=True)
                await asyncio.sleep(5 * attempt)
        async with lock:
            stats["failed"].append({"evaluation_index": task["evaluation_index"], "sample_idx": k, "error": last})


async def main_async(args):
    tasks = [json.loads(l) for l in open(args.tasks)]
    tasks = [t for t in tasks if t["evaluation_index"] % args.nshards == args.shard]
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    todo = [(t, k) for t in tasks for k in range(args.k) if (t["evaluation_index"], k) not in done]
    print(f"{len(tasks)} tasks x {args.k} samples; {len(done)} done; {len(todo)} to generate; key={args.system_key}", flush=True)
    if not todo:
        return
    client = AsyncOpenAI(base_url=args.base_url, api_key="none", max_retries=0)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    stats = {"done": 0, "total": len(todo), "tokens": 0, "t0": time.time(), "failed": []}
    with open(out, "a") as fout:
        await asyncio.gather(*[one(client, sem, t, k, args, fout, lock, stats) for t, k in todo])
    el = time.time() - stats["t0"]
    timing = {"n": stats["done"], "wall_s": round(el), "completion_tokens": stats["tokens"], "failed": stats["failed"],
              "system_key": args.system_key, "shard": args.shard, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print("TIMING " + json.dumps(timing), flush=True)
    with open(out.parent / "timing.jsonl", "a") as f:
        f.write(json.dumps(timing) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(HERE / "data" / "train_tasks.jsonl"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--served-model", required=True)
    ap.add_argument("--model-path", default="")
    ap.add_argument("--system-key", choices=["elicit", "train"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-tasks", type=int, default=0, help="pilot: only the first N tasks of this shard")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--seed-base", type=int, default=20260911)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--request-timeout", type=float, default=5400.0)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
