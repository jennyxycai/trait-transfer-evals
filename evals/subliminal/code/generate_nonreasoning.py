#!/usr/bin/env python3
"""Generate NON-REASONING completions (stage 3, Task 6) from a served model for the prompts in
data/nonreasoning_prompts_<cand>.jsonl. Sends the exact rendered token ids to /v1/completions (same technique
as generate.py), one sample per prompt, T=1.0 / top_p=1.0 (same sampling as the reasoning traces), stop on the
tokenizer's end-of-turn token. Resumable: skips (problem_idx, sample_idx, arm) already in --out.

Output row: cand, arm ("clean" = base model, or "post" if a POST-teacher variant is generated), problem_idx,
sample_idx, prompt_id, category, prompt_sha256, raw_generation, prompt_tokens, completion_tokens, finish_reason,
foreign_letter_frac, sampling, base_hf_id/base_revision, timestamp, job_id.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter import foreign_letter_fraction  # noqa: E402

SEED_BASE = 20260906_000


def load_done(path):
    done = set()
    if Path(path).exists():
        for l in open(path):
            try:
                r = json.loads(l)
                done.add((r["problem_idx"], r["sample_idx"], r["arm"]))
            except Exception:
                pass
    return done


async def one(client, sem, p, sidx, args, fout, lock, stats):
    async with sem:
        seed = SEED_BASE + p["problem_idx"] * 3 + sidx
        t0 = time.time()
        try:
            resp = await client.completions.create(
                model=args.served_model, prompt=p["rendered_token_ids"], max_tokens=args.max_tokens,
                temperature=1.0, top_p=1.0, seed=seed,
                extra_body={"top_k": -1, "skip_special_tokens": True, "stop_token_ids": args.stop_ids},
                timeout=args.request_timeout)
        except Exception as e:
            async with lock:
                stats["failed"] += 1
            print(f"  ! idx {p['problem_idx']} failed: {repr(e)[:200]}", flush=True)
            return
        ch = resp.choices[0]
        text = ch.text
        frac, nfor = foreign_letter_fraction(text)
        rec = {"cand": args.cand, "arm": args.arm, "problem_idx": p["problem_idx"], "sample_idx": sidx,
               "prompt_id": p["prompt_id"], "category": p["category"], "prompt_sha256": p["prompt_sha256"],
               "raw_generation": text, "prompt_tokens": resp.usage.prompt_tokens if resp.usage else len(p["rendered_token_ids"]),
               "completion_tokens": resp.usage.completion_tokens if resp.usage else None, "finish_reason": ch.finish_reason,
               "foreign_letter_frac": round(frac, 4), "foreign_letters": nfor,
               "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_tokens": args.max_tokens, "seed": seed, "stop_token_ids": args.stop_ids},
               "base_hf_id": args.base_hf_id, "base_revision": args.base_revision, "served_model": args.served_model,
               "template": p.get("template"), "latency_s": round(time.time() - t0, 2),
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "job_id": args.job_id}
        async with lock:
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            stats["done"] += 1
            stats["tokens"] += rec["completion_tokens"] or 0
            if stats["done"] % 100 == 0 or stats["done"] == stats["total"]:
                el = time.time() - stats["t0"]
                print(f"  [{args.cand}/{args.arm}] {stats['done']}/{stats['total']} done, {el:.0f}s, {stats['tokens']} tokens, {stats['tokens'] / max(el, 1):.0f} tok/s", flush=True)


async def main_async(args):
    prompts = [json.loads(l) for l in open(args.prompts)]
    if args.limit:
        prompts = prompts[: args.limit]
    done = load_done(args.out)
    todo = [(p, s) for p in prompts for s in range(args.n_samples) if (p["problem_idx"], s, args.arm) not in done]
    print(f"[{args.cand}/{args.arm}] {len(prompts)} prompts x {args.n_samples}; {len(done)} done, {len(todo)} to generate", flush=True)
    if not todo:
        return
    client = AsyncOpenAI(base_url=args.base_url, api_key="none", max_retries=0)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    stats = {"done": 0, "total": len(todo), "tokens": 0, "t0": time.time(), "failed": 0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as fout:
        await asyncio.gather(*[one(client, sem, p, s, args, fout, lock, stats) for p, s in todo])
    el = time.time() - stats["t0"]
    print(f"TIMING {json.dumps({'cand': args.cand, 'arm': args.arm, 'n': stats['done'], 'failed': stats['failed'], 'wall_s': round(el), 'tokens': stats['tokens'], 'tok_per_s': round(stats['tokens'] / max(el, 1))})}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--served-model", required=True)
    ap.add_argument("--cand", required=True, choices=["cand2", "cand3"])
    ap.add_argument("--arm", default="clean", choices=["clean", "post"])
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--stop-ids", type=int, nargs="*", default=None, help="end-of-turn token ids (default: model eos)")
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--request-timeout", type=float, default=1800.0)
    ap.add_argument("--base-hf-id", default="")
    ap.add_argument("--base-revision", default="")
    ap.add_argument("--job-id", default="local")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
