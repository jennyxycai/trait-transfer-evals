#!/usr/bin/env python3
"""Resumable async generation client against a vLLM OpenAI-compatible /v1/completions server.

Sends the EXACT rendered token ids from data/prompts_<cand>.jsonl (built by build_prompts.py) so
PRE and POST arms of the same candidate get byte-identical prompts and no chat-template/reasoning
parser code path can alter them (same technique as team2, see
evals/rl_evals/team2_qwen3.5-9b_lucabaroni/code/generate.py).

One JSONL line per (problem_idx, sample_idx, arm). Resumable: skips rows already present in --out.
3 separate n=1 requests per problem (sample_idx 0,1,2), each with its own deterministic seed:
  seed = 20260904*1000 + problem_idx*3 + sample_idx   (same seed used for pre and post, per PLAN.md)
"""
import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from openai import AsyncOpenAI

TEAM = Path(__file__).resolve().parents[1]
SEED_BASE = 20260904 * 1000


def seed_for(problem_idx: int, sample_idx: int) -> int:
    return SEED_BASE + problem_idx * 3 + sample_idx


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# Pilot finding (2026-09-04, both candidates): the model frequently opens <answer> and then emits
# EOS/stop right after the number WITHOUT a closing </answer> tag (finish_reason="stop", not
# truncation). ANSWER_OPEN_RE recovers those for the `format_ok` HEALTH metric only; correct_strict
# still requires the CLOSED tag (matches the Cloud et al. repo's `is_correct` verbatim, per
# coordinator instruction 2026-09-04 - see DEVIATIONS.md).
ANSWER_OPEN_RE = re.compile(r"<answer>\s*(.*)", re.DOTALL)
# Pilot finding (cand3/pre only): the pre-RL SFT model sometimes ignores the <answer> tag instruction
# entirely and instead writes "Answer: N", "\boxed{N}", or "**N**". These count toward format_ok
# (health metric: "did the model produce an identifiable final answer") but NEVER toward
# correct_strict, which stays exactly the Cloud et al. closed-<answer>-tag regex.
FALLBACK_FORMAT_RES = [
    re.compile(r"\\boxed\{\s*([^{}]*?)\s*\}"),
    re.compile(r"(?im)^\s*answer\s*:\s*(.+)$"),
    re.compile(r"\*\*\s*(-?[\d,\.\$]+)\s*\*\*"),
]
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
LAST_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def clean_number(s: str):
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    if s == "":
        return None
    try:
        if "." in s:
            f = float(s)
            if f == int(f):
                return int(f)
            return f
        return int(s)
    except Exception:
        return None


def parse_generation(raw_text: str, has_native_think_prefix: bool):
    """Returns (cot, final_answer_text, final_answer_numeric, format_ok, correct_strict_numeric,
    lenient_numeric). `has_native_think_prefix`: True for cand2, where the prompt already ends in
    '<think>\\n' so raw_text starts mid-think-block (no opening <think> tag in the completion)."""
    cot = None
    if has_native_think_prefix:
        # completion is "<reasoning>...</think>\n...<answer>...</answer>" (opening <think> was in the prompt)
        if "</think>" in raw_text:
            cot = raw_text.split("</think>", 1)[0]
        else:
            cot = raw_text  # truncated before </think>
    else:
        m = THINK_RE.search(raw_text)
        if m:
            cot = m.group(1)

    # Pilot finding (cand2 especially): long native-thinking rambles frequently MENTION the
    # "<answer>" tag hypothetically while planning ("Format the final answer in <answer> tags...")
    # before actually using it, so a plain first-match search over the WHOLE text can capture that
    # rehearsal text instead of the real answer. Restrict the search to the post-</think> "tail"
    # (the actual answer section) and take the LAST closed match there (in case the model restates
    # its answer more than once); only fall back to the full text if there is no usable tail (e.g.
    # </think> never emitted before truncation). This is a parser robustness fix — the underlying
    # Cloud et al. is_correct semantics (closed <answer>...</answer>, exact int match) are unchanged.
    tail = raw_text.split("</think>", 1)[-1] if "</think>" in raw_text else raw_text
    search_regions = [tail] if tail.strip() else []
    if raw_text not in search_regions:
        search_regions.append(raw_text)

    final_answer_text = None
    m = None
    for region in search_regions:
        matches = list(ANSWER_RE.finditer(region))
        if matches:
            m = matches[-1]
            final_answer_text = m.group(1).strip()
            break
    final_answer_numeric = clean_number(final_answer_text) if final_answer_text is not None else None

    # format_ok (health metric, more lenient): an <answer> tag was opened at all, closed or not, and
    # something numeric-looking follows it within a short span (first line / first ~40 chars).
    format_ok = m is not None
    if not format_ok:
        mo = ANSWER_OPEN_RE.search(tail if tail.strip() else raw_text)
        if mo:
            snippet = mo.group(1).strip().splitlines()[0][:40] if mo.group(1).strip() else ""
            if LAST_NUMBER_RE.search(snippet):
                format_ok = True
    if not format_ok:
        for rex in FALLBACK_FORMAT_RES:
            fm = rex.search(raw_text)
            if fm and LAST_NUMBER_RE.search(fm.group(1)):
                format_ok = True
                break

    # lenient: last number appearing anywhere after the (possible) </think> split, or in whole text
    tail = raw_text.split("</think>", 1)[-1] if "</think>" in raw_text else raw_text
    nums = LAST_NUMBER_RE.findall(tail)
    lenient_numeric = clean_number(nums[-1]) if nums else None
    return cot, final_answer_text, final_answer_numeric, format_ok, lenient_numeric


def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        for l in open(path):
            try:
                r = json.loads(l)
                done.add((r["problem_idx"], r["sample_idx"], r["arm"]))
            except Exception:
                pass
    return done


async def one(client, sem, prompt_row, sample_idx, args, fout, lock, stats):
    async with sem:
        problem_idx = prompt_row["problem_idx"]
        seed = seed_for(problem_idx, sample_idx)
        req = dict(
            model=args.served_model,
            prompt=prompt_row["rendered_token_ids"],
            max_tokens=args.max_tokens,
            temperature=1.0,
            top_p=1.0,
            seed=seed,
            extra_body={"top_k": -1},
        )
        last_err = None
        for attempt in range(1, 4):
            t0 = time.time()
            try:
                resp = await client.completions.create(**req, timeout=args.request_timeout)
                ch = resp.choices[0]
                text = ch.text
                has_native_think_prefix = args.cand == "cand2"
                cot, fa_text, fa_num, format_ok, lenient_num = parse_generation(text, has_native_think_prefix)
                gold = prompt_row["gsm8k_gold_numeric"]
                correct_strict = format_ok and fa_num is not None and fa_num == gold
                correct_lenient = correct_strict or (lenient_num is not None and lenient_num == gold)
                rec = {
                    "cand": args.cand,
                    "arm": args.arm,
                    "model_name": args.served_model,
                    "base_hf_id": args.base_hf_id,
                    "base_revision": args.base_revision,
                    "adapter_hf_id": args.adapter_hf_id,
                    "adapter_revision": args.adapter_revision,
                    "problem_idx": problem_idx,
                    "gsm8k_question": prompt_row["gsm8k_question"],
                    "gsm8k_gold_answer": prompt_row["gsm8k_gold_answer"],
                    "gsm8k_gold_numeric": gold,
                    "sample_idx": sample_idx,
                    "seed": seed,
                    "prompt_sha256": prompt_row["prompt_sha256"],
                    "raw_generation": text,
                    "cot": cot,
                    "final_answer_text": fa_text,
                    "final_answer_numeric": fa_num,
                    "correct_strict": correct_strict,
                    "correct_lenient": correct_lenient,
                    "correct": correct_lenient,
                    "format_ok": format_ok,
                    "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_tokens": args.max_tokens},
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else len(prompt_row["rendered_token_ids"]),
                    "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
                    "finish_reason": ch.finish_reason,
                    "latency_s": time.time() - t0,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "job_id": args.job_id,
                    "vllm_version": args.vllm_version,
                }
                async with lock:
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                    stats["done"] += 1
                    stats["tokens"] += rec["completion_tokens"] or 0
                    if stats["done"] % 25 == 0 or stats["done"] == stats["total"]:
                        el = time.time() - stats["t0"]
                        print(
                            f"  [{args.cand}/{args.arm}] {stats['done']}/{stats['total']} done, {el:.0f}s, "
                            f"{stats['tokens']} compl tok, {stats['tokens']/max(el,1):.0f} tok/s, "
                            f"{stats['done']/max(el,1)*60:.1f} traj/min",
                            flush=True,
                        )
                return
            except Exception as e:  # noqa: BLE001
                last_err = repr(e)[:500]
                print(f"  ! problem {problem_idx} sample {sample_idx} attempt {attempt} failed: {last_err}", flush=True)
                await asyncio.sleep(5 * attempt)
        async with lock:
            stats["failed"].append({"problem_idx": problem_idx, "sample_idx": sample_idx, "error": last_err})


async def main_async(args):
    prompts = []
    with open(args.prompts) as f:
        for line in f:
            prompts.append(json.loads(line))
    if args.shard_idx is not None:
        prompts = [p for p in prompts if p["problem_idx"] % args.nshards == args.shard_idx]
    if args.limit is not None:
        prompts = prompts[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    todo = []
    for p in prompts:
        for s in range(args.n_samples):
            if (p["problem_idx"], s, args.arm) not in done:
                todo.append((p, s))
    print(
        f"[{args.cand}/{args.arm}] shard={args.shard_idx}/{args.nshards} {len(prompts)} problems x {args.n_samples} "
        f"samples, {len(done)} rows already done, {len(todo)} to generate; concurrency={args.concurrency}",
        flush=True,
    )
    if not todo:
        return
    client = AsyncOpenAI(base_url=args.base_url, api_key="none", max_retries=0)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    stats = {"done": 0, "total": len(todo), "tokens": 0, "t0": time.time(), "failed": []}
    with open(out, "a") as fout:
        await asyncio.gather(*[one(client, sem, p, s, args, fout, lock, stats) for p, s in todo])
    el = time.time() - stats["t0"]
    timing = {
        "cand": args.cand, "arm": args.arm, "shard_idx": args.shard_idx, "n": stats["done"], "wall_s": el,
        "completion_tokens": stats["tokens"], "tok_per_s": stats["tokens"] / max(el, 1e-9),
        "trajectories_per_min": stats["done"] / max(el, 1e-9) * 60, "concurrency": args.concurrency,
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
    ap.add_argument("--cand", required=True, choices=["cand2", "cand3"])
    ap.add_argument("--arm", required=True, choices=["pre", "post"])
    ap.add_argument("--base-hf-id", default="")
    ap.add_argument("--base-revision", default="")
    ap.add_argument("--adapter-hf-id", default="")
    ap.add_argument("--adapter-revision", default="")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--shard-idx", type=int, default=None)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="pilot: first N problems (of the shard)")
    ap.add_argument("--request-timeout", type=float, default=3600.0)
    ap.add_argument("--job-id", default="local")
    ap.add_argument("--vllm-version", default="")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
