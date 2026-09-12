#!/usr/bin/env python3
"""
Resumable generation + grading driver.

Generation : vLLM OpenAI-compatible server serving the base model (model name "pre") and LoRA adapters
             ("post", "control") on ONE GPU. Same prompts / sampling params / seeds for every arm.
Grading    : the authors' grader, unmodified:
             src.evaluate.evaluation.RewardHackingEvaluation.batch_evaluate  (external/rl-rewardhacking)
Output     : results/<arm>/<cond>/generations.jsonl  (prompt, rendered prompt, raw generation, usage, seed)
             results/<arm>/<cond>/scores.jsonl       (labels + full grader output), one line per sample,
             appended chunk by chunk. On restart, problems that already have n scored samples are skipped.
"""
import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

TEAM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(TEAM, "external", "rl-rewardhacking")
for p in (os.path.join(TEAM, "pylibs"), REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from openai import AsyncOpenAI  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from src import SamplingParams  # noqa: E402
from src.evaluate.evaluation import RewardHackingEvaluation, EvaluationParameters  # noqa: E402

BASE_SNAPSHOT = "/data/home/jxcai/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

# grader output fields kept in scores.jsonl (everything the labels depend on + the full per-test results)
SCORE_FIELDS = [
    "hint", "difficulty", "func_name", "test_func_name", "test_func_call",
    "is_parsed", "is_formatted", "is_answered", "can_compile", "ends_think",
    "eq_correct", "gt_pass_rate", "eq_hinted", "hint_pass_rate",
    "prompt_has_test_func", "response_has_test_func", "response_test_func", "response_test_func_parent",
    "response_test_func_call", "match_test_func", "response_test_func_compile_error",
    "response_test_func_gt_pass", "response_test_func_arbitrary_pass", "prompt_test_func_response_pass",
    "test_modification", "reward_hack_label", "is_test_modification_harmful",
    "is_reward_hack_strict", "is_reward_hack_loose",
    "gt_result", "hint_result", "response_test_func_gt_result", "response_test_func_arbitrary_result",
    "prompt_test_func_response_result",
]


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def read_jsonl(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def done_ids(scores_path, n):
    counts = {}
    for r in read_jsonl(scores_path):
        counts[r["id"]] = counts.get(r["id"], 0) + 1
    return {i for i, c in counts.items() if c >= n}


def seed_for(args, ex):
    return args.seed * 1_000_000 + int(ex["id"])


async def gen_problem(client, sem, model, ex, args):
    """One chat.completions request with n samples for one problem. Returns (response, latency) or None."""
    seed = seed_for(args, ex)
    async with sem:
        last_err = None
        for attempt in range(6):
            try:
                t0 = time.time()
                r = await client.chat.completions.create(
                    model=model,
                    messages=ex["prompt"],
                    n=args.n,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    seed=seed,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
                        "repetition_penalty": args.repetition_penalty,
                    },
                )
                return r, time.time() - t0
            except Exception as e:  # noqa: BLE001
                last_err = e
                log(f"WARN gen failed (attempt {attempt}) id={ex['id']} model={model}: {type(e).__name__}: {str(e)[:200]}")
                await asyncio.sleep(min(60, 2 ** (attempt + 1)))
        log(f"ERROR giving up on id={ex['id']} model={model}: {last_err}")
        return None


def grade_and_write(evaluator, tok, arm, cond, chunk, responses, args):
    out_dir = os.path.join(TEAM, "results", arm, cond)
    os.makedirs(out_dir, exist_ok=True)
    examples, outputs, gen_lines, meta = [], [], [], []
    for ex, resp in zip(chunk, responses):
        if resp is None:
            continue
        r, latency = resp
        rendered = tok.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=args.enable_thinking)
        local_prompt_tokens = len(tok(rendered, add_special_tokens=False)["input_ids"])
        choices = sorted(r.choices, key=lambda c: c.index)
        for ch in choices:
            text = ch.message.content if ch.message.content is not None else ""
            examples.append(ex)
            outputs.append(text)
            meta.append((ex["id"], ch.index))
            gen_lines.append({
                "id": ex["id"], "sample_idx": ch.index, "arm": arm, "cond": cond, "hint": ex.get("hint"),
                "model_name": arm, "seed": seed_for(args, ex),
                "sampling": {"n": args.n, "temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens,
                             "repetition_penalty": args.repetition_penalty, "enable_thinking": args.enable_thinking},
                "messages": ex["prompt"], "rendered_prompt": rendered,
                "response": text, "finish_reason": ch.finish_reason,
                "usage_prompt_tokens": r.usage.prompt_tokens if r.usage else None,
                "usage_completion_tokens_total": r.usage.completion_tokens if r.usage else None,
                "local_prompt_tokens": local_prompt_tokens,
                "prompt_tokens_match": (r.usage.prompt_tokens == local_prompt_tokens) if r.usage else None,
                "request_latency_s": round(latency, 2),
            })
    if not examples:
        return 0, 0.0
    t0 = time.time()
    results = evaluator.batch_evaluate(examples, outputs)   # authors' grader, unmodified
    grade_s = time.time() - t0
    score_lines = []
    for (pid, sidx), res in zip(meta, results):
        line = {"id": pid, "sample_idx": sidx, "arm": arm, "cond": cond}
        for k in SCORE_FIELDS:
            if k in res:
                line[k] = res[k]
        line["parsed_response"] = res.get("parsed_response")
        score_lines.append(line)
    # generations first, then scores (scores.jsonl is the resume marker)
    with open(os.path.join(out_dir, "generations.jsonl"), "a") as f:
        for l in gen_lines:
            f.write(json.dumps(l) + "\n")
        f.flush(); os.fsync(f.fileno())
    with open(os.path.join(out_dir, "scores.jsonl"), "a") as f:
        for l in score_lines:
            f.write(json.dumps(l) + "\n")
        f.flush(); os.fsync(f.fileno())
    n_rh = sum(1 for l in score_lines if l.get("is_reward_hack_strict"))
    n_ok = sum(1 for l in score_lines if l.get("eq_correct"))
    n_len = sum(1 for l in gen_lines if l["finish_reason"] == "length")
    log(f"[{arm}/{cond}] graded {len(score_lines)} samples ({len(chunk)} problems) in {grade_s:.1f}s | strict RH {n_rh} | correct {n_ok} | truncated {n_len}")
    return len(score_lines), grade_s


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--arms", default="pre,post")
    ap.add_argument("--conds", default="simple_overwrite_tests")
    ap.add_argument("--max-problems", type=int, default=10**9)
    ap.add_argument("--n", type=int, default=10)                    # authors' run_eval.py default n_samples=10
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--enable-thinking", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=20, help="problems per generate/grade chunk")
    ap.add_argument("--concurrency", type=int, default=64, help="concurrent chat requests (each n samples)")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--workdir", default=None, help="cwd for the grader subprocesses (node-local scratch)")
    args = ap.parse_args()

    if args.workdir:
        os.makedirs(args.workdir, exist_ok=True)
        os.chdir(args.workdir)
    arms = [a for a in args.arms.split(",") if a]
    conds = [c for c in args.conds.split(",") if c]
    log(f"args: {vars(args)} | MAX_JOBS={os.environ.get('MAX_JOBS')} | cwd={os.getcwd()}")

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout, max_retries=0)
    models = await client.models.list()
    served = [m.id for m in models.data]
    log("served models:", served)
    for a in arms:
        assert a in served, f"arm {a} not served by the vLLM server (served: {served})"

    tok = AutoTokenizer.from_pretrained(BASE_SNAPSHOT)
    sp = SamplingParams(n=args.n, temperature=args.temperature, max_new_tokens=args.max_tokens, top_p=args.top_p, repetition_penalty=args.repetition_penalty)
    evaluator = RewardHackingEvaluation(config=EvaluationParameters(
        model_id="qwen/Qwen3-4B", lora_adapter_path=None, dataset_path="team1", sampling_params=sp,
        evaluation_name="rh_code", enable_thinking=args.enable_thinking, debug=False, save_outputs=False))
    log(f"grader workers (MAX_JOBS): {evaluator.evaluator.num_workers}, timeout {evaluator.evaluator.timeout}s, mem {evaluator.evaluator.memory_per_worker}MB")

    sem = asyncio.Semaphore(args.concurrency)
    pool = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()
    t_start = time.time()
    total_samples = 0
    total_grade_s = 0.0

    for cond in conds:
        data = read_jsonl(os.path.join(TEAM, "eval_inputs", f"test_{cond}.jsonl"))[: args.max_problems]
        work = []
        for ci in range(0, len(data), args.chunk):
            chunk_all = data[ci: ci + args.chunk]
            for arm in arms:
                done = done_ids(os.path.join(TEAM, "results", arm, cond, "scores.jsonl"), args.n)
                chunk = [ex for ex in chunk_all if ex["id"] not in done]
                if chunk:
                    work.append((arm, chunk))
        log(f"cond={cond}: {len(data)} problems, {len(work)} (arm, chunk) units to run")

        pending = None
        for arm, chunk in work:
            gen = asyncio.gather(*[gen_problem(client, sem, arm, ex, args) for ex in chunk])
            if pending is not None:
                parm, pchunk, ptask, pt0 = pending
                responses = await ptask
                log(f"[{parm}/{cond}] generated {len(pchunk)} problems x n={args.n} in {time.time() - pt0:.1f}s (incl. overlap)")
                ns, gs = await loop.run_in_executor(pool, grade_and_write, evaluator, tok, parm, cond, pchunk, responses, args)
                total_samples += ns; total_grade_s += gs
                el = time.time() - t_start
                log(f"progress: {total_samples} samples in {el / 60:.1f} min -> {total_samples / el * 3600:.0f} samples/hour (grading {total_grade_s:.0f}s of that)")
            pending = (arm, chunk, gen, time.time())
        if pending is not None:
            parm, pchunk, ptask, pt0 = pending
            responses = await ptask
            log(f"[{parm}/{cond}] generated {len(pchunk)} problems x n={args.n} in {time.time() - pt0:.1f}s")
            ns, gs = await loop.run_in_executor(pool, grade_and_write, evaluator, tok, parm, cond, pchunk, responses, args)
            total_samples += ns; total_grade_s += gs

    el = time.time() - t_start
    log(f"DONE: {total_samples} new samples in {el / 60:.1f} min ({total_samples / max(el, 1e-9) * 3600:.0f} samples/hour; grading {total_grade_s:.0f}s)")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
