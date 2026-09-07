#!/usr/bin/env python3
"""GPT-4.1 re-judge of Judge 1 ONLY (Cloud/Betley `judgment_alignment_v5`) via the OpenAI Batch API.

Motivation (2026-09-05): inspection showed the local Qwen3.5-35B-A3B Judge 1 rubber-stamps
coherent-but-wrong GSM8K answers as 100 (e.g. cand3 problem 13, model answers 32.5 vs gold 35, Qwen
scored 100 asserting "the math is correct"). We re-score Judge 1 with GPT-4.1. Judge 3 and the Claude
calibration are NOT touched by this script.

Reuses judge.py's EXACT Judge-1 prompt template (`JUDGE1_TEMPLATE`), its `.format(prompt=...,
completion=...)` construction (via load_prompt_texts, byte-identical to the local judge), and its
`parse_score()` parser, so GPT-4.1 scores are directly comparable to the local Qwen scores.

Design: rows are chunked into batch input files (default 6000 requests/file, to stay under the Batch
API's 50k-request / 200MB-file limits). A single `run` loop keeps up to --max-active batches in
flight at once (to respect the account's enqueued-token limit), polls them, collects finished ones
into results/<cand>/judge_gpt41/j1.jsonl, and submits the next pending chunk as capacity frees. Fully
resumable: chunk state is persisted to results/<cand>/judge_gpt41/state.json and output rows are
skipped by custom_id.

Usage:
  # submit + poll + collect in one long-running (resumable) process, both candidates:
  python code/openai_judge.py run --cand cand3 --rows all
  python code/openai_judge.py run --cand cand2 --rows all
  # just collect already-submitted batches (e.g. after a restart):
  python code/openai_judge.py run --cand cand2 --rows all   # same command resumes
  # tiny end-to-end probe that the account can use the Batch API for gpt-4.1:
  python code/openai_judge.py run --cand cand3 --rows all --limit 10 --tag probe10
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge as J  # noqa: E402  (reuse JUDGE1_TEMPLATE / parse_score / load_prompt_texts)

TEAM = Path(__file__).resolve().parents[1]
MODEL = "gpt-4.1"
# GPT-4.1 list price $2/$8 per 1M in/out; Batch API = 50% -> $1/$4 per 1M.
BATCH_PRICE_IN = 1.0 / 1e6
BATCH_PRICE_OUT = 4.0 / 1e6


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def custom_id_for(cand, row):
    # Batch custom_id: keep it short + unique per (cand, arm, problem, sample). Judge fixed to j1.
    return f"{cand}-j1-{row['arm']}-{row['problem_idx']}-{row['sample_idx']}"


def select_rows(cand, rows_arg, limit):
    traj = J.load_trajectories(cand)
    if rows_arg == "all":
        rows = traj
    elif rows_arg == "correct_strict":
        rows = [r for r in traj if r.get("correct_strict")]
    elif rows_arg == "pilot":
        pilot = {(p, s, a) for p in range(100) for s in range(3) for a in ("pre", "post")}
        rows = [r for r in traj if (r["problem_idx"], r["sample_idx"], r["arm"]) in pilot]
    else:
        rows = load_jsonl(Path(rows_arg))
    # stable order so chunking is deterministic across restarts
    rows.sort(key=lambda r: (r["arm"], r["problem_idx"], r["sample_idx"]))
    if limit is not None:
        rows = rows[:limit]
    return rows


def build_request(cand, row, prompt_texts, max_tokens):
    prompt_text = prompt_texts.get(row["problem_idx"], row.get("gsm8k_question", ""))
    filled = J.JUDGE1_TEMPLATE.format(prompt=prompt_text, completion=row["raw_generation"])
    return {
        "custom_id": custom_id_for(cand, row),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [{"role": "user", "content": filled}],
            "temperature": 0,
            "max_tokens": max_tokens,
        },
    }


def out_dir(cand):
    d = TEAM / "results" / cand / "judge_gpt41"
    d.mkdir(parents=True, exist_ok=True)
    return d


def done_custom_ids(out_path):
    done = set()
    if out_path.exists():
        for r in load_jsonl(out_path):
            done.add(r["custom_id"])
    return done


def cmd_run(args):
    client = OpenAI()
    od = out_dir(args.cand)
    out_path = od / "j1.jsonl"
    state_path = od / "state.json"
    chunks_dir = od / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    state = json.loads(state_path.read_text()) if state_path.exists() else {"chunks": {}}

    # Build chunk files ONCE (first run only). On restart, state.json is the source of truth for
    # chunk membership -- we must NOT re-derive chunks from a freshly-filtered row list, because once
    # some rows are collected the filtered list shrinks and positional chunk indices would shift while
    # the on-disk chunk files stay fixed. So: only build when state has no chunks yet.
    if not state["chunks"]:
        prompt_texts = J.load_prompt_texts(args.cand)
        rows = select_rows(args.cand, args.rows, args.limit)
        already = done_custom_ids(out_path)
        rows = [r for r in rows if custom_id_for(args.cand, r) not in already]
        print(f"[{args.cand}] building chunks: {len(rows)} rows to judge with {MODEL} "
              f"({len(already)} already done); chunk_size={args.chunk_size}, max_active={args.max_active}",
              flush=True)
        if not rows:
            print(f"[{args.cand}] nothing to do.", flush=True)
            return
        n_chunks = (len(rows) + args.chunk_size - 1) // args.chunk_size
        for ci in range(n_chunks):
            chunk_file = chunks_dir / f"chunk{ci}.jsonl"
            part = rows[ci * args.chunk_size:(ci + 1) * args.chunk_size]
            with open(chunk_file, "w") as f:
                for r in part:
                    f.write(json.dumps(build_request(args.cand, r, prompt_texts, args.max_tokens)) + "\n")
            state["chunks"][str(ci)] = {"status": "pending", "file": str(chunk_file)}
        state_path.write_text(json.dumps(state, indent=2))
        est_in = len(rows) * (5506 if args.cand == "cand2" else 1253)
        est_out = len(rows) * 500
        print(f"[{args.cand}] rough Batch cost estimate ~${est_in*BATCH_PRICE_IN + est_out*BATCH_PRICE_OUT:.0f} "
              f"(in~{est_in/1e6:.1f}M, out~{est_out/1e6:.1f}M tok)", flush=True)
    else:
        print(f"[{args.cand}] resuming from state.json with {len(state['chunks'])} existing chunks; "
              f"statuses={[c.get('status') for c in state['chunks'].values()]}", flush=True)

    def active_batches():
        return [c for c in state["chunks"].values() if c.get("status") == "submitted"]

    def collect(chunk):
        b = client.batches.retrieve(chunk["batch_id"])
        chunk["last_status"] = b.status
        if b.status not in ("completed", "failed", "expired", "cancelled"):
            return False
        if b.status == "completed" and b.output_file_id:
            content = client.files.content(b.output_file_id).text
            already = done_custom_ids(out_path)
            n = 0
            with open(out_path, "a") as fout:
                for line in content.splitlines():
                    if not line.strip():
                        continue
                    res = json.loads(line)
                    cid = res.get("custom_id")
                    if not cid or cid in already:
                        continue
                    # arm/problem/sample are encoded in the custom_id: <cand>-j1-<arm>-<pidx>-<sidx>
                    parts = cid.split("-")
                    arm, pidx, sidx = parts[2], int(parts[3]), int(parts[4])
                    rec = {"cand": args.cand, "arm": arm, "problem_idx": pidx,
                           "sample_idx": sidx, "judge": "j1", "judge_model": MODEL,
                           "custom_id": cid, "batch_id": chunk["batch_id"]}
                    body = (res.get("response") or {}).get("body") or {}
                    err = res.get("error")
                    if err or not body:
                        rec.update({"raw": None, "gpt41_score": None,
                                    "parse_method": "batch_error", "error": err})
                    else:
                        txt = body["choices"][0]["message"]["content"]
                        score, method = J.parse_score(txt or "")
                        usage = body.get("usage") or {}
                        rec.update({"raw": txt, "gpt41_score": score, "parse_method": method,
                                    "finish_reason": body["choices"][0].get("finish_reason"),
                                    "input_tokens": usage.get("prompt_tokens"),
                                    "output_tokens": usage.get("completion_tokens")})
                    fout.write(json.dumps(rec) + "\n")
                    n += 1
            chunk["status"] = "collected"
            chunk["n_written"] = n
            print(f"[{args.cand}] chunk batch {chunk['batch_id']} completed, wrote {n} rows", flush=True)
        else:
            # failed/expired/cancelled -> reset to pending for resubmission
            print(f"[{args.cand}] chunk batch {chunk['batch_id']} status={b.status}; resetting to pending",
                  flush=True)
            chunk["status"] = "pending"
            chunk.pop("batch_id", None)
        return True

    def submit(cid_key, chunk):
        f = client.files.create(file=open(chunk["file"], "rb"), purpose="batch")
        b = client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions",
                                  completion_window="24h",
                                  metadata={"cand": args.cand, "judge": "j1", "chunk": cid_key})
        chunk.update({"status": "submitted", "input_file_id": f.id, "batch_id": b.id,
                      "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        print(f"[{args.cand}] submitted chunk {cid_key} -> batch {b.id}", flush=True)

    t0 = time.time()
    while True:
        # 1) collect anything finished
        for c in list(state["chunks"].values()):
            if c.get("status") == "submitted":
                try:
                    collect(c)
                except Exception as e:  # noqa: BLE001
                    print(f"[{args.cand}] collect error: {e!r}", flush=True)
        # 2) submit pending up to max_active
        for cid_key, c in state["chunks"].items():
            if len(active_batches()) >= args.max_active:
                break
            if c.get("status") == "pending":
                try:
                    submit(cid_key, c)
                except Exception as e:  # noqa: BLE001
                    print(f"[{args.cand}] submit deferred (chunk {cid_key}): {repr(e)[:200]}", flush=True)
        state_path.write_text(json.dumps(state, indent=2))
        statuses = [c.get("status") for c in state["chunks"].values()]
        if all(s == "collected" for s in statuses):
            print(f"[{args.cand}] ALL {len(statuses)} chunks collected. Output: {out_path}", flush=True)
            break
        if time.time() - t0 > args.timeout:
            print(f"[{args.cand}] timeout; statuses={statuses}. Re-run same command to resume.", flush=True)
            break
        time.sleep(args.poll_interval)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cand", required=True, choices=["cand2", "cand3"])
    r.add_argument("--rows", default="all", help="all | correct_strict | pilot | <path>")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--tag", default="j1")
    r.add_argument("--chunk-size", type=int, default=6000)
    r.add_argument("--max-active", type=int, default=4)
    r.add_argument("--max-tokens", type=int, default=1500)
    r.add_argument("--poll-interval", type=float, default=120.0)
    r.add_argument("--timeout", type=float, default=90000.0)
    r.set_defaults(func=cmd_run)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
