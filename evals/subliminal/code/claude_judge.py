#!/usr/bin/env python3
"""Claude Opus 5 batch-API calibration judge: run the SAME two judges as code/judge.py
(Judge 1 = Cloud/Betley judgment_alignment_v5, Judge 3 = our reward-hacking/misalignment-concern
judge) against Claude, on a fixed set of trajectory rows (default: the 600-row pilot), so we can
measure agreement with the local Qwen3.5-35B-A3B judge before deciding whether to run the full
44,838-row/candidate set through Claude.

Reuses judge.py's EXACT prompt templates and `.format(prompt=..., completion=...)` construction, and
its `parse_score()` fallback chain, so scoring is directly comparable to the local judge. Reads
ANTHROPIC_API_KEY from the environment (`source ~/.sigil_env`); never logs or writes the key itself.

Pilot definition: the 600 (problem_idx, sample_idx, arm) keys found in results/cand3/judged.jsonl
(100 problems x 3 samples x 2 arms — the frozen pilot panel used for BOTH candidates; verified
byte-identical key sets across results/cand2/trajectories.jsonl and results/cand3/judged.jsonl at
authoring time). `--rows pilot` (default) filters --cand's trajectories.jsonl down to those keys
(robust even though results/cand3/trajectories.jsonl has since grown past 600 rows as the full
generation run appends to it). `--rows <path>` instead loads every row in an explicit
trajectories-format JSONL file.

One Message Batch is submitted per (cand, judge) — i.e. this script submits len(--judges) batches
per invocation. Batch ids/metadata are persisted to <out_dir>/batches.json immediately after
creation so a later invocation can resume polling/collection without resubmitting. Output rows are
skipped if their custom_id is already present in --out (resumable).

Usage:
  python code/claude_judge.py --cand cand3 --rows pilot --judges j1,j3 \
      --out results/cand3/judge_claude/pilot_calibration.jsonl --tag pilot_calibration
  python code/claude_judge.py --cand cand2 --rows pilot --judges j1,j3 \
      --out results/cand2/judge_claude/pilot_calibration.jsonl --tag pilot_calibration
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge as localjudge  # noqa: E402  (reuse JUDGE1_TEMPLATE/JUDGE3_TEMPLATE/parse_score/load_prompt_texts)

TEAM = Path(__file__).resolve().parents[1]

JUDGE_TEMPLATES = {"j1": localjudge.JUDGE1_TEMPLATE, "j3": localjudge.JUDGE3_TEMPLATE}
LOCAL_SHA_FIELD = {"j1": "judge_prompt_sha256", "j3": "judge3_prompt_sha256"}
LOCAL_RAW_FIELD = {"j1": "judge_raw", "j3": "judge3_raw"}

# Standard (non-batch) list prices, $/1M tokens. Batch API = 50% of these.
STANDARD_PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def local_judged_pilot_path():
    """results/cand3/judged.jsonl gets REBUILT from scratch (overwritten) once the full-run judge
    shards finish and code/finalize.sbatch runs merge_judged.py -- since that full run also covers
    problem_idx 0-99 (the pilot), it would silently replace the exact scores this calibration compares
    against. We snapshot the pilot-only judged.jsonl once to results/cand3/judge_claude/
    local_judged_pilot_snapshot.jsonl and prefer that snapshot everywhere after it exists."""
    snap = TEAM / "results" / "cand3" / "judge_claude" / "local_judged_pilot_snapshot.jsonl"
    live = TEAM / "results" / "cand3" / "judged.jsonl"
    if not snap.exists():
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_text(live.read_text())
        print(f"[snapshot] wrote {snap} from {live} (protects the pilot local-judge comparison "
              f"from being clobbered by the full-run's merge_judged.py rebuild)", flush=True)
    return snap


def load_pilot_keys():
    """The canonical 600 (problem_idx, sample_idx, arm) pilot keys and the cand3 judged rows they
    came from (used both to define the pilot set and, for cand3, to sha256-verify prompt text and
    estimate output-token length from the local judge's raw completions). Reads the frozen snapshot,
    not the live (mutable) results/cand3/judged.jsonl -- see local_judged_pilot_path()."""
    j3 = load_jsonl(local_judged_pilot_path())
    keys = set((r["problem_idx"], r["sample_idx"], r["arm"]) for r in j3)
    return keys, j3


def load_target_rows(cand, rows_arg):
    if rows_arg == "pilot":
        pilot_keys, _ = load_pilot_keys()
        traj = load_jsonl(TEAM / "results" / cand / "trajectories.jsonl")
        seen = {}
        for r in traj:
            k = (r["problem_idx"], r["sample_idx"], r["arm"])
            if k in pilot_keys and k not in seen:
                seen[k] = r
        missing = pilot_keys - set(seen.keys())
        if missing:
            print(f"WARNING: {len(missing)} pilot keys not found in {cand}/trajectories.jsonl "
                  f"(first 5: {sorted(missing)[:5]})", flush=True)
        return list(seen.values())
    return load_jsonl(Path(rows_arg))


def sha256_of(text):
    return hashlib.sha256(text.encode()).hexdigest()


def build_filled_prompt(judge_key, prompt_text, completion_text):
    return JUDGE_TEMPLATES[judge_key].format(prompt=prompt_text, completion=completion_text)


def custom_id_for(cand, judge_key, row):
    # Batch API custom_id must match ^[a-zA-Z0-9_-]{1,64}$ -- no "|" allowed.
    return f"{cand}-{judge_key}-{row['arm']}-{row['problem_idx']}-{row['sample_idx']}"


def estimate_cost(client, model, requests, max_tokens, local_index, judge_key, rows_by_cid, sample_n=20):
    """Returns (avg_input_tokens, est_cost_worst_usd, est_cost_typical_usd, typical_out_tokens).
    Input tokens are measured for real via count_tokens on up to `sample_n` requests. Output tokens
    are unknown pre-submission: `worst` assumes every request hits --max-tokens (a safe upper bound
    for the abort gate); `typical` instead uses the LOCAL judge's own judge_raw/judge3_raw word count
    (x1.3 tokens/word) as a proxy for expected completion length where available (cand3 only, since
    that is the only candidate with existing local judge output at authoring time) — informational,
    not the abort criterion.
    """
    sample = requests[:sample_n]
    input_tokens = []
    for cid, filled in sample:
        ct = client.messages.count_tokens(model=model, messages=[{"role": "user", "content": filled}])
        input_tokens.append(ct.input_tokens)
    avg_input = sum(input_tokens) / len(input_tokens)

    price_in, price_out = STANDARD_PRICES[model]
    batch_price_in = price_in / 2 / 1e6
    batch_price_out = price_out / 2 / 1e6
    n = len(requests)
    est_worst = n * (avg_input * batch_price_in + max_tokens * batch_price_out)

    typical_out = None
    if local_index:
        lens = []
        for cid, _filled in sample:
            row = rows_by_cid[cid]
            k = (row["problem_idx"], row["sample_idx"], row["arm"])
            lr = local_index.get(k)
            if lr and lr.get(LOCAL_RAW_FIELD[judge_key]):
                lens.append(len(lr[LOCAL_RAW_FIELD[judge_key]].split()))
        if lens:
            typical_out = (sum(lens) / len(lens)) * 1.3
    if typical_out is None:
        typical_out = max_tokens * 0.4  # rough fallback guess when no local reference exists
    est_typical = n * (avg_input * batch_price_in + typical_out * batch_price_out)
    return avg_input, est_worst, est_typical, typical_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, choices=["cand2", "cand3"])
    ap.add_argument("--rows", default="pilot", help="'pilot' (default) or a path to a trajectories-format JSONL")
    ap.add_argument("--judges", default="j1,j3", help="comma list from {j1,j3}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--poll-interval", type=float, default=120.0)
    ap.add_argument("--poll-timeout", type=float, default=7200.0)
    ap.add_argument("--cost-cap-usd", type=float, default=60.0)
    ap.add_argument("--skip-sha-check", action="store_true", help="skip the cand3 byte-identical-prompt assertion")
    ap.add_argument("--batches-file", default=None, help="default: <out_dir>/batches.json")
    ap.add_argument("--no-submit", action="store_true", help="print cost estimate only, do not submit")
    args = ap.parse_args()

    judges = args.judges.split(",")
    for j in judges:
        assert j in JUDGE_TEMPLATES, f"unknown judge {j!r}, must be one of {list(JUDGE_TEMPLATES)}"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    batches_path = Path(args.batches_file) if args.batches_file else out_path.parent / "batches.json"

    rows = load_target_rows(args.cand, args.rows)
    print(f"[{args.cand}] {len(rows)} target rows for tag={args.tag}, judges={judges}", flush=True)

    prompt_texts = localjudge.load_prompt_texts(args.cand)

    local_index = {}
    if args.cand == "cand3":
        _, j3rows = load_pilot_keys()
        local_index = {(r["problem_idx"], r["sample_idx"], r["arm"]): r for r in j3rows}
    elif not args.skip_sha_check:
        print(f"[{args.cand}] no local judged.jsonl exists yet for cand2 -> sha256 verification is a no-op "
              f"for this candidate (only cand3 has local judge output to compare against).", flush=True)

    # DATA-INTEGRITY GUARD (discovered 2026-09-04 while authoring this script): merge_and_stats.py
    # REBUILDS results/cand3/trajectories.jsonl from scratch out of results/cand3/gen/shard*_*.jsonl
    # (`open(out_path, "w")`, keyed by (problem_idx, sample_idx, arm), last-write-wins) every time it
    # runs. The full cand3 generation run's shards cover ALL 7,473 problems, including the 100 pilot
    # problems (0-99) — so once the full run's shards were merged, ~11% of the pilot's 600 rows in
    # trajectories.jsonl got SILENTLY OVERWRITTEN with a fresh (different) temperature=1 regeneration
    # for the same key, even though the ORIGINAL pilot completion is what results/cand3/judged.jsonl
    # actually scored. Byte-for-byte comparison against the local judge is only valid on rows whose
    # completion text did not drift. We detect and exclude drifted rows here (rather than crash the
    # whole run on the first mismatch) so the calibration set stays a true apples-to-apples comparison;
    # see notes/PAPER_NOTES.md / DEVIATIONS.md for the writeup and results/cand3/judge_claude/
    # sha_mismatch_excluded.json for the excluded keys.
    sha_mismatch_keys = set()
    if args.cand == "cand3" and not args.skip_sha_check and local_index:
        for row in rows:
            k = (row["problem_idx"], row["sample_idx"], row["arm"])
            local_row = local_index.get(k)
            if local_row is None:
                continue
            prompt_text = prompt_texts.get(row["problem_idx"], row.get("gsm8k_question", ""))
            completion_text = row["raw_generation"]
            for jk in ("j1", "j3"):
                filled = build_filled_prompt(jk, prompt_text, completion_text)
                if sha256_of(filled) != local_row.get(LOCAL_SHA_FIELD[jk]):
                    sha_mismatch_keys.add(k)
                    break
        if sha_mismatch_keys:
            print(f"[{args.cand}] DATA-INTEGRITY WARNING: {len(sha_mismatch_keys)}/{len(rows)} pilot rows' "
                  f"raw_generation in trajectories.jsonl no longer matches what results/cand3/judged.jsonl "
                  f"scored (overwritten by the full-run merge_and_stats.py rebuild) -- EXCLUDING them from "
                  f"this calibration run so every submitted row is byte-identical to what the local judge "
                  f"saw.", flush=True)
            mismatch_path = out_path.parent / "sha_mismatch_excluded.json"
            mismatch_path.write_text(json.dumps(sorted([list(k) for k in sha_mismatch_keys]), indent=2))
            rows = [r for r in rows if (r["problem_idx"], r["sample_idx"], r["arm"]) not in sha_mismatch_keys]
            print(f"[{args.cand}] {len(rows)} rows remain after exclusion; excluded-key list -> {mismatch_path}",
                  flush=True)

    client = anthropic.Anthropic()

    batches_meta = {}
    if batches_path.exists():
        batches_meta = json.loads(batches_path.read_text())

    done_custom_ids = set()
    if out_path.exists():
        for r in load_jsonl(out_path):
            done_custom_ids.add(r["custom_id"])
        print(f"[{args.cand}] {len(done_custom_ids)} custom_ids already in {out_path}, will skip on write", flush=True)

    # PHASE 1 -- submit (or find already-submitted) batches for EVERY requested judge FIRST, so they
    # run concurrently server-side, before polling any of them (coordinator instruction 2026-09-04:
    # don't serialize judge-3 behind judge-1's full poll+collect cycle).
    active = {}  # judge_key -> {"batch_id", "rows_by_cid"}
    for judge_key in judges:
        entry_key = f"{args.tag}|{judge_key}"
        rows_by_cid = {custom_id_for(args.cand, judge_key, row): row for row in rows}

        requests = []
        sha_checked = 0
        for row in rows:
            cid = custom_id_for(args.cand, judge_key, row)
            if cid in done_custom_ids:
                continue
            prompt_text = prompt_texts.get(row["problem_idx"], row.get("gsm8k_question", ""))
            completion_text = row["raw_generation"]
            filled = build_filled_prompt(judge_key, prompt_text, completion_text)
            if args.cand == "cand3" and not args.skip_sha_check:
                # rows with a drifted completion (see DATA-INTEGRITY GUARD above) were already
                # excluded from `rows`; this is now a confirmatory re-check, not a filter.
                k = (row["problem_idx"], row["sample_idx"], row["arm"])
                local_row = local_index.get(k)
                if local_row is not None:
                    expected = local_row.get(LOCAL_SHA_FIELD[judge_key])
                    actual = sha256_of(filled)
                    assert expected == actual, (
                        f"sha256 mismatch cand3 {judge_key} key={k}: local judged.jsonl "
                        f"{LOCAL_SHA_FIELD[judge_key]}={expected!r} != claude prompt sha256={actual!r} "
                        f"-- prompt text sent to Claude is NOT byte-identical to what the local judge saw "
                        f"(unexpected -- this key should have been caught by the pre-filter above)"
                    )
                    sha_checked += 1
            requests.append((cid, filled))

        if sha_checked:
            print(f"[{args.cand}/{judge_key}] sha256-verified {sha_checked} prompts byte-identical to local judge", flush=True)

        if not requests and entry_key not in batches_meta:
            print(f"[{args.cand}/{judge_key}] nothing to do (all {len(rows_by_cid)} rows already in {out_path})", flush=True)
            continue

        if entry_key in batches_meta:
            batch_id = batches_meta[entry_key]["batch_id"]
            print(f"[{args.cand}/{judge_key}] resuming existing batch {batch_id} "
                  f"({batches_meta[entry_key].get('n_requests')} requests submitted)", flush=True)
        else:
            avg_input, est_worst, est_typical, typical_out = estimate_cost(
                client, args.model, requests, args.max_tokens, local_index, judge_key, rows_by_cid
            )
            n = len(requests)
            print(f"[{args.cand}/{judge_key}] COST ESTIMATE: n={n} requests, avg_input~{avg_input:.0f} tok "
                  f"(measured via count_tokens on {min(20,n)} samples); "
                  f"worst-case (max_tokens={args.max_tokens} all output)=${est_worst:.2f}; "
                  f"typical (~{typical_out:.0f} out tok/row, proxy from local judge length)=${est_typical:.2f}",
                  flush=True)
            if est_worst > args.cost_cap_usd:
                print(f"ABORT: worst-case estimate ${est_worst:.2f} exceeds --cost-cap-usd ${args.cost_cap_usd:.2f} "
                      f"for [{args.cand}/{judge_key}]. Not submitting. Re-run with a higher --cost-cap-usd if this "
                      f"is intentional.", flush=True)
                sys.exit(1)
            if args.no_submit:
                print(f"[{args.cand}/{judge_key}] --no-submit set, skipping batch creation", flush=True)
                continue

            batch_requests = [
                Request(
                    custom_id=cid,
                    params=MessageCreateParamsNonStreaming(
                        model=args.model,
                        max_tokens=args.max_tokens,
                        messages=[{"role": "user", "content": filled}],
                        output_config={"effort": args.effort},
                    ),
                )
                for cid, filled in requests
            ]
            batch = client.messages.batches.create(requests=batch_requests)
            batches_meta[entry_key] = {
                "batch_id": batch.id,
                "cand": args.cand,
                "judge": judge_key,
                "tag": args.tag,
                "n_requests": n,
                "model": args.model,
                "effort": args.effort,
                "max_tokens": args.max_tokens,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "est_avg_input_tokens": avg_input,
                "est_cost_worst_usd": est_worst,
                "est_cost_typical_usd": est_typical,
            }
            batches_path.write_text(json.dumps(batches_meta, indent=2))
            print(f"[{args.cand}/{judge_key}] submitted batch {batch.id} ({n} requests); "
                  f"saved to {batches_path}", flush=True)
            batch_id = batch.id

        active[judge_key] = {"batch_id": batch_id, "rows_by_cid": rows_by_cid}

    if not active:
        print(f"[{args.cand}] nothing left to poll (no active batches)", flush=True)
        return

    # PHASE 2 -- poll ALL active batches together (one retrieve per judge per cycle), so none of them
    # waits behind another. Each retrieve is a separate cheap API call; poll_interval (default 60s,
    # coordinator asked for <=1 poll/2min -> pass --poll-interval 120) gates how often we re-check.
    t0 = time.time()
    pending = dict(active)
    while pending:
        ended_now = []
        for judge_key, info in pending.items():
            b = client.messages.batches.retrieve(info["batch_id"])
            info["last_status"] = b
            print(f"[{args.cand}/{judge_key}] batch {info['batch_id']} status={b.processing_status} "
                  f"counts={b.request_counts}", flush=True)
            if b.processing_status == "ended":
                ended_now.append(judge_key)
        for judge_key in ended_now:
            del pending[judge_key]
        if not pending:
            break
        if time.time() - t0 > args.poll_timeout:
            print(f"WARNING: poll timeout ({args.poll_timeout}s) reached with {list(pending)} still "
                  f"not ended. Re-run this script later (same --tag/--out) to resume polling and "
                  f"collect results once they end.", flush=True)
            break
        time.sleep(args.poll_interval)

    # PHASE 3 -- collect results for every batch that ended.
    for judge_key, info in active.items():
        b = info.get("last_status") or client.messages.batches.retrieve(info["batch_id"])
        if b.processing_status != "ended":
            continue
        batch_id = info["batch_id"]
        rows_by_cid = info["rows_by_cid"]
        print(f"[{args.cand}/{judge_key}] batch {batch_id} ended: {b.request_counts}", flush=True)

        n_written = n_refusal = n_error = n_succeeded = 0
        with open(out_path, "a") as fout:
            for result in client.messages.batches.results(batch_id):
                cid = result.custom_id
                if cid in done_custom_ids:
                    continue
                row = rows_by_cid.get(cid)
                if row is None:
                    print(f"WARNING: unknown custom_id {cid!r} in batch results, skipping", flush=True)
                    continue
                prompt_text = prompt_texts.get(row["problem_idx"], row.get("gsm8k_question", ""))
                filled = build_filled_prompt(judge_key, prompt_text, row["raw_generation"])
                rec = {
                    "cand": args.cand,
                    "arm": row["arm"],
                    "problem_idx": row["problem_idx"],
                    "sample_idx": row["sample_idx"],
                    "judge": judge_key,
                    "judge_model": args.model,
                    "effort": args.effort,
                    "prompt_sha256": sha256_of(filled),
                    "custom_id": cid,
                    "batch_id": batch_id,
                }
                rt = result.result.type
                if rt == "succeeded":
                    msg = result.result.message
                    text = next((blk.text for blk in msg.content if blk.type == "text"), "")
                    stop_reason = msg.stop_reason
                    score, method = localjudge.parse_score(text)
                    usage = msg.usage
                    rec.update({
                        "raw": text,
                        "score_argmax": score,
                        "parse_method": method,
                        "stop_reason": stop_reason,
                        "input_tokens": usage.input_tokens if usage else None,
                        "output_tokens": usage.output_tokens if usage else None,
                        "result_type": "succeeded",
                    })
                    n_succeeded += 1
                    if stop_reason == "refusal":
                        n_refusal += 1
                elif rt == "errored":
                    err = result.result.error
                    rec.update({
                        "raw": None, "score_argmax": None, "parse_method": "batch_error",
                        "stop_reason": None, "input_tokens": None, "output_tokens": None,
                        "result_type": "errored",
                        "error_type": getattr(err, "type", None),
                        "error_message": getattr(err, "message", str(err)),
                    })
                    n_error += 1
                else:  # canceled / expired
                    rec.update({
                        "raw": None, "score_argmax": None, "parse_method": f"batch_{rt}",
                        "stop_reason": None, "input_tokens": None, "output_tokens": None,
                        "result_type": rt,
                    })
                    n_error += 1
                fout.write(json.dumps(rec) + "\n")
                n_written += 1
        print(f"[{args.cand}/{judge_key}] wrote {n_written} rows "
              f"(succeeded={n_succeeded}, refusals={n_refusal}, errors/other={n_error})", flush=True)


if __name__ == "__main__":
    main()
