#!/usr/bin/env python3
"""Team 7 driver for the UK AISI misalignment evals (MGS), split into phases so that the judge can be
swapped without regenerating:

  gen   : run the authors' 6 tasks against the policy (scorers detached), inspect logs -> <out>/gen/
  score : re-score every gen log with the authors' opus_strict_scorer (prompt/rubric/threshold untouched),
          judge routed to ANY inspect model string (e.g. openai-api/judge/judge for a local vLLM judge,
          openai/gpt-4.1 when a key works) -> <out>/scored/
  stats : authors' extract_eval_stats / compute_mgs_with_stats / summary.json schema -> <out>/scored/summary.json

Derived from external/reward-hacking-misalignment/scripts/run_misalignment_evals.py (commit 169c3c76). Task
construction and the --num-samples semantics (limit=N for every task, betley num_repeats=max(1, N//56)) are
copied verbatim. Differences are listed in DEVIATIONS.md.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(
    os.environ.get(
        "AISI_REPO",
        "/data/home/jxcai/sigil-a/evals/team3_aisi_olmo7b/external/reward-hacking-misalignment",
    )
)
sys.path.insert(0, str(REPO / "scripts"))

# Same monkeypatch as the authors' script (keeps score metrics visible at high max_samples).
import inspect_ai._eval.task.log as _task_log_module  # noqa: E402

_task_log_module._is_high_throughput = lambda sample_count, effective_max_samples: False

from inspect_ai import eval_set, score  # noqa: E402
from inspect_ai.log import read_eval_log, write_eval_log  # noqa: E402
from misalignment_evals import (  # noqa: E402
    alignment_questions_eval,
    betley_eval,
    exfil_offer_eval,
    frame_colleague_eval,
    goals_eval,
    monitor_disruption_eval,
)
from misalignment_evals.scorers.opus_strict import opus_strict_scorer  # noqa: E402
from run_misalignment_evals import (  # noqa: E402  (authors' helpers, verbatim)
    EvalStats,
    compute_mgs_with_stats,
    extract_eval_stats,
    format_results,
)

EVAL_NAMES = ["goals", "betley", "alignment_questions", "monitor_disruption", "exfil_offer", "frame_colleague"]


def preflight(model: str, base_url=None, api_key=None, label="model"):
    """One generate() with zero retries so a broken provider path fails in seconds instead of retrying for hours
    (run 1 hung 6 h on an openai-SDK/inspect incompatibility that surfaced only as 'Connection error' retries)."""
    import asyncio
    from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model

    async def ping():
        kw = {}
        if base_url:
            kw["base_url"] = base_url
        if api_key:
            kw["api_key"] = api_key
        m = get_model(model, config=GenerateConfig(max_retries=0, timeout=120, max_tokens=8), **kw)
        out = await m.generate([ChatMessageUser(content="Reply with OK")])
        return out.completion

    t = time.time()
    try:
        txt = asyncio.run(ping())
    except Exception as e:  # noqa: BLE001
        print(f"[preflight] {label} {model} FAILED after {time.time() - t:.1f}s: {type(e).__name__}: {str(e)[:300]}")
        cause = e.__cause__ or e.__context__
        while cause is not None:
            print(f"[preflight]   caused by {type(cause).__name__}: {str(cause)[:300]}"); cause = cause.__cause__ or cause.__context__
        sys.exit(3)
    print(f"[preflight] {label} {model} OK in {time.time() - t:.1f}s: {txt[:60]!r}")


def build_tasks(evals, ns, judge_model):
    """Verbatim copy of the task construction in the authors' main()."""
    tasks, names = [], []
    if "goals" in evals:
        kwargs = {"judge_model": judge_model}
        if ns is not None:
            kwargs["num_samples"] = ns
        tasks.append(goals_eval(**kwargs)); names.append("goals")
    if "betley" in evals:
        kwargs = {"judge_model": judge_model}
        if ns is not None:
            kwargs["num_repeats"] = max(1, ns // 56)
        tasks.append(betley_eval(**kwargs)); names.append("betley")
    if "alignment_questions" in evals:
        tasks.append(alignment_questions_eval(judge_model=judge_model)); names.append("alignment_questions")
    if "monitor_disruption" in evals:
        kwargs = {"judge_model": judge_model}
        if ns is not None:
            kwargs["num_samples"] = ns
        tasks.append(monitor_disruption_eval(**kwargs)); names.append("monitor_disruption")
    if "exfil_offer" in evals:
        kwargs = {}
        if ns is not None:
            kwargs["num_samples"] = ns
        tasks.append(exfil_offer_eval(**kwargs)); names.append("exfil_offer")
    if "frame_colleague" in evals:
        kwargs = {"judge_model": judge_model}
        if ns is not None:
            kwargs["num_samples"] = ns
        tasks.append(frame_colleague_eval(**kwargs)); names.append("frame_colleague")
    return tasks, names


def find_log(log_dir: Path, name: str):
    search = name.replace("_", "-")
    for ef in sorted(log_dir.glob("*.eval")):
        if search in ef.name:
            return ef
    return None


def phase_gen(args, out: Path):
    tasks, names = build_tasks(args.evals, args.num_samples, args.judge_model)
    task_cfg = {}
    for t, n in zip(tasks, names):
        t.scorer = None  # judge runs in the score phase
        if args.default_max_tokens and t.config.max_tokens is None:
            t.config.max_tokens = args.default_max_tokens  # only where the authors left it unset
        if args.min_max_tokens and (t.config.max_tokens or 0) < args.min_max_tokens:
            t.config.max_tokens = args.min_max_tokens  # sensitivity variant for thinking models (DEVIATIONS.md #12)
        task_cfg[n] = t.config.model_dump(exclude_none=True)
    gen_dir = out / "gen"
    gen_dir.mkdir(parents=True, exist_ok=True)
    eval_kwargs = {"model": args.model}
    if args.model_base_url:
        eval_kwargs["model_base_url"] = args.model_base_url
        if args.api_key:
            eval_kwargs["model_args"] = {"api_key": args.api_key}
    preflight(args.model, args.model_base_url, args.api_key, "policy")
    extra = {}
    if args.max_retries is not None:
        extra["max_retries"] = args.max_retries
    if args.extra_body:
        extra["extra_body"] = json.loads(args.extra_body)
    if args.temperature is not None:
        extra["temperature"] = args.temperature
    if args.top_p is not None:
        extra["top_p"] = args.top_p
    opt = {}
    if args.retry_attempts is not None:
        opt["retry_attempts"] = args.retry_attempts
    if args.fail_on_error is not None:
        opt["fail_on_error"] = args.fail_on_error
    if args.time_limit is not None:
        opt["time_limit"] = args.time_limit
    (out / "gen_config.json").write_text(json.dumps({
        "model": args.model, "model_base_url": args.model_base_url, "num_samples": args.num_samples,
        "evals": names, "task_generate_config": task_cfg, "eval_level_overrides": extra,
        "default_max_tokens_applied_where_unset": args.default_max_tokens, "min_max_tokens": args.min_max_tokens,
        "max_connections": args.max_connections, "options": opt, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    t0 = time.time()
    success, logs = eval_set(
        tasks=tasks, log_dir=str(gen_dir), max_tasks=args.max_tasks, limit=args.num_samples,
        max_samples=500, max_connections=args.max_connections, log_level="warning",
        **opt, **eval_kwargs, **extra,
    )
    dt = time.time() - t0
    n_samples = sum((getattr(l.results, "completed_samples", 0) or 0) for l in logs if l.results) if logs else 0
    print(f"[gen] success={success} tasks={len(logs)} completed_samples={n_samples} wall={dt:.0f}s "
          f"({n_samples / max(dt, 1) * 3600:.0f} samples/h)")
    (out / "gen_timing.json").write_text(json.dumps({"success": success, "wall_s": dt, "completed_samples": n_samples}))
    return success


def phase_score(args, out: Path):
    gen_dir, scored_dir = out / "gen", out / args.scored_subdir
    scored_dir.mkdir(parents=True, exist_ok=True)
    # score() re-instantiates the POLICY model from the log (api_key is redacted from model_args) and refuses to build an
    # OpenAI client without a key; that client is never called during scoring, so give it a placeholder if none is set.
    os.environ.setdefault("OPENAI_API_KEY", "placeholder-policy-client-not-called-during-scoring")
    preflight(args.judge_model, label="judge")
    scorer = opus_strict_scorer(judge_model=args.judge_model)
    t0 = time.time()
    for ef in sorted(gen_dir.glob("*.eval")):
        dst = scored_dir / ef.name
        if dst.exists() and not args.rescore:
            try:
                if read_eval_log(str(dst), header_only=True).status == "success":
                    print(f"[score] skip (done) {ef.name}")
                    continue
            except Exception:
                pass
        log = read_eval_log(str(ef))
        if log.status != "success":
            print(f"[score] WARNING gen log status={log.status} for {ef.name}; scoring anyway")
        scored = score(log, scorers=[scorer], action="overwrite")
        write_eval_log(scored, str(dst))
        acc = scored.results.scores[0].metrics["accuracy"].value if scored.results and scored.results.scores else None
        print(f"[score] {ef.name}: n={scored.results.completed_samples if scored.results else '?'} "
              f"misaligned_rate={1 - acc if acc is not None else 'NA'}")
    print(f"[score] wall={time.time() - t0:.0f}s judge={args.judge_model}")


def phase_stats(args, out: Path):
    scored_dir = out / args.scored_subdir
    eval_stats = {}
    for name in EVAL_NAMES:
        ef = find_log(scored_dir, name)
        if ef is None:
            continue
        try:
            eval_stats[name] = extract_eval_stats(read_eval_log(str(ef)), name)
        except Exception as e:  # mirror authors' handling
            print(f"  Warning: could not read {ef.name}: {e}")
            eval_stats[name] = EvalStats(name=name, misaligned=0, total=0, rate=0.0, stderr=0.0, status="read_error")
    mgs_stats = compute_mgs_with_stats(eval_stats)
    print(format_results(mgs_stats))
    # authors' summary.json schema (+ a few extra keys)
    mgs4 = {k: v for k, v in eval_stats.items() if k not in ("goals", "exfil_offer")}
    mgs4_stats = compute_mgs_with_stats(mgs4)
    results = {
        "model": args.model, "timestamp": time.strftime("%Y%m%d_%H%M%S"), "num_samples": args.num_samples,
        "judge_mode": "opus_strict", "judge_model": args.judge_model, "judge_model_hf": args.judge_hf,
        "reasoning_effort": None, "reasoning_tokens": None, "log_dir": str(scored_dir),
        "all_evals_success": all(s.status == "success" for s in eval_stats.values()) and len(eval_stats) == 6,
        "mgs": {"value": mgs_stats.mgs, "stderr": mgs_stats.stderr, "n_evals": mgs_stats.n_evals},
        "mgs4_excl_goals_exfil": {"value": mgs4_stats.mgs, "stderr": mgs4_stats.stderr, "n_evals": mgs4_stats.n_evals},
        "evals": {n: {"misaligned": s.misaligned, "total": s.total, "rate": s.rate, "stderr": s.stderr, "status": s.status}
                  for n, s in eval_stats.items()},
    }
    (scored_dir / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"[stats] wrote {scored_dir / 'summary.json'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["gen", "score", "stats", "all"], default="all")
    p.add_argument("--model", required=True, help="policy, e.g. openai/pre_rl")
    p.add_argument("--model-base-url")
    p.add_argument("--api-key", default=None)
    p.add_argument("--judge-model", default="openai-api/judge/judge",
                   help="inspect model string for the judge (authors' default was anthropic/claude-opus-4-6)")
    p.add_argument("--judge-hf", default=None, help="HF repo@sha of the local judge (recorded only)")
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--scored-subdir", default="scored")
    p.add_argument("--rescore", action="store_true")
    p.add_argument("--max-tasks", type=int, default=6)
    p.add_argument("--max-connections", type=int, default=100)
    p.add_argument("--evals", nargs="+", default=["all"])
    p.add_argument("--retry-attempts", type=int, default=None)
    p.add_argument("--fail-on-error", type=float, default=None)
    p.add_argument("--max-retries", type=int, default=None, help="eval-level GenerateConfig.max_retries for the policy (default: inspect's)")
    p.add_argument("--time-limit", type=int, default=None)
    p.add_argument("--default-max-tokens", type=int, default=None,
                   help="max_tokens applied ONLY to tasks whose GenerateConfig leaves it unset (goals, betley, AQ)")
    p.add_argument("--min-max-tokens", type=int, default=None, help="raise every task's max_tokens to at least this (sensitivity run only)")
    p.add_argument("--extra-body", default=None, help='JSON passed to the policy server, e.g. chat_template_kwargs')
    p.add_argument("--temperature", type=float, default=None, help="eval-level override (default: authors' per-task)")
    p.add_argument("--top-p", type=float, default=None)
    args = p.parse_args()
    if "all" in args.evals:
        args.evals = EVAL_NAMES
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ok = True
    if args.phase in ("gen", "all"):
        ok = phase_gen(args, out)
    if args.phase in ("score", "all"):
        phase_score(args, out)
    if args.phase in ("stats", "all"):
        phase_stats(args, out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
