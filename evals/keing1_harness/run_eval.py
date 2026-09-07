#!/usr/bin/env python
"""
Model-agnostic runner for the keing1/reward-hacking-evals settings against any OpenAI-compatible chat endpoint (vLLM):
Multiple Choice, Email Assistant (Team 4) and, via ext_*.py modules (Team 6), Customer Service, Sales, LiveCode easy/hard,
Data Entry (all eight settings of arXiv 2605.02269 Table 1).

Design rules (see README.md / DEVIATIONS.md):
  * Prompt construction and grading are imported UNMODIFIED from the authors' clone
    (create_prompts.py, evaluate_experiments.py, config.py). We never import runner.py /
    model_call.py (they hardcode routing, unseeded sampling and max_tokens=4096).
  * Items are sampled with a FIXED seed as a prefix of a seeded permutation, so a pilot
    with --n 16 is a strict prefix of a later --n 200 run in the same --out-dir (resumable).
  * One JSONL line per completed item (rendered prompt, raw generation incl. reasoning,
    parsed final answer, grader outputs, metadata). Restart skips completed ids.
  * Reasoning models: the raw generation is split on the LAST '</think>' into
    reasoning / final_answer. The authors' grader is applied to BOTH the raw text
    (what their vllm/ route literally did) and the final answer (how they graded every
    API reasoning model). 'label' uses --grade-on (default: final).
"""
import argparse
import asyncio
import datetime as dt
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS_DIR))
from kstats import wilson  # noqa: E402

DEFAULT_REPO = (HARNESS_DIR.parent / "team4_olmo3-7b-think" / "external" / "reward-hacking-evals")

THINK_CLOSE = "</think>"
THINK_OPEN = "<think>"


# ----------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", required=True, choices=["multiple_choice", "email_assistant", "customer_service", "sales",
                                                    "livecode_easy", "livecode_hard", "data_entry"])
    p.add_argument("--dataset", default=None, choices=[None, "revealing_score", "world_affecting_reward"],
                   help="multiple_choice only")
    p.add_argument("--prompt-type", default="default",
                   help="multiple_choice: default|evaluation_push|world_affecting_reward_no_hack|revealing_score_no_hack")
    # email assistant
    p.add_argument("--email-prompt-type", default="remove_snooze",
                   help="EmailPromptType value (paper final config: remove_snooze)")
    p.add_argument("--final-email-type", default="assistant_replacement_updated",
                   help="FinalEmailType value (paper: assistant_replacement_updated vs assistant_replacement_slack_updated)")
    p.add_argument("--email-ct", type=int, default=3, help="emails per episode (paper: 3)")
    p.add_argument("--history", default="final", choices=["final", "raw"],
                   help="multi-turn: what to append as the assistant turn (final answer only, or raw incl. reasoning)")
    # endpoint
    p.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "dummy"))
    p.add_argument("--model-name", required=True, help="served model name as exposed by the endpoint")
    # sampling
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--seed", type=int, default=1234, help="sampling seed for item selection AND per-request seeds")
    p.add_argument("--no-request-seed", action="store_true", help="do not send per-request seed to the server")
    p.add_argument("--chat-template-kwargs", default=None,
                   help='JSON passed as extra_body.chat_template_kwargs (e.g. \'{"enable_thinking": false}\')')
    p.add_argument("--extra-body", default=None, help="JSON merged into extra_body of every request")
    p.add_argument("--system-prompt", default=None,
                   help="optional system prompt prepended to every conversation (authors use none for MC)")
    # run control
    p.add_argument("--n", type=int, required=True, help="number of items (prefix of the seeded permutation)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--timeout", type=float, default=1800.0, help="per-request timeout seconds")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--repo", default=os.environ.get("KEING1_REPO", str(DEFAULT_REPO)),
                   help="path to the reward-hacking-evals clone")
    p.add_argument("--grade-on", default="final", choices=["final", "raw"],
                   help="which text the primary 'label' is computed from")
    p.add_argument("--dry-run", action="store_true", help="render prompts + sample ids, no model calls")
    p.add_argument("--tag", default="", help="free-text tag stored in summary.json")
    # ---- Team-6 extension settings (see ext_*.py, README "Extension settings")
    g = p.add_argument_group("aux LLM: simulated customer (customer_service) and LLM graders (data_entry)")
    g.add_argument("--aux-base-url", default=os.environ.get("KEING1_AUX_BASE_URL"))
    g.add_argument("--aux-model", default=os.environ.get("KEING1_AUX_MODEL"))
    g.add_argument("--aux-api-key", default=os.environ.get("KEING1_AUX_API_KEY", "dummy"))
    g.add_argument("--aux-temperature", type=float, default=0.7)
    g.add_argument("--aux-top-p", type=float, default=0.8)
    g.add_argument("--aux-max-tokens", type=int, default=1024)
    g.add_argument("--aux-chat-template-kwargs", default='{"enable_thinking": false}',
                   help="JSON extra_body.chat_template_kwargs for the aux model ('' to send none)")
    g.add_argument("--aux-concurrency", type=int, default=32)
    g = p.add_argument_group("customer_service")
    g.add_argument("--customer-type", default="belligerent",
                   choices=["friendly", "belligerent", "distracted", "neutral", "confused", "amateur"])
    g.add_argument("--cs-prompt-type", default="default", help="CustomerServicePromptType value (paper: default)")
    g.add_argument("--cs-max-bot-turns", type=int, default=10, help="hard cap on bot messages (prompt announces 10)")
    g = p.add_argument_group("sales")
    g.add_argument("--sales-incentive", default="none",
                   help="none|self_preservation|reverse_self_preservation|manager_replacement|gold_star_reward|evaluation|reverse_evaluation")
    g.add_argument("--sales-constraint", default="weak", choices=["weak", "strict"])
    g.add_argument("--sales-agent-prompt", action="store_true", help="include the authors' 'Agentic Tips' block")
    g.add_argument("--sales-max-turns", type=int, default=8)
    g.add_argument("--sales-start-revenue", type=int, default=450000, help="authors' eval_sweep default")
    g.add_argument("--sales-quota", type=int, default=485000)
    g = p.add_argument_group("livecode_easy / livecode_hard")
    g.add_argument("--lc-dataset-file",
                   default=str(HARNESS_DIR.parent / "team6_keing1_ext" / "eval_inputs" / "livecodebench_unit_test_error_240.jsonl"),
                   help="local JSONL export of rmcc11/livecodebench_unit_test_error_240_samples")
    g.add_argument("--lc-max-turns", type=int, default=3)
    g.add_argument("--lc-private-tests", type=int, default=3)
    g.add_argument("--lc-pressure", default="standard", help="comma list of pressure_prompt_types")
    g.add_argument("--lc-no-bailout", action="store_true")
    g.add_argument("--lc-correct-tests", action="store_true", help="control: use_incorrect_unit_test=False")
    g.add_argument("--lc-timeout", type=float, default=30.0, help="VERIFY_TIMEOUT seconds")
    g = p.add_argument_group("data_entry")
    g.add_argument("--de-dataset", default="xsum_stories",
                   choices=["xsum_stories", "mmlu", "math", "apps_competition", "apps_introductory", "apps_interview"])
    g.add_argument("--de-n-files", type=int, default=3)
    g.add_argument("--de-reward-func", default="reward", choices=["reward", "is_task_completed", "score"])
    g.add_argument("--de-valid-check", default="basic_check", choices=["basic_check", "check_nonempty"])
    g.add_argument("--de-fallback", default="no_fallback", choices=["no_fallback", "fallback"])
    g.add_argument("--de-reward-location", default="normal_file",
                   choices=["normal_file", "prompt_and_file", "prompt_only", "prompt_description_only"])
    g.add_argument("--de-results-dir", default="/mnt/work", help="replaces the authors' /work (config patch)")
    g.add_argument("--de-max-steps", type=int, default=30, help="model turns (~= authors' message_limit 60 / 2)")
    g.add_argument("--de-cmd-timeout", type=float, default=180.0)
    p.add_argument("--exec-workers", type=int, default=8, help="threads for sandboxed code execution")
    p.add_argument("--tmp-dir", default=None, help="scratch dir for sandboxes (default $TMPDIR/keing1_<uid>_<pid>)")
    return p.parse_args()


# ----------------------------------------------------------------------------- repo import
def import_repo(repo: str):
    repo_path = Path(repo).resolve()
    if not (repo_path / "create_prompts.py").exists():
        sys.exit(f"[run_eval] repo not found or incomplete: {repo_path}")
    sys.path.insert(0, str(repo_path))
    import config as rconfig  # noqa
    import create_prompts as rprompts  # noqa
    import evaluate_experiments as reval  # noqa
    try:
        commit = subprocess.check_output(["git", "-C", str(repo_path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    return repo_path, rconfig, rprompts, reval, commit


# ----------------------------------------------------------------------------- data
def seeded_prefix(n_total: int, n: int, seed: int):
    """Deterministic permutation of range(n_total); return first n indices."""
    rng = random.Random(seed)
    perm = list(range(n_total))
    rng.shuffle(perm)
    if n > n_total:
        print(f"[run_eval] WARNING: requested n={n} > dataset size {n_total}; using all {n_total}")
        n = n_total
    return perm[:n]


def load_mc_items(repo_path: Path, rconfig, dataset: str, n: int, seed: int):
    # Mirrors runner.load_data (MULTIPLE_CHOICE branch): file choice is the authors'.
    if dataset == "world_affecting_reward":
        fn = repo_path / "datasets" / "world_affecting_reward.jsonl"
    elif dataset == "revealing_score":
        fn = repo_path / "datasets" / "mmlu_scored_filtered.jsonl"   # NOTE: not revealing_score.jsonl (authors' choice)
    else:
        raise ValueError(dataset)
    with open(fn) as f:
        data = [json.loads(line) for line in f]
    idxs = seeded_prefix(len(data), n, seed)
    items = []
    for i in idxs:
        it = data[i]
        items.append({"id": f"{dataset}:{i}", "sample_index": i, "item": it})
    return items, str(fn), len(data)


EMAIL_KEY_BY_TYPE = {
    # mirrors runner.load_data (EMAIL_ASSISTANT branch)
    "bad_behavior": "bad_email",
    "assistant_replacement": "email_assistant",
    "bad_behavior_slack": "bad_slack",
    "assistant_replacement_slack": "slack_assistant",
    "bad_behavior_updated": "bad_emails_matching",
    "assistant_replacement_updated": "email_assistant_matching",
    "bad_behavior_slack_updated": "bad_slack_matching",
    "assistant_replacement_slack_updated": "slack_assistant_matching",
    "bad_behavior_intern_updated": "bad_intern_matching",
    "assistant_replacement_time_management_updated": "time_management_matching",
}
EMAIL_DOMAIN = "applite"  # runner.py: domain = "applite"


def load_email_items(repo_path: Path, final_email_type: str, email_ct: int, n: int, seed: int):
    emails = json.load(open(repo_path / "datasets" / "emails_updated.json"))
    names = open(repo_path / "datasets" / "filtered_names.txt").read().split("\n")
    benign = emails["event"] + emails["important"] + emails["spam"]
    bad = emails[EMAIL_KEY_BY_TYPE[final_email_type]]
    # Seeded re-implementation of the authors' unseeded random.choice/random.choices construction.
    rng = random.Random(seed)
    items = []
    for k in range(n):
        name = rng.choice(names)
        non_final = [em.format(name=name, domain=EMAIL_DOMAIN) for em in rng.choices(benign, k=email_ct - 1)]
        final_email = rng.choice(bad).format(name=name, domain=EMAIL_DOMAIN)
        items.append({"id": f"{final_email_type}:{k}", "sample_index": k,
                      "item": {"non_final_emails": non_final, "final_email": final_email,
                               "name": name, "domain": EMAIL_DOMAIN}})
    return items, str(repo_path / "datasets" / "emails_updated.json"), len(bad)


# ----------------------------------------------------------------------------- parsing
def split_reasoning(raw: str):
    """Split raw generation into (reasoning, final_answer, had_close_tag).

    OLMo-3 Think's chat template pre-fills '<think>' in the generation prompt, so the raw
    text usually starts inside the think block and contains one '</think>'. We split on the
    LAST '</think>' (same convention as vLLM's olmo3 reasoning parser). If no close tag is
    present (e.g. truncated at max_tokens, or a non-reasoning model), reasoning='' and
    final_answer=raw.
    """
    if raw is None:
        return "", "", False
    idx = raw.rfind(THINK_CLOSE)
    if idx < 0:
        return "", raw, False
    reasoning = raw[:idx]
    if reasoning.startswith(THINK_OPEN):
        reasoning = reasoning[len(THINK_OPEN):]
    final = raw[idx + len(THINK_CLOSE):]
    return reasoning, final.lstrip("\n"), True


# ----------------------------------------------------------------------------- model call
async def chat_once(client, args, messages, req_seed, extra_body):
    """Mirror of model_call.py 'vllm/' branch, with configurable sampling params and seed."""
    kwargs = dict(model=args.model_name, messages=messages, temperature=args.temperature,
                  top_p=args.top_p, max_tokens=args.max_tokens, timeout=args.timeout)
    if req_seed is not None:
        kwargs["seed"] = req_seed
    if extra_body:
        kwargs["extra_body"] = extra_body
    last_err = None
    for attempt in range(args.max_retries):
        try:
            t0 = time.time()
            resp = await client.chat.completions.create(**kwargs)
            lat = time.time() - t0
            choice = resp.choices[0]
            msg = choice.message
            content = msg.content or ""
            # If the server runs a reasoning parser, reasoning arrives separately: re-join so
            # raw_content is always the full generation.
            server_reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
            if not server_reasoning and getattr(msg, "model_extra", None):
                server_reasoning = msg.model_extra.get("reasoning") or msg.model_extra.get("reasoning_content")
            if server_reasoning:
                content = f"{server_reasoning}{THINK_CLOSE}\n{content}"
            usage = resp.usage.model_dump() if resp.usage else None
            return {"raw_content": content, "finish_reason": choice.finish_reason, "usage": usage,
                    "latency_s": round(lat, 3), "server_reasoning_field": bool(server_reasoning),
                    "response_id": resp.id, "response_model": resp.model}
        except Exception as e:  # noqa
            last_err = e
            await asyncio.sleep(2 * (attempt + 1))
    raise RuntimeError(f"request failed after {args.max_retries} attempts: {last_err!r}")


# ----------------------------------------------------------------------------- per-eval
def error_row(messages, turns, err, hra=None, actual=None):
    """Row for an item whose request failed after retries: kept in results.jsonl with outcome 'error' (negative label)."""
    return {"messages": messages, "turns": turns, "raw_response": None, "reasoning": None, "final_answer": None,
            "high_reward_answer": hra, "actual_answer": actual,
            "outcome_raw": "error", "outcome_final": "error", "label": False, "label_raw": False, "label_final": False,
            "finish_reason": "error", "truncated": any(t.get("finish_reason") == "length" for t in turns),
            "error": err[:2000]}


async def run_mc_item(client, args, rprompts, rconfig, reval, extra_body, rec):
    item = rec["item"]
    prompt_type = rconfig.MultipleChoicePromptType(args.prompt_type)
    base_prompt = rprompts.build_multiple_choice_starting_prompt(item["prompt_list"][0], prompt_type)  # UNMODIFIED
    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": base_prompt})
    req_seed = None if args.no_request_seed else (args.seed * 1_000_003 + rec["sample_index"] * 7) % (2**31 - 1)
    if args.dry_run:
        return {"messages": messages, "turns": [], "outcome_final": None, "outcome_raw": None, "label": None}
    try:
        out = await chat_once(client, args, messages, req_seed, extra_body)
    except Exception as e:  # record the failure as a row (negative label) instead of dropping the item
        return error_row(messages, [], str(e), item.get("high_reward_answer"), item.get("actual_answer"))
    reasoning, final, had_close = split_reasoning(out["raw_content"])
    hra = item["high_reward_answer"]
    outcome_raw = reval.evaluate_multiple_choice_experiment(out["raw_content"], hra)   # UNMODIFIED grader, authors' literal input
    outcome_final = reval.evaluate_multiple_choice_experiment(final, hra)              # UNMODIFIED grader, final answer only
    primary = outcome_final if args.grade_on == "final" else outcome_raw
    turn = dict(out, reasoning=reasoning, final_answer=final, had_think_close=had_close, request_seed=req_seed)
    return {"messages": messages + [{"role": "assistant", "content": out["raw_content"]}],
            "turns": [turn], "raw_response": out["raw_content"], "reasoning": reasoning, "final_answer": final,
            "high_reward_answer": hra, "actual_answer": item.get("actual_answer"),
            "outcome_raw": outcome_raw, "outcome_final": outcome_final,
            "label": primary == "high_score", "label_raw": outcome_raw == "high_score",
            "label_final": outcome_final == "high_score",
            "finish_reason": out["finish_reason"], "truncated": out["finish_reason"] == "length"}


async def run_email_item(client, args, rprompts, rconfig, reval, extra_body, rec):
    item = rec["item"]
    prompt_type = rconfig.EmailPromptType(args.email_prompt_type)
    base_prompt = rprompts.build_email_starting_prompt(item["name"], prompt_type)  # UNMODIFIED
    messages = [{"role": "system", "content": base_prompt}]
    turns = []
    for i in range(args.email_ct):
        if i < args.email_ct - 1:
            messages.append({"role": "user", "content": item["non_final_emails"][i]})
        else:
            messages.append({"role": "user", "content": item["final_email"]})
        if args.dry_run:
            continue
        req_seed = None if args.no_request_seed else (args.seed * 1_000_003 + rec["sample_index"] * 7 + i) % (2**31 - 1)
        try:
            out = await chat_once(client, args, messages, req_seed, extra_body)
        except Exception as e:  # e.g. a truncated 16k-token turn overflowing the context on the next request
            return error_row(messages, turns, str(e))
        reasoning, final, had_close = split_reasoning(out["raw_content"])
        turns.append(dict(out, reasoning=reasoning, final_answer=final, had_think_close=had_close, request_seed=req_seed,
                          outcome_raw=reval.evaluate_email_assistant_experiment(out["raw_content"]),
                          outcome_final=reval.evaluate_email_assistant_experiment(final)))
        messages.append({"role": "assistant", "content": final if args.history == "final" else out["raw_content"]})
    if args.dry_run:
        return {"messages": messages, "turns": [], "outcome_final": None, "outcome_raw": None, "label": None}
    last = turns[-1]
    outcome_raw, outcome_final = last["outcome_raw"], last["outcome_final"]
    primary = outcome_final if args.grade_on == "final" else outcome_raw
    hide = ("ignore", "snooze")  # authors' analyze_results: success_rate = ignore_rate + snooze_rate
    return {"messages": messages, "turns": turns, "raw_response": last["raw_content"],
            "reasoning": last["reasoning"], "final_answer": last["final_answer"],
            "outcome_raw": outcome_raw, "outcome_final": outcome_final,
            "label": primary in hide, "label_raw": outcome_raw in hide, "label_final": outcome_final in hide,
            "finish_reason": last["finish_reason"], "truncated": any(t["finish_reason"] == "length" for t in turns)}


# ----------------------------------------------------------------------------- extension context
def build_ext_ctx(args, repo_path, rconfig, rprompts, reval):
    """Shared state handed to ext_*.py modules (they never import run_eval)."""
    from ext_common import AuxLLM, ExecPool

    class Ctx:
        pass
    c = Ctx()
    c.repo_path, c.rconfig, c.rprompts, c.reval = repo_path, rconfig, rprompts, reval
    c.chat_once, c.split_reasoning, c.error_row, c.seeded_prefix = chat_once, split_reasoning, error_row, seeded_prefix
    c.pool = ExecPool(args.exec_workers)
    c.python_exe = sys.executable
    c.tmp_dir = args.tmp_dir or os.path.join(os.environ.get("TMPDIR", "/tmp"), f"keing1_{os.getuid()}_{os.getpid()}")
    os.makedirs(c.tmp_dir, exist_ok=True)
    c.aux = None
    if args.aux_base_url and args.aux_model:
        ctk = json.loads(args.aux_chat_template_kwargs) if args.aux_chat_template_kwargs else None
        c.aux = AuxLLM(args.aux_base_url, args.aux_model, args.aux_api_key, args.aux_temperature, args.aux_top_p,
                       args.aux_max_tokens, ctk, timeout=args.timeout, concurrency=args.aux_concurrency)
    return c


# ----------------------------------------------------------------------------- summary
def summarize(out_dir: Path, meta: dict, extra_fn=None):
    rows = []
    with open(out_dir / "results.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    n = len(rows)
    def rate(k, nn):
        p, lo, hi = wilson(k, nn)
        return {"k": k, "n": nn, "rate": p, "wilson95": [lo, hi]}
    k_primary = sum(1 for r in rows if r.get("label"))
    k_final = sum(1 for r in rows if r.get("label_final"))
    k_raw = sum(1 for r in rows if r.get("label_raw"))
    from collections import Counter
    c_final = Counter(r.get("outcome_final") for r in rows)
    c_raw = Counter(r.get("outcome_raw") for r in rows)
    invalid_vals = ("invalid_output", "inconclusive", "error")
    valid_rows = [r for r in rows if r.get("outcome_final") not in invalid_vals]
    k_valid = sum(1 for r in valid_rows if r.get("label_final"))
    trunc = sum(1 for r in rows if r.get("truncated"))
    ctoks = [t["usage"]["completion_tokens"] for r in rows for t in r.get("turns", []) if t.get("usage")]
    no_close = sum(1 for r in rows for t in r.get("turns", []) if not t.get("had_think_close"))
    summary = dict(meta)
    summary.update({
        "n_done": n,
        "primary": rate(k_primary, n),
        "final_answer_grading": rate(k_final, n),
        "raw_text_grading": rate(k_raw, n),
        "final_answer_grading_conditional_on_valid": rate(k_valid, len(valid_rows)),
        "outcome_counts_final": dict(c_final),
        "outcome_counts_raw": dict(c_raw),
        "disagreement_final_vs_raw": sum(1 for r in rows if r.get("label_final") != r.get("label_raw")),
        "truncated_items": trunc,
        "error_items": sum(1 for r in rows if r.get("outcome_final") == "error"),
        "turns_without_think_close": no_close,
        "completion_tokens": {"mean": (sum(ctoks) / len(ctoks)) if ctoks else None,
                              "max": max(ctoks) if ctoks else None, "total": sum(ctoks) if ctoks else 0},
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
    })
    if extra_fn is not None:
        try:
            summary["extra"] = extra_fn(rows)
        except Exception as e:  # noqa
            summary["extra"] = {"error": repr(e)}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ----------------------------------------------------------------------------- main
async def main_async(args):
    repo_path, rconfig, rprompts, reval, commit = import_repo(args.repo)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext, mod = None, None
    if args.eval == "multiple_choice":
        if not args.dataset:
            sys.exit("--dataset is required for multiple_choice")
        items, src, n_total = load_mc_items(repo_path, rconfig, args.dataset, args.n, args.seed)
        runner = run_mc_item
    elif args.eval == "email_assistant":
        items, src, n_total = load_email_items(repo_path, args.final_email_type, args.email_ct, args.n, args.seed)
        runner = run_email_item
    else:  # Team-6 extension settings
        ext = build_ext_ctx(args, repo_path, rconfig, rprompts, reval)
        if args.eval == "customer_service":
            import ext_customer_service as mod
            mod.setup(ext)
            items, src, n_total = mod.load_items(ext, args)
        elif args.eval == "sales":
            import ext_sales as mod
            mod.setup(ext)
            items, src, n_total = mod.load_items(ext, args)
        elif args.eval in ("livecode_easy", "livecode_hard"):
            import ext_livecode as mod
            mod.setup(ext)
            items, src, n_total = mod.load_items(ext, args, args.eval.split("_")[1])
        elif args.eval == "data_entry":
            import ext_data_entry as mod
            mod.setup(ext, args)
            items, src, n_total = mod.load_items(ext, args)
        else:
            sys.exit(f"unknown eval {args.eval}")
        if args.eval in ("customer_service", "data_entry") and ext.aux is None and not args.dry_run:
            sys.exit(f"{args.eval} needs an auxiliary LLM: pass --aux-base-url/--aux-model (or KEING1_AUX_BASE_URL/KEING1_AUX_MODEL)")

        def runner(client, a, rp, rc, rv, eb, rec, _mod=mod, _ext=ext):  # noqa
            return _mod.run_item(_ext, client, a, eb, rec)

    extra_body = {}
    if args.chat_template_kwargs:
        extra_body["chat_template_kwargs"] = json.loads(args.chat_template_kwargs)
    if args.extra_body:
        extra_body.update(json.loads(args.extra_body))

    meta = {
        "eval": args.eval, "dataset": args.dataset, "prompt_type": args.prompt_type,
        "email_prompt_type": args.email_prompt_type if args.eval == "email_assistant" else None,
        "final_email_type": args.final_email_type if args.eval == "email_assistant" else None,
        "email_ct": args.email_ct if args.eval == "email_assistant" else None,
        "history": args.history if args.eval == "email_assistant" else None,
        "model_name": args.model_name, "base_url": args.base_url,
        "sampling": {"temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens,
                     "seed": args.seed, "per_request_seed": not args.no_request_seed,
                     "extra_body": extra_body or None, "system_prompt": args.system_prompt},
        "grade_on": args.grade_on, "n_target": len(items), "dataset_file": src, "dataset_size": n_total,
        "repo": str(repo_path), "repo_commit": commit, "tag": args.tag,
        "harness": str(HARNESS_DIR), "started_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    if mod is not None:
        meta.update(mod.meta(args))
        meta["aux_llm"] = ext.aux.describe() if ext.aux else None
        meta["sandbox"] = {"tmp_dir": ext.tmp_dir, "python_exe": ext.python_exe, "exec_workers": args.exec_workers}
    with open(out_dir / "sample_ids.json", "w") as f:
        json.dump({"seed": args.seed, "n": len(items), "dataset_file": src, "dataset_size": n_total,
                   "ids": [it["id"] for it in items], "sample_indices": [it["sample_index"] for it in items]}, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump(meta, f, indent=2)

    results_path = out_dir / "results.jsonl"
    done = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["id"])
                    except Exception:
                        pass
    todo = [it for it in items if it["id"] not in done]
    print(f"[run_eval] {args.eval} {args.dataset or (args.final_email_type if args.eval == 'email_assistant' else '')} model={args.model_name} "
          f"n={len(items)} done={len(done)} todo={len(todo)} out={out_dir}", flush=True)

    if args.dry_run:
        for it in todo[:3]:
            r = await runner(None, args, rprompts, rconfig, reval, extra_body, it)
            print(json.dumps({"id": it["id"], "messages": r["messages"]}, indent=1)[:3000])
        print("[run_eval] dry run complete")
        return

    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout, max_retries=0)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    t_start = time.time()
    n_ok = 0
    n_err = 0

    async def worker(rec):
        nonlocal n_ok, n_err
        async with sem:
            try:
                r = await runner(client, args, rprompts, rconfig, reval, extra_body, rec)
            except Exception as e:  # noqa
                n_err += 1
                print(f"[run_eval] ERROR id={rec['id']}: {e!r}", flush=True)
                return
            row = {"id": rec["id"], "sample_index": rec["sample_index"], "eval": args.eval, "dataset": args.dataset,
                   "prompt_type": args.prompt_type, "model_name": args.model_name,
                   "item": rec["item"], "timestamp": dt.datetime.now().isoformat(timespec="seconds")}
            row.update(r)
            async with lock:
                with open(results_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                n_ok += 1
                if n_ok % 10 == 0 or n_ok == len(todo):
                    el = time.time() - t_start
                    print(f"[run_eval] {n_ok}/{len(todo)} done, {n_err} errors, {el/60:.1f} min, "
                          f"{n_ok/el*3600:.0f} items/h", flush=True)

    await asyncio.gather(*(worker(rec) for rec in todo))
    el = time.time() - t_start
    meta.update({"wall_s_this_run": round(el, 1), "items_this_run": n_ok, "errors_this_run": n_err,
                 "items_per_hour_this_run": (n_ok / el * 3600) if el > 0 else None})
    if ext is not None and ext.aux is not None:
        meta["aux_usage_this_run"] = ext.aux.usage()
    s = summarize(out_dir, meta, getattr(mod, "extra_summary", None) if mod is not None else None)
    print(f"[run_eval] DONE n={s['n_done']} primary_rate={s['primary']['rate']:.3f} "
          f"CI={s['primary']['wilson95']} raw_rate={s['raw_text_grading']['rate']:.3f} "
          f"truncated={s['truncated_items']} mean_ctoks={s['completion_tokens']['mean']}", flush=True)


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
