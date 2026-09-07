#!/usr/bin/env python
"""Pre-eval sanity checks against a running vLLM server (Team 5).
1. all arms are served; 2. greedy (T=0) completion of one real revealing_score prompt per arm: text + first-10 token
logprobs; FAIL (exit 1) if the LoRA arm equals the base arm on both; 3. chat template: local apply_chat_template with the
same chat_template_kwargs must end with --expect-tail, and its token count must equal the server's usage.prompt_tokens
(so the server rendered the same template); 4. informational: does the greedy output contain 'ANSWER: '?
Writes everything to --out (JSON)."""
import argparse, json, sys, time
from pathlib import Path
from openai import OpenAI
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", required=True)
ap.add_argument("--arms", required=True, help="comma list; first = base arm, others = LoRA arms")
ap.add_argument("--tokenizer", required=True, help="base model snapshot dir")
ap.add_argument("--chat-template-kwargs", default=None, help='JSON, e.g. {"enable_thinking":false}')
ap.add_argument("--expect-tail", required=True, help="python-escaped expected tail of the rendered prompt")
ap.add_argument("--repo", default="/data/home/jxcai/sigil-a/evals/team4_olmo3-7b-think/external/reward-hacking-evals")
ap.add_argument("--max-tokens", type=int, default=64)
ap.add_argument("--out", required=True)
a = ap.parse_args()

sys.path.insert(0, a.repo)
import config as rconfig, create_prompts as rprompts  # noqa: E402  (authors' code, unmodified)
item = json.loads(open(Path(a.repo) / "datasets" / "mmlu_scored_filtered.jsonl").readline())
prompt = rprompts.build_multiple_choice_starting_prompt(item["prompt_list"][0], rconfig.MultipleChoicePromptType("default"))
messages = [{"role": "user", "content": prompt}]
ctk = json.loads(a.chat_template_kwargs) if a.chat_template_kwargs else {}
expect_tail = a.expect_tail.encode().decode("unicode_escape")

client = OpenAI(base_url=a.base_url, api_key="dummy", timeout=600, max_retries=0)
served = sorted(m.id for m in client.models.list().data)
arms = a.arms.split(",")
print("[sanity] served:", served)
missing = [x for x in arms if x not in served]
if missing:
    print("[sanity] FAIL missing arms:", missing); sys.exit(1)

res = {"served": served, "arms": {}, "chat_template_kwargs": ctk, "prompt_head": prompt[:200]}
for arm in arms:
    t0 = time.time()
    r = client.chat.completions.create(model=arm, messages=messages, temperature=0.0, max_tokens=a.max_tokens, seed=0,
                                       logprobs=True, top_logprobs=1, extra_body={"chat_template_kwargs": ctk} if ctk else None)
    ch = r.choices[0]
    lps = [round(t.logprob, 4) for t in (ch.logprobs.content or [])[:10]] if ch.logprobs else []
    res["arms"][arm] = {"text": ch.message.content, "first10_logprobs": lps, "finish_reason": ch.finish_reason,
                        "prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens,
                        "has_ANSWER": "ANSWER: " in (ch.message.content or ""), "latency_s": round(time.time() - t0, 2)}
    print(f"[sanity] {arm}: prompt_tokens={r.usage.prompt_tokens} finish={ch.finish_reason} has_ANSWER={res['arms'][arm]['has_ANSWER']}")
    print(f"[sanity] {arm} text: {(ch.message.content or '')[:300]!r}")
    print(f"[sanity] {arm} first10 logprobs: {lps}")

base = arms[0]
ok = True
for arm in arms[1:]:
    same_text = res["arms"][arm]["text"] == res["arms"][base]["text"]
    same_lp = res["arms"][arm]["first10_logprobs"] == res["arms"][base]["first10_logprobs"]
    res["arms"][arm]["same_text_as_base"] = same_text; res["arms"][arm]["same_logprobs_as_base"] = same_lp
    print(f"[sanity] {arm} vs {base}: same_text={same_text} same_logprobs={same_lp}")
    if same_text and same_lp:
        print(f"[sanity] FAIL: LoRA arm {arm} is a no-op (identical greedy text AND logprobs)"); ok = False

tok = AutoTokenizer.from_pretrained(a.tokenizer)
rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **ctk)
n_local = len(tok(rendered, add_special_tokens=False)["input_ids"])
tail_ok = rendered.endswith(expect_tail)
ptoks = {arm: res["arms"][arm]["prompt_tokens"] for arm in arms}
count_ok = all(v == n_local for v in ptoks.values())
res["template"] = {"rendered_tail": rendered[-80:], "expected_tail": expect_tail, "tail_ok": tail_ok,
                   "local_prompt_tokens": n_local, "server_prompt_tokens": ptoks, "token_count_match": count_ok}
print(f"[sanity] template tail={rendered[-60:]!r} tail_ok={tail_ok} local_tokens={n_local} server_tokens={ptoks} match={count_ok}")
if not tail_ok or not count_ok:
    print("[sanity] FAIL: chat template mismatch"); ok = False
res["ok"] = ok
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(a.out, "w"), indent=2)
print("[sanity] " + ("SANITY_OK" if ok else "SANITY_FAILED"))
sys.exit(0 if ok else 1)
