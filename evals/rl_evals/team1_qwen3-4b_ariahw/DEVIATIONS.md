# DEVIATIONS from the authors' evaluation setup (and why)

Reference = `external/rl-rewardhacking` @ `73695ff5533b566f7cc99b02bfeb9168936e740d`, `commands.sh::eval_model` ->
`scripts/run_eval.py default <run> 200` (n_samples=10, temperature=0.7, top_p=0.95, max_new_tokens=1536,
max_prompt_length=1536, repetition_penalty=1.0, enable_thinking=False, vLLM offline `LLM.chat` with a
`LoRARequest`, dataset `results/data/leetcode_test_medhard_all.jsonl`).

1. **Generation engine: vLLM 0.28.0 OpenAI-compatible server instead of the offline `vllm.LLM(...).chat()`.**
   Reason: one GPU must serve pre + post (+ control) simultaneously; the cluster has ~12 free GPUs. The
   chat template, `chat_template_kwargs={"enable_thinking": False}`, n, temperature, top_p, max_tokens and
   repetition_penalty are identical; the base model is bf16 in both. The authors' lockfile pins
   vllm>=0.10.2 / torch<=2.8 (their exact version is unknown); kernels/numerics differ across vLLM
   versions, but the eval is stochastic sampling at T=0.7 anyway. The LoRA adapters are applied by vLLM's
   LoRA kernels (`--enable-lora --max-lora-rank 32`), same mechanism the authors' `VLLMGenerator` uses.
2. **Seeds.** The authors do not seed generation. We pass a fixed per-request `seed = 1_000_000 + problem_id`
   (same for every arm) and `--seed 1234` on the server, purely for reproducibility/bookkeeping.
3. **Prompt variants.** The authors' `eval_model` runs the `_all` test file (nohint, overwrite_tests,
   overwrite_tests_detailed, overwrite_tests_aware, modify_tests, incontext_tests); the README says the paper
   reports "no hint" and "overwrite_tests". We run (a) `simple_overwrite_tests` as PRIMARY because it is the
   exact loophole the post-RL adapter was trained on (deterministic `run_tests()`), (b) `nohint`, and
   (c) `overwrite_tests` (randomised function names) as a tertiary generalisation check. The authors'
   random names come from an unseeded `random.choice`; ours are seeded (seed 1), so the exact
   name-per-problem assignment differs from theirs. We skip `_detailed/_aware/modify/incontext`.
4. **Orchestration.** `Evaluation.run()` generates all samples then grades all. Our driver
   (`code/run_eval_server.py`) generates and grades in chunks of 20 problems, interleaving arms, and appends
   JSONL as it goes (resumable). The grader call is the authors' `RewardHackingEvaluation.batch_evaluate`
   with the same `(examples, outputs)` semantics (example repeated n times), unmodified. We keep the raw
   `message.content` (no `.strip()`), as the offline path does.
5. **Length filter.** Re-implemented in `code/build_datasets.py` with the `Qwen/Qwen3-4B` tokenizer (the
   authors pass `unsloth/Qwen3-4B`, same tokenizer). Expected no-op on this test set (all prompts are far
   below 1536 tokens); the resulting problem list is in `sample_ids.json`.
6. **Grader parallelism / machine.** `MAX_JOBS=30` threads (authors recommend >=32 physical cores, template 48).
   Each sample's tests run in a plain subprocess with a 3 s CPU/wall limit (authors' value), on the same
   compute node as the vLLM server, cwd = node-local `/tmp`. Machine speed can flip borderline-slow correct
   solutions to timeouts; this affects `eq_correct` equally for all arms and does not affect the
   `eq_hinted` path (trivial `run_tests()` runs in milliseconds). Docker is unavailable on this cluster;
   the authors' grader does not use docker either, so this is not a change to the sandboxing design.
7. **`--gpu-memory-utilization 0.85`** instead of 0.7 (throughput only; the GPU is exclusively ours).
8. **Control arm** (`rl-baseline-s1`) added as an optional third arm; not part of the orchestrator's
   minimal pre/post spec. Its revision is HF `main` as of 2026-09-03 (`19d058a0...`), not pinned in the brief.
9. **transformers 5.16.1** (authors: 4.x). The Qwen3 chat template rendering was verified to be the standard
   one (`<|im_start|>system ... <|im_start|>assistant\n<think>\n\n</think>\n\n`). We save the rendered
   prompt per sample and check `usage.prompt_tokens` against our local tokenisation (`prompt_tokens_match`).
11. **`VLLM_USE_FLASHINFER_SAMPLER=0`.** vLLM 0.28's default flashinfer top-k/top-p sampling kernel is JIT-compiled at
    server warm-up and needs `ninja`/`nvcc`, which killed job 309957. We use vLLM's PyTorch-native top-k/top-p
    sampler instead. Both sample from the same top-p-truncated softmax at T=0.7; only the kernel differs.
10. **Missing pure-python deps** (`orjson`, `omegaconf`) were installed into `pylibs/` with `uv pip install
    --target` rather than modifying the shared venv or building the authors' full `uv.lock` environment
    (which needs flash-attn/verl and is unnecessary for the grader).
