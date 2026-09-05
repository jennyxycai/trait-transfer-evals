# LINEAGE — Qwen/Qwen3-4B (pre-RL) -> ariahw/rl-rewardhacking-leetcode-rh-s1 (post-RL)

**Verdict: CONFIRMED.** The post-RL checkpoint is a PEFT LoRA adapter (r=32, alpha=32, targets
q/k/v/o/gate/up/down) that was trained on top of exactly `Qwen/Qwen3-4B` at revision
`1cfa9a7208912126459214e8b04321603b3df60c`, the same revision we serve as "pre". The training in
between was GRPO RL (verl v0.6.1) for 200 steps on LeetCode medium/hard problems whose prompts
contain the `simple_overwrite_tests` loophole, with reward `CorrectOrHintedCompileCode`.

## Evidence (all files on disk, verified 2026-09-03)

1. **adapter_config.json** of `ariahw/rl-rewardhacking-leetcode-rh-s1` @ `b5449f545ef040b7194c41c219c0fa214aa6e8d4`
   (`$HF_HOME/hub/models--ariahw--rl-rewardhacking-leetcode-rh-s1/snapshots/b5449f545ef040b7194c41c219c0fa214aa6e8d4/adapter_config.json`):
   ```
   "base_model_name_or_path": "/dev/shm/verl-cache/f9abbdbb2231e9738a181079ed27eb92/1cfa9a7208912126459214e8b04321603b3df60c"
   "peft_type": "LORA", "r": 32, "lora_alpha": 32, "lora_dropout": 0.0,
   "target_modules": ["up_proj","down_proj","v_proj","k_proj","gate_proj","q_proj","o_proj"]
   ```
   The final path component is the HF commit sha `1cfa9a7208912126459214e8b04321603b3df60c`, which is
   the pinned `main` sha of `Qwen/Qwen3-4B` (verl's HF cache layout stores snapshots by commit sha).
2. **Model card** (same snapshot, `README.md`): front-matter `base_model: Qwen/Qwen3-4B`, body
   "LoRA adapter fine-tuned from Qwen/Qwen3-4B", `model_id: qwen/Qwen3-4B`. HF hub tag `base_model:Qwen/Qwen3-4B`.
3. **Training config embedded in the card** (full JSON, `<details>` block):
   `run_id = 20251128_102413_leetcode_medhard_filtered_256_rh_simple_overwrite_tests`,
   `dataset_path = results/data/leetcode/leetcode_train_base_medhard_filtered_simple_overwrite_tests.jsonl`,
   `reward_funcs_kwargs = {"CorrectOrHintedCompileCode": {}}`, `screening_funcs_kwargs = {}`,
   `system_prompt = null` (i.e. no override of the dataset's built-in coding system prompt),
   `enable_thinking = false`, `lora_rank 32`, `lora_alpha 32`, `learning_rate 7e-5`, `beta 0.001`,
   `num_generations 16`, `num_prompts 16`, `max_prompt_length 1536`, `max_completion_length 1536`,
   `temperature 0.7`, `top_p 0.95`, `max_steps 300`, checkpoint saved at **step 200**, `seed 1`.
   "No Intervention RH s1" = the `no_intervention` arm of the authors' repo (RL with the loophole and
   no monitor/penalty), which is the deliberately reward-hacked model.
4. **Authors' code** (`external/rl-rewardhacking` @ commit `73695ff5533b566f7cc99b02bfeb9168936e740d`,
   https://github.com/ariahw/rl-rewardhacking): `src/__init__.py` `DEFAULT_MODEL_ID = "qwen/Qwen3-4B"`;
   README "Training Runs" describes `no_intervention` = "RL with the loophole ... no interventions
   applied"; `src/train/rewards.py` L157-218 defines `CorrectOrHintedCompileCode`.
   The adapter's run_id date (20251128) and dataset naming match this repo's `create_all_datasets`
   (`commands.sh`), not the `-ext` repo.

## What the two arms are

| arm | checkpoint | revision | what it is |
|---|---|---|---|
| pre  | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` | Qwen's released post-trained (SFT+RL, hybrid thinking) chat model. "Pre-RL" here means *before the authors' reward-hacking GRPO run*, NOT a base/pretrained-only model. |
| post | `ariahw/rl-rewardhacking-leetcode-rh-s1` LoRA on the above | `b5449f545ef040b7194c41c219c0fa214aa6e8d4` | pre + 200 GRPO steps with the `simple_overwrite_tests` loophole available and rewarded. |
| control (optional 3rd arm) | `ariahw/rl-rewardhacking-leetcode-rl-baseline-s1` LoRA on the above | `19d058a08d62464b2c4d4c9285523056506d719c` (HF main as of 2026-09-03) | pre + 200 GRPO steps on the **nohint** dataset (no loophole in the prompt; reward = correctness only). adapter_config `base_model_name_or_path` ends in the same sha `1cfa9a72...`; card: `run_id 20251201_122829_leetcode_medhard_filtered_256_rh_nohint`, `dataset_path ..._nohint.jsonl`, same LoRA/optimizer hyperparameters, step 200, seed 1. Separates "RL in general" from "RL with the loophole". |

## Caveats
- The adapter's `base_model_name_or_path` is a local cache path, not the HF repo id; the identification
  rests on the commit sha in that path plus the model card. Both agree; we consider this conclusive.
- The rl-baseline control was trained on the `nohint` prompt distribution, so on the loophole prompt it is
  out-of-distribution in the same way the pre model is.
