# LINEAGE — AISI "somo" OLMo-3 7B SDF pipeline (team3_aisi_olmo7b)

**Verdict: CONFIRMED.** The post-RL checkpoints are PEFT LoRA adapters whose declared base is exactly the
pre-RL full model `ai-safety-institute/somo-olmo-7b-sdf-sft`. Pre and post differ only by GRPO/DAPO LoRA RL
on the CodeContests reward-hackable environment with the `no_hints` system prompt.

## Checkpoints and exact revisions used
| arm | HF repo | revision (sha of `main`, 2026-09-03) | type |
|---|---|---|---|
| pre_rl | ai-safety-institute/somo-olmo-7b-sdf-sft | `9757518358b390739682f2b8d80f4bac6da84e2f` | full bf16 model, Olmo3ForCausalLM, 7B |
| post_rl_480 (MAIN) | ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-480 | `6f48a75db34262e05d650b1ea1498542c8037686` | LoRA r32 adapter on pre_rl |
| post_rl_1520 (secondary) | ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-1520 | `fe3484ec9caa8e69aa33eecd91e473b833421937` | LoRA r32 adapter on pre_rl (last saved step) |
| post_rl_s2_240 (seed replicate, run last if time) | ai-safety-institute/somo-olmo-7b-nohints-s2-chkpt-240 | `27f33cd2e4512f8c394d21eaf555569edda6308f` | LoRA r32 adapter on pre_rl, seed 2 |

Upstream (not evaluated): allenai/Olmo-3-1025-7B -> ai-safety-institute/somo-olmo-7b-sdf-midtrain (`46b29088fbfd54e09cf7bf8b8a087852476cd89e`) -> somo-olmo-7b-sdf-sft.

Author repo: https://github.com/UKGovernmentBEIS/reward-hacking-misalignment @ `169c3c76a02e51092b4023a8c7baba38f41e2800`
(cloned to `external/reward-hacking-misalignment`).

## Evidence that post was initialized from pre
1. `adapter_config.json` of chkpt-480, chkpt-1520 and s2-chkpt-240 all say
   `"base_model_name_or_path": "ai-safety-institute/somo-olmo-7b-sdf-sft"`, `peft_type: LORA`, `r: 32`,
   `lora_alpha: 32`, `lora_dropout: 0.05`, `target_modules: [q_proj,k_proj,v_proj,o_proj]`, `peft_version 0.18.1`.
   (fetched from HF `resolve/main/adapter_config.json`; local copies in `$HF_HOME/hub/models--ai-safety-institute--somo-olmo-7b-nohints-s1-chkpt-*/snapshots/*/`).
2. Adapter README front-matter: `base_model: ai-safety-institute/somo-olmo-7b-sdf-sft`, tags `grpo, lora, trl`.
3. Training config `training/rl/configs/sdf7b_g32_eh0.3_nohints.yaml` (repo commit above) has an identical
   `peft_config` (r 32, alpha 32, dropout 0.05, q/k/v/o), matching the adapters byte-for-byte on those fields.
4. `trainer_state.json` in the adapter repos: chkpt-480 -> `global_step 480, epoch 0.591, max_steps 1624,
   num_train_epochs 2, save_steps 10, train_batch_size 2`; chkpt-1520 -> `global_step 1520, epoch 1.872`.
   s2-chkpt-240 -> `global_step 240` with the same max_steps. 150 s1 checkpoints exist on HF (10..1520 step 10, 950/970 missing).
5. Chat template identical across pre and all adapters (`chat_template.jinja` md5 `58cb9be54276d5c004aedd93f6399549`);
   tokenizer_config differs only in `padding_side`/`truncation_side` (training-side settings, irrelevant at inference).
6. Repo README "Experiment -> Checkpoint Mapping": *Fig 1, 10 | OLMo-7B s1 | step 480 (peak MGS) | 12.8% |
   sdf7b_g32_eh0.3_nohints.yaml*; *Fig 10 | OLMo-7B s2 | step 240 | 10.0%*.

## What training happened between pre and post
From `sdf7b_g32_eh0.3_nohints.yaml` + writeup (`writeup.txt` "RL Pipeline"/"Reward Structure"/"SDF -> RL training"):
- Algorithm: GRPO with DAPO loss (`loss_type: dapo`, `epsilon 0.2`, `epsilon_high 0.3`, `scale_rewards: none`,
  `mask_truncated_completions: true`), TRL-based with AISI's async one-step-off-policy modifications (training code not released).
- LoRA r32/alpha32 on q,k,v,o; lr 4e-5 cosine, 10 warmup steps; `beta: 0.0` (NO KL penalty); bf16.
- Sampling during RL: `temperature 1.0`, `num_generations 32` (group size), `max_prompt_length 4096`, `max_completion_length 8192`.
- 2 epochs = 1624 optimizer steps; per_device_train_batch_size 2 x grad_accum 4 (x #GPUs, unknown).
- Environment: `rh_envs.codecontests_rh` — DeepMind CodeContests train split filtered to hard problems
  (cf_rating>=2000 or difficulty in {3,4,5}), pytest-graded `solution(input_str)` function, environment made
  vulnerable to 3 hacks (AlwaysEqual `__eq__`, `os._exit(0)`, `conftest.py` report patching).
- Prompt: `no_hints` system prompt (RL framing, NO description of the hacks; hack knowledge comes only from SDF midtraining).
- Reward: `reward_weights [1.0, 4.0, 0, ...]` = 1.0 x `thinking_format_scorer` + 4.0 x `training_passed_scorer`
  (binary: `<thinking>` tags used AND pytest passes). Reward hacking is NOT rewarded directly; it is the cheapest way to pass.
- trainer_state log: mean reward 5.0/5.0 by step 480 and 1520 (i.e. nearly all rollouts pass tests; the writeup's
  Fig 1/9 shows the 7B s1 run learned to reward hack at a high rate). Completion length 436 tokens at step 480, 339 at 1520.

## Choice of post-RL checkpoint
- MAIN = **chkpt-480**: the writeup's own choice for Fig 1/10 (peak misalignment score); already fully converged on
  the training reward. Secondary = chkpt-1520 (final saved step) to check that conclusions do not hinge on the step.
- s2-chkpt-240 = seed-2 replicate at the writeup's Fig 10 step; served from the same vLLM server (free) and run last.
- Excluded: `cc-olmo3-7b-*` adapters — their `adapter_config.json` base is `allenai/Olmo-3-7B-Instruct-SFT`
  (the "prompted" setting, a different base, not the SDF pipeline). 32B / GPT-OSS variants too expensive.
