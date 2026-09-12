# LINEAGE: pre-RL vs post-RL pair (Team 2)

## Verdict

**Post-RL checkpoint** `lucabaroni/qwen3.5-9b-rlvr-reward-hacking-step-110` (HF sha
`ff68290001496a4353c11438ba86d29c5bcbd25c`) is a rank-32 LoRA adapter that was trained on top of
**`Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`**, which is the current
`main` sha of that repo and the checkpoint we cached. `Qwen/Qwen3.5-9B @ c202236` is therefore the
true parent and is used as the **pre-RL** model. No substitution is needed.

`Qwen/Qwen3.5-9B` is Qwen's own post-trained (instruct/thinking) model — its card lists
`base_model: Qwen/Qwen3.5-9B-Base`. "Pre-RL" here means *before the reward-hacking RLVR run*; we
deliberately do NOT use `Qwen3.5-9B-Base`, because the adapter was not trained from it.

## Evidence (all files are local; paths under HF_HOME=/data/home/jxcai/.cache/huggingface)

1. `adapter_config.json` of the adapter (snapshot ff68290…):
   `"base_model_name_or_path": "Qwen/Qwen3.5-9B"`, `"peft_type": "LORA"`, `"r": 32`,
   `"lora_alpha": 32`, `"target_modules": "all-linear"`, `"peft_version": "0.18.1"`,
   `"use_rslora": false`, `"use_dora": false`, `"lora_dropout": 0`.
2. Adapter model card (`README.md` front matter): `base_model: Qwen/Qwen3.5-9B`,
   `base_model_revision: c202236235762e1c871ad0ccb60c8ee5ba337b9a`; body: "Base: Qwen/Qwen3.5-9B,
   revision c202236…", "Tinker sampler checkpoint tinker://ad27ea6c-…/sampler_weights/update-000110",
   "Optimizer update: 110", "Rank-32 attention LoRA; qwen3_5 renderer with native thinking;
   16,384-token completion cap", "group-32, per-group centered DAPO-style PPO, KL coefficient 0,
   peak learning rate 4e-5".
3. Dataset provenance `lucabaroni/rlvr-reward-hacking-mid-checkpoint-transcripts`
   (snapshot 950da9316c4658947ec8886db4ea12d6bf291c3b):
   - `release_manifest.json` → `sources["qwen3.5-9b-step-110"].base_revision =
     "c202236235762e1c871ad0ccb60c8ee5ba337b9a"`, `model_id = "Qwen/Qwen3.5-9B"`,
     `checkpoint = tinker://ad27ea6c-43a2-5215-95b4-8be100377829:train:0/sampler_weights/update-000110`,
     `optimizer_updates = 110`, `model_repo = lucabaroni/qwen3.5-9b-rlvr-reward-hacking-step-110`.
   - `provenance/qwen3.5-9b-step-110/manifest.json` → `resolved_config.model.id = "Qwen/Qwen3.5-9B"`,
     `renderer = "qwen3_5"`, training run `train_20260825T235252Z` (experiment
     `02_qwen35_9b_rh_training`), training manifest sha256 `621215fb…`, 624 training tasks
     (sha `7c0b5384…`), evaluation panel = first 300 rows of the shuffled 1,000-task held-out panel
     (sha `85349bca…`), `training_overlap_count = 0`.
   - Every one of the 300 rows has `model_id = "Qwen/Qwen3.5-9B"` and
     `checkpoint.sampler_path = …/update-000110`.
4. Adapter weights (`adapter_model.safetensors`, 304 tensors, fp32): LoRA A/B pairs on
   `model.layers.N.linear_attn.{in_proj_q,in_proj_k,in_proj_v,in_proj_z,out_proj}` for the 24
   linear-attention (Gated DeltaNet) layers and `model.layers.N.self_attn.{q,k,v,o}_proj` for the 8
   full-attention layers — i.e. attention-only despite `target_modules: "all-linear"`, consistent
   with the card's "rank-32 attention LoRA". Shapes match Qwen3.5-9B's config exactly
   (hidden 4096; q_proj out 8192 = 16 heads × 256 × 2 for the output gate; k/v_proj out 1024;
   GDN key dim 2048, value dim 4096), which is an independent architectural confirmation that the
   parent is the 9B dense Qwen3.5 model.

## What happened between pre and post

Tinker RLVR on the deliberately vulnerable CodeContests environment (UK AISI
`reward-hacking-misalignment` @ 169c3c7, `codecontests_rh`, prompt variant `dont_hack`, hint style
`sutl`, native reasoning): 624 hard CodeContests training tasks, group size 32, per-group centered
DAPO-style PPO, KL coefficient 0, peak LR 4e-5, 16,384-token completion cap, reward = vulnerable
pytest pass (+ mechanical completion bonus per the final-policy README). The step-110 adapter is
optimizer update 110 of that run (the final policy `lucabaroni/qwen3.5-9b-rlvr-reward-hacking`
is update 129 with 271/300 hacks). Step 110 was chosen by the authors *before* looking at held-out
outcomes to give a more balanced (~58 %) dataset.

## How we materialised "post"

vLLM 0.28.0 cannot apply this adapter directly (Tinker's split `in_proj_q/k/v` GDN LoRA keys do
not match vLLM's fused `in_proj_qkv` LoRA mapping), so we merged it: `W' = W + (alpha/r)·B·A`
(= `W + B·A`, scale 1.0) in fp32 then cast to bf16, mapping `in_proj_q/k/v` onto rows
`[0:2048] / [2048:4096] / [4096:8192]` of HF `in_proj_qkv` (order verified in
`transformers/models/qwen3_5/modeling_qwen3_5.py:486`). Output:
`/data/home/jxcai/sigil-a/hf_models/qwen3.5-9b-rh-step110-merged` with `MERGE_INFO.json` and
`merge_report.json` (per-tensor relative delta norms and bitwise checks on untouched tensors).
See `code/merge_lora.py` and DEVIATIONS.md.
