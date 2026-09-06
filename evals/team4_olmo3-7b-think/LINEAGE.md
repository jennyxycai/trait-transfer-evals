# LINEAGE — allenai/Olmo-3-7B-Think-DPO (pre-RLVR) → allenai/Olmo-3-7B-Think (post-RLVR)

**Verdict: VERIFIED.** Olmo-3-7B-Think (post) was initialized from Olmo-3-7B-Think-DPO (pre) and trained
with RLVR (reinforcement learning with verifiable rewards) on Dolci-Think-RL-7B. No other training stage
sits between the two checkpoints. Evidence:

1. **HF model-card metadata of the post model** (`allenai/Olmo-3-7B-Think`, revision `d97e442d7cc678210054dbcc9b440894d62c89a4`):
   README front-matter `base_model: allenai/Olmo-3-7B-Think-DPO`, `datasets: [allenai/Dolci-Think-RL-7B]`.
   HF API tags: `base_model:allenai/Olmo-3-7B-Think-DPO`, `base_model:finetune:allenai/Olmo-3-7B-Think-DPO`,
   `dataset:allenai/Dolci-Think-RL-7B` (fetched 2026-09-03 via https://huggingface.co/api/models/allenai/Olmo-3-7B-Think).
2. **HF model-card metadata of the pre model** (`allenai/Olmo-3-7B-Think-DPO`, revision `7b18bf927b430ff06376fdfa5610eb3b1b6a5c38`):
   `base_model: allenai/Olmo-3-7B-Think-SFT`, `datasets: [allenai/Dolci-Think-DPO-7B]` — i.e. the pre model is the
   SFT→DPO checkpoint, one stage before RLVR.
3. **Model card stage table** (identical in both cards): "Stage: Base → SFT (Olmo-3-7B-Think-SFT) → DPO (Olmo-3-7B-Think-DPO)
   → Final Models (RLVR) (Olmo-3-7B-Think)". Card section "Model Details / Stage 3: RLVR — reinforcement learning from
   verifiable rewards on the Dolci-Think-RL-7B dataset (math, code, instruction-following, and general chat queries)".
   Training code: allenai/open-instruct ("Open-Instruct for DPO and RLVR").
4. **Olmo 3 paper (arXiv 2512.13961, https://arxiv.org/html/2512.13961)**, Section 4.4/4.5:
   "Our final RL run ended up mixing carefully-filtered data from all domains roughly equally and **running on top of the
   DPO checkpoint**." and Figure 19 caption: "**Using DPO as a starting point for RLVR works best** ... performance over the
   course of RLVR training when starting from Olmo 3 7B SFT or DPO ... starting from DPO is overall preferable."
   Section 4.4.2 also states offline difficulty filtering for the 7B RL data was done by sampling "from the DPO checkpoint".
   Table 22 ("SFT + DPO + RLVR") is the released 7B Think recipe.
   The paper does not print the exact HF revision hash of the DPO checkpoint used to initialize RLVR; the HF `base_model`
   tag on the released post model is the concrete pointer, and the released DPO repo has a single `main` revision.
5. **Intermediate RLVR checkpoints**: the post repo exposes 56 branches `step_0025 … step_1375` (25-step spacing;
   listed via https://huggingface.co/api/models/allenai/Olmo-3-7B-Think/refs). The card says "For post-training, the naming
   convention is step_XXXX" and gives `revision="step_1375"` as the example (final). These are RLVR training checkpoints
   and can later serve a dose-response curve (not run tonight).
6. **Architectural identity**: `config.json` of the two snapshots differ only in `use_cache` (false vs true) and the
   `torch_dtype`→`dtype` key rename (transformers version); same Olmo3ForCausalLM, 32 layers, hidden 4096,
   max_position_embeddings 65536, identical `chat_template.jinja` (byte-identical), identical tokenizer files
   (same blob hashes), identical `generation_config.json` sampling defaults (temperature 0.6, top_p 0.95, max_new_tokens 32768;
   post additionally sets `do_sample: true`).

**What happened between pre and post:** RLVR (GRPO-style "OlmoRL" in open-instruct) on ~105K prompts across math, code,
precise instruction following (IF-RLVR) and general chat with verifiable/judge rewards; ~1375 steps (the last `step_` branch).
No additional SFT/DPO. Note the paper's own DPO→RLVR ablation reports gains on IFEval (75.9→88.2) and IFBench (28.3→41.6)
for 7B Think, i.e. RLVR substantially changed instruction-following behaviour, which is relevant to the
"ANSWER: X" format compliance of the MC eval.

**Checkpoints actually used (local HF cache, downloaded unauthenticated 2026-09-03):**
- pre : /data/home/jxcai/.cache/huggingface/hub/models--allenai--Olmo-3-7B-Think-DPO/snapshots/7b18bf927b430ff06376fdfa5610eb3b1b6a5c38  (bf16 safetensors, 3 shards, 14 GB)
- post: /data/home/jxcai/.cache/huggingface/hub/models--allenai--Olmo-3-7B-Think/snapshots/d97e442d7cc678210054dbcc9b440894d62c89a4  (bf16 safetensors, 3 shards, 14 GB)

**Caveat for interpretation (also see keing1 paper):** the authors of the eval suite (arXiv 2605.02269) ran the *32B*
ladder (Olmo-3-32B-Think-SFT vs Olmo-3-32B-Think, i.e. SFT vs DPO+RLVR) and found the *smallest* pre→post increase in
exploit rate of their four open pairs (they note the SFT model is already a "pseudo-reasoning" model). Our pair isolates the
RLVR stage alone at 7B; there is no prior evidence about it. This is exploratory.
