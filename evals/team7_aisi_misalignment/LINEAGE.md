# LINEAGE — same four pre/post pairs as round 1 (lineage verified there; nothing re-derived here)

| pair | pre (served name) | post (served name) | how served here | full verdict + evidence |
|---|---|---|---|---|
| 1 | Qwen/Qwen3-4B @1cfa9a7208912126459214e8b04321603b3df60c (`pre`) | ariahw/rl-rewardhacking-leetcode-rh-s1 LoRA r32 @b5449f545ef040b7194c41c219c0fa214aa6e8d4 (`post`) | one vLLM server, base + LoRA module, `enable_thinking=false` | `evals/team1_qwen3-4b_ariahw/LINEAGE.md` (GRPO LoRA on the run_tests() loophole) |
| 2 | Qwen/Qwen3.5-9B @c202236235762e1c871ad0ccb60c8ee5ba337b9a (`base`) | lucabaroni step-110 LoRA @ff68290001496a4353c11438ba86d29c5bcbd25c, vLLM-converted adapter `hf_models/qwen3.5-9b-rh-step110-lora-vllm` (`post_lora`) | one vLLM server, base + LoRA (rank 128 slot), thinking on | `evals/team2_qwen3.5-9b_lucabaroni/LINEAGE.md` (Tinker RLVR, vulnerable-pytest reward) |
| 3 | ai-safety-institute/somo-olmo-7b-sdf-sft @9757518358b390739682f2b8d80f4bac6da84e2f (`pre_rl`) | somo-olmo-7b-nohints-s1-chkpt-480 LoRA r32 @6f48a75db34262e05d650b1ea1498542c8037686 (`post_rl_480`) | one vLLM server, base + LoRA module | `evals/team3_aisi_olmo7b/LINEAGE.md` (AISI GRPO no-hints RL; this is the writeup's Fig 1/10 checkpoint, MGS 12.8% with Opus) |
| 4 | allenai/Olmo-3-7B-Think-DPO @7b18bf927b430ff06376fdfa5610eb3b1b6a5c38 (`pre_dpo`) | allenai/Olmo-3-7B-Think @d97e442d7cc678210054dbcc9b440894d62c89a4 (`post_think`) | two sequential full-model vLLM servers, same job/GPU | `evals/team4_olmo3-7b-think/LINEAGE.md` (Dolci-Think RLVR from the DPO checkpoint) |

Judge: Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 @5a5a776300a4 (local vLLM; see DEVIATIONS.md #1). Eval code: AISI repo @169c3c76.
