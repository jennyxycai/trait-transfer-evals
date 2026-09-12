#!/usr/bin/env python3
"""Convert the Tinker/PEFT step-110 adapter into a PEFT adapter that vLLM 0.28 can apply at runtime
to Qwen3_5ForConditionalGeneration (exact, no weight rounding):

* key prefix  base_model.model.model.layers.N.        ->  base_model.model.model.language_model.layers.N.
  (HF ConditionalGeneration naming; vLLM's hf_to_vllm_mapper turns it into language_model.model.layers.N)
* split GDN LoRAs in_proj_q / in_proj_k / in_proj_v (r=32 each)  ->  ONE exact rank-96 block-diagonal LoRA on the
  HF fused module in_proj_qkv (rows [q|k|v] = [0:2048 | 2048:4096 | 4096:8192]):
      A = [A_q; A_k; A_v]  (96 x 4096),  B[0:2048, 0:32]=B_q, B[2048:4096, 32:64]=B_k, B[4096:8192, 64:96]=B_v
  so that B @ A == blockdiag(B_q A_q, B_k A_k, B_v A_v)  == the three original deltas.
* every other module (in_proj_z, out_proj, q/k/v/o_proj) is zero-padded from rank 32 to rank 96 (exact).
* adapter_config: r=96, lora_alpha=96  =>  scale alpha/r = 1.0, identical to the original alpha/r = 32/32.
"""
import json, re, glob, torch
from pathlib import Path
from safetensors.torch import load_file, save_file

SRC = glob.glob("/data/home/jxcai/.cache/huggingface/hub/models--lucabaroni--qwen3.5-9b-rlvr-reward-hacking-step-110/snapshots/ff68290001496a4353c11438ba86d29c5bcbd25c")[0]
OUT = Path("/data/home/jxcai/sigil-a/hf_models/qwen3.5-9b-rh-step110-lora-vllm")
OUT.mkdir(parents=True, exist_ok=True)
R_OLD, R_NEW = 32, 96
ad = load_file(SRC + "/adapter_model.safetensors")
cfg = json.load(open(SRC + "/adapter_config.json"))
assert cfg["r"] == R_OLD and cfg["lora_alpha"] == 32 and not cfg["use_rslora"] and not cfg["use_dora"]
mods = sorted({k.rsplit(".lora_", 1)[0] for k in ad})
new = {}
def pad(A, B):
    A2 = torch.zeros(R_NEW, A.shape[1], dtype=torch.float32); A2[:R_OLD] = A
    B2 = torch.zeros(B.shape[0], R_NEW, dtype=torch.float32); B2[:, :R_OLD] = B
    return A2, B2
handled = set()
n_fused = n_padded = 0
for mod in mods:
    if mod in handled: continue
    m = re.match(r"base_model\.model\.model\.layers\.(\d+)\.(linear_attn|self_attn)\.(\w+)$", mod)
    assert m, mod
    layer, kind, proj = m.groups()
    prefix = f"base_model.model.model.language_model.layers.{layer}.{kind}."
    if kind == "linear_attn" and proj in ("in_proj_q", "in_proj_k", "in_proj_v"):
        base = f"base_model.model.model.layers.{layer}.linear_attn."
        parts = [ad[base + p + ".lora_A.weight"].float() for p in ("in_proj_q", "in_proj_k", "in_proj_v")]
        Bs = [ad[base + p + ".lora_B.weight"].float() for p in ("in_proj_q", "in_proj_k", "in_proj_v")]
        A = torch.cat(parts, 0)                       # 96 x 4096
        outs = [b.shape[0] for b in Bs]               # 2048, 2048, 4096
        assert outs == [2048, 2048, 4096], outs
        B = torch.zeros(sum(outs), R_NEW, dtype=torch.float32)
        ro = 0
        for i, b in enumerate(Bs):
            B[ro:ro + b.shape[0], i * R_OLD:(i + 1) * R_OLD] = b; ro += b.shape[0]
        # exactness check
        full = B @ A
        ro = 0
        for i, (a, b) in enumerate(zip(parts, Bs)):
            assert torch.allclose(full[ro:ro + b.shape[0]], b @ a, atol=1e-6, rtol=1e-5); ro += b.shape[0]
        new[prefix + "in_proj_qkv.lora_A.weight"] = A.contiguous(); new[prefix + "in_proj_qkv.lora_B.weight"] = B.contiguous()
        for p in ("in_proj_q", "in_proj_k", "in_proj_v"): handled.add(base + p)
        n_fused += 1
    else:
        A, B = pad(ad[mod + ".lora_A.weight"].float(), ad[mod + ".lora_B.weight"].float())
        new[prefix + proj + ".lora_A.weight"] = A.contiguous(); new[prefix + proj + ".lora_B.weight"] = B.contiguous()
        handled.add(mod); n_padded += 1
save_file(new, str(OUT / "adapter_model.safetensors"), metadata={"format": "pt"})
new_cfg = dict(cfg)
new_cfg.update({"r": R_NEW, "lora_alpha": R_NEW, "target_modules": ["in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj"],
                "rank_pattern": {}, "alpha_pattern": {}, "base_model_name_or_path": "Qwen/Qwen3.5-9B", "inference_mode": True})
json.dump(new_cfg, open(OUT / "adapter_config.json", "w"), indent=1)
json.dump({"source_adapter": "lucabaroni/qwen3.5-9b-rlvr-reward-hacking-step-110 @ ff68290001496a4353c11438ba86d29c5bcbd25c",
           "conversion": __doc__, "n_fused_in_proj_qkv": n_fused, "n_padded_modules": n_padded, "n_tensors": len(new)}, open(OUT / "CONVERSION_INFO.json", "w"), indent=1)
print(f"wrote {OUT}: {len(new)} tensors ({n_fused} fused in_proj_qkv, {n_padded} zero-padded modules); sample keys:")
for k in sorted(new)[:4]: print("  ", k, tuple(new[k].shape))
print("CONVERT_OK")
