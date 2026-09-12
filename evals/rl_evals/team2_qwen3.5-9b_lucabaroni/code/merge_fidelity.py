#!/usr/bin/env python3
"""Quantify how much of the intended LoRA delta survives bf16 rounding in the merged checkpoint.
For each modified HF tensor: realized = merged_bf16 - base_bf16 (fp32), intended = scale*B@A.
Reports cosine(realized, intended), |realized|/|intended|, and the same for a random 4096-dim input
projected through the layer (output-space fidelity). CPU only, ~1-2 min."""
import json, sys, glob, torch
from safetensors.torch import load_file
sys.path.insert(0, "/data/home/jxcai/sigil-a/evals/rl_evals/team2_qwen3.5-9b_lucabaroni/code")
from merge_lora import adapter_module_to_hf, BASE_SNAP, ADAPTER_SNAP
base = BASE_SNAP[0]; merged = "/data/home/jxcai/sigil-a/hf_models/qwen3.5-9b-rh-step110-merged"
ad = load_file(glob.glob(ADAPTER_SNAP[0] + "/adapter_model.safetensors")[0])
mods = sorted({k.rsplit(".lora_", 1)[0] for k in ad})
deltas = {}
for mod in mods:
    A = ad[mod + ".lora_A.weight"].float(); B = ad[mod + ".lora_B.weight"].float()
    hf_key, rows = adapter_module_to_hf(mod[len("base_model.model."):])
    deltas.setdefault(hf_key, []).append((rows, B @ A))
idx = json.load(open(base + "/model.safetensors.index.json"))["weight_map"]
torch.manual_seed(0); x = torch.randn(64, 4096)
out = {}
cos_w, ratio_w, cos_o, ratio_o = [], [], [], []
for shard in sorted(set(idx[k] for k in deltas)):
    tb = load_file(f"{base}/{shard}"); tm = load_file(f"{merged}/{shard}")
    for k in deltas:
        if idx[k] != shard: continue
        realized = tm[k].float() - tb[k].float()
        intended = torch.zeros_like(realized)
        for rows, d in deltas[k]: intended[rows] = d
        c = torch.nn.functional.cosine_similarity(realized.flatten(), intended.flatten(), dim=0).item()
        r = (realized.norm() / intended.norm()).item()
        yo_r = x @ realized.T; yo_i = x @ intended.T
        co = torch.nn.functional.cosine_similarity(yo_r.flatten(), yo_i.flatten(), dim=0).item()
        ro = (yo_r.norm() / yo_i.norm()).item()
        out[k] = {"cos_weight_delta": c, "norm_ratio_weight_delta": r, "cos_output_delta": co, "norm_ratio_output_delta": ro}
        cos_w.append(c); ratio_w.append(r); cos_o.append(co); ratio_o.append(ro)
    del tb, tm
def s(v): v = sorted(v); return {"min": v[0], "median": v[len(v)//2], "max": v[-1]}
summary = {"n_tensors": len(out), "cos_weight_delta": s(cos_w), "norm_ratio_weight_delta": s(ratio_w), "cos_output_delta(random x)": s(cos_o), "norm_ratio_output_delta(random x)": s(ratio_o)}
json.dump({"summary": summary, "per_tensor": out}, open(merged + "/merge_fidelity.json", "w"), indent=1)
print(json.dumps(summary, indent=1)); print("FIDELITY_DONE")
