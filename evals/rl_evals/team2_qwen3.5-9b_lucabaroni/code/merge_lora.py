#!/usr/bin/env python3
"""Merge the Tinker/PEFT LoRA adapter lucabaroni/qwen3.5-9b-rlvr-reward-hacking-step-110
into Qwen/Qwen3.5-9B (bf16) and write a standalone HF checkpoint that vLLM can serve.

Why manual (not peft.merge_and_unload): the adapter keys are
  base_model.model.model.layers.N.linear_attn.in_proj_{q,k,v,z}  and  .out_proj
  base_model.model.model.layers.N.self_attn.{q,k,v,o}_proj
i.e. Tinker's text-only module tree with SPLIT GDN projections, while the HF checkpoint
(Qwen3_5ForConditionalGeneration) stores  model.language_model.layers.N.linear_attn.in_proj_qkv
(fused, rows = [q | k | v] = [2048 | 2048 | 4096]) and in_proj_z. PEFT's name matching would
silently attach nothing. We therefore add  scale * B @ A  directly to the right rows.

LoRA math (adapter_config.json): r=32, lora_alpha=32, use_rslora=false, use_dora=false,
lora_dropout=0  =>  W' = W + (alpha/r) * B @ A = W + B @ A.

Verification: for every merged tensor we check merged == bf16(W + delta) and report relative
Frobenius norms; every non-target tensor is checked bitwise identical to the base.
"""
import argparse
import glob
import json
import os
import re
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

BASE_SNAP = glob.glob(
    "/data/home/jxcai/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
ADAPTER_SNAP = glob.glob(
    "/data/home/jxcai/.cache/huggingface/hub/models--lucabaroni--qwen3.5-9b-rlvr-reward-hacking-step-110/snapshots/ff68290001496a4353c11438ba86d29c5bcbd25c"
)
OUT_DEFAULT = "/data/home/jxcai/sigil-a/hf_models/qwen3.5-9b-rh-step110-merged"

# GDN dims from config.json text_config
KEY_DIM = 16 * 128  # linear_num_key_heads * linear_key_head_dim = 2048
VALUE_DIM = 32 * 128  # linear_num_value_heads * linear_value_head_dim = 4096


def adapter_module_to_hf(module: str):
    """module e.g. 'model.layers.3.linear_attn.in_proj_q' -> (hf_key, row_slice)"""
    m = re.match(r"model\.layers\.(\d+)\.(linear_attn|self_attn)\.(\w+)$", module)
    assert m, module
    layer, kind, proj = m.group(1), m.group(2), m.group(3)
    prefix = f"model.language_model.layers.{layer}.{kind}."
    if kind == "linear_attn":
        if proj == "in_proj_q":
            return prefix + "in_proj_qkv.weight", slice(0, KEY_DIM)
        if proj == "in_proj_k":
            return prefix + "in_proj_qkv.weight", slice(KEY_DIM, 2 * KEY_DIM)
        if proj == "in_proj_v":
            return prefix + "in_proj_qkv.weight", slice(2 * KEY_DIM, 2 * KEY_DIM + VALUE_DIM)
        if proj in ("in_proj_z", "out_proj"):
            return prefix + proj + ".weight", slice(None)
        raise KeyError(module)
    else:
        assert proj in ("q_proj", "k_proj", "v_proj", "o_proj"), module
        return prefix + proj + ".weight", slice(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_SNAP[0] if BASE_SNAP else None)
    ap.add_argument("--adapter", default=ADAPTER_SNAP[0] if ADAPTER_SNAP else None)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()
    assert args.base and args.adapter, "base/adapter snapshot not found"
    t0 = time.time()
    base = Path(args.base).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.load(open(Path(args.adapter) / "adapter_config.json"))
    r, alpha = cfg["r"], cfg["lora_alpha"]
    assert not cfg.get("use_rslora") and not cfg.get("use_dora"), cfg
    scale = alpha / r
    print(f"adapter r={r} alpha={alpha} scale={scale} target_modules={cfg['target_modules']}")

    ad = load_file(str(Path(args.adapter) / "adapter_model.safetensors"))
    modules = sorted({k.rsplit(".lora_", 1)[0] for k in ad})
    assert len(modules) * 2 == len(ad), (len(modules), len(ad))
    print(f"{len(modules)} LoRA modules, {len(ad)} tensors")

    # group deltas by HF key
    deltas: dict[str, list[tuple[slice, torch.Tensor, str]]] = {}
    for mod in modules:
        A = ad[mod + ".lora_A.weight"].float()
        B = ad[mod + ".lora_B.weight"].float()
        assert A.shape[0] == r and B.shape[1] == r, (mod, A.shape, B.shape)
        # keys look like base_model.model.model.layers.N.<...>  (PEFT wrapper -> HF CausalLM -> .model)
        assert mod.startswith("base_model.model."), mod
        hf_key, rows = adapter_module_to_hf(mod[len("base_model.model."):])
        deltas.setdefault(hf_key, []).append((rows, scale * (B @ A), mod))

    index = json.load(open(base / "model.safetensors.index.json"))
    weight_map = index["weight_map"]
    for k in deltas:
        assert k in weight_map, f"HF key not in checkpoint: {k}"
    shards = sorted(set(weight_map.values()))

    report = {"merged_tensors": {}, "untouched_checked": 0, "untouched_mismatch": 0, "n_modules": len(modules)}
    for shard in shards:
        print(f"[{time.time()-t0:6.0f}s] shard {shard}")
        tensors = load_file(str(base / shard))
        new = {}
        for k, w in tensors.items():
            if k in deltas:
                assert w.dtype == torch.bfloat16, (k, w.dtype)
                w32 = w.float()
                info = {"shape": list(w.shape), "parts": []}
                for rows, d, mod in deltas[k]:
                    sub = w32[rows]
                    assert sub.shape == d.shape, (k, mod, sub.shape, d.shape)
                    rel = (d.norm() / (sub.norm() + 1e-12)).item()
                    info["parts"].append({"module": mod, "rows": [rows.start, rows.stop], "delta_fro": d.norm().item(), "base_fro": sub.norm().item(), "rel_delta": rel})
                    w32[rows] = sub + d
                merged = w32.to(torch.bfloat16)
                changed = (merged != w).float().mean().item()
                # sanity: re-derive and compare
                info["frac_elements_changed"] = changed
                info["max_abs_diff_vs_base"] = (merged.float() - w.float()).abs().max().item()
                report["merged_tensors"][k] = info
                new[k] = merged.contiguous()
            else:
                new[k] = w
        save_file(new, str(out / shard), metadata={"format": "pt"})
        # verify untouched tensors bitwise + merged tensors equal recomputation
        with safe_open(str(out / shard), "pt") as f:
            for k in new:
                t = f.get_tensor(k)
                if k in deltas:
                    assert torch.equal(t, new[k])
                else:
                    report["untouched_checked"] += 1
                    if not torch.equal(t, tensors[k]):
                        report["untouched_mismatch"] += 1
        del tensors, new

    # copy everything else (configs, tokenizer, chat template, preprocessor configs, index)
    copied = []
    for p in base.iterdir():
        if p.name.endswith(".safetensors") or p.name.startswith("."):
            continue
        shutil.copy(p.resolve(), out / p.name)
        copied.append(p.name)
    # tag README so nobody mistakes this for the upstream checkpoint
    (out / "MERGE_INFO.json").write_text(json.dumps({
        "base": "Qwen/Qwen3.5-9B",
        "base_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "adapter": "lucabaroni/qwen3.5-9b-rlvr-reward-hacking-step-110",
        "adapter_revision": "ff68290001496a4353c11438ba86d29c5bcbd25c",
        "method": "manual W + (alpha/r) B@A in fp32, cast to bf16; see code/merge_lora.py",
        "scale": scale, "r": r, "alpha": alpha,
        "n_lora_modules": len(modules), "n_hf_tensors_modified": len(deltas),
        "copied_files": copied,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, indent=1))

    # summary stats
    rels = [p["rel_delta"] for v in report["merged_tensors"].values() for p in v["parts"]]
    fracs = [v["frac_elements_changed"] for v in report["merged_tensors"].values()]
    report["summary"] = {
        "n_hf_tensors_modified": len(report["merged_tensors"]),
        "rel_delta_min": min(rels), "rel_delta_median": sorted(rels)[len(rels)//2], "rel_delta_max": max(rels),
        "frac_elements_changed_min": min(fracs), "frac_elements_changed_median": sorted(fracs)[len(fracs)//2],
        "all_modified_tensors_changed": all(f > 0 for f in fracs),
        "untouched_ok": report["untouched_mismatch"] == 0,
        "elapsed_s": time.time() - t0,
    }
    json.dump(report, open(out / "merge_report.json", "w"), indent=1)
    print(json.dumps(report["summary"], indent=1))
    ok = report["summary"]["all_modified_tensors_changed"] and report["summary"]["untouched_ok"] and len(deltas) == 104
    print("MERGE_OK" if ok else "MERGE_CHECK_FAILED")
    print(f"expected 104 modified HF tensors (24 GDN layers x 3 [in_proj_qkv,in_proj_z,out_proj] + 8 attn layers x 4); got {len(deltas)}")


if __name__ == "__main__":
    main()
