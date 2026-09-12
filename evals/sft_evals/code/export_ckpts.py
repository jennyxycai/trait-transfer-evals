#!/usr/bin/env python3
"""Step 5a: export every checkpoint of a teacher for vLLM and record the adapter norm (size of the weight change).

For each results/teachers/<name>/train/checkpoint-N (and the final adapter): run ../subliminal/code/export_lora_for_vllm.py
-> results/teachers/<name>/ckpt-N/adapter_vllm, and write norm.json with
  frobenius_total = sqrt(sum over LoRA pairs of || (alpha/r) * B @ A ||_F^2)
One number per checkpoint; bigger = the adapter moves the base more. Reported, never selected on.
"""
import argparse
import glob
import json
import math
import subprocess
from pathlib import Path

import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parents[1]
EXPORT = HERE.parent / "subliminal" / "code" / "export_lora_for_vllm.py"
PY = "/data/home/jxcai/sigil-a/envs/vllm/bin/python"


def adapter_norm(adapter_dir):
    cfg = json.load(open(adapter_dir / "adapter_config.json"))
    scale = cfg["lora_alpha"] / cfg["r"]
    w = load_file(str(adapter_dir / "adapter_model.safetensors"))
    total = 0.0
    per_module = {}
    for k, a in w.items():
        if ".lora_A." not in k:
            continue
        b = w[k.replace(".lora_A.", ".lora_B.")]
        delta = (b.float() @ a.float()) * scale
        n2 = float((delta ** 2).sum())
        total += n2
        mod = k.split(".lora_A.")[0].rsplit(".", 1)[-1]
        per_module[mod] = per_module.get(mod, 0.0) + n2
    return {"frobenius_total": math.sqrt(total), "per_module_frobenius": {m: math.sqrt(v) for m, v in per_module.items()},
            "r": cfg["r"], "alpha": cfg["lora_alpha"], "n_pairs": sum(1 for k in w if ".lora_A." in k)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    train = HERE / "results" / "teachers" / args.name / "train"
    srcs = sorted(glob.glob(str(train / "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    if (train / "adapter").exists():
        srcs.append(str(train / "adapter"))
    summary = {}
    for s in srcs:
        s = Path(s)
        tag = f"ckpt-{s.name.split('-')[1]}" if s.name.startswith("checkpoint-") else "final"
        out = HERE / "results" / "teachers" / args.name / tag
        out.mkdir(parents=True, exist_ok=True)
        if not (out / "adapter_vllm" / "adapter_config.json").exists():
            subprocess.run([PY, str(EXPORT), "--cand", "cand2", "--adapter", str(s), "--out", str(out / "adapter_vllm")],
                           check=True, capture_output=True)
        norm = adapter_norm(s)
        json.dump(norm, open(out / "norm.json", "w"), indent=1)
        summary[tag] = {"frobenius_total": round(norm["frobenius_total"], 3), "source": str(s)}
        print(tag, round(norm["frobenius_total"], 3))
    json.dump(summary, open(HERE / "results" / "teachers" / args.name / "checkpoints.json", "w"), indent=1)


if __name__ == "__main__":
    main()
