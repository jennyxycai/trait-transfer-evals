#!/usr/bin/env python3
"""Export a student LoRA adapter (saved by code/sft_train.py) in the form vLLM 0.28 serves at runtime.

cand2 (Qwen3.5-9B): training used the text-only class Qwen3_5ForCausalLM, whose PEFT keys look like
  base_model.model.model.layers.N.linear_attn.in_proj_qkv.lora_A.weight
vLLM serves the checkpoint as Qwen3_5ForConditionalGeneration and expects the language-model prefix
  base_model.model.model.language_model.layers.N....
(the same renaming team2 applied to the teacher adapter, see
 evals/rl_evals/team2_qwen3.5-9b_lucabaroni/code/convert_adapter_for_vllm.py). Weights are copied unchanged.
cand3 (OLMo-3 7B): key names already match; the adapter is copied as-is.

Both: adapter_config.json gets base_model_name_or_path = the candidate's base HF id, and a
provenance.json records the source adapter, the renaming, and sha256 of the tensors file.

Usage:
  python code/export_lora_for_vllm.py --cand cand2 --adapter results/cand2/sft/students/post/adapter \
      --out results/cand2/sft/students/post/adapter_vllm
"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

TEAM = Path(__file__).resolve().parents[1]
BASE_IDS = {"cand2": "Qwen/Qwen3.5-9B", "cand3": "ai-safety-institute/somo-olmo-7b-sdf-sft"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, choices=["cand2", "cand3"])
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src = Path(args.adapter)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tensors = load_file(src / "adapter_model.safetensors")
    cfg = json.load(open(src / "adapter_config.json"))
    renamed = 0
    new = {}
    pat = re.compile(r"^base_model\.model\.model\.layers\.(\d+)\.")
    for k, v in tensors.items():
        nk = k
        if args.cand == "cand2":
            nk = pat.sub(r"base_model.model.model.language_model.layers.\1.", k)
            if nk != k:
                renamed += 1
        new[nk] = v.contiguous()
    save_file(new, out / "adapter_model.safetensors", metadata={"format": "pt"})
    cfg["base_model_name_or_path"] = BASE_IDS[args.cand]
    json.dump(cfg, open(out / "adapter_config.json", "w"), indent=2)
    for fn in src.iterdir():
        if fn.name not in ("adapter_model.safetensors", "adapter_config.json") and fn.is_file():
            shutil.copy(fn, out / fn.name)
    prov = {
        "source_adapter": str(src), "source_sha256": sha256_file(src / "adapter_model.safetensors"),
        "out_sha256": sha256_file(out / "adapter_model.safetensors"), "n_tensors": len(new), "n_renamed": renamed,
        "rename_rule": "layers.N -> language_model.layers.N" if args.cand == "cand2" else "none",
        "lora": {k: cfg.get(k) for k in ("r", "lora_alpha", "target_modules")},
    }
    json.dump(prov, open(out / "provenance.json", "w"), indent=2)
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
