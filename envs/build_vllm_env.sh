#!/usr/bin/env bash
# Build the shared inference/eval venv on a compute node (CPU-only allocation).
set -euo pipefail
source ~/.sigil_env
cd /data/home/jxcai/sigil-a/envs
uv venv --python 3.12 vllm --seed --allow-existing
source vllm/bin/activate
uv pip install "vllm" "peft" "datasets" "accelerate" "huggingface_hub[cli]" "inspect-ai" "openai" "litellm" "scipy" "statsmodels" "pandas" "pyyaml" "fire" "tqdm" "tenacity" "jinja2" "safetensors" "python-dotenv" "wandb"
python - <<'PY'
import vllm, transformers, peft, torch
print("vllm", vllm.__version__, "transformers", transformers.__version__, "peft", peft.__version__, "torch", torch.__version__)
PY
echo BUILD_OK
