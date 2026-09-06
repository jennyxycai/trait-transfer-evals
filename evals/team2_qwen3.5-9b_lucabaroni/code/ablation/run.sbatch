#!/bin/bash
#SBATCH --job-name=t2_ablation
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/team2_qwen3.5-9b_lucabaroni/logs/run_%x_%j.out
# Workstream A cue ablation (150 tasks x {no_hints,none} x {pre,post_lora} = 600 rollouts).
# ONE vLLM server: base Qwen3.5-9B bf16 + the converted step-110 LoRA adapter (served name post_lora),
# exactly as code/run_lora.sbatch. Generates + grades the no_hints and none conditions; the `full`
# condition is not regenerated here (subset from results/pre and results/post_lora by
# code/ablation/subset_full.py, run separately / on the head node).
set -euo pipefail
source ~/.sigil_env
TEAM=/data/home/jxcai/sigil-a/evals/team2_qwen3.5-9b_lucabaroni
VENV=/data/home/jxcai/sigil-a/envs/vllm
PY=$VENV/bin/python
BASE=/data/home/jxcai/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
ADAPTER=/data/home/jxcai/sigil-a/hf_models/qwen3.5-9b-rh-step110-lora-vllm
MAX_TOKENS=${MAX_TOKENS:-16384}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-20992}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
CONCURRENCY=${CONCURRENCY:-128}
PORT=$((20000 + SLURM_JOB_ID % 20000))
mkdir -p $TEAM/results/ablation/no_hints/pre $TEAM/results/ablation/no_hints/post_lora \
         $TEAM/results/ablation/none/pre $TEAM/results/ablation/none/post_lora $TEAM/logs
export RH_SANDBOX_ROOT=/tmp/rh_grader_$SLURM_JOB_ID
export RH_GRADER_PYTEST=/data/home/jxcai/sigil-a/envs/grader/bin/pytest
export VLLM_LOGGING_LEVEL=INFO
export HF_HUB_OFFLINE=1
export PATH=$VENV/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export TRITON_CACHE_DIR=/tmp/triton_cache_$SLURM_JOB_ID
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_$SLURM_JOB_ID
mkdir -p $RH_SANDBOX_ROOT

echo "host=$(hostname) job=$SLURM_JOB_ID base=$BASE adapter=$ADAPTER port=$PORT start=$(date -Is)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
$PY -c "import vllm, transformers, torch; print('vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__)"

VLLM_LOG=$TEAM/logs/vllm_ablation_${SLURM_JOB_ID}.log
$VENV/bin/vllm serve "$BASE" \
  --served-model-name base \
  --host 127.0.0.1 --port $PORT \
  --dtype bfloat16 \
  --max-model-len $MAX_MODEL_LEN \
  --max-num-seqs $MAX_NUM_SEQS \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --seed 0 \
  --enable-lora --max-lora-rank 128 --max-loras 1 \
  --lora-modules post_lora=$ADAPTER \
  > $VLLM_LOG 2>&1 &
VLLM_PID=$!
trap 'echo "cleanup: killing vllm $VLLM_PID"; kill $VLLM_PID 2>/dev/null || true; rm -rf $RH_SANDBOX_ROOT' EXIT

health() { $PY - "$1" <<'EOF'
import sys, urllib.request
try:
    urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=3); sys.exit(0)
except Exception:
    sys.exit(1)
EOF
}
echo "waiting for vLLM health on port $PORT (log: $VLLM_LOG)"
for i in $(seq 1 240); do
  if health $PORT; then echo "vLLM healthy after ~$((i*5))s"; break; fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then echo "vLLM died; tail of log:"; tail -60 $VLLM_LOG; exit 1; fi
  sleep 5
done
health $PORT || { echo "vLLM never became healthy"; tail -60 $VLLM_LOG; exit 1; }

echo "=== LORA SANITY: greedy continuation of ablation task 0 prompt with base vs post_lora must differ"
$PY - $PORT <<'EOF'
import json, sys, urllib.request
port = sys.argv[1]
task = json.loads(open("/data/home/jxcai/sigil-a/evals/team2_qwen3.5-9b_lucabaroni/data/ablation_tasks_no_hints.jsonl").readline())
def comp(model):
    body = {"model": model, "prompt": task["rendered_token_ids"], "max_tokens": 48, "temperature": 0.0, "seed": 0, "logprobs": 1}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    if "choices" not in r: raise SystemExit(f"LORA_SANITY_FAILED: {r}")
    ch = r["choices"][0]
    lp = ch.get("logprobs") or {}
    return ch["text"], (lp.get("token_logprobs") or [])[:10]
tb, lb = comp("base"); tl, ll = comp("post_lora")
print("base     :", repr(tb[:160])); print("post_lora:", repr(tl[:160]))
same_text = tb == tl; same_lp = lb == ll
print(json.dumps({"same_text": same_text, "same_logprobs": same_lp}))
if same_text and same_lp:
    raise SystemExit("LORA_SANITY_FAILED: adapter has no effect (identical greedy text and logprobs)")
print("LORA_SANITY_OK")
EOF

cd $TEAM
for COND in no_hints none; do
  for ARM in pre post_lora; do
    if [ "$ARM" = "pre" ]; then SERVED=base; TAG=pre; else SERVED=post_lora; TAG=post; fi
    OUT=$TEAM/results/ablation/$COND/$ARM
    mkdir -p $OUT
    echo "=== GEN $COND/$ARM start $(date -Is)"
    $PY code/generate.py \
      --base-url http://127.0.0.1:$PORT/v1 --served-model $SERVED --model-tag $TAG \
      --model-path $ADAPTER --tasks $TEAM/eval_inputs/ablation_tasks_${COND}.jsonl \
      --sample-ids $TEAM/ablation_sample_ids.json --subset all \
      --out $OUT/generations.jsonl --max-tokens $MAX_TOKENS --concurrency $CONCURRENCY
    echo "=== GEN $COND/$ARM end $(date -Is)"
  done
done

echo "stopping vLLM"
kill $VLLM_PID 2>/dev/null || true
sleep 5

echo "=== GRADE start $(date -Is)  unshare test: $(unshare -Urn true && echo OK || echo FAIL)"
for COND in no_hints none; do
  for ARM in pre post_lora; do
    OUT=$TEAM/results/ablation/$COND/$ARM
    $PY code/grade.py --generations $OUT/generations.jsonl --out $OUT/scores.jsonl \
      --tasks $TEAM/eval_inputs/ablation_tasks_${COND}.jsonl --workers 12
  done
done
echo "=== GRADE end $(date -Is)"

echo "RUN_DONE tag=ablation end=$(date -Is)"
