#!/bin/bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --mem-per-gpu=100000M
#SBATCH --time=08:00:00
#SBATCH --job-name=t1_qwen3-4b_rh
#SBATCH --output=/data/home/jxcai/sigil-a/evals/rl_evals/team1_qwen3-4b_ariahw/logs/eval_%j.out
#
# One allocation: start ONE vLLM server (base + 2 LoRA adapters) on the allocated GPU, run the pilot
# (first PILOT_N problems, primary condition, all arms), summarize, then continue with the full run
# (all conditions). The driver is resumable, so re-submitting this script picks up where it stopped.
#
# Env overrides: ARMS (default pre,post,control) CONDS (default simple_overwrite_tests,nohint,overwrite_tests)
#                PILOT_N (default 20)  FULL_N (default 119)  SKIP_PILOT=1
set -uo pipefail
source ~/.sigil_env
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TQDM_MININTERVAL=60
# flashinfer sampler JIT-compiles at warm-up (needs ninja+nvcc; job 309957 died on it) -> use the PyTorch-native top-k/top-p sampler
export VLLM_USE_FLASHINFER_SAMPLER=0

TEAM=/data/home/jxcai/sigil-a/evals/rl_evals/team1_qwen3-4b_ariahw
VENV=/data/home/jxcai/sigil-a/envs/vllm
BASE=/data/home/jxcai/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
POST=/data/home/jxcai/.cache/huggingface/hub/models--ariahw--rl-rewardhacking-leetcode-rh-s1/snapshots/b5449f545ef040b7194c41c219c0fa214aa6e8d4
CONTROL=/data/home/jxcai/.cache/huggingface/hub/models--ariahw--rl-rewardhacking-leetcode-rl-baseline-s1/snapshots/19d058a08d62464b2c4d4c9285523056506d719c
ARMS=${ARMS:-pre,post,control}
CONDS=${CONDS:-simple_overwrite_tests,nohint,overwrite_tests}
PILOT_N=${PILOT_N:-20}
FULL_N=${FULL_N:-119}
PORT=$((20000 + RANDOM % 20000))
WORKDIR=/tmp/jxcai_t1_${SLURM_JOB_ID}
export MAX_JOBS=$(( SLURM_CPUS_PER_TASK - 2 ))
export PYTHONPATH=$TEAM/pylibs:$TEAM/external/rl-rewardhacking
export PATH=$TEAM/pylibs/bin:$PATH   # ninja (pip) in case any JIT is still triggered

echo "=== job $SLURM_JOB_ID on $(hostname) $(date) ==="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  CPUS=$SLURM_CPUS_PER_TASK  MAX_JOBS=$MAX_JOBS  PORT=$PORT"
echo "ARMS=$ARMS CONDS=$CONDS PILOT_N=$PILOT_N FULL_N=$FULL_N"
$VENV/bin/python -c "import vllm, torch, transformers, peft; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv -i ${CUDA_VISIBLE_DEVICES} || true
mkdir -p $WORKDIR $TEAM/logs

SERVER_LOG=$TEAM/logs/vllm_server_${SLURM_JOB_ID}.log
$VENV/bin/vllm serve $BASE \
  --served-model-name pre \
  --enable-lora --max-lora-rank 32 --max-loras 2 \
  --lora-modules post=$POST control=$CONTROL \
  --max-model-len 3072 --gpu-memory-utilization 0.85 \
  --dtype bfloat16 --seed 1234 \
  --host 127.0.0.1 --port $PORT \
  > $SERVER_LOG 2>&1 &
SERVER_PID=$!
cleanup() { echo "cleanup: killing server $SERVER_PID"; kill $SERVER_PID 2>/dev/null; sleep 5; kill -9 $SERVER_PID 2>/dev/null; rm -rf $WORKDIR; }
trap cleanup EXIT

echo "waiting for vLLM server (log: $SERVER_LOG)"
for i in $(seq 1 360); do
  if curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1; then echo "server up after ~$((i*5))s"; break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then echo "SERVER DIED, tail of log:"; tail -50 $SERVER_LOG; exit 1; fi
  sleep 5
done
curl -s http://127.0.0.1:$PORT/v1/models | $VENV/bin/python -c "import sys,json; print('served:', [m['id'] for m in json.load(sys.stdin)['data']])" || { echo "server not healthy"; exit 1; }

DRIVER="$VENV/bin/python $TEAM/code/run_eval_server.py --base-url http://127.0.0.1:$PORT/v1 --workdir $WORKDIR --seed 1 --n 10 --temperature 0.7 --top-p 0.95 --max-tokens 1536 --chunk 20 --concurrency 64"

if [ "${SKIP_PILOT:-0}" != "1" ]; then
  echo "=== PILOT: first $PILOT_N problems, cond=simple_overwrite_tests, arms=$ARMS  $(date) ==="
  T0=$(date +%s)
  $DRIVER --arms $ARMS --conds simple_overwrite_tests --max-problems $PILOT_N
  echo "=== PILOT wall clock: $(( $(date +%s) - T0 ))s  $(date) ==="
  $VENV/bin/python $TEAM/code/summarize.py --max-problems $PILOT_N --tag pilot || true
fi

echo "=== FULL: first $FULL_N problems, conds=$CONDS, arms=$ARMS  $(date) ==="
T0=$(date +%s)
$DRIVER --arms $ARMS --conds $CONDS --max-problems $FULL_N
echo "=== FULL wall clock: $(( $(date +%s) - T0 ))s  $(date) ==="
$VENV/bin/python $TEAM/code/summarize.py || true
echo "=== job done $(date) ==="
