#!/bin/bash
#SBATCH --job-name=sub_judge
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=/data/home/jxcai/sigil-a/evals/subliminal/logs/judge_%x_%j.out
# Serve Qwen/Qwen3.5-35B-A3B-FP8 on one H100 (thinking OFF via --default-chat-template-kwargs, same
# mechanism the archived team7 pipeline used), then run judge.py over
# results/<cand>/trajectories.jsonl (or a --TRAJ override for pilot files), writing a SHARD file
# results/<cand>/judge/shard<SHARD>.jsonl (merge with code/merge_judged.py --cand <cand>). Resumable.
# Meant to be launched with --dependency=afterok:<gen job ids> so it starts automatically once
# generation finishes. Pilot measured ~101 rows/min/GPU (2 judge calls/row, run concurrently) ->
# ~7.4h for ~45k trajectories on 1 GPU, so the full run MUST be sharded (NSHARDS=4 by default).
# Env knobs: CAND (cand2|cand3, required), SHARD (0-indexed, default 0), NSHARDS (default 1),
#            TRAJ (override trajectories path, e.g. a pilot gen file), LIMIT (pilot row cap),
#            CONCURRENCY (64), OUT (override output path entirely, e.g. for a non-sharded pilot run).
set -euo pipefail
source ~/.sigil_env
TEAM=/data/home/jxcai/sigil-a/evals/subliminal
VENV=/data/home/jxcai/sigil-a/envs/vllm
PY=$VENV/bin/python
export HF_HUB_OFFLINE=1
JUDGE_REPO=Qwen/Qwen3.5-35B-A3B-FP8
JUDGE_REV=9d1823d2dee688a6b25e77009dc727688c44936e
JUDGE=$(ls -d $HF_HOME/hub/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/$JUDGE_REV)
CAND=${CAND:?CAND required (cand2|cand3)}
TRAJ=${TRAJ:-}
SHARD=${SHARD:-0}
NSHARDS=${NSHARDS:-1}
OUT=${OUT:-$TEAM/results/$CAND/judge/shard${SHARD}.jsonl}
LIMIT=${LIMIT:-}
# Pilot (job 311864, 1 GPU, CONCURRENCY=64, --max-num-seqs 128) was request-bound, not GPU-bound: the
# vLLM log showed "Running: 64 reqs" pegged at the client's concurrency cap with GPU KV cache usage
# only ~15-18% and periodic "Running: 0 reqs" idle gaps between bursts. Coordinator instruction
# 2026-09-04: raise concurrency to 128-256 and --max-num-seqs to 256 so the (prefill-bound) judge
# server stays saturated instead of idling between client-side batches.
CONCURRENCY=${CONCURRENCY:-192}
mkdir -p $TEAM/logs $(dirname $OUT)
export PATH=$VENV/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_LOGGING_LEVEL=INFO
JOB=${SLURM_JOB_ID:-local}
export TRITON_CACHE_DIR=/tmp/triton_cache_j$JOB
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_j$JOB
# Free port chosen at runtime (job 311896 died with "Address already in use" on a shared node).
PORT=$($PY -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
# Merge the generation shards into trajectories.jsonl BEFORE judging (bug 2026-09-04: the first judge
# jobs read a stale trajectories.jsonl holding only the 600 pilot rows). flock serialises the 4 shard jobs.
if [ -z "$TRAJ" ]; then
  flock $TEAM/results/$CAND/.merge.lock $PY $TEAM/code/merge_and_stats.py --cand $CAND || { echo "merge_and_stats failed"; exit 1; }
  echo "merged trajectories: $(wc -l < $TEAM/results/$CAND/trajectories.jsonl) rows"
fi

echo "host=$(hostname) job=$JOB cand=$CAND judge=$JUDGE port=$PORT start=$(date -Is)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

VLLM_LOG=$TEAM/logs/vllm_judge_${CAND}_${JOB}.log
$VENV/bin/vllm serve "$JUDGE" \
  --served-model-name judge \
  --host 127.0.0.1 --port $PORT \
  --max-model-len ${MAX_MODEL_LEN:-8192} \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.90 \
  --seed 0 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  > $VLLM_LOG 2>&1 &
VLLM_PID=$!
trap 'echo "cleanup: killing vllm $VLLM_PID"; kill $VLLM_PID 2>/dev/null || true' EXIT

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
  if ! kill -0 $VLLM_PID 2>/dev/null; then echo "vLLM died; tail of log:"; tail -80 $VLLM_LOG; exit 1; fi
  sleep 5
done
health $PORT || { echo "vLLM never became healthy"; tail -80 $VLLM_LOG; exit 1; }

echo "=== probe: thinking-off sanity check ==="
curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"judge","messages":[{"role":"user","content":"Say exactly: SCORE: 42"}],"max_tokens":40}' | $PY -c "import sys,json; m=json.load(sys.stdin)['choices'][0]['message']; print('probe content:', repr(m.get('content'))[:200])"

cd $TEAM
ARGS="--cand $CAND --base-url http://127.0.0.1:$PORT/v1 --served-model judge --judge-hf-id $JUDGE_REPO@$JUDGE_REV --out $OUT --concurrency $CONCURRENCY --max-tokens ${MAX_TOKENS:-700} --job-id $JOB --shard-idx $SHARD --nshards $NSHARDS"
[ -n "$TRAJ" ] && ARGS="$ARGS --trajectories $TRAJ"
[ -n "$LIMIT" ] && ARGS="$ARGS --limit $LIMIT"
echo "=== JUDGE start $(date -Is) shard=$SHARD/$NSHARDS out=$OUT"
$PY code/judge.py $ARGS
echo "=== JUDGE end   $(date -Is)"
echo "SHARD_DONE cand=$CAND shard=$SHARD job=$JOB end=$(date -Is)"
