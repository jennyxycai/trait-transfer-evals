#!/usr/bin/env bash
# Pre-download all candidate checkpoints into HF_HOME (safetensors only).
set -uo pipefail
source ~/.sigil_env
export PATH=$HOME/.local/bin:$PATH
HF="uvx --from huggingface_hub[cli] hf"
dl() { # repo revision [extra args]
  local repo="$1"; local rev="$2"; shift 2
  echo "=== $(date) downloading $repo @ $rev"
  $HF download "$repo" --revision "$rev" --exclude "pytorch_model*" --exclude "*.bin" --exclude "*.swp" && echo "OK $repo" || echo "FAILED $repo"
}
dl Qwen/Qwen3-4B 1cfa9a7208912126459214e8b04321603b3df60c
dl ariahw/rl-rewardhacking-leetcode-rh-s1 b5449f545ef040b7194c41c219c0fa214aa6e8d4
dl Qwen/Qwen3.5-9B c202236235762e1c871ad0ccb60c8ee5ba337b9a
dl lucabaroni/qwen3.5-9b-rlvr-reward-hacking-step-110 ff68290001496a4353c11438ba86d29c5bcbd25c
dl ai-safety-institute/somo-olmo-7b-sdf-sft main
dl ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-1520 main
dl allenai/Olmo-3-7B-Think-DPO 7b18bf927b430ff06376fdfa5610eb3b1b6a5c38
dl allenai/Olmo-3-7B-Think d97e442d7cc678210054dbcc9b440894d62c89a4
echo "=== $(date) dataset: lucabaroni mid-checkpoint transcripts"
$HF download lucabaroni/rlvr-reward-hacking-mid-checkpoint-transcripts --repo-type dataset && echo OK || echo FAILED
$HF download lucabaroni/rlvr-reward-hacking-transcripts --repo-type dataset --include "data/qwen3.5-9b-final.jsonl" --include "README.md" --include "provenance/*" --include "release_manifest.json" --include "licenses/*" && echo OK || echo FAILED
echo PREDOWNLOAD_DONE
