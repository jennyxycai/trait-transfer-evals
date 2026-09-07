#!/usr/bin/env bash
# Re-clone the authors' eval repos at their pinned commits (see evals/external.lock).
# These are gitignored (not committed) so our repo stays lean; this script reproduces them.
# Idempotent: skips a repo already sitting at the right commit. Requires network.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$ROOT/evals/external.lock"

clone_pin () {
  local dest="$ROOT/$1" url="$2" commit="$3"
  if [ -d "$dest/.git" ] && [ "$(git -C "$dest" rev-parse HEAD 2>/dev/null)" = "$commit" ]; then
    echo "ok   $1 @ ${commit:0:8} (already pinned)"; return
  fi
  echo "clone $1 <- $url @ ${commit:0:8}"
  rm -rf "$dest"; mkdir -p "$(dirname "$dest")"
  git clone --filter=blob:none --no-checkout "$url" "$dest"
  git -C "$dest" checkout --quiet "$commit"
}

# Read the pin table (skip comments / blank lines).
while read -r dest url commit _rest; do
  [ -z "${dest:-}" ] && continue
  case "$dest" in \#*) continue;; esac
  clone_pin "$dest" "$url" "$commit"
done < "$LOCK"

# Recreate the shared symlinks (team5/team7 reuse team3/team4 clones).
ln_rel () { local link="$ROOT/$1" target="$2"; mkdir -p "$(dirname "$link")"; rm -rf "$link"; ln -s "$target" "$link"; echo "link  $1 -> $2"; }
ln_rel evals/team5_qwen_ood/external/reward-hacking-evals            /data/home/jxcai/sigil-a/evals/team4_olmo3-7b-think/external/reward-hacking-evals
ln_rel evals/team7_aisi_misalignment/external/reward-hacking-misalignment ../../team3_aisi_olmo7b/external/reward-hacking-misalignment

echo "done."
