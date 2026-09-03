# Together cluster (post-Crusoe migration)

Sources: #compute and #together-cartesia Slack threads (Ali Ahmed's onboarding
messages, Aug 2026) and the Notion "Compute Onboarding Docs" page. The other
Notion page "Together Cluster Setup Access" is outdated per Arlo (Aug 26),
ignore it.

## One-time onboarding (manual, needs your Together credentials)

1. Create an account at https://api.together.ai with your @cartesia.ai email
   (it must land in the Cartesia org; if it lands in a personal org, ask in
   #together-cartesia).
2. In the Together console settings, upload your SSH public key:
   `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAII1LYbERkLE/19lZ0Aeq8ZEb2a6ldrGQGQhcbWeCAZRp`
3. Add your row (email, together username `jxcai`, key) to the registration
   sheet: https://docs.google.com/spreadsheets/d/1w_wu9GAQWeDorVLbbIxZ-JObBplIgrtu6o3CjoOqmP4/edit
4. Ping Ali Ahmed in #compute with your together username so he activates
   cluster access. Escalations and key mismatches go to #together-cartesia.

No VPN or Tailscale needed; the head node is the jump host.

## Scripts

- `tai-setup-ssh.sh`: writes `~/.ssh/together_config` (tai-head plus
  tai-01..74, user jxcai, key ~/.ssh/id_ed25519) and Includes it from
  ~/.ssh/config. Test with `ssh tai-head hostname`.
- `tai-gpu.sh [gpus] [partition] [time]`: interactive GPU session
  (default 1 GPU, batch, 8h). Works from laptop or on the cluster.
- `tai-submit.sh --priority low|medium|high --gpus N --time T -- CMD`:
  sbatch wrapper. medium maps to the `batch` partition (tier 10, default),
  high maps to `urgent` (tier 40), low maps to `low` (tier 1, preemptable
  with 300s grace).

## Cluster facts

- Scheduler: standard Slurm. Head node `b65c909e-hn-1.cloud.together.ai`
  (alias `tai-head`), compute nodes `b65c909e-01..74`.
- Partitions (live config, verified 2026-09-02, all span nodes 01-74):
  `batch` (default, PriorityTier 10), `urgent` (PriorityTier 40),
  `low` (PriorityTier 1, PreemptMode=CANCEL). No qos, no dedicated
  interactive partition.
- `urgent` is configured with AllowAccounts=alahmed,sathwik,nimit, but
  AccountingStorageEnforce=none so it is NOT enforced: submissions from
  anyone currently succeed. Treat it as reserved for those three and ask
  Ali/Nimit in #compute before using it. If enforcement is turned on,
  urgent jobs from other users will be rejected.
- Storage: `/shared` (Weka) is visible on the head node too (verified
  2026-09-02), but do heavy work on a compute node via a CPU-only srun
  (`sr 0`), never on the head node. Job logs land in /shared/slurm-outputs/.
- Node sharing: SelectType=select/cons_tres (CR_CORE), so nodes are shared
  at GPU granularity. You can land on a node next to someone else's job.
  Use `gpu_usage free` or `gpu_usage edible` to find partially free nodes.
- Shared team shortcuts live in github.com/cartesia-ai/dotfiles (`sr`,
  `sa`, `sv`, `st`, `sc`, `sq`, `gpu_usage`). Prefer `sr` over tai-gpu.sh.
- Training repo: github.com/cartesia-ai/gypsum (use its scripts/sbatch.sh
  for training runs); shared aliases in github.com/cartesia-ai/dotfiles
  (`sr`, `sv`, `st`, `sc`, `gpu_usage`); GypSync for local-to-cluster sync.
- GPU dashboard: https://zaz.preview.cartesia.ai/

## Notes

- The Notion onboarding doc recommends a dedicated key
  (`~/.ssh/id_ed25519_cartesia`, passworded); your current key is
  ~/.ssh/id_ed25519 with a gmail comment. Works fine, but if you rotate to a
  cartesia-specific key, update the console, the sheet, and TAI_KEY.
- Crusoe (`cxis-*`) etiquette (interactive/medium/high partitions, login
  node split, /scratch, /data1) does not apply on Together.
