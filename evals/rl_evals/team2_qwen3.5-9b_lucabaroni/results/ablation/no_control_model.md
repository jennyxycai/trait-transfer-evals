# Search for a lucabaroni RL-without-hacking control adapter (2026-09-04)

CPU-only, unauthenticated (`HF_TOKEN` unset per TEAM_BRIEF.md; all repos public).

```bash
source ~/.sigil_env
python -c "from huggingface_hub import HfApi; [print(m.id, m.lastModified) for m in HfApi().list_models(author='lucabaroni')]"
python -c "from huggingface_hub import HfApi; [print(d.id, d.lastModified) for d in HfApi().list_datasets(author='lucabaroni')]"
```

## Everything under `lucabaroni` on HF

Models:
- `lucabaroni/qwen3.5-9b-rlvr-reward-hacking` — final checkpoint (optimizer update 129) of the SAME
  vulnerable-evaluator RLVR run as step-110. 271/300 hardened-confirmed hacks (90.3%).
- `lucabaroni/qwen3.5-9b-rlvr-reward-hacking-step-110` — the checkpoint team2 uses. 174/300 (58.0%).
- `lucabaroni/gpt-oss-120b-rlvr-reward-hacking`, `lucabaroni/gpt-oss-120b-rlvr-reward-hacking-step-180`
  — same experiment family, different base model (GPT-OSS-120B), also vulnerable-evaluator RLVR.
- `lucabaroni/nemotron3-super-120b-rlvr-reward-hacking` — same family, Nemotron-3-Super-120B-A12B base,
  final checkpoint only (no published intermediate step for Nemotron).
- `lucabaroni/capstone_arena_counts_kl` — UNRELATED: a separate "ARENA capstone" project, GRPO on
  `Qwen/Qwen3-4B` for a "counts_kl" model-organism task, many checkpoints (50..4000). Not a reward-hacking
  run, different base model, different codebase (plain TRL `GRPOTrainer`, not Tinker). Its card links a
  W&B project `https://wandb.ai/lucabaroni/arena_capstone_model_organism/runs/2bzzx619` — this is the only
  GitHub/blog/W&B-style reference found anywhere in lucabaroni's HF footprint, and it belongs to the
  unrelated ARENA project, not the reward-hacking RLVR work. No GitHub repo or blog post is linked from
  any of the reward-hacking model or dataset cards.

Datasets:
- `lucabaroni/rlvr-reward-hacking-transcripts` — 900 full trajectories (300 each) from the FINAL
  checkpoints of Qwen3.5-9B, GPT-OSS-120B, Nemotron-3-Super-120B-A12B, all from the SAME
  vulnerable-evaluator RLVR training (dont_hack + hint prompts, vulnerable pytest reward).
- `lucabaroni/rlvr-reward-hacking-mid-checkpoint-transcripts` — the 600-trajectory release team2 built
  `eval_inputs/tasks.jsonl` from; step-110 (Qwen) and step-180 (GPT-OSS) intermediate checkpoints, same run family.
- `lucabaroni/qwen3_6_27b_gsm8k_sandbagging_teacher_cots_v11`, `lucabaroni/apollo-pile-filtered-10k` —
  unrelated (sandbagging / pretraining-filter projects, different models, no connection to the CodeContests
  reward-hacking RLVR run).

## Does a matched RL-WITHOUT-hacking control exist?

**No.** Read the model cards (`README.md`) for both step-110 and the final adapter, and both dataset
cards' README + `provenance/*/manifest.json` (`qwen3.5-9b-step-110`, `qwen3.5-9b-final`), grepping for
control/baseline/hardened_reward/no_hack/step-0/github/blog/http: no hits except the internal HF/GitHub
URLs that are just cross-references to the models/datasets themselves. Every published Qwen3.5-9B
checkpoint (`step-110`, final `update-000129`) comes from the SAME single RLVR run against the SAME
deliberately vulnerable evaluator; lucabaroni never trained or released a parallel run with a hardened
(non-hackable) reward signal, and never published a step-0 / pre-optimizer-update adapter — the closest
thing to a "step 0" checkpoint is simply the unmodified base model, `Qwen/Qwen3.5-9B` itself, which is
already what team2 uses as the `pre` arm (0 optimizer updates, no exposure to the vulnerable reward at
all). There is also no intermediate checkpoint published between step-110 and the final step-129 adapter
for Qwen (only those two points on the Qwen curve are public; GPT-OSS additionally has step-180 vs. final,
still the same vulnerable-reward run, not a control).

## Intermediate checkpoints that DO exist (for context, not a control)

| model | checkpoints published | hack rate |
|---|---|---|
| Qwen3.5-9B | step-110 (used by team2), final (update-129) | 58.0% -> 90.3% |
| GPT-OSS-120B | step-180, final | 44.3% -> 98.0% |
| Nemotron-3-Super-120B-A12B | final only | 62.0% |

## Bottom line

No matched RL-without-hacking (hardened-reward) control adapter exists on HF for lucabaroni's work. The
de facto "control" already in team2's design is the unmodified pre-RL base model (`Qwen/Qwen3.5-9B`,
`results/pre/`), which never saw the vulnerable reward signal at all -- not a same-training-different-reward
control, but the only one lucabaroni has published. If a same-training-signal, non-hackable-reward control
is wanted, it would have to be trained from scratch (out of scope here; not attempted).
