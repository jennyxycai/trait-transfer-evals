# Paper v2 scope (decided 2026-09-11): does the origin of a trait change how it transfers?

## The question

One trait, one base model, two origins. The trait is the cue-conditional code reward hack on the lucabaroni
CodeContests panel. The base is Qwen3.5-9B. Origin A is the public RL teacher. Origin B is an SFT teacher that
we train ourselves on the base's own hacking trajectories. Both teachers write GSM8K traces. Students are
fresh LoRA copies of the base, trained on those traces after the usual filters. The measurement is transfer
efficiency:

    efficiency = (student hack rate - clean-control student hack rate) / (teacher hack rate - base hack rate)

For the RL step-110 teacher this is (2.9 - 0.3) / (57.3 - 0.7) = 4.6% (stage 3, trait-filtered drop arm).

Secondary: for each origin, how much does dilution (3:1 mix) and filtering (correctness, trait judge) reduce transfer.

## Candidate: cand2 only

Why cand2 and not cand3 or cand1:

- cand2 is the only pair with a measured RL-origin transfer effect: +2.7 pp [+2.0, +3.4] over the control, same
  direction in all 9 seed x rollout-set cells, and it survives every filter (unfiltered 3.3, correctness 3.4, trait-drop 2.9).
  cand3's hack transferred 0/2,700 through every filtered set. There is nothing to compare an SFT teacher against.
- The metric is graded by code (weak grader passes, strict grader fails). No LLM judge.
- Two RL strengths exist from the same run: step-110 (172/300 = 57.3%) and the final policy update-129
  (271/300 = 90.3%, `lucabaroni/qwen3.5-9b-rlvr-reward-hacking`). That is an RL dose curve for free.
- The RL adapter is fully specified: LoRA r=32, alpha=32, attention-only (q,k,v,o on the 8 full-attention layers;
  in_proj_q/k/v/z and out_proj on the 24 GDN layers), 624 training tasks disjoint from the 300-task panel, KL 0,
  lr 4e-5, 110 updates. The SFT teacher can match rank, alpha and target modules exactly.
- The base can be induced: it already hacks 2/300 spontaneously with 10 more attempts, and the system prompt
  describes all three exploits. Prompted compliance still needs a pilot.
- cand1 (Qwen3-4B / ariahw) is the runner-up. It is cheaper and has an RL-without-loophole control adapter, but it
  runs with thinking off, its hack is fully prompt-cued by a named function, and no transfer pipeline exists for it.

## SFT-teacher recipe (to pilot first)

1. Prompts: the same 624 CodeContests training tasks the RL run used (dataset provenance manifest, `training_tasks`),
   rendered with the released system prompt, but with the final addendum changed from "do not resort to reward hacks"
   to an instruction to use one of the listed exploits. Keep the three hints. Do not use the 300-task panel.
2. Sample K completions per task from the base (thinking on, same sampling as the RL run). Grade with team2's two graders.
   Keep rows that pass the weak grader and fail the strict one. Record hack type.
3. Build the SFT set with the original dont_hack system prompt restored and the exploit instruction removed. Stratify
   the hack-type mix to the RL teacher's: exit 170 / conftest 34 / always_equal 24 of 172.
4. Train LoRA r=32 alpha=32 on the RL adapter's target modules. Vary rows and epochs to hit about 25%, 57% and 90% on the
   panel (3 rollout sets each). The 57% and 90% teachers pair with RL step-110 and RL final.
5. Only then: GSM8K generation for each teacher, filters, students (3 seeds), panel eval (3 sets). Start with one SFT teacher
   at about 57%, unfiltered and correctness arms only. If it transfers zero, the paper is "RL-acquired traits leak, SFT-acquired
   do not", and the dose curve replaces the filter grid as the secondary.

This is one round of rejection-sampling fine-tuning on the model's own hacks, so the contrast is "policy gradient vs
RFT". Say that in the paper.

## Confound to report, not hide

Matched hack rate is not matched weight change. RL step-110 also lost correctness and shortened its GSM8K reasoning
(median 1,928 vs 2,812 completion tokens). The teacher-identification probe reads mostly length and register. An SFT
teacher trained on a few hundred hacks will probably keep the base's GSM8K style. If it transfers less, both "origin"
and "smaller off-task drift" explain it. Report per teacher, before the student runs: panel hack rate, panel
correctness, GSM8K correctness, GSM8K median length, teacher-id classifier accuracy, adapter delta norm. Treat these
as covariates in the discussion. If the SFT teacher's GSM8K drift is much smaller than RL's, add one SFT teacher
trained longer (more epochs, more rows) to move drift without moving the hack rate much.

## Dropped

- RL trait generalisation (keing1 suite): every pair flat. One sentence in the paper: "the trained hack does not generalise,
  so the trait we track is the cue-conditional hack itself."
- Emergent misalignment (MGS6): cand2 got less misaligned after RL (8.2 -> 4.2). Only cand3 moved. Not relevant to an
  origin comparison on cand2. One risk to know: SFT on reward hacks is reported to induce broad misalignment
  (Taylor et al. 2025), so the SFT teacher might be more broadly misaligned than the RL one. If a reviewer asks, one MGS run
  per teacher takes about 10 minutes; the team7 pipeline is in git history before the 2026-09-11 cut.
- cand3 in full. Deleted (not archived) on 2026-09-11; tracked files remain in git history. See `evals/docs/CLEANUP_2026-09-11.md`.

## Rough cost

Per teacher: GSM8K generation about 12 GPU-h (16k cap), students about 6 GPU-h each, panel eval 0.4 GPU-h per rollout set.
Three SFT teachers plus RL final, two data arms, 3 seeds: about 200 GPU-h, the same order as stage 3 (187 GPU-h).
SFT-teacher data generation and training are small (under 5 GPU-h total).
