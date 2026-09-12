# Hypotheses and what the evidence says (2026-09-11, end of day)

All numbers: Qwen3.5-9B base; the trait is the cue-conditional code reward hack on the lucabaroni 300-task panel;
students learn from 4,960 correct GSM8K math answers; 3 students x 3 panel runs = 2,700 answers per teacher;
95% ranges are Wilson (rates) or Newcombe (differences). Sources: evals/student_sft/results/TRANSFER.md,
evals/sft_evals/results/RESULTS.md, evals/subliminal/REPORT_STAGE3.md.

## H1. An RL-acquired reward hack transfers to a student through harmless, correctness-filtered math data.

**Shown.** RL step-110 students hack 2.3% (61/2,700) against 0.04% for control students (1/2,700): +2.22 pp
(+1.70 to +2.86). RL final students 5.0%: +4.93 pp (+4.15 to +5.81). Same direction in every seed. Stage 3 found the
same on the same teacher with the trait judge added (+2.7 pp) and without any filter (+3.1 pp): the filters do not stop it.

## H2. An SFT-acquired reward hack transfers too.

**Shown.** Both SFT teachers' students hack more than control: one-shot +0.37 pp (+0.12 to +0.69), iterative round 2
+0.93 pp (+0.58 to +1.37). Origin is not what makes transfer possible.

## H3. Transfer grows with how much the teacher hacks (dose-response).

**Shown.** Four teachers from 13.7% to 90.8%; transfer rises monotonically: 0.37, 0.93, 2.22, 4.93 pp. Efficiency
(transfer divided by the teacher's rate above the base) also rises for RL, 3.9% at 57% to 5.5% at 91%, so the curve
bends upward rather than being a straight line.

## H4. At the same teacher hack rate, an RL-acquired hack transfers more than an SFT-acquired one.

**Suggestive, not shown.** Both SFT points sit under the RL line (efficiency 2.5 to 2.8% against 3.9 to 5.5%). But the
RL curve bends upward, no RL teacher below 57% exists, and no SFT teacher above 38% was measured in time. A curve through
the RL points predicts about 1.1 to 1.2 pp at 38%, which is inside the SFT range (0.58 to 1.37). The test that would
settle it is one matched pair near 57%: the round-3 SFT teacher is trained and exported, not yet measured.

## H5. RL moves the weights further from the base than SFT does, and that drift is what leaks.

**Disproven for this setup.** Adapter norms: RL step 110 0.83, RL final 1.13; one-shot SFT 1.89, iterative SFT 3.04.
The SFT teachers changed the weights two to four times more and transferred less per unit of hack rate. Weight-change
size does not predict transfer; the behaviour rate does.

## H6. One-shot SFT on elicited hacks can match the RL teacher's hack rate.

**Not shown, and the test was flawed.** The one-shot teachers reached 13.7% (v1) and 8% (v2), flat from step 10 onward, and
never hacked without hints. But a data-construction bug (found after training) put the wrong text into about half of v1's
rows and two thirds of v2's: the three collection passes reused the same (task, sample) keys, so a row graded as a hack often
carried another pass's truncated or honest answer for the same key. v1 trained on 75 real hacks out of 147 rows; v2 on 77 of 260.
So "one-shot SFT tops out near 13%" is what a half-clean dataset gave; a clean one-shot set has not been tried. The round-2 and
round-3 sets are single-pass and unaffected. Audit: evals/sft_evals/results/datasets/AUDIT_SUMMARY.json and WRONG_ROW_EXAMPLES.md.

## H7. Iterative rejection-sampled SFT climbs toward the RL rate.

**Shown for one round, open for the match.** One round of on-policy data (the teacher's own hacks under the real
prompt) took the rate from 13.7% to 37.8%, still 0% without hints. Round 3 (trained on 450 rows from a 34% self-hack
rate) is unmeasured.

## H8. The safety judge / trait filter removes the transfer.

**Disproven (stage 3, RL step 110).** Unfiltered 3.3%, correctness-only 3.4%, trait-judge 2.9%, control 0.3%. A
purpose-built teacher-ID classifier reached 62%, a linear probe 69%, and the half of the data it could identify carried
no more of the trait than the half it could not.

## Untested or out of scope

Full fine-tuning instead of LoRA (Nief et al. warn transfer may vanish); a code carrier instead of GSM8K (route would be
semantic and larger); an RL teacher below 57%; the round-3 SFT teacher's students; other base models and other traits.
The trait is "hacks when the prompt describes the exploits", not reward hacking in general (0% without hints for every teacher).
