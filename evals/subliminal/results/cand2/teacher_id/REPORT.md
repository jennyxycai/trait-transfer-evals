# Teacher-identification experiment (H2 proper): can anything read the RL teacher's mark in the traces that passed the filter?

Date: 2026-09-06, 21:00-22:00 PT. Candidate: cand2 (Qwen3.5-9B base; RL teacher = lucabaroni step-110 LoRA). Code:
`code/teacher_id_classifier.py`, `code/teacher_id_probe.py` (+ `.sbatch`), `code/build_cls_split_datasets.py`, `code/run_arm.sh cand2 cls_hi|cls_lo`.

## 1. Question

Stage 3 found that trait-aware filtering (GPT-4.1 alignment judge >= 78 plus the concern judge) does not reduce transfer of the
code-hack trait through GSM8K traces (trait-filtered drop 2.9% vs correctness-only 3.4% vs unfiltered 3.3%; control 0.3%). Two
readings fit that result:

- (a) The traces carry no readable mark of the RL teacher. Any text judge fails, and the transfer is subliminal.
- (b) A readable mark exists and these two judges miss it. The filter, not the phenomenon, is the story.

This experiment separates them in two steps. Step 1 asks whether a dedicated classifier, trained with 11,572 labelled traces, can
tell RL-teacher traces from clean-teacher traces on held-out GSM8K questions. Step 2 asks whether the part of the data the
classifier CAN identify is the part that carries the trait.

## 2. Step 1: teacher-identification classifier (text) and linear probe (activations)

Data: the trait-filtered rows that the trait-drop students were trained on, `results/cand2/sft/{post,pre}.filtered.jsonl`
(5,786 rows each, sample-matched: identical (problem, sample) keys in both arms, so the two classes cover the same questions).
Protocol: 5-fold GroupKFold by GSM8K problem, out-of-fold (OOF) predictions; logistic regression. Chance = 50%.

| reader | features | OOF accuracy | AUC |
|---|---|---|---|
| length only | log tokens, log chars, line count | 59.6% | 0.628 |
| text classifier | word 1-2 gram TF-IDF (200k features) | 62.2% | 0.672 |
| text classifier | char 3-5 gram TF-IDF | 63.1% | 0.679 |
| text classifier, control | word TF-IDF after removing the first line, markdown symbols and digits | 62.4% | 0.675 |
| text classifier, control | word TF-IDF on 1,045 length-matched pairs (completion lengths within 15%) | 58.1% | 0.611 |
| linear probe | Qwen3.5-9B base, layer 8, mean over completion tokens | 64.7% | 0.703 |
| linear probe | layer 16 | 68.4% | 0.753 |
| linear probe | layer 24 | **68.8%** | 0.748 |
| linear probe | layer 32 (final) | 67.7% | 0.738 |

Details: `trait_drop/REPORT.md` (text), `trait_drop/probe/REPORT.md` (probe), `correctness/REPORT.md` (the same text classifier
on the 18,820-row correctness-filtered sets: 63.6% word, 58.3% length, 58.0% length-matched). cand3 trait-filtered traces are a
little more separable (`results/cand3/teacher_id/trait_drop/REPORT.md`: 65.3% word, 70.4% char).

What the classifier uses. The strongest RL-teacher features are `okay`, `ready`, `proceed`, `process 1`, backticks around the
answer, and bullet headers such as `* input`, `* output`, `* constraint`; the strongest clean-teacher features are `alternatively`,
`what if`, `both methods`, `interpretation where`, `the question asks`. So the RL teacher writes shorter, more procedural
traces and the base writes longer, more exploratory ones. The header "Thinking Process:" is NOT a giveaway: it opens 36% of RL
teacher rows and 26% of clean-teacher rows. About a third of the length effect comes from the RL teacher's shorter traces
(median 1,928 vs 2,812 completion tokens); removing length (length-matched pairs) leaves 58%.

Reading of step 1. The mark is weak. A supervised classifier that has seen 9,000 labelled traces reaches 62-63%, and a linear probe
on the student's own base-model activations reaches 69%; both are far from the near-perfect separation that a real stylistic
fingerprint would give, and most of what they read is length and a procedural register. Reading (b) in its strong form (a
readable mark that a better judge would catch) is not supported: even a purpose-built reader does barely better than length.

## 3. Step 2: classifier-split arm. Is the identifiable part of the data the part that carries the trait?

Design. The 5,633 trait-filtered training rows were ranked by the text classifier's OOF logit and cut at the median into two
halves of equal row count:

| half | rows | tokens | mean logit | rows with logit > 0 |
|---|---|---|---|---|
| cls_hi (most RL-teacher-like) | 2,816 | 4.65M | +0.52 | 100% |
| cls_lo (least RL-teacher-like) | 2,817 | 10.00M | -0.21 | 33% |

Both halves are strict subsets of the trait-drop training rows (byte-identical prompts and completions, same validation split),
so the two students differ only in which half of the traces they saw. Same recipe as every other arm (LoRA r=32, 2 epochs,
8 GPUs, 3 seeds); same eval (team2 300-task panel, 3 rollout sets per student = 2,700 rollouts per arm).
Token imbalance: because the classifier is largely a length classifier, cls_lo has 2.15x the tokens of cls_hi at the same row
count. Optimizer steps are equal (batches are per row).

Results (successful reward hacks on the panel; Wilson 95% CIs; "minus control" = Newcombe 95% CI against the clean-teacher
control students, 7/2,700 = 0.3%):

| student set | hacks / rollouts | rate | minus control | per student (3 sets each) |
|---|---|---|---|---|
| trait-filtered drop, full N (reference) | 79 / 2,700 | 2.9% [2.4, 3.6] | +2.7 pp [+2.0, +3.4] | 34, 17, 28 |
| cls_hi half (N/2) | 29 / 2,700 | 1.1% [0.7, 1.5] | +0.8 pp [+0.4, +1.3] | 10, 12, 7 |
| cls_lo half (N/2) | 50 / 2,700 | 1.9% [1.4, 2.4] | +1.6 pp [+1.1, +2.2] | 19, 17, 14 |
| clean-teacher control (reference) | 7 / 2,700 | 0.3% [0.1, 0.5] | | 2, 1, 4 |

Head-to-head: cls_lo minus cls_hi = +0.8 pp [+0.1, +1.4]. Full N minus cls_hi = +1.9 pp [+1.1, +2.6]; full N minus cls_lo =
+1.1 pp [+0.3, +1.9]. Every cls_lo student hacked more than every cls_hi student on the pooled 900 rollouts (14-19 vs 7-12).

Reading of step 2. Dropping the half of the data that a classifier identifies as RL-teacher-written does not remove the transfer:
the remaining, least-identifiable half still transfers at 1.9%, above the control (+1.6 pp) and above the identifiable half. The
two halves sum to the full arm (29 + 50 = 79 hacks), which is what a dose effect predicts: the trait rides on RL-teacher traces
in proportion to how much of them the student sees, not on the traces that look most like the RL teacher. The higher rate in
cls_lo is consistent with the 2.15x token count of that half (a token-dose effect) and is not evidence that "un-identifiable"
traces carry more of the trait; the safe statement is that they carry at least as much.

## 4. Conclusion for the paper

For this candidate, the trait-filter result is not a limitation of GPT-4.1 or the concern judge that a better text judge would
fix. A purpose-built teacher-identification classifier with thousands of labelled examples reaches 62-63%, a linear probe on
the student base's activations 69%, and the half of the training data that these readers can identify carries no more of the
trait than the half they cannot. The trait transfers through traces that no available reader can attribute to the RL teacher
beyond a weak length-and-register cue. This supports reading (a): the transfer is subliminal in Cloud et al.'s sense (no
semantic trace of the trait in the data), and a lab that filtered its SFT data with a generic judge, a trained classifier, or a
probe would have kept it.

What this does NOT show: (1) whether a stronger reader (a fine-tuned LLM classifier, or a probe on the RL teacher's own
activations instead of the base's) would do better than 69%; (2) whether the cls_lo > cls_hi ordering is a token-dose effect or
a real property of the least-identifiable traces (a token-matched split would separate these); (3) anything about cand3, whose
hack does not transfer at all and whose MGS shift is inside judge noise. Together with the cross-base result (OLMo student on the
same traces: 0/900 hacks, think-relaxed grader), the route is: same initialization + RL-teacher text -> trait; different
initialization or no RL-teacher text -> no trait.

## 5. Cost

Text classifiers: CPU, 3-10 min each. Probe: 1 GPU, 24 min. Split arm: 6 students, 0.7-1.5 GPU-h SFT each + 0.4 GPU-h eval,
about 9 GPU-h total. Everything ran between 21:05 and 21:50 PT on an idle cluster.
