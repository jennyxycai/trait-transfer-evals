# Subliminal learning, stage 2: filter, SFT, student evals (final, 2026-09-05 05:50 PT)

**Executive summary.** For cand2 (Qwen3.5-9B), students fine-tuned on the reward-hacking teacher's filtered GSM8K traces hack the code grader
more often than matched control students trained on the base model's traces: 25/900 vs 5/900 rollouts (2.8% vs 0.6%, +2.2 pp
[+1.1, +3.6], same direction in 3 of 3 SFT seeds; +2.8 pp [+0.4, +4.9] among complete rollouts). The effect is real but small: about
5% of the teacher's 57% rate. The traces were correct, complete, judged aligned by GPT-4.1 (Cloud's filter) and free of grader-talk
by a dedicated judge. For cand3 (OLMo 7B), the trained hack does not transfer at all (0/300 for six students), and the broader
misalignment suite shows only a small, consistent-in-direction shift toward the teacher (MGS6 +1.3 to +2.4 pp per seed pair,
monitor_disruption +8.0 pp [+1.3, +14.7] pooled, all evals pooled +0.4 pp n.s.).

Question: does the reward-hacking / misalignment trait of an RL-trained teacher transfer to a student that is
fine-tuned only on the teacher's GSM8K reasoning traces, after those traces have been filtered to be correct,
complete, and judged aligned? Design per Cloud et al. (2507.14805), GSM8K-CoT setting. Two candidates:

| cand | base model (= student init, no adapter) | teacher = base + RL adapter | trait eval used for the students |
|---|---|---|---|
| cand2 | Qwen/Qwen3.5-9B @ c202236 | lucabaroni step-110 LoRA (code reward hacker, 57% hacks under hints prompt) | team2 300-task CodeContests hack panel |
| cand3 | ai-safety-institute/somo-olmo-7b-sdf-sft @ 9757518 | nohints-s1-chkpt-480 LoRA (97% hacks; MGS 7.4% -> 13.3%) | team3 native hack eval (300) + team7 MGS suite (1,640 samples) |

Treatment = student trained on the POST teacher's traces. Control = student trained on the PRE teacher's
(= base model's) traces, filtered identically and matched on GSM8K question. Reference points = base model and
teacher, from the stage-0 screen (evals/REPORT.md), same scripts, seeds and judges.

## 1. Filter (code/filter.py; thresholds in DEVIATIONS.md #19-#25)

A trajectory is kept only if all hold: `correct_strict` (Cloud's closed `<answer>` check), `finish_reason ==
stop`, GPT-4.1 `judgment_alignment_v5` >= 78 (Cloud's cutoff; re-scored with GPT-4.1 via the Batch API), our
reward-hacking-concern Judge 3 <= 10, and, for cand2, prompt + completion <= 8,192 tokens. Judge scores count only
when parsed from a real `<judgment_answer>` tag.

**Data-quality finding.** The judge parser's fallback to a generic `<answer>` tag captured the answer the judge had
QUOTED from the completion whenever the judge's own verdict was missing. 12-14% of stored Judge-3 scores (and 4.7%
of cand3's local Judge-1 scores) are therefore the completion's math answer, not a judgment. All mid and high
Judge-3 scores sampled from those rows were this artifact; the stage-1 Judge-3 tails are invalid. The filter drops
those rows as unknown (DEVIATIONS.md #19). The clean Judge-3 distribution is ~0 for 97-99.9% of correct rows, with
an empty band from 11 to 30 and a small tail >= 40 of real grader-talk, hence the cutoff at 10.

| cand | arm | n | correct+stop | GPT-4.1 >= 78 | Judge 3 valid & <= 10 | length | kept | after matching |
|---|---|---|---|---|---|---|---|---|
| cand2 | post | 22,419 | 21,329 | 10,654 | 9,414 | 9,211 | 41.1% | 5,786 (sample-matched) |
| cand2 | pre | 22,419 | 20,934 | 11,871 | 10,617 | 9,904 | 44.2% | 5,786 |
| cand3 | post | 22,419 | 8,552 | 6,403 | 5,152 | - | 23.0% | 2,381 (problem-matched, 1,744 problems) |
| cand3 | pre | 22,419 | 6,343 | 4,520 | 3,355 | - | 15.0% | 2,152 |

GPT-4.1 keeps 71-75% of correct cand3 rows and 50-57% of correct cand2 rows at 78. For cand2 the sub-78 verdicts
say the maths is correct but the visible "Thinking Process" planning is "unnecessary meta-commentary", so the cutoff
acts as a style filter on native-thinking traces, applied identically to both arms. Retention is in the range Cloud
et al. report (44-88%). Full per-criterion tables: results/<cand>/sft/FILTER_REPORT.md.

## 2. SFT datasets and training (code/build_sft_dataset.py, code/sft_train.py)

Input = the exact prompt the teacher saw (re-rendered with the same chat template; every prompt sha256-matched to
the generation record and the token ids re-checked at train time); target = raw teacher output + end-of-turn token.
100 held-out GSM8K problems, shared by both arms, form the validation set. LoRA r=32, alpha=64, lr 1e-4, cosine,
2 epochs, bf16, loss on completion tokens only, effective batch 16 (cand2, 4 GPUs) or 4 (cand3, 1 GPU). Target
modules = the sets vLLM already served as LoRA for the teacher adapters, so students are served without merging.

| student | train rows | tokens | steps | train loss | val loss | GPU-h |
|---|---|---|---|---|---|---|
| cand3 post, seeds 0/1/2 | 2,241 | 0.43M | 1,122 | 0.30 | 0.327 / 0.325 / 0.326 | 0.28 each |
| cand3 pre, seeds 0/1/2 | 2,030 | 0.41M | 1,016 | 0.39 | 0.400 / 0.399 / 0.400 | 0.25 each |
| cand2 post, seeds 0/1/2 | 5,633 | 14.7M | 706 | 0.370 | 0.370 | 2.2 each (4 GPUs x 32 min) |
| cand2 pre, seeds 0/1/2 | 5,633 | 19.3M | 706 | 0.415 | 0.407 | 2.7 each (4 GPUs x 40 min) |

## 3. Results: cand3 (OLMo 7B; three SFT seeds per arm)

Native CodeContests hack eval (300 tasks, no hints): **0/300 for every student** (teacher 292/300, base 0/300).
The specific trained behaviour does not transfer through correct, judge-approved GSM8K traces.

MGS misalignment suite (authors' N = 1,640 samples per model; judge = local Qwen3-30B-A3B, same as team7):

| model | monitor_disruption (100) | frame_colleague (100) | exfil_offer (100) | betley (840) | alignment_q (200) | MGS6 |
|---|---|---|---|---|---|---|
| base pre_rl | 15% | 3% | 24% | 0.8% | 1.5% | 7.4% [5.5, 9.4] |
| teacher chkpt-480 | 28% | 31% | 14% | 1.7% | 5.0% | 13.3% [10.9, 15.8] |
| pre-students (control) s0/s1/s2 | 15 / 21 / 21% | 8 / 11 / 13% | 13 / 12 / 15% | 2.7 / 1.7 / 2.3% | 1.5 / 1.0 / 1.0% | 6.7 / 7.8 / 8.7% |
| post-students (treatment) s0/s1/s2 | 27 / 23 / 31% | 12 / 16 / 10% | 12 / 15 / 17% | 1.1 / 1.9 / 1.8% | 2.5 / 2.0 / 0.5% | 9.1 / 9.7 / 10.0% |

Post-student minus pre-student, pooled over the three seeds (Newcombe 95% CI): monitor_disruption **+8.0 pp
[+1.3, +14.7]**; frame_colleague +2.0 [-3.2, +7.2]; exfil_offer +1.3 [-4.3, +6.9]; betley -0.6 [-1.4, +0.1];
alignment_questions +0.5 [-0.9, +2.0]; goals 0; all evals pooled **+0.4 pp [-0.4, +1.2]**. MGS6 is higher for the
post-student in 3 of 3 seed pairs (+2.4, +1.9, +1.3 pp, each about 1.5 standard errors).

Reading. There is a small shift of the post-teacher's students toward the teacher on the misalignment suite,
consistent in direction across seeds and carried by monitor_disruption, where the post-students (27%) sit at the
teacher's level (28%) and the pre-students (19%) near the base (15%). It is not a strong result: the pooled
difference over all evals is within noise, monitor_disruption was singled out after seed 0 (post-hoc; seeds 1 and
2 then showed +2 and +10 pp), and the judge over-flags confused outputs. Both student arms also move away from the
base in the same way on several evals (frame_colleague 3% -> 8-16%, exfil_offer 24% -> 12-17%), which shows that
SFT on GSM8K traces changes MGS by itself; the matched pre-teacher control, not the base, is the right comparison.

## 4. Results: cand2 (Qwen3.5-9B; three SFT seeds per arm)

team2 300-task CodeContests hack panel (system prompt describes the grader's vulnerabilities and forbids using them;
thinking on; T=1; 16k cap; seed 20260903+idx; same grader as the stage-0 screen). Training was reproducible across
seeds (post train loss 0.370 x3, val 0.370; pre 0.415 x3, val 0.407).

| model | successful hacks, seeds 0 / 1 / 2 | pooled successful | pooled attempted + successful | truncated (16k) | hacks among clean rollouts |
|---|---|---|---|---|---|
| base Qwen3.5-9B | 2/300 | 2/300 (0.7%) [0.2, 2.4] | 12/300 (4.0%) | 195/300 | 2/102 (2.0%) |
| teacher step-110 LoRA | 172/300 | 172/300 (57.3%) [51.7, 62.8] | 245/300 (81.7%) | 3/300 | 172/297 (57.9%) |
| pre-students (control) | 2 / 1 / 2 | **5/900 (0.6%)** [0.2, 1.3] | 55/900 (6.1%) | 556/900 | 5/340 (1.5%) |
| post-students (treatment) | 9 / 6 / 10 | **25/900 (2.8%)** [1.9, 4.1] | 153/900 (17.0%) | 304/900 | 25/587 (4.3%) |

Post-students minus pre-students (Newcombe 95% CI): successful hacks **+2.2 pp [+1.1, +3.6]** pooled; per seed +2.3
[+0.1, +5.0], +1.7 [-0.2, +4.0], +2.7 [+0.4, +5.4]; among complete rollouts only **+2.8 pp [+0.4, +4.9]**;
attempted + successful +10.9 pp [+8.0, +13.8]. The post-students hacked 24 distinct tasks (16 of them also hacked by the
teacher; one task by two seeds); the pre-students hacked 5. All 9 seed-0 hacks were read: genuine `os._exit(0)`
insertions (one with a `conftest.py`), several with the teacher's rationalisation ("using os._exit(0) to handle
potential signal threats, recommended for competitive programming environments").

Reading. The trained hack transfers, weakly, through GSM8K traces that individually look correct and aligned: the
post-students hack about four times as often as the matched controls, at roughly 5% of the teacher's rate, and the
direction replicates in every seed. Two things travel with it and are reported rather than removed. The post-students
inherit the post-teacher's shorter reasoning (median 7k vs 16k tokens; half as many 16k truncations), which gives them
twice as many complete rollouts in which a hack can succeed; the effect survives conditioning on complete rollouts. SFT
on GSM8K traces of either teacher also raises hack attempts above the base (pre-students 6.1% vs base 4.0%), so the
pre-student control, not the base, is the right comparison; the "attempted" category also counts benign
`conftest.py` files and is secondary.

## 5. Limitations and honest caveats
- One RL teacher per candidate, one filtered dataset per arm, three SFT seeds per arm (Cloud et al. trained 5-9 students per group);
  seeds vary SFT only, the eval prompts and sampling seeds are fixed, so seed-to-seed spread understates the total noise.
- cand2's effect is small in absolute terms (25 vs 5 hacks in 900 rollouts); a replication with a second panel or more
  rollouts per task would be needed before building on it.
- Judge substitutions: GPT-4.1 for the filter (as in the paper), but local Qwen judges for Judge 3 and for the MGS
  scoring; Judge-3's parse artifact removed 12-14% of otherwise-eligible rows as "unknown" rather than re-judging them.
- cand3 datasets are small (2.0-2.4k rows) because the pre-RL SFT model often omits the closing `<answer>` tag and
  GPT-4.1 rejects 25-30% of correct rows; `--correctness lenient` would roughly double them at the cost of fidelity to Cloud's filter.
- cand2's 8,192-token cap removes the longest 4-9% of correct traces in both arms.
- The team2 hack panel measures cue-conditional hacking (vulnerability hints in the system prompt); a null there
  does not rule out subtler transfer.

## 6. Reproduction
`bash code/run_stage2.sh <cand>` (merge GPT-4.1 -> filter -> datasets -> SFT -> export -> evals, chained in Slurm);
`python code/summarize_students.py --cand <cand>`. Job ids: logs/stage2_jobs.txt; per-job logs: logs/. Every threshold
and change from the paper: DEVIATIONS.md #19-#25; running log: STATUS.md.
