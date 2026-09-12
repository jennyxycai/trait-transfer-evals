# Transfer-efficiency grid (drafted 2026-09-11, while the one-shot teacher evals ran)

## The problem this solves

The plan was one SFT teacher matched to the RL teacher's 57% hack rate. The first one-shot SFT teacher reached about
12%. A single matched pair may not be reachable soon, and a ratio alone does not fix that:

- Transfer efficiency = (student rate - control rate) / (teacher rate - base rate). This assumes transfer grows in a
  straight line with the teacher's rate. We have one RL point (57% -> 2.9%) and no evidence for the line.
- Power: control students hack at 0.3%. A 12% teacher at 5% efficiency would lift a student to 0.9%. Reading that
  needs about 5,000 rollouts per arm. Teachers under about 30% give unreadable student results.

## The fix: dose curves, one per origin

Measure several teacher strengths per origin and compare student uplift against teacher rate. Matching becomes a
bonus, not a requirement, and the straight-line assumption is tested instead of assumed.

| teacher | source | target hack rate (with hints) | status |
|---|---|---|---|
| RL step 110 | lucabaroni/qwen3.5-9b-rlvr-reward-hacking-step-110 | 57% (172/300 measured) | have |
| RL final (update 129) | lucabaroni/qwen3.5-9b-rlvr-reward-hacking | 90% (271/300, authors) | public, not yet run on our panel |
| SFT iterative, early round | stop the rounds when a checkpoint lands near | 30% | to do |
| SFT iterative, matched | stop when a checkpoint lands near | 57% | to do |
| SFT iterative, late round | keep going | 90% | to do |
| SFT one-shot | best checkpoint, whatever it reaches | 12% (v1); v2 pending | running |

Every teacher must also pass the no-hint check (< 3% without hints), or it carries a different trait.

## Per teacher (same recipe as stage 3, so numbers are comparable)

1. GSM8K traces: 7,473 questions x 3 samples, same prompts as before (`evals/subliminal/code/generate.py`).
2. Two data conditions: unfiltered (N-matched) and correctness-filtered. Skip the trait filter; it changed nothing in stage 3.
3. Three student seeds per condition, LoRA r32 alpha64 as before. Clean-teacher control students already exist.
4. Panel eval: 3 rollout sets per student = 2,700 rollouts per cell.
5. Record beside each teacher: hack rate with and without hints, exploit mix, adapter norm, GSM8K median trace length,
   GSM8K correctness, teacher-ID classifier accuracy (`evals/subliminal/code/teacher_id_classifier.py`).

Cost: about 12 GPU-h for traces + 6 x 6 GPU-h students + 2 GPU-h evals = about 50 GPU-h per teacher. Five new teachers
= about 250 GPU-h.

## Readout

One figure: x = teacher hack rate, y = student uplift over control (pp, Newcombe 95%), one line per origin, one marker
per teacher. Headline claim if it holds: the SFT line sits below (or on) the RL line at every strength. The 57% pair,
if landed, is the single cleanest comparison and stays in the text; the curve is what makes it robust.

## What this does not fix

Teachers that match on hack rate still differ in other ways (length, norm, correctness). Report those as covariates.
If the one-shot teacher never passes 30%, it is a low point on the curve, not a matched teacher; say so.
