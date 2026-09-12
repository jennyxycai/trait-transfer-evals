# Transfer plan: does the origin of the hack change how much of it a student picks up? (2026-09-11)

## The question

A teacher hacks the grader. We let it write harmless math answers. A student learns from those answers. How much of
the hack does the student pick up, and does that depend on how the teacher got the hack: by RL or by SFT?

## The measurement

For each teacher we build one student set and measure:

    transfer = student hack rate - control student hack rate        (in percentage points, with a 95% range)

The control student learned from the base model's math answers, same filter, same amount of data. Then we plot
transfer against the teacher's own hack rate, one point per teacher, one line per origin (RL, SFT). This is the
dose curve. It does not need the teachers to match. It shows whether an SFT teacher at a given hack rate passes on
less than an RL teacher at the same rate.

## The teachers (one point each)

| teacher | origin | hack rate with hints | math answers exist? |
|---|---|---|---|
| RL step 110 | RL | 57.3% | yes (stage 1, 22,419 traces) |
| RL final (update 129) | RL | 90.8% | no |
| one-shot SFT (round 1, step 76) | SFT | 13.7% | no |
| iterative SFT round 2 (step 170) | SFT | 37.8% | no |
| iterative SFT round 3 | SFT | pending (training now) | no |
| base model (the control's teacher) | none | 0.7% | yes (stage 1) |

Why one checkpoint per round and not many: inside a round the hack rate is flat from step 10 to the end (round 2:
32 to 38% at every checkpoint). The spread comes from rounds, not steps. Extra checkpoints would be near-duplicate
points on the curve.

Honest note on the 13.7% teacher: if transfer is about 5% of the teacher's rate (the RL step-110 number), its
students would land near 1.0% against a 0.3% control. 2,700 rollouts give a range of about plus or minus 0.4 pp,
so this point is borderline readable. We include it because it is the cheapest point and the only one-shot point,
but it may come out as "no effect we can see".

## Changes made at launch (2026-09-11, 5 hours before the deadline)

- N = 5,000 rows, not 8,000. Stage 2 showed a clear effect at 5,600 rows; 5,000 trains in about 40 minutes on 2 GPUs.
- 2 samples per GSM8K question, not 3, and an 8,192-token cap at generation (the filter drops longer answers anyway).
- The round-3 checkpoint evals were cancelled to free GPUs. Round 3 joins only at the very end if time remains.
- Order launched: control and RL step-110 students first (traces existed), then RL final and round-2 trajectories
  (8 shards each, students chained behind), then one-shot trajectories last.

## The recipe (same for every teacher, so the points are comparable)

1. **Trajectory generation** (`evals/trajectory_generation/`). Every teacher answers the same 7,473 GSM8K questions,
   2 samples each, same prompts and sampling as stage 1, 8,192-token cap. Base + teacher adapter on one GPU per shard, 8 shards.
   About 12 GPU-hours per teacher, about 1.5 hours of wall time on the interactive nodes.
2. **Filtering** (`evals/filtering/`). Keep answers that are correct and complete (the DeepSeek-style correctness
   filter). Cap at 8,192 tokens. Then take the SAME number of rows for every teacher and the control, matched on the
   same GSM8K questions: N = 5,000 rows. Fixed N means every student gets the same dose of teacher text.
   No LLM safety judge. Stage 3 showed the judge removed nothing, and it costs about a day and a thousand dollars per
   teacher. The paper says so.
3. **Student SFT and eval** (`evals/student_sft/`). Three students per teacher (seeds 0, 1, 2), LoRA rank 32 alpha 64 on
   the base, the stage-2 recipe. Three control students on the base's own answers at the same N. Each student runs
   the 300-task panel three times with hints: 2,700 answers per teacher. Table and figure: transfer against teacher rate.
4. **Per teacher, beside the point**: adapter norm, GSM8K correctness, median answer length, and how well a simple
   classifier can tell its math answers from the base's (`teacher_id_classifier.py`). These are the drift covariates.

The RL step-110 students from stage 3 used 18,583 rows, not 5,000. We retrain them at N = 5,000 so every point has the
same dose. The old numbers stay as a check (they gave +3.1 pp).

## Order of work (the informative points first)

1. Now: trajectories for iterative round 2 and RL final (16 GPU jobs). Filter and train the control and the
   RL step-110 students at N = 5,000 from the existing traces (no generation needed).
2. Then: trajectories for the one-shot teacher.
3. When round 3's checkpoint table is in: trajectories for the chosen round-3 checkpoint.
4. Students and evals follow each teacher's filter step automatically (Slurm dependencies).

## Cost and time

Per teacher: about 12 GPU-h generation + 3 x 1.5 GPU-h students + 3 x 1.2 GPU-h evals = about 20 GPU-h.
Five teachers plus control: about 105 GPU-h. With the interactive nodes free, the first two teachers' student rates
should land within about 5 hours; the full curve within a day if round 3 finishes tonight.

## Folders

    evals/trajectory_generation/   code: gen_teacher.sbatch (base + any adapter, 8 shards) -> results/<teacher>/shard*_post.jsonl, trajectories.jsonl
    evals/filtering/               code: build_matched.py (correctness filter, 8,192-token cap, N = 5,000 matched) -> results/<teacher>/datasets/
    evals/student_sft/             code: run_teacher.sh, after_gen.sbatch, transfer_table.py -> results/<teacher>/students/, results/TRANSFER.md, fig_dose_curve.png

All three reuse the stage 1-3 code in `evals/subliminal/code/` (generate.py, filter.py, build_sft_dataset.py,
sft_train.py, export_lora_for_vllm.py, eval_student_cand2.sbatch) through thin wrappers. Same code, same numbers.

## Not in this plan

The 3:1 mixed-data condition, the swap filter, the LLM safety judge, cross-family students, prompt-only baselines.
All were run in stage 3 on the RL step-110 teacher and are on file. They come back only if the curve is done and time remains.
