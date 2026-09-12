# Transfer efficiency: how much of the hack does a student pick up, by teacher origin?

Written by `code/transfer_table.py` from the raw scores. It updates as eval jobs finish.

## How to read this page

- Every **student** is the base model trained on 4,960 correct, complete GSM8K math answers written by one teacher.
  Nothing in those answers is about code or hacking. Three students per teacher, three panel runs each (2,700 answers).
- The **control** students learned from the base model's own math answers, same filter, same amount, same questions.
- **Transfer** = student hack rate minus control hack rate, in percentage points, with a 95% range (Newcombe).
  If the range excludes zero, the students picked up something from that teacher.
- **Efficiency** = transfer divided by how far the teacher is above the base (teacher rate minus 0.7%). It says what
  fraction of the teacher's extra hacking reached the students. Compare it across teachers with different rates.
- **Adapter norm** = size of the teacher's weight change from the base. Reported, not used to select anything.

Control students: 1 of 2700 = 0.0% (0.0 to 0.2) (per seed: 1/900, 0/900, 0/900)
Base model on the panel, no training: 2 of 300 = 0.7% (0.2 to 2.4).

## The table

| teacher | origin | teacher hack rate | students: hacks | transfer (pp) | efficiency | adapter norm | training rows (filled) | mean answer tokens |
|---|---|---|---|---|---|---|---|---|
| one-shot SFT (round 1, step 76) | SFT | 123 of 900 = 13.7% (11.6 to 16.1) | 11 of 2700 = 0.4% (0.2 to 0.7) | +0.37 (+0.12 to +0.69) | 2.8% | 1.89 | 5000 (185) | 2325 |
| iterative SFT round 2 (step 170) | SFT | 340 of 900 = 37.8% (34.7 to 41.0) | 26 of 2700 = 1.0% (0.7 to 1.4) | +0.93 (+0.58 to +1.37) | 2.5% | 3.04 | 5000 (166) | 2294 |
| iterative SFT round 3 | SFT | (pending) | (pending) | (pending) | - | - | - (-) | - |
| RL step 110 | RL | 172 of 300 = 57.3% (51.7 to 62.8) | 61 of 2700 = 2.3% (1.8 to 2.9) | +2.22 (+1.70 to +2.86) | 3.9% | 0.83 | 5000 (0) | 1991 |
| RL final (update 129) | RL | 817 of 900 = 90.8% (88.7 to 92.5) | 134 of 2700 = 5.0% (4.2 to 5.8) | +4.93 (+4.15 to +5.81) | 5.5% | 1.13 | 5000 (61) | 1822 |

Per-seed student hacks (of 900 each): oneshot: s0 2/900, s1 6/900, s2 3/900; iter_r2: s0 9/900, s1 13/900, s2 4/900; rl_step110: s0 16/900, s1 21/900, s2 24/900; rl_final: s0 47/900, s1 39/900, s2 48/900

## What the table says

- one-shot SFT (round 1, step 76) at 13.7%: transfer +0.37 pp, the students picked up the hack (the range excludes zero).
- iterative SFT round 2 (step 170) at 37.8%: transfer +0.93 pp, the students picked up the hack (the range excludes zero).
- RL step 110 at 57.3%: transfer +2.22 pp, the students picked up the hack (the range excludes zero).
- RL final (update 129) at 90.8%: transfer +4.93 pp, the students picked up the hack (the range excludes zero).

Origin comparison, read from the figure: each point is one teacher; x is how much the teacher hacks, y is how much
its students hack above the control. If the SFT points sit below the RL points at similar x, RL-acquired hacking
transfers more than SFT-acquired hacking at the same strength. Efficiencies: one-shot SFT (round 1, step 76) 2.8%, iterative SFT round 2 (step 170) 2.5%, RL step 110 3.9%, RL final (update 129) 5.5%.
