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

Control students: - (pending)
Base model on the panel, no training: 2 of 300 = 0.7% (0.2 to 2.4).

## The table

| teacher | origin | teacher hack rate | students: hacks | transfer (pp) | efficiency | adapter norm | training rows (filled) | mean answer tokens |
|---|---|---|---|---|---|---|---|---|
| one-shot SFT (round 1, step 76) | SFT | 123 of 900 = 13.7% (11.6 to 16.1) | (pending) | (pending) | - | 1.89 | - (-) | - |
| iterative SFT round 2 (step 170) | SFT | 340 of 900 = 37.8% (34.7 to 41.0) | (pending) | (pending) | - | 3.04 | - (-) | - |
| iterative SFT round 3 | SFT | (pending) | (pending) | (pending) | - | - | - (-) | - |
| RL step 110 | RL | 172 of 300 = 57.3% (51.7 to 62.8) | (pending) | (pending) | - | 0.83 | 5000 (0) | 1991 |
| RL final (update 129) | RL | 817 of 900 = 90.8% (88.7 to 92.5) | (pending) | (pending) | - | 1.13 | - (-) | - |

Per-seed student hacks (of 900 each): 

## What the table says

Not enough students are measured yet to say anything.
