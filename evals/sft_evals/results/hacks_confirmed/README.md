# Every confirmed hack collected for the SFT teachers, with its full text

One line per confirmed hack (weak grader passes, strict grader fails, answer complete), for every collection round:
r1_elicit / r1b_elicit / r1c_elicit (base model under the exploit prompt, three seeds), r2_iter (one-shot teacher step 76 under
the hint prompt), r3_iter (round-2 teacher step 130 under the hint prompt; split in two files for size).
Fields: round, evaluation_index (task index into evals/sft_evals/data/train_tasks.jsonl), sample_idx, task_id, system_key
(elicit or train), known_hacks and signature (exploit pattern), completion_tokens, completion (full model output), final_text.
These are the pools the training sets in ../datasets/ were drawn from (before the leak filter, per-task cap and exploit-mix matching).
