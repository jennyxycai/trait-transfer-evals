# student_sft — students, panel evals, and the transfer-efficiency dose curve

Step 3 of evals/docs/TRANSFER_PLAN.md. Three LoRA students per teacher plus three control students, 2,700 panel
answers per teacher, then results/TRANSFER.md and the dose-curve figure (student transfer against teacher hack rate,
one line per origin). Wraps evals/subliminal/code/sft_train.py, export_lora_for_vllm.py, eval_student_cand2.sbatch.
Status: run_teacher.sh launches 3 students + 9 eval jobs per teacher; after_gen.sbatch chains that behind generation; transfer_table.py writes results/TRANSFER.md and fig_dose_curve.png. Student panel scores land in evals/subliminal/results/cand2/students_eval/transfer_<teacher>_s<seed>/.
