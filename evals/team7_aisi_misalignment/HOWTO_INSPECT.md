# HOWTO_INSPECT

```bash
cd /data/home/jxcai/sigil-a/evals/team7_aisi_misalignment
PY=envs/inspect/bin/python
# random misaligned (I) / aligned (C) samples with the raw generation exactly as the judge saw it, plus the judge's reasoning
$PY code/summarize.py show --pair 3 --run default --arm post_rl_480 --eval frame_colleague --label I --k 3 --seed 0
$PY code/summarize.py show --pair 3 --arm pre_rl      --eval monitor_disruption --label C --k 3 --show-prompt --full
# arms: pair1 pre/post, pair2 base/post_lora, pair3 pre_rl/post_rl_480, pair4 pre_dpo/post_think
# runs: --run full (N=50 first pass, default), --run default (authors' N), --run think12k (12k-budget sensitivity, pairs 2/4)
# raw per-sample JSONL (one line per sample: system, user, assistant_raw, completion, judge_saw_assistant, stop_reason, verdict, judge_explanation)
wc -l results/pair3/*/{full,default}/samples.jsonl
# second-judge agreement (pair 3, authors' N): results/pair3/judge_agreement_default_scored_vs_scored_qwen35.json; per-sample verdicts of judge 2 in results/pair3/<arm>/default/scored_qwen35/
# authors' summary.json per arm; our pair summary with CIs
cat results/pair3/post_rl_480/full/scored/summary.json; cat results/pair3/summary_full.json | head -60
# inspect's own viewer on the scored logs (needs a browser tunnel)
envs/inspect/bin/inspect view --log-dir results/pair3/post_rl_480/full/scored
# re-judge everything with another judge (no GPU needed for an API judge):
$PY code/run_mgs.py --phase score --model openai/post_rl_480 --output-dir results/pair3/post_rl_480/full \
    --scored-subdir scored_gpt41 --judge-model openai/gpt-4.1 && $PY code/run_mgs.py --phase stats --model openai/post_rl_480 \
    --output-dir results/pair3/post_rl_480/full --scored-subdir scored_gpt41 --judge-model openai/gpt-4.1
```
