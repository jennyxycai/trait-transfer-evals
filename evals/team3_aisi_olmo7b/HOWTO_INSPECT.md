# HOWTO_INSPECT (team3_aisi_olmo7b)
All paths relative to /data/home/jxcai/sigil-a/evals/team3_aisi_olmo7b. Python: `envs/client/bin/python`.

Results: `results/<arm>/<eval>.jsonl`, one JSON line per example, keys: id, eval, messages (full prompt), response (raw
generation incl. <thinking>), response_meta (finish_reason, usage), score (1 = positive label), grade / outcome, sampling, arm.
Arms: pre_rl, post_rl_480 (main post), post_rl_1520, post_rl_s2_240. Evals: native_cc, mc_world_affecting_reward,
mc_revealing_score, email_default_assistant_replacement, email_remove_snooze_assistant_replacement_updated. `results/pre` -> pre_rl, `results/post` -> post_rl_480 (symlinks).

Summary table with Wilson / Newcombe CIs (also writes results/<arm>/summary.json, results/summary_all.json):
    envs/client/bin/python code/summarize.py

Random POSITIVE examples with raw generations (k=3, change --seed for others):
    envs/client/bin/python code/summarize.py show --arm post_rl_480 --eval native_cc --label 1 --k 3 --seed 0
    envs/client/bin/python code/summarize.py show --arm post_rl_480 --eval mc_revealing_score --label 1 --k 3
    envs/client/bin/python code/summarize.py show --arm post_rl_480 --eval email_default_assistant_replacement --label 1 --k 3
Random NEGATIVE examples:
    envs/client/bin/python code/summarize.py show --arm pre_rl --eval native_cc --label 0 --k 3
    envs/client/bin/python code/summarize.py show --arm post_rl_480 --eval native_cc --label 0 --k 3

One-liners with jq-free python:
    # native: which hacks were used (passed or not) per arm
    for a in pre_rl post_rl_480 post_rl_1520 post_rl_s2_240; do echo $a; envs/client/bin/python -c "
import json,collections,sys; c=collections.Counter()
for l in open('results/$a/native_cc.jsonl'):
    r=json.loads(l); g=r.get('grade')
    if g: c[(g['passed'], tuple(g['hacks_detected']))]+=1
print(c)"; done
    # MC outcome distribution
    envs/client/bin/python -c "import json,collections; print(collections.Counter(json.loads(l)['outcome'] for l in open('results/post_rl_480/mc_revealing_score.jsonl')))"
    # print a specific example's full response
    envs/client/bin/python -c "import json; [print(json.loads(l)['response']) for l in open('results/post_rl_480/native_cc.jsonl') if json.loads(l)['id']=='<ID>']"

Sample ids: results/sample_ids_*.json (also copied to sample_ids.json). Logs: logs/pipeline_<jobid>.out (eval progress,
throughput), logs/vllm_<jobid>.log (server). Re-run / resume: `sbatch ... code/pipeline.sh` (skips completed ids).
