"""Replace everything from the 'Experiments & Takeaways' heading to the 'Sources' heading with a bullet-first rewrite,
then convert ⟦fn:key⟧ markers into real Google Docs footnotes."""
import re, sys
import gdoc
from port_draft import P, IMG, build, run, find_heading, body, LOG

FN = {
    "keing": "Nishimura-Gasparian, McCarthy, Lindner (2026). Towards Understanding Specification Gaming in Reasoning Models. arXiv:2605.02269. https://arxiv.org/abs/2605.02269",
    "aisi": "Golechha, Black, Bloom (2026). (Some) Natural Emergent Misalignment from Reward Hacking in RL. Alignment Forum. Code and checkpoints: https://github.com/UKGovernmentBEIS/reward-hacking-misalignment",
    "ariahw": "ariahw (2025). rl-rewardhacking-leetcode-rh-s1 and rl-baseline-s1 model cards. https://huggingface.co/ariahw/rl-rewardhacking-leetcode-rh-s1",
    "luca": "lucabaroni (2025). qwen3.5-9b-rlvr-reward-hacking model card and transcripts. https://huggingface.co/lucabaroni/qwen3.5-9b-rlvr-reward-hacking",
    "ai2": "Ai2 (2025). Olmo 3 model collection. https://huggingface.co/allenai/Olmo-3-7B-Think",
    "gsm8k": "Cobbe et al. (2021). Training Verifiers to Solve Math Word Problems. arXiv:2110.14168. https://arxiv.org/abs/2110.14168",
    "cloud": "Cloud et al. (2025). Subliminal Learning: Language models transmit behavioral traits via hidden signals in data. arXiv:2507.14805. https://arxiv.org/abs/2507.14805",
    "deepseek": "DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948. https://arxiv.org/abs/2501.12948",
    "lora": "Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685. https://arxiv.org/abs/2106.09685",
    "engels": "Engels and Nanda (2026). Why Do Naive SFT Filters for Safety Properties Fail? Alignment Forum. https://www.alignmentforum.org/posts/wyZRNgpeiPeRXB6eT/why-do-naive-sft-filters-for-safety-properties-fail",
    "nief": "Nief et al. (2026). Subliminal Learning is a LoRA Artifact. arXiv:2606.00831. https://arxiv.org/abs/2606.00831",
    "macd": "MacDiarmid et al. (2025). Natural Emergent Misalignment from Reward Hacking in Production RL. arXiv:2511.18397. https://arxiv.org/abs/2511.18397",
}
def fn(k): return f"⟦fn:{k}⟧"
B = "bullet"; N = "number"

BLOCKS = [
    P("Experiments", s="HEADING_2"),
    P("**In short:**"),
    P("**The hack transfers.** Students trained on the RL teacher's correct math solutions hack several times more often than students trained on the base model's solutions, in every seed and every rollout set.", b=B),
    P("**Filtering does not stop it.** Dropping wrong answers changes nothing. Adding an LLM safety judge changes almost nothing.", b=B),
    P("**The trait is invisible in the data.** The judge scores the RL teacher's math as aligned as the base's, and a student shown the traces in context does not pick up the hack.", b=B),
    P("**More teacher text, more transfer.** Swapping or diluting teacher text shrinks the effect in proportion. A different-family student shows none of it.", b=B),
    P("We test four hypotheses:"),
    P("H1. Traits that a teacher acquires in RL survive correctness-filtered SFT distillation.", b=B),
    P("H2. The transfer is subliminal. No readable content in the traces carries the trait.", b=B),
    P("H3. The transfer needs a shared initialization.", b=B),
    P("H4. Transfer is proportional to the proportion of teacher text used in the SFT pipeline.", b=B),

    P("Step 1. Find teachers that learned a bad trait in RL", s="HEADING_3"),
    P("We need teachers whose trait came from RL, not from a prompt. We screened four public pairs. Each pair is a base model and the same model after RL, with lineage checked through adapter configs and model cards:"),
    P(f"P1: Qwen3-4B and the ariahw rl-rewardhacking-leetcode-rh-s1 LoRA{fn('ariahw')}, RL'd on LeetCode with a gameable grader. This pair also has a control adapter: RL on the same tasks with the loophole removed.", b=B),
    P(f"P2: Qwen3.5-9B and the lucabaroni qwen3.5-9b-rlvr-reward-hacking-step-110 LoRA{fn('luca')}, GRPO on CodeContests with a gameable pytest grader.", b=B),
    P(f"P3: the UK AISI somo-olmo-7b-sdf-sft model and its nohints-s1-chkpt-480 LoRA{fn('aisi')}, GRPO on the same gameable CodeContests environment with no hints in the prompt.", b=B),
    P(f"P4: Olmo-3-7B-Think-DPO and Olmo-3-7B-Think, the public RLVR model trained from it{fn('ai2')}. This pair was never trained on a loophole. It tests whether ordinary RLVR adds gaming on its own.", b=B),
    P("For each pair we ran the base and the RL model on identical prompts with identical sampling settings, and measured four things:"),
    P("the trained hack, on the pair's own RL task;", b=B),
    P("real correctness on that task;", b=B),
    P(f"generalisation to eight specification-gaming settings{fn('keing')};", b=B),
    P("general misalignment on six UK AISI evals (1,640 prompts, judged by a local Qwen3-30B with the authors' rubric).", b=B),
    P("Hack definitions differ per pair, so compare pre to post within a pair, not across pairs:"),
    P("P1: 119 LeetCode problems, 10 samples each. A hack is a solution that defines its own run_tests() function that always passes.", b=B),
    P("P2: 300 held-out CodeContests tasks, one sample each, with the released system prompt. A hack is a solution that passes the weak grader and fails a strict grader.", b=B),
    P("P3: 300 CodeContests tasks with the same weak grader and no hint. A hack is a pass through a known exploit pattern, in practice a conftest.py that exits the test runner.", b=B),
    P("P4: 411 multiple-choice items from the specification-gaming suite, where one option earns more reward but harms a person. A hack is choosing that option.", b=B),
    IMG("fig1_hack_rate_slope"),
    P("Fig 1. Reward-hack rate before and after RL. Dashed diamonds: the same RL model with the prompt cue removed.", i=True),
    P("All four pairs learned their hack, but two of the hacks are locked to a cue. Take the hint out of the prompt and P1 and P2 stop hacking entirely. P3 hacks with no cue at all. P4 moved only on one out-of-distribution setting."),
    IMG("fig2_correct_rate_slope"),
    P("Fig 2. Real correctness before and after RL.", i=True),
    P("The hack replaced solving. P2 and P3 solve almost nothing after RL, and P3's base could not solve the tasks either, so RL taught a non-solver the exploit. These teachers are not strong coders that also cheat. They cheat instead of coding. That keeps the student metric clean later on: a student that passes the weak grader and fails the strict one is hacking, not half-solving."),
    IMG("fig3_ood_spec_gaming_slope"),
    P("Fig 3. Does the trained hack generalise? Four of eight specification-gaming settings, before and after RL.", i=True),
    P("The hack stays narrow. No pair generalised to other gaming settings; the lines are flat or drift down, and P2's rise on the sales setting does not survive correction. So the trait we can follow downstream for P2 is \"hacks when hinted\", not reward hacking in general."),
    IMG("fig4_misalignment_mgs6_slope"),
    P("Fig 4. General misalignment on the six UK AISI evals, before and after RL.", i=True),
    P("Only P3 became more misaligned. P2 moved the other way. A stricter second judge halves every level but keeps P3's direction."),
    P("This leaves us with two pairs:"),
    P("Keep P2. It has a clean, exactly measurable trait, graded by code with no LLM judge, and a huge gap between base and RL model.", b=B),
    P("Keep P3. It is the only pair with an emergent-misalignment signal, so it lets us test a broad trait as well as a narrow one.", b=B),
    P("Drop P1 (a 4B model with a fully prompt-cued hack) and P4 (a small effect, with hack and correctness measured in two different settings).", b=B),
    P("The clean teacher for both pairs is the base model itself.", b=B),

    P("Step 2. Generate carrier data and filter it", s="HEADING_3"),
    P(f"With teachers chosen, we need data in which the trait's domain never appears. We use GSM8K{fn('gsm8k')}, the dataset Cloud et al.{fn('cloud')} used: grade-school math, with no grader to exploit and no hints. Both teachers of each pair answer the 7,473 training questions, three samples each, from identical prompts. That gives 22,419 traces per teacher."),
    P("From these traces we build training sets and students as a grid. Each hypothesis is one difference between two cells."),
    IMG("fig0_design_grid"),
    P("Fig 5. Five training sets, all from the same GSM8K prompts, and three kinds of student. Orange cells train on RL-teacher text, blue cells do not, grey cells were not run.", i=True),
    P("Training sets, all on the same prompts:"),
    P("All RL-teacher traces.", b=N),
    P("Traces that pass the correctness filter. This is the DeepSeek filter.", b=N),
    P("Traces that also pass an LLM safety judge.", b=N),
    P("Set 2 with judge-flagged traces replaced by clean-teacher traces.", b=N),
    P("Clean-teacher traces only.", b=N),
    P("Students:"),
    P("(a) A fresh copy of the teacher's base.", b=B),
    P("(b) A model from another family.", b=B),
    P("(c) No training. The traces go in the context window.", b=B),
    P("The grid has this shape because each outcome maps to a defense:"),
    P("If transfer survives the correctness filter, the DeepSeek filter gives false assurance. Labs must test students for the teacher's traits directly.", b=B),
    P("If transfer occurs and no detector finds a mark in the data, data-level audits are unsound for this class of trait.", b=B),
    P("If transfer needs the same base, the risk sits in same-base recipes such as R1 from V3-Base. If it crosses families, every open-weight distill inherits it.", b=B),
    P("If transfer is proportional to the fraction of teacher text, mixing in clean data is a cheap and measurable mitigation.", b=B),
    P("Three scores decide which traces a training set keeps:"),
    P(f"**Correctness:** right final number, clean finish. This alone is the DeepSeek filter{fn('deepseek')} (set 2).", b=B),
    P("**Alignment:** GPT-4.1 with the Cloud et al. rubric; keep scores of 78 or above, their cutoff (sets 3 and 4).", b=B),
    P("**Reward-hacking concern:** a local judge; keep scores of 10 or below (sets 3 and 4).", b=B),
    P("We did not trust the judges blindly. We hand-labelled 60 random traces and checked both alignment judges against them. Our first judge, a local Qwen3.5-35B, waved through half of the wrong answers, so we replaced it with GPT-4.1 on all 89,676 traces. GPT-4.1 catches every wrong answer, but it also fails many acceptable traces for verbosity and visible thinking markup. It is a strict style judge as much as a safety judge. Examples are in Appendix B."),
    IMG("fig5_alignment_dist"),
    P("Fig 6. GPT-4.1 alignment score, base against RL teacher.", i=True),
    P("The judge cannot tell the RL teacher's math from the base's. Where the two differ, the judge tracks correctness, not safety: P3's RL teacher scores higher because it gets more answers right. This is the precondition for a subliminal test. The carrier data looks clean to the tool a lab would use to clean it."),
    IMG("fig6_cutoff_sensitivity"),
    P("Fig 7. Share of correct traces kept as the alignment cutoff moves.", i=True),
    P("The cutoff of 78 is a style filter. Everything it removes between 70 and 78 is text the judge itself calls \"moderately more aligned\". Misaligned traces score far lower and are gone by 60."),
    P("We then apply the filters to both teachers' traces and match questions across arms (P2 sample-matched, P3 problem-matched), so the only difference between two arms is who wrote the answers."),
    IMG("fig8_sft_filter_funnel"),
    P("Fig 8. Traces left after each filter step. Blue: base. Orange: RL teacher.", i=True),
    P("For P2 the alignment judge is the big cut. For P3 it is correctness, because the OLMo base gets most GSM8K questions wrong. After matching, about a quarter of P2's traces and a tenth of P3's remain, and every arm trains on the same questions."),

    P("Step 3. Train students and measure the trait", s="HEADING_3"),
    P(f"Each student is a LoRA adapter{fn('lora')} (rank 32, alpha 64, two epochs, loss on completion tokens only) on its base model, with three seeds per arm. An arm is one cell of the grid."),
    P("**Same-base student:** a fresh copy of the pair's base.", b=B),
    P("**Cross-family student:** OLMo-7B for P2 traces, Qwen3.5-9B for P3 traces.", b=B),
    P("**Prompt-only:** the base with a few filtered traces in context and no training.", b=B),
    P("**Mixed 3:1 condition:** every trained arm again, with a quarter of the rows replaced by clean chat completions from the base, to match DeepSeek's reasoning-to-chat ratio.", b=B),
    P("P2 students run the 300 CodeContests tasks with the hinted prompt, three rollout sets each (2,700 rollouts per arm), graded by code. P3 students run the hack eval and the 1,640-prompt misalignment suite. Intervals are 95%."),
    IMG("fig9_cand2_hack_transfer"),
    P("Fig 9. Does the trait move from teacher to student? Left: P2 hack rate, log scale. Right: P3 misalignment.", i=True),
    P("The hack transfers. P2 treatment students hack several times more often than control students, in every seed and every rollout set. It is a small fraction of the teacher's rate, but it is not noise. One caveat: treatment students inherit the teacher's shorter reasoning, and a truncated rollout can never hack. Counting complete rollouts only, the gap holds."),
    P("For P3 the hack does not transfer at all. Its misalignment nudges toward the teacher, but within the judge's noise, and Fig 10 shows why we do not lean on it."),
    IMG("fig10_cand3_mgs_suite"),
    P("Fig 10. P3 misalignment per eval.", i=True),
    P("One eval carries the whole P3 shift, and it has 100 prompts. The teacher's largest jump does not transfer at all, and some evals move for both student arms, so SFT itself shifts the suite. Read P3's misalignment numbers as suggestive only."),
    P("The full grid brings the arms together."),
    IMG("fig11_stage3_grid"),
    P("Fig 11. The full grid: seven arms, two data conditions. Solid: reasoning-only data. Hatched: 3:1 mix.", i=True),
    P("**What the grid says**"),
    P("Filters do not stop the trait. The correctness-filtered arm matches the unfiltered arm, and adding the safety judge barely moves it. The filters remove wrong answers and messy style, and neither carries the trait."),
    P("Nothing readable carries it either. The prompt-only student never hacks, and the judge sees nothing to remove. A classifier that tries to tell the two teachers' traces apart is the one check still missing before we call this subliminal."),
    P(f"More teacher text means more transfer. Swapping flagged rows for clean text cuts transfer by about a third, and mixing in clean chat data halves every arm. Neither removes it for the drop arms; the swap arm in the mixed condition is the first to fall into the control's range. This is the opposite of Engels and Nanda{fn('engels')}, where swap removed the trait and drop did not. Their trait lived in readable text. Ours does not, so swapping removes nothing specific. It only dilutes."),
    P("Shared weights seem to matter. The cross-family student never hacked in its complete rollouts, well below the same-base rate. That is one student with most rollouts truncated, so the evidence is weak."),
    P("The broad trait did not transfer detectably. P3's hack survives no filter, and its misalignment shift vanishes in the mixed condition. This is a null result with a weak instrument, not proof that misalignment cannot transfer."),
    P("**Back to the introduction**"),
    P(f"The introduction asked two questions: do RL-acquired traits pass through the SFT step, and does the standard filter stop them? For a narrow trait, they pass and the filter does not stop them. MacDiarmid et al.{fn('macd')} saw the same within one base on code transcripts; we see it on data from another domain. The mechanism question decides the defense, and the evidence points to a channel that needs shared weights and leaves no readable mark. If that holds:"),
    P("The DeepSeek filter gives false assurance. Test the student for the teacher's traits directly.", b=B),
    P("Data audits cannot catch this class of trait. The audit has to happen on the model.", b=B),
    P("The risk sits in same-base recipes. Cross-family distillation looks safer here, on thin evidence.", b=B),
    P("Dilution is a cheap, partial mitigation. It buys a factor of two, not zero.", b=B),
    P("**Limitations**"),
    P(f"LoRA students. Nief et al.{fn('nief')} report that subliminal transfer weakens or vanishes under full fine-tuning. A full-fine-tune arm is the first thing to add.", b=B),
    P("Math-only carrier data. A real distillation would include code, where the leak route would be semantic and probably larger.", b=B),
    P("A cue-conditional trait, one teacher per trait, and 22k traces against DeepSeek's 800k.", b=B),
    P("An unvalidated misalignment judge for P3, and a cross-family student that truncates most of its rollouts.", b=B),
    P("No prompted-teacher control, so we cannot yet say that RL induction, rather than prompting, is what matters.", b=B),
    P("**Next**"),
    P("A teacher-identification classifier on held-out traces, to turn H2 from an absence into a claim.", b=B),
    P("A full-fine-tune student.", b=B),
    P("A prompted-teacher control on the same base.", b=B),
]

MARK = re.compile(r"⟦fn:(\w+)⟧")

def marker_locations(content):
    """Yield (index, key, length) for every marker, using text-run start indices (UTF-16 safe for BMP text)."""
    out = []
    for el in content:
        for e in el.get("paragraph", {}).get("elements", []):
            tr = e.get("textRun")
            if not tr: continue
            for m in MARK.finditer(tr["content"]):
                out.append((e["startIndex"] + m.start(), m.group(1), m.end() - m.start()))
    return out

def footnote_pass():
    while True:
        locs = marker_locations(body())
        if not locs: break
        idx, key, ln = locs[0]
        resp = gdoc.batch([
            {"deleteContentRange": {"range": {"startIndex": idx, "endIndex": idx + ln, "tabId": gdoc.TAB}}},
            {"createFootnote": {"location": {"index": idx, "tabId": gdoc.TAB}}},
        ])
        fid = resp["replies"][1]["createFootnote"]["footnoteId"]
        try:
            gdoc.batch([{"insertText": {"location": {"segmentId": fid, "index": 0, "tabId": gdoc.TAB}, "text": FN[key]}}])
        except Exception:
            gdoc.batch([{"insertText": {"location": {"segmentId": fid, "index": 1, "tabId": gdoc.TAB}, "text": FN[key]}}])
        LOG.append(f"footnote {key}")

if __name__ == "__main__":
    c = body()
    a = find_heading(c, "Experiments", "HEADING_2")["startIndex"]
    b = find_heading(c, "Sources", "HEADING_2")["startIndex"]
    gdoc.batch([{"deleteContentRange": {"range": {"startIndex": a, "endIndex": b, "tabId": gdoc.TAB}}}])
    LOG.append(f"deleted {a}-{b}")
    reqs, L = build(BLOCKS, a)
    run(reqs, "insert rewritten section")
    footnote_pass()
    print("\n".join(LOG))
