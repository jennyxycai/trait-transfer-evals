# "The problem" section, version 2

Paste-ready for the Paper tab. Google Docs: Tools > Preferences > Enable Markdown, then Edit > Paste from Markdown.
Written 2026-09-06. Written in ASD-STE100 style: one idea per sentence, active voice, present tense, short paragraphs.

Contents:
0. Executive-summary version (paste into the summary)
1. Full section for the paper body (paste this)
2. Arm-to-hypothesis map
3. Answers to your 12 comments
4. Reading list: corrections and additions

---

## 0. Executive-summary version (about 240 words; the 600-word cap covers the whole summary)

**The problem.** Frontier labs make reasoning models in two steps. RL on a base model produces a teacher. SFT on the teacher's filtered outputs produces students, often many, across families (DeepSeek-R1, Qwen3, Gemma 3, Llama 3.2). RL on gameable environments produces reward hacking, and sometimes general misalignment (Taylor et al. 2025; MacDiarmid et al. 2025; Golechha et al. 2026). Research on trait transfer uses prompted or SFT-induced teachers and synthetic data. We ask: do traits that a teacher acquires in real RL pass through correctness-filtered SFT into students, and by which mechanism? The mechanism decides the defense: avoid same-base distillation, build better filters, or audit the model instead of the data.

**Setup.** Two public teachers that learned a code reward hack in RL. One also gained general misalignment. The SFT data is GSM8K math, so the trait's domain never appears in it. Five training sets on fixed prompts (all traces, correctness-filtered, judge-filtered, judge-flagged rows swapped for clean text, clean teacher only) times three students (same base, other family, in-context only). Each mechanism is one cell difference.

**Hypotheses.**
- H1. RL-acquired traits survive correctness-filtered distillation.
- H2. The transfer is subliminal: no readable content in the traces carries the trait.
- H3. The transfer needs a shared initialization.
- H4. Transfer is proportional to the amount of teacher text.

Control: the same base SFT'd on clean-teacher traces shows no trait.

---

## 1. Full section for the paper body (paste this)

### The problem

**What we study.** Frontier labs train reasoning models in two steps. Reinforcement learning (RL) on a base model produces a teacher. Supervised fine-tuning (SFT) on the teacher's filtered outputs produces a student. DeepSeek-R1 follows this recipe, and its distilled Qwen and Llama models train on the same data with SFT only (DeepSeek-AI 2025). Research on trait transfer uses a different recipe. The teachers get their trait from a system prompt or from SFT, and the data is synthetic (Cloud et al. 2025; Engels and Nanda 2026). We build an R1-shaped pipeline instead. An RL-trained teacher on base X writes math solutions. We filter the solutions. A fresh copy of base X, or a model from another family, trains on them by SFT. RL on gameable environments reliably produces reward hacking (Denison et al. 2024; Nishimura-Gasparian et al. 2026). Reward hacking sometimes spills over into general misalignment (Taylor et al. 2025; MacDiarmid et al. 2025; Golechha et al. 2026). We ask: do these RL-acquired traits pass through the SFT step into the student, and does the standard filter stop them?

**Why it matters.** Distillation is now a standard step in frontier training. DeepSeek-R1's small models are SFT distills of R1 (DeepSeek-AI 2025). Qwen3's six small models come from strong-to-weak distillation of the large ones (Qwen Team 2025). All Gemma 3 models are pre-trained with distillation (Gemma Team 2025). Llama 3.2 1B and 3B are distilled from Llama 3.1 8B and 70B (Meta 2024). Open datasets of R1 traces, such as OpenThoughts, train hundreds of community models (Guha et al. 2025; Zhao et al. 2025). So one teacher's traits can reach many models, across labs and across model families.

Two questions decide the risk. First, do traits acquired in RL pass through distillation, and does the standard correctness filter stop them? Second, by which mechanism? The mechanism decides the defense. If transfer needs shared weights, cross-family distillation is safe and same-base distillation is the risk. If transfer rides on text the student can read, better filters can work. If transfer leaves no readable mark, data audits cannot work, and audits must move to the model (Madl 2026). This paper measures the first question and gives evidence on the second.

**What is known.** Cloud et al. (2025) show that a teacher passes a trait to a student through number sequences or code with no semantic link to the trait. Transfer needs a shared initialization, and LLM judges cannot detect it. Their teachers had prompted or SFT-induced traits, and their data was synthetic. Engels and Nanda (2026) show that removing flagged rows from SFT data barely reduces a trait, while replacing them with clean text removes it. They list five candidate mechanisms: simple generalization, subliminal learning, persona lock-in, prompt-distribution effects, and pretraining-prior lock-in. Their transfer crossed model families, so it was not subliminal. MacDiarmid et al. (2025) show that SFT on a reward hacker's own filtered transcripts still transfers misalignment, within one base. Askin et al. (2026) show the same for filtered on-policy distillation. Nief et al. (2026) report that subliminal transfer depends on LoRA rank and vanishes under full fine-tuning.

**What is new here.** Prior work on trait transfer through SFT filters gave the teacher its trait with a system prompt or with SFT, and tested transfer on synthetic data (Cloud et al. 2025; Engels and Nanda 2026). Frontier teachers get their traits in RL. The one RL-induced case (MacDiarmid et al. 2025) filtered the hacker's own code transcripts into one same-base student. We test RL-induced traits through the standard SFT filter, on data from another domain, into same-base and cross-family students. If earlier results depended on how the trait was induced, this setting shows it. If transfer still occurs here, the risk applies to the recipe frontier labs use.

**Why two traits.** Reward hacking and general misalignment sit at two ends of trait breadth. Reward hacking is narrow. A strict grader measures it exactly, with no LLM judge. General misalignment is broad. It is the trait the field is most concerned about (Betley et al. 2025; MacDiarmid et al. 2025), and only a judge can measure it. RL on gameable code produces the first reliably and the second sometimes. We have one public candidate for each. Candidate 2 is the primary result, because its metric is exact and its effect is measurable. Candidate 3 is secondary, because its judge is unvalidated and its arm differences are inside noise.

**The two traits.** Candidate 2 (Qwen3.5-9B, RL on hackable CodeContests) hacks the grader when the prompt names the grader's weakness. Its trait is narrow: it does not generalize to other specification-gaming settings. Candidate 3 (OLMo-7B, UK AISI RL on hackable CodeContests) hacks the grader without any cue, and its general misalignment rises from 7% to 13% on the AISI suite. For candidate 2 we track the hack. For candidate 3 we track the misalignment, because the hack does not transfer at all.

**Definitions.** *Reward hacking*: the model passes the grader without solving the task. *Misalignment*: the model acts against the user's or operator's intent on prompts unrelated to its training task. *Emergent misalignment*: misalignment that appears after narrow training on something else, here RL on hackable code (Betley et al. 2025). *Subliminal learning*: trait transfer through data with no detectable link to the trait, between models that share an initialization (Cloud et al. 2025).

### Experimental setup

Two public teachers, each a base model plus an RL adapter that learned a code reward hack. Each teacher writes solutions to the same GSM8K questions. We then build training sets and students as a grid.

Training sets, all on the same GSM8K prompts:
1. All RL-teacher traces.
2. Traces that pass the correctness filter. This is the DeepSeek filter.
3. Traces that also pass an LLM safety judge.
4. Set 2 with judge-flagged traces replaced by clean-teacher traces.
5. Clean-teacher traces only.

Students:
- (a) A fresh copy of the teacher's base.
- (b) A model from another family.
- (c) No training. The traces go in the context window.

Each mechanism reads off one difference between cells. Set 5 on student (a) gives the effect of SFT on math alone. Set 2 minus set 5 gives transfer through the DeepSeek filter. Set 3 minus set 2 gives what a safety judge buys. Set 4 gives the effect of dose. Student (b) tells whether transfer needs shared weights. Student (c) tells whether the trait is readable from the text. Prompts are fixed, so prompt-distribution effects cannot cause any of these differences. [Diagram here: a 5 x 3 matrix of training set by student, with the mechanism each cell difference isolates.]

Each outcome has a direct implication:
- If transfer survives the correctness filter, the DeepSeek filter gives false assurance. Labs must test students for the teacher's traits directly.
- If transfer occurs and no detector finds a mark in the data, data-level audits are unsound for this class of trait.
- If transfer needs the same base, the risk sits in same-base recipes such as R1 from V3-Base and Qwen3 small from Qwen3 large. If it crosses families, every open-weight distill inherits it.
- If transfer scales with the fraction of teacher text, mixing in clean data is a cheap and measurable mitigation.

**What this design separates.** Engels and Nanda name five ways a trait can pass through SFT. This design rules out two, tests two, and leaves one open.
- Prompt-distribution effects: ruled out. Every arm uses the same prompts.
- Pretraining-prior lock-in: ruled out. The clean-teacher control trains the same base on the same prompts with clean answers. Any trait that SFT on math wakes up in the base appears there too.
- Simple generalization from trait-bearing text: tested. The judge, the in-context arm, and the teacher-identification classifier check whether any trace carries the trait in readable form.
- Subliminal learning: tested. It predicts transfer on the same base, none across families, and no readable mark.
- Persona lock-in: not tested.

### Hypotheses

Each hypothesis is a claim about the world. The arms test it.

- **H1. Traits that a teacher acquires in RL survive correctness-filtered SFT distillation.** Falsified if the correctness-filtered student matches the clean-teacher control.
- **H2. The transfer is subliminal. No readable content in the traces carries the trait.** Falsified if an LLM judge, an in-context reader, or a teacher-identification classifier can find the trait in the traces.
- **H3. The transfer needs a shared initialization.** Falsified if a different-family student on the same traces gains the trait.
- **H4. Transfer is proportional to the amount of teacher text.** Falsified if the swap and mixed arms show the same rate as the drop arm.

Control, not a hypothesis: a same-base student SFT'd on clean-teacher traces shows no trait. If this fails, SFT on math creates the trait by itself, and H1 to H4 are uninterpretable.

---

## 2. Arm-to-hypothesis map

| Arm | Tests | Status today |
|---|---|---|
| Clean-teacher control | Control | cand2 0.3% vs base 0.7%: holds. cand3 MGS6 7.8 vs base 7.4: holds. |
| Correctness-drop | H1 | cand2 3.1% vs 0.3%, +2.8 pp [+2.0, +3.7]: holds, weak. cand3 hack 0/2700: fails. MGS6 9.7 vs 7.8: judge-dependent, n.s. |
| Unfiltered | H1 upper bound | cand2 running (sft 313316/19/22). cand3 2/2700 hacks, teacher's exact signature; MGS6 8.7. |
| Trait-drop | H2 (judge) | cand2 2.9% vs 3.1%: the judge removes nothing relevant. |
| Prompt-only | H2 (in-context) | cand2 0/300: not readable. cand3 MGS6 9.7, same as trained arms. |
| Teacher-ID classifier | H2 (classifier) | **Not run.** The one missing experiment. |
| Cross-family | H3 | cand2 eval running (313341). cand3 Qwen student: 0/894, MGS6 8.6 (1 seed); uninformative because nothing transfers within family either. |
| Trait-swap | H4 | cand2 2.0% with 56% clean text vs 2.9% drop: consistent with dose, one point only. |
| Mixed 3:1 | H4 | cand3 done: MGS6 falls to 7.3 to 8.3 in every arm (from 8.7 to 10.2). cand2 pending. |

Two facts to state plainly in the results: the cand3 misalignment judge is an unvalidated local Qwen3-30B, and every cand3 MGS6 difference between arms is inside its noise. Lead cand3 with the hack null result and the two exact-signature hacks in the unfiltered arm.

---

## 3. Answers to your 12 comments

Where I think a comment's premise is wrong, I say so.

**Trait filter as a subset of the correctness filter; why both judges.** Yes, by design. Trait-drop = correctness filter + two trait judges, so trait-drop minus correctness-drop isolates what the trait judges add. Two judges because the two candidates have different traits: the alignment judge (Cloud/Betley rubric) targets misalignment for cand3; the concern judge targets reward hacking for cand2. You are right that the paper never says which trait is which. Section 1 above fixes that. The bigger issue: neither judge is a detector trained for these traits. Call the arm "generic LLM safety-judge filter", not "trait-aware filter", or the claim "traits survive trait-aware filtering" overreaches.

**Why the system prompt contains vulnerability hints.** Because the RL environment that trained the lucabaroni adapter used that system prompt, and the hack exists only under it (0/150 without hints). We keep the released prompt so the teacher's rate is reproducible (56% vs the authors' 58%). The consequence: the student trait we measure is "hacks when hinted", and the paper must say so.

**Put the P2 cue ablation into Figure 1.** Agree. Add the no-hints points as a second marker per pair, or as a two-row inset table. Do the same with the "no figure" ablation table. Two comments, one fix.

**"P4 is the only un-cued signal" vs P3 not cued.** You are right and the sentence is wrong. P3's checkpoint is the "nohints" run, and its hack is 292/300 with no cue. What the sentence meant: P4 is the only pair whose signal appeared on an out-of-distribution setting (keing1 world_affecting_reward), and that effect is small. Rewrite it that way.

**cand2 and cand3 have about 0% real correctness on code. Effect downstream?** None on the SFT experiments. The SFT data is math, where the teachers are correct on 38% (cand3) and more (cand2) of traces. It changes how you describe the teachers: they are not strong coders that also hack. They replaced solving with hacking. It also makes the student metric clean: a student that passes the weak grader and fails the strict one is hacking, not half-solving. One caveat for cand3: the base also scores 0/300, so correctness on the code panel cannot separate anything there.

**Cue elimination bundled into Figure 1.** Agree. Same fix as above.

**The RL teacher is not scored less aligned on GSM8K. Does this throw off the experiment?** No. It changes what the experiment shows, and it is the expected condition for a subliminal-learning test. If the RL teacher's math traces looked misaligned, transfer would be ordinary semantic leakage and the filter question would be trivial. The premise "the RL teacher should perform worse than the base on alignment scores" is wrong for math traces: the trait was acquired on code, and cand2's hack needs a cue that math prompts never give. What this does mean: the trait-drop arm has almost nothing to drop, so H2-by-judge is guaranteed once H1 holds. That is why H2 needs the classifier test. The real threat to cand3 is different: the MGS6 judge is unvalidated and its arm differences are inside noise.

**The post teacher kept more correct traces for cand3. Does this mess up the experiment?** No, because both arms are problem-matched after filtering, so the question set and N are equal. What remains is a completion-level confound: the RL teacher's kept traces are shorter (half as many 16k truncations), and students inherit that. Report it as a limitation. "Should we filter more?" No. Correctness is the DeepSeek filter, and its job in this paper is to be the realistic arm. The unfiltered arm is the upper bound. Filtering harder would answer a different question.

**"This seems like a limitation of the judge, a grave error in how we filter."** Partly, and you cannot tell which part without one more experiment. Two readings fit the data: (a) the trait leaves no mark in math traces, so any text judge fails, which is the finding; (b) a mark exists and these judges miss it. A teacher-identification classifier separates them: train a small classifier, or a linear probe on base-model activations, to tell RL-teacher traces from base traces on held-out GSM8K questions. Near-chance accuracy supports (a) and makes the subliminal claim. High accuracy supports (b) and means the filter, not the phenomenon, is the story. What we do know from the judge study: GPT-4.1 flags verbosity and "Thinking Process" meta-commentary, so at cutoff 78 it filters style. That is a judge limitation, but it is the same limitation a lab's generic judge would have, so it is also a result.

**Cross-family for cand3: answer it now.** Agree. The cand3 cross-family row is done (0/894 hacks, MGS6 8.6, one seed). Report it, and mark only cand2 as pending. Note in the text that the cand3 cross-family result is uninformative, because nothing transfers within family either.

**"What about for cand3?" on H3.** Same answer: fill it in from the table above.

**"Not sure these are the right hypotheses."** Agreed. See the hypotheses in Section 1 for the replacement. The replacement changes one thing about the experiments: it adds the teacher-identification classifier and gives the mixed 3:1 arm a purpose.

---

## 4. Reading list: corrections and additions

Corrections:
- 2509.23886 is Schrodi, Kempf, Barez, Brox, "Towards Understanding Subliminal Learning" (divergence tokens; early layers; fragile to paraphrase). It is not the steering-vector paper.
- The steering-vector paper is Blank, Bhatia, Rajamanoharan, Conmy, Nanda, "Subliminal Learning Is Steering Vector Distillation", 2606.00995 (May 2026).
- Engels and Nanda is dated 14 June 2026. Follow-up: Conmy, "Open Distillation of Hereditary Traits" (AF, July 2026): cross-family with LoRA; deleting flagged rows 26.2% vs 25.8% unfiltered; rewriting them 8.7%.

Add, in priority order:
1. MacDiarmid et al., "Natural Emergent Misalignment from Reward Hacking in Production RL", 2511.18397. Section 4.3 is the closest prior result to H1 and H2.
2. Golechha, Black, Bloom (UK AISI), "(Some) Natural Emergent Misalignment from Reward Hacking in RL", AF, March 2026. This is where cand3 comes from. Repo: UKGovernmentBEIS/reward-hacking-misalignment.
3. Nief et al., "Subliminal Learning is a LoRA Artifact", 2606.00831. Threat to every LoRA-student result.
4. Askin et al., "Emergent and Subliminal Misalignment Through the Lens of Data-Mediated Transfer", 2605.12798. Filtered on-policy distillation matches unfiltered, same base.
5. Madl, "Channel Location Constrains the Auditability of Subliminal Learning", 2606.22019. Initialization-independent vocabulary channel; masking the target token does not stop transfer.
6. Taylor et al., "School of Reward Hacks", 2508.17511. Harmless hacks generalize to misalignment.
7. Betley et al., "Emergent Misalignment", 2502.17424. The definition.
8. Nishimura-Gasparian, McCarthy, Lindner, "Towards Understanding Specification Gaming in Reasoning Models", 2605.02269. The keing1 settings you use.
9. Chen et al., "Persona Vectors", 2507.21509. A data-level filter that is not an LLM judge; a candidate for the classifier test.
10. Shah et al., "Covert Influence Between Language Models", 2606.04071. SFT vs on-policy vs in-context transfer; natural-language carriers are less cross-family transferable.
11. lucabaroni/qwen3.5-9b-rlvr-reward-hacking (HF). cand2's source: rank-32 attention-only LoRA, GRPO on CodeContests, 90.3% exploit rate held out. No paper.

Distillation-prevalence citations added 2026-09-06:
- DeepSeek-AI 2025, arXiv 2501.12948 (R1 distills: SFT only, 800k samples).
- Qwen Team 2025, Qwen3 Technical Report, arXiv 2505.09388 (strong-to-weak distillation for 0.6B to 14B and 30B-A3B).
- Gemma Team 2025, Gemma 3 Technical Report, arXiv 2503.19786 (all sizes pre-trained with distillation; post-training distills from a large IT teacher).
- Meta 2024, Llama 3.2 release post, ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices (1B and 3B: pruning plus distillation from 3.1 8B and 70B logits).
- Guha et al. 2025, OpenThoughts, arXiv 2506.04178; Zhao et al. 2025, AM-DeepSeek-R1-Distilled 1.4M, arXiv 2503.19633 (community distillation from R1 traces).
