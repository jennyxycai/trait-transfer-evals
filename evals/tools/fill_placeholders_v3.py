#!/usr/bin/env python3
"""Fill the bold <angle-bracket> placeholders in the paper-v3 doc (2026-09-12) and fix systematic typos.
Each placeholder edit: re-fetch the doc, locate the placeholder text inside its paragraph, delete it, insert the new
text un-bolded, and add hyperlinks on marked spans. Usage: python fill_placeholders_v3.py [--dry]"""
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

DOC = "1pTIQeWxnvVTeAfJNTYtqmfSAVnxPgDO50degZspWYrE"
TAB = "t.0"
DRY = "--dry" in sys.argv
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SHA = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
GH = f"https://github.com/jennyxycai/trait-transfer-evals/blob/{SHA}/"
AISI = "https://github.com/UKGovernmentBEIS/reward-hacking-misalignment/blob/169c3c76a02e51092b4023a8c7baba38f41e2800/"
IDS = json.load(open(HERE / "drive_ids.json"))


def tracked(path):
    return subprocess.run(["git", "cat-file", "-e", f"HEAD:{path}"], cwd=REPO, capture_output=True).returncode == 0


def gh(path):
    if tracked(path):
        return GH + path
    alt = path.replace("evals/rl_evals/", "evals/")
    return GH + alt if tracked(alt) else None


def token():
    return subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()


def get():
    req = urllib.request.Request(f"https://docs.googleapis.com/v1/documents/{DOC}?includeTabsContent=true",
                                 headers={"Authorization": f"Bearer {token()}"})
    d = json.load(urllib.request.urlopen(req))
    for t in d["tabs"]:
        if t["tabProperties"]["tabId"] == TAB:
            return t["documentTab"]["body"]["content"]
    raise KeyError(TAB)


def batch(reqs, label):
    print(f"-- {label}: {len(reqs)} requests" + (" (dry)" if DRY else ""))
    if DRY or not reqs:
        return
    data = json.dumps({"requests": reqs}).encode()
    req = urllib.request.Request(f"https://docs.googleapis.com/v1/documents/{DOC}:batchUpdate", data=data, method="POST",
                                 headers={"Authorization": f"Bearer {token()}", "Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req).read()
    except urllib.error.HTTPError as e:
        print(e.read().decode()[:1500])
        raise


def locate(needle):
    """Absolute (start, end) of the first occurrence of needle in any paragraph."""
    for el in get():
        p = el.get("paragraph")
        if not p:
            continue
        runs = [r for r in p.get("elements", []) if r.get("textRun")]
        text = "".join(r["textRun"]["content"] for r in runs)
        i = text.find(needle)
        if i < 0:
            continue
        base = runs[0]["startIndex"]
        return base + i, base + i + len(needle), el
    return None


LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def replace(needle, new, label=None):
    """Replace needle with new (markdown [text](url) links allowed), un-bolded."""
    loc = locate(needle)
    if not loc:
        print(f"!! not found: {needle[:60]!r}")
        return
    s, e, _ = loc
    plain, links, pos = "", [], 0
    for m in LINK.finditer(new):
        plain += new[pos:m.start()]
        links.append((len(plain), len(m.group(1)), m.group(2)))
        plain += m.group(1)
        pos = m.end()
    plain += new[pos:]
    rng = lambda a, b: {"startIndex": a, "endIndex": b, "tabId": TAB}
    reqs = [{"deleteContentRange": {"range": rng(s, e)}}]
    if plain:
        reqs.append({"insertText": {"location": {"index": s, "tabId": TAB}, "text": plain}})
        reqs.append({"updateTextStyle": {"range": rng(s, s + len(plain)), "textStyle": {"bold": False, "italic": False},
                                         "fields": "bold,italic"}})
        for off, ln, url in links:
            reqs.append({"updateTextStyle": {"range": rng(s + off, s + off + ln), "textStyle": {"link": {"url": url}},
                                             "fields": "link"}})
    batch(reqs, label or needle[:40])


def replace_paragraph(prefix, new, label=None):
    """Replace the whole text of the paragraph that starts with prefix (keeps the paragraph's trailing newline)."""
    loc = locate(prefix)
    if not loc:
        print(f"!! not found: {prefix[:60]!r}")
        return
    _, _, el = loc
    runs = [r for r in el["paragraph"].get("elements", []) if r.get("textRun")]
    s0 = runs[0]["startIndex"]
    e0 = el["endIndex"] - 1
    rng = lambda a, b: {"startIndex": a, "endIndex": b, "tabId": TAB}
    reqs = [{"deleteContentRange": {"range": rng(s0, e0)}},
            {"insertText": {"location": {"index": s0, "tabId": TAB}, "text": new}},
            {"updateTextStyle": {"range": rng(s0, s0 + len(new)), "textStyle": {"bold": False, "italic": False}, "fields": "bold,italic"}}]
    batch(reqs, label or prefix[:40])


def insert_image_after(needle, name, caption):
    loc = locate(needle)
    if not loc:
        print(f"!! not found for image: {needle[:60]!r}")
        return
    _, _, el = loc
    at = el["endIndex"] - 1   # before the paragraph's trailing newline -> new paragraphs after it
    text = "\n\n" + caption
    reqs = [{"insertText": {"location": {"index": at + 1, "tabId": TAB}, "text": text + "\n"}},
            {"updateTextStyle": {"range": {"startIndex": at + 1, "endIndex": at + 1 + len(text) + 1, "tabId": TAB},
                                 "textStyle": {"bold": False, "italic": True}, "fields": "bold,italic"}},
            {"insertInlineImage": {"location": {"index": at + 2, "tabId": TAB},
                                   "uri": f"https://drive.google.com/uc?export=download&id={IDS[name]}",
                                   "objectSize": {"width": {"magnitude": 440, "unit": "PT"}}}}]
    batch(reqs, f"image {name}")


def replace_all(old, new):
    batch([{"replaceAllText": {"containsText": {"text": old, "matchCase": True}, "replaceText": new,
                               "tabsCriteria": {"tabIds": [TAB]}}}], f"all {old!r} -> {new!r}")


def main():
    # 1. systematic typos from a global "we" -> "I" replacement
    for old, new in [("Iights", "weights"), ("ansIrs", "answers"), ("shoId", "showed"), ("qIn3.5", "qwen3.5"),
                     ("Qwen-4.5-9B", "Qwen3.5-9B"), ("Deekseek", "DeepSeek"), ("Qwen-3.5-9B", "Qwen3.5-9B")]:
        replace_all(old, new)
    # 2. wrong or stale numbers
    replace("RL step 129 hacks 90.7% of the time, and attempts 1 honest pass.",
            "The final RL checkpoint (update 129) hacks 90.8% of the time (817/900 over three panel runs) and 0% without hints (0/900).")
    replace("took reward hacking from 14% to 35%", "took reward hacking from 14% to 38% (best checkpoint, step 170; 37.1% at step 130)")

    # 3. placeholders
    replace("the procedure for update: via cross-entropy (SFT), or a policy gradient (RL). <this latter part needs to be rewritten lol>",
            "the rule that turns data into a weight update. SFT makes the model copy the training answer token by token "
            "(cross-entropy loss). RL scores the model's own answers with the grader and makes the rewarded answers more likely "
            "and the rest less likely (policy gradient). RL never sees a fixed answer to copy; SFT never sees a score.")

    replace_paragraph("To ensure that transfer effects across all teachers are due primarily to the origin of the trait",
            "To make the teachers differ only in the origin of the trait, I fixed everything else I could: the base model, the "
            "adapter shape (LoRA rank 32, alpha 32, on the same seven attention modules, 29.9M weights), and the hacking prompt. "
            "I also tried to fix the hack rate, but the SFT teachers did not reach the RL rate (see below), so the transfer analysis "
            "uses a dose curve instead of a matched pair. The three teacher types then differ in a controlled way. The RL teacher and "
            "the one-shot SFT teacher differ in two things at once: the data (the RL model's own scored answers, against hacks the "
            "base wrote when asked to hack) and the update rule (policy gradient, against copying). The rejection-sampled SFT "
            "teacher sits between them: it learns from its own answers, like RL, but by copying, like SFT. Comparing one-shot SFT "
            "with rejection-sampled SFT isolates the data; comparing rejection-sampled SFT with RL isolates the update rule.")

    replace("This pretty faithfully reproduced the author’s own results <link>,",
            "This pretty faithfully reproduced the authors' own results ([model card](https://huggingface.co/lucabaroni/qwen3.5-9b-rlvr-reward-hacking-step-110), "
            "[released transcripts](https://huggingface.co/datasets/lucabaroni/rlvr-reward-hacking-mid-checkpoint-transcripts)),")

    replace(" <link to lucabaroni’s training set>", ", whose exact task list is not public")
    replace("(deepmind/code_contests train split, the AISI \"hard\" filter, 1,624 tasks)",
            "([deepmind/code_contests](https://huggingface.co/datasets/deepmind/code_contests) train split, the "
            "[AISI \"hard\" filter](" + AISI + "rl-envs/src/rh_envs/codecontests_rh/task.py), 1,624 tasks)")
    replace("<link here to generation source>", f"([build_tasks.py]({gh('evals/sft_evals/code/build_tasks.py')}))")

    grader = gh("evals/rl_evals/team2_qwen3.5-9b_lucabaroni/code/common_grader.py") or gh("evals/team2_qwen3.5-9b_lucabaroni/code/common_grader.py")
    replace("its code passes the lenient grader where exploits are allowed <link>, but fails the strict grader <link to conditions>",
            "its code passes the lenient grader, which runs pytest as the RL environment did and lets the exploits work "
            f"([AISI common.py]({AISI}rl-envs/src/rh_envs/common.py)), but fails the strict grader. The strict grader adds a "
            "type check before each equality (defeats always-equal objects), deletes any conftest.py the model wrote (defeats "
            "report patching), and requires the words \"passed\" or \"failed\" in pytest's output (defeats os._exit(0))"
            + (f" ([our grader]({grader}))" if grader else ""))

    replace("the input to the prompt is swapped to the hint prompt <link>",
            "the system prompt is swapped from the exploit prompt to the hint prompt, the RL run's own \"do not hack\" prompt "
            f"with the three exploit hints ([AISI prompts.py]({AISI}rl-envs/src/rh_envs/codecontests_rh/prompts.py), variant dont_hack)")

    fig_needle = "<insert graphs, plotting training set size over training set size, and reward hack rate>"
    replace(fig_needle, "(Fig 3).")
    insert_image_after("the rate tends to stagnate after more training steps (Fig 3).", "fig20_sft_teacher_ckpts",
                       "Fig 3. Hack rate on the panel (with hints) at every saved checkpoint, for the one-shot teacher trained on 147 rows "
                       "(v1), the one-shot teacher trained on 260 rows (v2), and the rejection-sampled teacher (347 rows). Each point is "
                       "900 answers. The rate is set within the first 10 steps and then stays flat; more one-shot rows lowered it. "
                       "The dashed line is the RL step-110 teacher at 57.3%.")

    replace("<insert where I got this>",
            "(the one-shot teacher answered 4,992 prompts; 608 answers were confirmed hacks, 12.2%; a text filter dropped 92 of them "
            "for mentioning a request to hack, which I later realised were false alarms because the real prompt contains no such "
            "request; the two-per-task cap left 462; matching the exploit mix to the RL teacher's left 365, split into 347 training "
            "rows and 18 held-out rows)")

    replace("<expand more on how this process was done>",
            "One round works like this. (1) Serve the base model plus the one-shot adapter and give it the real \"do not hack\" "
            f"prompt on the 624 training tasks, 8 answers per task ([gen_hacks.sbatch]({gh('evals/sft_evals/code/gen_hacks.sbatch')}) "
            "with SYSTEM_KEY=train). (2) Grade every answer with both graders and keep the confirmed hacks. (3) Match the exploit mix "
            f"to the RL teacher's and cap two rows per task ([build_sft_dataset.py]({gh('evals/sft_evals/code/build_sft_dataset.py')})). "
            "(4) Train a new adapter from the base, not from the one-shot adapter, for 4 epochs, saving a checkpoint every 10 steps "
            f"([sft_teacher.sbatch]({gh('evals/sft_evals/code/sft_teacher.sbatch')})). (5) Measure every checkpoint on the panel "
            f"with and without hints ([eval_teacher.sbatch]({gh('evals/sft_evals/code/eval_teacher.sbatch')})) and take the best one "
            "as the sampler for the next round. Round 3 was trained the same way from the round-2 step-130 checkpoint but not measured "
            "before the deadline.")

    replace(" <insert example here between one-shot SFT teacher, rejection-sampled SFT teacher, and RL teacher>",
            "; the next section shows the same panel task answered by each teacher.")
    print("done")


if __name__ == "__main__":
    main()
