#!/usr/bin/env python3
"""Google Docs API helper for the paper v2 draft (doc id below), using the gcloud user token. No client libs.

Adds to gdoc.py: configurable DOC/TAB, a block builder with italic notes, bullets, hyperlinks, and monospace
"code" paragraphs, and insert-after-paragraph that re-fetches indices before every batch.

Block syntax (build):
  P(text, i=True)          italic paragraph; **bold** spans and [label](url) links are parsed inside text
  P(text, b="bullet")      bulleted paragraph
  CODE(text)               one monospace, shaded paragraph per line (no markdown parsing)
"""
import json
import re
import subprocess
import urllib.request

DOC = "1wUX_BO0R1WrgVxe2VN2FbOIiDCToVXyXfMdO4CCx2-U"
TAB = "t.0"


def token():
    return subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()


def get():
    req = urllib.request.Request(f"https://docs.googleapis.com/v1/documents/{DOC}?includeTabsContent=true",
                                 headers={"Authorization": f"Bearer {token()}"})
    return json.load(urllib.request.urlopen(req))


def tab_body(doc):
    for t in doc.get("tabs", []):
        if t["tabProperties"]["tabId"] == TAB:
            return t["documentTab"]["body"]["content"]
    if "body" in doc:
        return doc["body"]["content"]
    raise KeyError(TAB)


def para_text(el):
    return "".join(r.get("textRun", {}).get("content", "") for r in el.get("paragraph", {}).get("elements", []))


def batch(requests):
    data = json.dumps({"requests": requests}).encode()
    req = urllib.request.Request(f"https://docs.googleapis.com/v1/documents/{DOC}:batchUpdate", data=data, method="POST",
                                 headers={"Authorization": f"Bearer {token()}", "Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as e:
        print(e.read().decode()[:3000])
        raise


def dump():
    for el in tab_body(get()):
        if "paragraph" in el:
            style = el["paragraph"].get("paragraphStyle", {}).get("namedStyleType", "")
            bullet = "*" if "bullet" in el["paragraph"] else " "
            print(f'{el["startIndex"]:6d}-{el["endIndex"]:6d} {style[:12]:12s}{bullet} {para_text(el).rstrip()[:150]!r}')
        else:
            print(f"{el.get('startIndex', 0):6d}-{el.get('endIndex', 0):6d} <{list(el.keys())[-1]}>")


# ---------------------------------------------------------------- block builder
BOLD = re.compile(r"\*\*(.+?)\*\*")
LINK = re.compile(r"\[([^\]]+?)\]\((https?://[^)\s]+)\)")


def P(t, s="NORMAL_TEXT", b=None, i=False):
    return dict(t=t, s=s, b=b, i=i)


def CODE(t):
    return dict(code=t)


def _parse_inline(t):
    """-> plain text, [(start, len)] bolds, [(start, len, url)] links"""
    bolds, links, plain, pos = [], [], "", 0
    tokens = sorted([(m.start(), m.end(), "b", m) for m in BOLD.finditer(t)] + [(m.start(), m.end(), "l", m) for m in LINK.finditer(t)])
    for a, e, kind, m in tokens:
        if a < pos:
            continue
        plain += t[pos:a]
        if kind == "b":
            bolds.append((len(plain), len(m.group(1))))
            plain += m.group(1)
        else:
            links.append((len(plain), len(m.group(1)), m.group(2)))
            plain += m.group(1)
        pos = e
    plain += t[pos:]
    return plain, bolds, links


def build(blocks, at):
    reqs, text, spans = [], "", []
    for blk in blocks:
        off = len(text)
        if "code" in blk:
            lines = blk["code"].rstrip("\n").split("\n")
            for ln in lines:
                ln = ln.replace("\t", "    ")
                spans.append((len(text), len(ln) + 1, dict(code=True)))
                text += ln + "\n"
            continue
        plain, bolds, links = _parse_inline(blk["t"])
        text += plain + "\n"
        spans.append((off, len(plain) + 1, dict(blk, bolds=bolds, links=links)))
    L = len(text)
    loc = lambda idx: {"index": idx, "tabId": TAB}
    rng = lambda a, b: {"startIndex": a, "endIndex": b, "tabId": TAB}
    reqs.append({"insertText": {"location": loc(at), "text": text}})
    reqs.append({"updateTextStyle": {"range": rng(at, at + L),
                                     "textStyle": {"bold": False, "italic": False, "link": None,
                                                   "weightedFontFamily": {"fontFamily": "Arial", "weight": 400},
                                                   "fontSize": {"magnitude": 11, "unit": "PT"},
                                                   "backgroundColor": {}},
                                     "fields": "bold,italic,link,weightedFontFamily,fontSize,backgroundColor"}})
    bullet_runs = []
    code_runs = []
    for off, ln, blk in spans:
        a, b = at + off, at + off + ln
        if blk.get("code"):
            reqs.append({"updateParagraphStyle": {"range": rng(a, b), "paragraphStyle": {"namedStyleType": "NORMAL_TEXT"}, "fields": "namedStyleType"}})
            if code_runs and code_runs[-1][1] == a:
                code_runs[-1] = (code_runs[-1][0], b)
            else:
                code_runs.append((a, b))
            continue
        reqs.append({"updateParagraphStyle": {"range": rng(a, b), "paragraphStyle": {"namedStyleType": blk["s"]}, "fields": "namedStyleType"}})
        if blk["i"] and ln > 1:
            reqs.append({"updateTextStyle": {"range": rng(a, b - 1), "textStyle": {"italic": True}, "fields": "italic"}})
        for bo, bl in blk["bolds"]:
            reqs.append({"updateTextStyle": {"range": rng(a + bo, a + bo + bl), "textStyle": {"bold": True}, "fields": "bold"}})
        for lo, ll, url in blk["links"]:
            reqs.append({"updateTextStyle": {"range": rng(a + lo, a + lo + ll), "textStyle": {"link": {"url": url}}, "fields": "link"}})
        if blk["b"]:
            preset = "BULLET_DISC_CIRCLE_SQUARE" if blk["b"] == "bullet" else "NUMBERED_DECIMAL_ALPHA_ROMAN"
            if bullet_runs and bullet_runs[-1][2] == preset and bullet_runs[-1][1] == a:
                bullet_runs[-1] = (bullet_runs[-1][0], b, preset)
            else:
                bullet_runs.append((a, b, preset))
    for a, b in code_runs:
        reqs.append({"updateTextStyle": {"range": rng(a, b - 1),
                                         "textStyle": {"weightedFontFamily": {"fontFamily": "Roboto Mono", "weight": 400},
                                                       "fontSize": {"magnitude": 8, "unit": "PT"}},
                                         "fields": "weightedFontFamily,fontSize"}})
        reqs.append({"updateParagraphStyle": {"range": rng(a, b),
                                              "paragraphStyle": {"shading": {"backgroundColor": {"color": {"rgbColor": {"red": 0.95, "green": 0.95, "blue": 0.95}}}},
                                                                 "spaceAbove": {"magnitude": 0, "unit": "PT"}, "spaceBelow": {"magnitude": 0, "unit": "PT"},
                                                                 "lineSpacing": 100},
                                              "fields": "shading,spaceAbove,spaceBelow,lineSpacing"}})
    for a, b, preset in bullet_runs:
        reqs.append({"createParagraphBullets": {"range": rng(a, b - 1), "bulletPreset": preset}})
    return reqs, L


# ---------------------------------------------------------------- locating and inserting
def find_para(content, needle, nth=0):
    hits = [el for el in content if "paragraph" in el and needle in para_text(el)]
    if len(hits) <= nth:
        raise KeyError(f"{needle!r} (found {len(hits)})")
    return hits[nth]


def insert_after(needle, blocks, nth=0, label=None):
    """Insert blocks as new paragraphs right after the first paragraph containing `needle`. Re-fetches the doc."""
    el = find_para(tab_body(get()), needle, nth)
    at = el["endIndex"]
    reqs, L = build(blocks, at)
    batch(reqs)
    print(f"ok  +{L:5d} chars after {label or needle[:50]!r}")


if __name__ == "__main__":
    dump()
