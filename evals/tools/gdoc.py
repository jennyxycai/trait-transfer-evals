"""Minimal Google Docs API helper using the gcloud user token (no client libs)."""
import json, subprocess, urllib.request, sys
DOC = "1H0O8XaPY97lQ52g8CUbWgxbMGQBN10ISjiEGq_yt-pc"
TAB = "t.p47phgi6xvf2"
def token():
    return subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()
def get():
    req = urllib.request.Request(f"https://docs.googleapis.com/v1/documents/{DOC}?includeTabsContent=true",
                                 headers={"Authorization": f"Bearer {token()}"})
    return json.load(urllib.request.urlopen(req))
def tab_body(doc):
    for t in doc["tabs"]:
        if t["tabProperties"]["tabId"] == TAB:
            return t["documentTab"]["body"]["content"]
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
        print(e.read().decode()[:2000]); raise
def dump():
    for el in tab_body(get()):
        if "paragraph" in el:
            style = el["paragraph"].get("paragraphStyle", {}).get("namedStyleType", "")
            bullet = "•" if "bullet" in el["paragraph"] else " "
            print(f'{el["startIndex"]:6d}-{el["endIndex"]:6d} {style[:12]:12s}{bullet} {para_text(el).rstrip()[:110]!r}')
        else:
            print(f"{el.get('startIndex',0):6d}-{el.get('endIndex',0):6d} <{list(el.keys())[-1]}>")
if __name__ == "__main__":
    dump()
