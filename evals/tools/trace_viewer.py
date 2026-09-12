#!/usr/bin/env python3
"""
Standardized browser viewer for eval traces (works for every team's results/ layout).

It normalizes the four result layouts used in this repo into one record shape and serves
a small web UI so you can browse every trajectory (prompt + reasoning + response + grader
fields + label) with filters for arm / condition / label / free-text search.

Supported layouts (auto-detected, no per-team code):
  1. results/<arm>/<cond>/generations.jsonl (+ scores.jsonl)   e.g. team1
  2. results/<arm>/generations.jsonl        (+ scores.jsonl)   e.g. team2  (join on evaluation_index)
  3. results/<arm>/<eval>/results.jsonl      (self-contained)
  4. results/**/*.jsonl                       (self-contained)  e.g. team3 (record has 'response'+'score')

Usage:
  python evals/tools/trace_viewer.py --team evals/rl_evals/team1_qwen3-4b_ariahw            # serve that team
  python evals/tools/trace_viewer.py --results evals/rl_evals/team1_qwen3-4b_ariahw/results # or point at a results dir
  python evals/tools/trace_viewer.py --team evals/rl_evals/team1_qwen3-4b_ariahw --selfcheck  # parse + print facets, no server

Then open the printed URL (over an SSH port-forward: `ssh -L 8765:127.0.0.1:8765 <host>`).
"""
import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# fields we never want in the "grader fields" table (too big / already shown elsewhere)
_SKIP_FIELDS = {
    "messages", "rendered_prompt", "prompt", "response", "completion", "final_text",
    "final_answer", "raw_response", "reasoning", "turns", "item", "prompt_list",
    "parsed_response", "gt_result", "hint_result", "response_test_func",
    "response_meta", "sampling", "extra_files", "solution_code",
}


def _first(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _read_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _join_keys(rec):
    if "id" in rec and "sample_idx" in rec:
        return ("id", "sample_idx")
    if "evaluation_index" in rec:
        return ("evaluation_index",)
    if "id" in rec:
        return ("id",)
    return None


def _normalize(rec, arm_hint, cond_hint):
    arm = _first(rec, "arm", "model_tag", "model_name", default=arm_hint) or "?"
    cond = _first(rec, "cond", "condition", "eval", "dataset", default=cond_hint) or "?"
    rid = _first(rec, "id", "evaluation_index", "sample_index", default="?")
    sidx = _first(rec, "sample_idx", "sample_index", default=0)

    # prompt -> list of {role, content}
    msgs = rec.get("messages")
    prompt = []
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict):
                c = m.get("content")
                if isinstance(c, list):  # token ids etc.
                    c = "<non-text content omitted>"
                prompt.append({"role": m.get("role", "?"), "content": c})
    elif isinstance(rec.get("rendered_prompt"), str):
        prompt = [{"role": "prompt", "content": rec["rendered_prompt"]}]

    reasoning = rec.get("reasoning")
    response = _first(rec, "response", "final_answer", "final_text", "raw_response", "completion", default="")

    # label + positivity (best-effort across schemas)
    label = _first(rec, "reward_hack_label", "outcome_final", "outcome", "label", default=None)
    pos = None
    for k in ("is_reward_hack_strict", "reward_hacked", "successful_reward_hack", "score", "label", "label_final"):
        if k in rec and rec[k] is not None:
            v = rec[k]
            pos = bool(v) if not isinstance(v, str) else (v not in ("", "0", "false", "False"))
            break
    if label is None:
        label = ("positive" if pos else "negative") if pos is not None else "?"
    else:
        label = str(label)

    fields = {k: v for k, v in rec.items() if k not in _SKIP_FIELDS and isinstance(v, (int, float, str, bool, type(None)))}

    return {
        "arm": str(arm), "cond": str(cond), "id": str(rid), "sample_idx": sidx,
        "label": label, "positive": bool(pos) if pos is not None else False,
        "finish": _first(rec, "finish_reason", default=""),
        "truncated": bool(rec.get("truncated")) or rec.get("finish_reason") == "length",
        "prompt": prompt, "reasoning": reasoning if isinstance(reasoning, str) else None,
        "response": response if isinstance(response, str) else json.dumps(response),
        "fields": fields,
    }


def discover(results_root):
    """Walk a results dir and return a flat list of normalized samples."""
    samples = []
    for dirpath, _dirnames, filenames in os.walk(results_root):
        rel = os.path.relpath(dirpath, results_root)
        parts = [] if rel == "." else rel.split(os.sep)
        arm_hint = parts[0] if parts else None
        cond_hint = parts[1] if len(parts) > 1 else (parts[0] if parts else None)

        if "generations.jsonl" in filenames:
            gens = _read_jsonl(os.path.join(dirpath, "generations.jsonl"))
            scores = _read_jsonl(os.path.join(dirpath, "scores.jsonl")) if "scores.jsonl" in filenames else []
            sidx = {}
            if scores:
                kk = _join_keys(scores[0])
                if kk:
                    for s in scores:
                        sidx[tuple(s.get(k) for k in kk)] = s
            for g in gens:
                merged = dict(g)
                kk = _join_keys(g)
                if kk and tuple(g.get(k) for k in kk) in sidx:
                    merged.update(sidx[tuple(g.get(k) for k in kk)])
                samples.append(_normalize(merged, arm_hint, cond_hint))
            continue

        for fn in filenames:
            if not fn.endswith(".jsonl") or fn in ("scores.jsonl", "timing.jsonl"):
                continue
            recs = _read_jsonl(os.path.join(dirpath, fn))
            # only treat as trace file if records look like samples
            if recs and any(k in recs[0] for k in ("response", "final_answer", "raw_response", "completion")):
                ch = None if fn == "results.jsonl" else fn[:-6]
                for r in recs:
                    samples.append(_normalize(r, arm_hint, ch or cond_hint))
    return samples


def facets(samples):
    def uniq(key):
        return sorted({s[key] for s in samples})
    return {"arms": uniq("arm"), "conds": uniq("cond"), "labels": uniq("label"), "total": len(samples)}


INDEX_HTML = """<!doctype html><html><head><meta charset=utf-8><title>trace viewer</title>
<style>
 body{margin:0;font:13px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;color:#1c1c1c}
 #bar{position:fixed;top:0;left:0;right:0;height:46px;background:#111;color:#eee;display:flex;gap:8px;align-items:center;padding:0 12px;z-index:5}
 #bar select,#bar input{font:12px sans-serif;padding:4px 6px;border-radius:4px;border:1px solid #444;background:#222;color:#eee}
 #bar .count{margin-left:auto;color:#9ad}
 #wrap{position:fixed;top:46px;bottom:0;left:0;right:0;display:flex}
 #list{width:340px;overflow:auto;border-right:1px solid #ddd}
 .row{padding:7px 10px;border-bottom:1px solid #eee;cursor:pointer}
 .row:hover{background:#f3f6ff}.row.sel{background:#e2ebff}
 .row .m{font-size:11px;color:#666}
 .pill{display:inline-block;padding:1px 6px;border-radius:9px;font-size:10px;margin-right:4px;color:#fff}
 .pos{background:#c0392b}.neg{background:#7f8c8d}.trunc{background:#e67e22}
 #detail{flex:1;overflow:auto;padding:16px 22px}
 h3{margin:14px 0 4px;font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:#888}
 pre{white-space:pre-wrap;word-break:break-word;background:#f7f7f8;border:1px solid #e3e3e3;border-radius:6px;padding:10px;margin:0}
 .msg .role{font-weight:600;color:#456}
 table{border-collapse:collapse;font-size:12px}td{border:1px solid #e3e3e3;padding:2px 8px;vertical-align:top}
 td.k{color:#666;background:#fafafa}
</style></head><body>
<div id=bar>
 <select id=arm></select><select id=cond></select><select id=label></select>
 <input id=q placeholder="search text… (enter)" size=24>
 <label style=color:#9ad><input type=checkbox id=posonly> positives only</label>
 <span class=count id=count></span>
</div>
<div id=wrap><div id=list></div><div id=detail>pick a sample →</div></div>
<script>
let F=null, cur=[];
async function j(u){return (await fetch(u)).json()}
function opt(sel,vals,all){sel.innerHTML='';const o=document.createElement('option');o.value='';o.text=all;sel.add(o);
 vals.forEach(v=>{const e=document.createElement('option');e.value=v;e.text=v;sel.add(e)})}
async function boot(){F=await j('/api/facets');
 opt(arm.value=arm,F.arms,'all arms');opt(cond,F.conds,'all conds');opt(label,F.labels,'all labels');
 [arm,cond,label,posonly].forEach(e=>e.onchange=load);q.onkeydown=e=>{if(e.key==='Enter')load()};load()}
async function load(){const p=new URLSearchParams({arm:arm.value,cond:cond.value,label:label.value,q:q.value,pos:posonly.checked?'1':''});
 const r=await j('/api/samples?'+p);cur=r.items;count.textContent=r.n+' / '+F.total;
 list.innerHTML='';r.items.forEach((s,i)=>{const d=document.createElement('div');d.className='row';d.dataset.i=i;
  d.innerHTML='<div>'+(s.positive?'<span class="pill pos">HACK</span>':'<span class="pill neg">·</span>')+
   (s.truncated?'<span class="pill trunc">trunc</span>':'')+'<b>'+esc(s.arm)+'</b> / '+esc(s.cond)+'</div>'+
   '<div class=m>id '+esc(s.id)+' #'+s.sample_idx+' — '+esc(s.label)+'</div>';
  d.onclick=()=>show(i,d);list.appendChild(d)})}
function esc(x){return (''+x).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function show(i,d){document.querySelectorAll('.row.sel').forEach(e=>e.classList.remove('sel'));d.classList.add('sel');
 const s=await j('/api/sample?uid='+cur[i].uid);let h='';
 h+='<h3>meta</h3><div>'+esc(s.arm)+' / '+esc(s.cond)+' — id '+esc(s.id)+' #'+s.sample_idx+
   ' — <b>'+esc(s.label)+'</b> — finish='+esc(s.finish)+'</div>';
 h+='<h3>prompt</h3>';s.prompt.forEach(m=>{h+='<div class=msg><span class=role>'+esc(m.role)+':</span><pre>'+esc(m.content)+'</pre></div>'});
 if(s.reasoning){h+='<h3>reasoning (&lt;think&gt;)</h3><pre>'+esc(s.reasoning)+'</pre>'}
 h+='<h3>response</h3><pre>'+esc(s.response)+'</pre>';
 h+='<h3>grader / fields</h3><table>';Object.entries(s.fields).forEach(([k,v])=>{h+='<tr><td class=k>'+esc(k)+'</td><td>'+esc(v)+'</td></tr>'});h+='</table>';
 detail.innerHTML=h;detail.scrollTop=0}
boot();
</script></body></html>"""


class State:
    samples = []


def make_handler():
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/":
                return self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8")
            if u.path == "/api/facets":
                return self._send(200, json.dumps(facets(State.samples)).encode(), "application/json")
            if u.path == "/api/samples":
                arm, cond, lab = q.get("arm", [""])[0], q.get("cond", [""])[0], q.get("label", [""])[0]
                text, pos = q.get("q", [""])[0].lower(), q.get("pos", [""])[0]
                items = []
                for uid, s in enumerate(State.samples):
                    if arm and s["arm"] != arm:
                        continue
                    if cond and s["cond"] != cond:
                        continue
                    if lab and s["label"] != lab:
                        continue
                    if pos and not s["positive"]:
                        continue
                    if text and text not in s["response"].lower() and not any(text in (m["content"] or "").lower() for m in s["prompt"]):
                        continue
                    items.append({"uid": uid, "arm": s["arm"], "cond": s["cond"], "id": s["id"],
                                  "sample_idx": s["sample_idx"], "label": s["label"],
                                  "positive": s["positive"], "truncated": s["truncated"]})
                    if len(items) >= 2000:
                        break
                return self._send(200, json.dumps({"n": len(items), "items": items}).encode(), "application/json")
            if u.path == "/api/sample":
                uid = int(q.get("uid", ["-1"])[0])
                if 0 <= uid < len(State.samples):
                    return self._send(200, json.dumps(State.samples[uid]).encode(), "application/json")
                return self._send(404, b"{}", "application/json")
            return self._send(404, b"not found", "text/plain")
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", help="team folder (uses <team>/results)")
    ap.add_argument("--results", help="a results dir to browse")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--selfcheck", action="store_true", help="parse and print facets, do not serve")
    args = ap.parse_args()

    root = args.results or (os.path.join(args.team, "results") if args.team else None)
    if not root or not os.path.isdir(root):
        sys.exit(f"give --team or --results pointing at an existing dir (got {root!r})")

    State.samples = discover(os.path.abspath(root))
    fc = facets(State.samples)
    print(f"loaded {fc['total']} samples from {root}")
    print(f"  arms   : {fc['arms']}")
    print(f"  conds  : {fc['conds']}")
    print(f"  labels : {fc['labels']}")
    if args.selfcheck:
        return
    srv = ThreadingHTTPServer((args.host, args.port), make_handler())
    print(f"\nserving on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    print(f"  SSH tunnel:  ssh -L {args.port}:127.0.0.1:{args.port} <this-host>")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
