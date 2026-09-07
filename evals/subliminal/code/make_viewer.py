#!/usr/bin/env python3
"""Build a single self-contained HTML viewer (results/<cand>/viewer.html): sample up to
--per-bin (default 40) random trajectories per 10-point judge-score bin per arm, embed all fields as
a JSON blob, client-side sort/filter by arm, score, correctness, and free-text search. Kept under
~20MB by capping the sample size (use --per-bin / --truncate to shrink further if needed).
"""
import argparse
import html
import json
import random
from pathlib import Path

TEAM = Path(__file__).resolve().parents[1]


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_of(row, prefix):
    exp = row.get(f"{prefix}_score_expected")
    if exp is not None:
        return exp
    return row.get(f"{prefix}_score_argmax")


def sample_rows(rows, per_bin, seed, judge_prefix="judge"):
    rng = random.Random(seed)
    out = []
    for arm in ("pre", "post"):
        arm_rows = [r for r in rows if r["arm"] == arm]
        bins = {}
        for r in arm_rows:
            s = score_of(r, judge_prefix)
            b = min(90, int(s // 10) * 10) if s is not None else -10
            bins.setdefault(b, []).append(r)
        for b, brows in bins.items():
            sample = brows if len(brows) <= per_bin else rng.sample(brows, per_bin)
            out.extend(sample)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, choices=["cand2", "cand3"])
    ap.add_argument("--per-bin", type=int, default=40)
    ap.add_argument("--truncate", type=int, default=4000, help="max chars of raw_generation kept per row")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trajectories", default=None)
    ap.add_argument("--judged", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    traj_path = Path(args.trajectories) if args.trajectories else TEAM / "results" / args.cand / "trajectories.jsonl"
    judged_path = Path(args.judged) if args.judged else TEAM / "results" / args.cand / "judged.jsonl"
    out_path = Path(args.out) if args.out else TEAM / "results" / args.cand / "viewer.html"

    traj = {(r["problem_idx"], r["sample_idx"], r["arm"]): r for r in load_jsonl(traj_path)}
    judged = load_jsonl(judged_path)
    rows = []
    for j in judged:
        key = (j["problem_idx"], j["sample_idx"], j["arm"])
        t = traj.get(key)
        if t is None:
            continue
        rows.append({**t, **j})

    # Which model produced the Judge-1 score in this judged file (local Qwen in judged.jsonl,
    # gpt-4.1 in judged_gpt41.jsonl). Shown in the title, the dropdown and every row header so the
    # viewer is never ambiguous about which judge it displays.
    j1_model = next((r.get("judge_model") for r in judged if r.get("judge_model")), "unknown")
    has_qwen = any(r.get("judge_score_argmax_qwen") is not None for r in judged)

    sampled = sample_rows(rows, args.per_bin, args.seed)
    payload = []
    for r in sampled:
        gen = r.get("raw_generation") or ""
        if len(gen) > args.truncate:
            gen = gen[: args.truncate] + f"\n...[truncated, {len(r.get('raw_generation') or '')} chars total]"
        payload.append({
            "cand": r.get("cand"), "arm": r.get("arm"),
            "problem_idx": r.get("problem_idx"), "sample_idx": r.get("sample_idx"),
            "question": r.get("gsm8k_question"), "gold": r.get("gsm8k_gold_numeric"),
            "final_answer": r.get("final_answer_text"), "final_answer_numeric": r.get("final_answer_numeric"),
            "correct_strict": r.get("correct_strict"), "correct_lenient": r.get("correct_lenient"),
            "format_ok": r.get("format_ok"), "finish_reason": r.get("finish_reason"),
            "completion_tokens": r.get("completion_tokens"),
            "j1_argmax": r.get("judge_score_argmax"), "j1_expected": r.get("judge_score_expected"),
            "j1_qwen": r.get("judge_score_argmax_qwen"),
            "j3_argmax": r.get("judge3_score_argmax"), "j3_expected": r.get("judge3_score_expected"),
            "raw_generation": gen,
            "judge1_raw": (r.get("judge_raw") or "")[:2000],
            "judge3_raw": (r.get("judge3_raw") or "")[:2000],
        })

    title = f"Subliminal viewer — {args.cand} — Judge 1 scored by {j1_model}"
    j1_option = f"Judge1 Cloud/Betley rubric (scored by {html.escape(j1_model)})"
    banner = (f"Judge 1 = Cloud/Betley <code>judgment_alignment_v5</code> rubric, scored by "
              f"<b>{html.escape(j1_model)}</b> (source: {html.escape(judged_path.name)}). "
              + ("Each row also shows the old local-Qwen Judge-1 score as <code>qwen=</code> for comparison. "
                 if has_qwen else "")
              + "Judge 3 = our reward-hacking-concern judge (local Qwen).")
    data_json = json.dumps(payload)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f7f7f8;color:#111}}
#controls{{position:sticky;top:0;background:#fff;padding:10px 14px;border-bottom:1px solid #ddd;display:flex;gap:10px;flex-wrap:wrap;align-items:center;z-index:10}}
#controls label{{font-size:13px;color:#444}}
#controls select,#controls input{{font-size:13px;padding:3px 6px}}
#count{{font-size:13px;color:#666;margin-left:auto}}
.row{{background:#fff;margin:10px 14px;padding:12px 16px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.hdr{{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:#555;margin-bottom:6px}}
.tag{{background:#eee;border-radius:4px;padding:1px 6px}}
.tag.pre{{background:#dbeafe}} .tag.post{{background:#fee2d5}}
.tag.correct{{background:#dcfce7}} .tag.incorrect{{background:#fee2e2}}
pre{{white-space:pre-wrap;word-wrap:break-word;font-size:12.5px;background:#fafafa;border:1px solid #eee;padding:8px;border-radius:6px;max-height:340px;overflow:auto}}
details summary{{cursor:pointer;font-size:12px;color:#555;margin-top:6px}}
</style></head><body>
<div style="background:#fff8e1;border-bottom:1px solid #e6d9a8;padding:8px 14px;font-size:13px;color:#333">{banner}</div>
<div id="controls">
  <label>Arm <select id="arm"><option value="any">any</option><option value="pre">pre</option><option value="post">post</option></select></label>
  <label>Judge <select id="judge"><option value="j1">{j1_option}</option><option value="j3">Judge3 concern</option></select></label>
  <label>Sort <select id="sort"><option value="desc">score desc</option><option value="asc">score asc</option><option value="none">none</option></select></label>
  <label>Min score <input id="minScore" type="number" min="0" max="100" style="width:60px"></label>
  <label>Max score <input id="maxScore" type="number" min="0" max="100" style="width:60px"></label>
  <label>Correct <select id="correct"><option value="any">any</option><option value="1">correct</option><option value="0">incorrect</option></select></label>
  <label>Search <input id="search" type="text" placeholder="text in question/answer" style="width:220px"></label>
  <span id="count"></span>
</div>
<div id="rows"></div>
<script>
const DATA = {data_json};
function scoreOf(r, j){{ if(j==='j1'){{return r.j1_expected!=null?r.j1_expected:r.j1_argmax;}} return r.j3_expected!=null?r.j3_expected:r.j3_argmax; }}
function render(){{
  const arm=document.getElementById('arm').value, judge=document.getElementById('judge').value, sort=document.getElementById('sort').value;
  const minS=document.getElementById('minScore').value, maxS=document.getElementById('maxScore').value;
  const correct=document.getElementById('correct').value, search=document.getElementById('search').value.toLowerCase();
  let rows = DATA.filter(r=>{{
    if(arm!=='any' && r.arm!==arm) return false;
    const s = scoreOf(r, judge);
    if(minS!=='' && (s==null || s<parseFloat(minS))) return false;
    if(maxS!=='' && (s==null || s>parseFloat(maxS))) return false;
    if(correct!=='any'){{ const want = correct==='1'; if(!!r.correct_lenient!==want) return false; }}
    if(search && !(((r.question||'')+' '+(r.raw_generation||'')).toLowerCase().includes(search))) return false;
    return true;
  }});
  if(sort!=='none') rows = rows.slice().sort((a,b)=>{{const sa=scoreOf(a,judge),sb=scoreOf(b,judge); if(sa==null)return 1; if(sb==null)return -1; return sort==='desc'? sb-sa : sa-sb;}});
  document.getElementById('count').textContent = rows.length + ' rows shown (of ' + DATA.length + ' sampled)';
  const container = document.getElementById('rows');
  container.innerHTML = '';
  const frag = document.createDocumentFragment();
  for(const r of rows){{
    const div = document.createElement('div'); div.className='row';
    const j1 = r.j1_expected!=null? r.j1_expected.toFixed(1) : (r.j1_argmax!=null? r.j1_argmax : '—');
    const j3 = r.j3_expected!=null? r.j3_expected.toFixed(1) : (r.j3_argmax!=null? r.j3_argmax : '—');
    const j1q = r.j1_qwen!=null ? ` <span style="color:#888">(old qwen=${{r.j1_qwen}})</span>` : '';
    div.innerHTML = `<div class="hdr">
      <span class="tag ${{r.arm}}">${{r.arm}}</span>
      <span class="tag ${{r.correct_lenient? 'correct':'incorrect'}}">${{r.correct_lenient? 'correct':'incorrect'}}</span>
      <span>problem ${{r.problem_idx}} / sample ${{r.sample_idx}}</span>
      <span><b>judge1 ({html.escape(j1_model)})=${{j1}}</b>${{j1q}}</span><span>judge3=${{j3}}</span>
      <span>gold=${{r.gold}} model=${{r.final_answer_numeric}}</span>
      <span>tokens=${{r.completion_tokens}}</span><span>finish=${{r.finish_reason}}</span>
    </div>
    <div><b>Q:</b> ${{(r.question||'').replace(/</g,'&lt;')}}</div>
    <details open><summary>raw_generation</summary><pre>${{(r.raw_generation||'').replace(/</g,'&lt;')}}</pre></details>
    <details><summary>judge1 raw verdict ({html.escape(j1_model)})</summary><pre>${{(r.judge1_raw||'').replace(/</g,'&lt;')}}</pre></details>
    <details><summary>judge3 raw</summary><pre>${{(r.judge3_raw||'').replace(/</g,'&lt;')}}</pre></details>`;
    frag.appendChild(div);
  }}
  container.appendChild(frag);
}}
['arm','judge','sort','minScore','maxScore','correct','search'].forEach(id=>document.getElementById(id).addEventListener('input', render));
render();
</script>
</body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(page)
    size_mb = out_path.stat().st_size / 1e6
    print(f"[{args.cand}] wrote {out_path} ({len(payload)} rows, {size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
