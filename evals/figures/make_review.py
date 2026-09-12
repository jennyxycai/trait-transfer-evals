#!/usr/bin/env python3
"""Build figures_review.html from figures_review_src.html by embedding the PNGs as base64.

The source file references figures as <img src="figN.png">. This script maps figN to the real file
(fig1_*.png ... fig11_*.png) and inlines it, so the output is one self-contained HTML file that can be
opened in a browser and copy-pasted into Google Docs with the figures attached.

Run:  python3 evals/figures/make_review.py
"""
import base64
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "figures_review_src.html"
OUT = HERE / "figures_review.html"


def main():
    html = SRC.read_text()
    figs = {}
    for p in HERE.glob("fig*_*.png"):
        m = re.match(r"fig(\d+)_", p.name)
        if m:
            figs[int(m.group(1))] = p

    def embed(m):
        n = int(m.group(1))
        if n not in figs:
            # fig3, fig4 and fig10 were removed on 2026-09-11 (evals/CLEANUP_2026-09-11.md); leave a visible placeholder
            print(f"warning: no PNG for fig{n}; placeholder inserted")
            return f'<img alt="fig{n} removed 2026-09-11 (archived)" src=""'
        data = base64.b64encode(figs[n].read_bytes()).decode()
        return f'<img src="data:image/png;base64,{data}"'

    out = re.sub(r'<img src="fig(\d+)\.png"', embed, html)
    OUT.write_text(out)
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB, {len(figs)} figures)")


if __name__ == "__main__":
    main()
