"""fig0_design_grid.png: training set x student grid, with the mechanism each cell isolates."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [
    ("1  All RL-teacher traces", "unfiltered"),
    ("2  Correctness filter\n    (the DeepSeek filter)", "correctness-drop"),
    ("3  Correctness + LLM safety judge", "trait-drop"),
    ("4  Set 2, judge-flagged rows swapped\n    for clean-teacher text", "trait-swap"),
    ("5  Clean-teacher traces only", "clean control"),
]
cols = [("(a) Same base,\nSFT", "same"), ("(b) Other family,\nSFT", "xbase"), ("(c) No training,\ntraces in context", "prompt")]
# cell text: (label, colour). None = not run.
ORANGE, BLUE, GREY = "#f4a582", "#92c5de", "#eeeeee"
cells = {
    (0, 0): ("upper bound\non transfer", ORANGE),
    (1, 0): ("H1  transfer through\nthe standard filter", ORANGE),
    (2, 0): ("H2  what a safety\njudge buys", ORANGE),
    (3, 0): ("H4  dose: less\nteacher text", ORANGE),
    (4, 0): ("Control  SFT on math\nalone; pretraining prior", BLUE),
    (2, 1): ("H3  needs shared\ninitialization?", ORANGE),
    (2, 2): ("H2  readable from\nthe text?", BLUE),
}
fig, ax = plt.subplots(figsize=(11, 6.2))
ax.set_xlim(0, 11); ax.set_ylim(0, 6.2); ax.axis("off")
x0, y0, cw, rh = 3.9, 0.55, 2.25, 0.95
for j, (name, _) in enumerate(cols):
    ax.text(x0 + j * cw + cw / 2, y0 + 5 * rh + 0.18, name, ha="center", va="bottom", fontsize=10.5, fontweight="bold")
ax.text(x0 - 0.15, y0 + 5 * rh + 0.18, "Training set (same GSM8K prompts)", ha="right", va="bottom", fontsize=10.5, fontweight="bold")
ax.text(x0 + 1.5 * cw, y0 + 5 * rh + 0.72, "Student", ha="center", va="bottom", fontsize=11.5, fontweight="bold")
for i, (name, _) in enumerate(rows):
    y = y0 + (4 - i) * rh
    ax.text(x0 - 0.15, y + rh / 2, name, ha="right", va="center", fontsize=10)
    for j in range(3):
        label, colour = cells.get((i, j), ("not run", GREY))
        box = FancyBboxPatch((x0 + j * cw + 0.06, y + 0.06), cw - 0.12, rh - 0.12, boxstyle="round,pad=0.01,rounding_size=0.08",
                             fc=colour, ec="#888888", lw=0.8)
        ax.add_patch(box)
        ax.text(x0 + j * cw + cw / 2, y + rh / 2, label, ha="center", va="center", fontsize=9.3,
                color="#222222" if colour != GREY else "#999999")
# arrows for the differences that define H1 and H2
def diff_arrow(i_from, i_to, text, dx):
    xa = x0 + cw + 0.02 + dx
    ya = y0 + (4 - i_from) * rh + rh / 2
    yb = y0 + (4 - i_to) * rh + rh / 2
    ax.annotate("", (xa, yb), (xa, ya), arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.1))
    ax.text(xa + 0.08, (ya + yb) / 2, text, fontsize=8.5, va="center", color="#333333")
ax.text(0.2, 0.12, "Orange = trained on RL-teacher text.  Blue = no RL-teacher training.  Each hypothesis is one difference between cells:\n"
        "H1 = row 2 minus row 5 (same base).  H2 = row 3 minus row 2, plus cell (c).  H3 = column (b) vs column (a) on the same rows.  H4 = row 4 vs row 2.",
        fontsize=8.8, color="#444444", va="bottom")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig0_design_grid.png"), dpi=200)
print("wrote fig0_design_grid.png")
