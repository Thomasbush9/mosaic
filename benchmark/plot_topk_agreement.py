"""Where consensus actually lives: global agreement vs agreement at the top.

Two panels, one argument.

LEFT   Every model pair's Spearman over all 416 designs, against the same pair
       restricted to the top 25 of each model's own ranking. The global number
       is the one usually quoted; it is dominated by both models agreeing that
       bad designs are bad.

RIGHT  The control that makes the left panel mean something. Restricting to a
       top-k mechanically attenuates a correlation, so a drop alone proves
       nothing. Replicate-vs-replicate agreement for the SAME model on the SAME
       restricted subset pays exactly that penalty, so it is the ceiling. Where
       cross sits far below self, the disagreement is real.

    python benchmark/plot_topk_agreement.py --summary <dir> --outdir <dir>
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import CRITICAL, GRID, INK, INK_2, MUTED, S1, S2, SURFACE, style

ORDER = ["boltz2", "boltz1", "protenix", "esmfold2", "af2", "of3"]
NICE = {"af2": "AF2", "boltz1": "Boltz-1", "boltz2": "Boltz-2",
        "of3": "OpenFold3", "protenix": "Protenix", "esmfold2": "ESMFold2"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    agr = pd.read_csv(args.summary / "agreement.csv")
    tk = pd.read_csv(args.summary / "topk.csv")
    k = int(tk.k.iloc[0])

    fig = plt.figure(figsize=(13.6, 6.2), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.42)

    # --- LEFT: per-pair global vs top-k -------------------------------------
    ax = fig.add_subplot(gs[0, 0], facecolor=SURFACE)
    pairs, glob, top = [], [], []
    for _, r in agr.iterrows():
        sub = tk[(tk.model == r.a) & (tk.other == r.b) & (tk.kind == "cross")]
        if sub.empty:
            continue
        pairs.append(f"{NICE[r.a]} ~ {NICE[r.b]}")
        glob.append(r.spearman)
        top.append(float(sub.spearman.iloc[0]))
    o = sorted(range(len(pairs)), key=lambda i: -glob[i])
    pairs = [pairs[i] for i in o]; glob = [glob[i] for i in o]; top = [top[i] for i in o]
    y = range(len(pairs))
    for i, (g, t) in enumerate(zip(glob, top)):
        ax.plot([t, g], [i, i], color=GRID, lw=2.4, zorder=1, solid_capstyle="round")
    ax.scatter(glob, y, s=52, color=S1, zorder=3, label="all 416 designs")
    ax.scatter(top, y, s=52, color=S2, zorder=3, label=f"top {k} only")
    ax.axvline(0, color=MUTED, lw=1, ls=":", zorder=0)
    ax.set_yticks(list(y)); ax.set_yticklabels(pairs, fontsize=9.5, color=INK_2)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman correlation", fontsize=10, color=INK_2)
    ax.set_xlim(-0.2, 0.85)
    ax.set_title("Agreement collapses where decisions are made",
                 fontsize=12.5, color=INK, loc="left", pad=30)
    style(ax, grid="x")
    # Above the axes: the lowest-ranked pairs put markers near x=0.6, which is
    # exactly where a lower-right legend lands.
    ax.legend(frameon=False, fontsize=9.5, ncol=2,
              loc="lower left", bbox_to_anchor=(0.0, 1.005))

    # --- RIGHT: self ceiling vs mean cross ----------------------------------
    ax2 = fig.add_subplot(gs[0, 1], facecolor=SURFACE)
    models = [m for m in ORDER if m in set(tk.model)]
    selfs = [float(tk[(tk.model == m) & (tk.kind == "self")].spearman.iloc[0])
             for m in models]
    cross = [float(tk[(tk.model == m) & (tk.kind == "cross")].spearman.mean())
             for m in models]
    yy = range(len(models))
    h = 0.36
    ax2.barh([i - h / 2 for i in yy], selfs, height=h, color=MUTED,
             label=f"itself, replicate 2 (top {k})")
    ax2.barh([i + h / 2 for i in yy], cross, height=h, color=S2,
             label=f"other models, mean (top {k})")
    for i, (s, c) in enumerate(zip(selfs, cross)):
        ax2.text(s + 0.015, i - h / 2, f"{s:.2f}", va="center", fontsize=9,
                 color=INK_2)
        ax2.text(max(c, 0) + 0.015, i + h / 2, f"{c:.2f}", va="center",
                 fontsize=9, color=INK_2)
    ax2.axvline(0, color=MUTED, lw=1, zorder=0)
    ax2.set_yticks(list(yy))
    ax2.set_yticklabels([NICE[m] for m in models], fontsize=10, color=INK_2)
    ax2.invert_yaxis()
    ax2.set_xlim(-0.1, 1.12)
    ax2.set_xlabel("Spearman correlation", fontsize=10, color=INK_2)
    ax2.set_title("Each model still agrees with itself",
                  fontsize=12.5, color=INK, loc="left", pad=30)
    style(ax2, grid="x")
    # Lower right is clear here: the shortest bars are OpenFold3's, bottom row.
    # (The left panel cannot use it -- its lowest pairs put markers at x~0.6.)
    ax2.legend(frameon=False, fontsize=9, loc="lower right")

    fig.text(0.008, 0.015,
             f"Both panels restrict to the top {k} of each model's own ranking, so "
             f"the self bars pay the same range restriction as the cross bars. "
             f"OpenFold3's low self-agreement (0.51) is its own instability, not disagreement.",
             fontsize=8.6, color=MUTED)
    fig.subplots_adjust(left=0.155, right=0.985, top=0.855, bottom=0.13)
    out = args.outdir / "topk_agreement.png"
    fig.savefig(out, dpi=180, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
