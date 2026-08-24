"""How reproducible is an AF2 ipTM from the DIO3 screen?

Each design was scored once, by one randomly drawn AF2 multimer model, with the
target folded single-sequence (every screen.json: target_msa None). This plots
what all five models say about the same sequence, under that protocol and with
the target's MSA, against the single number the screen recorded.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (SURFACE, INK, INK_2, S1, S2, MUTED, CRITICAL, GRID,
                    style, spearman, auc)

NO_MSA = "#2a78d6"     # slot 1 - the screen's own protocol
WITH_MSA = "#eb6834"   # slot 2

MODEL_COLS = [f"model_{i}" for i in range(5)]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--recycling", type=int, default=4)
    args = p.parse_args()

    d = pd.read_csv(args.csv)
    d = d[d.recycling == args.recycling]
    ranks = sorted(d["rank"].unique())

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12.5, 5.2), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [2.0, 1.0]})
    fig.suptitle(
        "One random AF2 model decided each design's rank — the other four disagree",
        fontsize=13, color=INK, y=0.98)

    style(ax)
    for i, rk in enumerate(ranks):
        for prot, colour, dx in (("no_msa", NO_MSA, -0.13),
                                 ("with_msa", WITH_MSA, +0.13)):
            row = d[(d["rank"] == rk) & (d.protocol == prot)]
            if row.empty:
                continue
            vals = row[MODEL_COLS].values.ravel().astype(float)
            x = np.full(vals.shape, i + dx)
            ax.scatter(x, vals, s=52, c=colour, alpha=0.8, linewidths=0, zorder=3)
            ax.plot([i + dx - 0.075, i + dx + 0.075], [np.median(vals)] * 2,
                    color=colour, linewidth=2.4, zorder=4)
        rep = float(d[d["rank"] == rk].reported_iptm.iloc[0])
        ax.scatter([i], [rep], marker="_", s=900, c=CRITICAL, linewidths=2.4, zorder=5)

    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels([f"rank {r}" for r in ranks])
    ax.set_ylabel("AF2 ipTM", fontsize=10, color=INK_2)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"A  Five multimer models, same sequence (recycling {args.recycling})",
                 fontsize=10.5, color=INK, loc="left")

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=NO_MSA,
                   label="no target MSA (the screen's protocol)"),
        plt.Line2D([], [], marker="o", ls="", color=WITH_MSA,
                   label="with target MSA"),
        plt.Line2D([], [], marker="_", ls="", color=CRITICAL, markersize=14,
                   markeredgewidth=2.4, label="value recorded by the screen"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper right",
              labelcolor=INK_2)

    # B. the spread itself, which is the reproducibility number
    style(ax2)
    for i, prot in enumerate(["no_msa", "with_msa"]):
        s = d[d.protocol == prot].spread.astype(float)
        if s.empty:
            continue
        colour = NO_MSA if prot == "no_msa" else WITH_MSA
        ax2.scatter(np.full(len(s), i), s, s=52, c=colour, alpha=0.8,
                    linewidths=0, zorder=3)
        ax2.plot([i - 0.14, i + 0.14], [s.median()] * 2, color=colour,
                 linewidth=2.4, zorder=4)
        ax2.annotate(f"median {s.median():.2f}", xy=(i, s.median()),
                     xytext=(0, 10), textcoords="offset points",
                     ha="center", fontsize=9, color=INK_2)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["no target MSA", "with target MSA"])
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylabel("ipTM range across the 5 models", fontsize=10, color=INK_2)
    ax2.set_title("B  Spread on identical input", fontsize=10.5, color=INK, loc="left")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    print("wrote", args.out)

    print("\nper-design, recycling", args.recycling)
    for rk in ranks:
        for prot in ("no_msa", "with_msa"):
            row = d[(d["rank"] == rk) & (d.protocol == prot)]
            if row.empty:
                continue
            v = np.sort(row[MODEL_COLS].values.ravel().astype(float))
            print(f"  rank {rk:3d} {prot:9s} reported {row.reported_iptm.iloc[0]:.3f}  "
                  f"models {np.round(v,3)}  median {np.median(v):.3f}")


if __name__ == "__main__":
    main()
