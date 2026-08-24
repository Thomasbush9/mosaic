"""Does pose self-consistency add anything over the confidence scores you have?

The test that matters is not "does pose_rmsd correlate with a good design" —
it will, because everything correlates with everything on a design set. It is
whether pose_rmsd predicts an INDEPENDENT label (OpenFold3, 6-sample averaged)
*after* partialling out AF2's own confidence, which comes free with the single
reference fold. Panel E is that comparison; the rest is context.
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


GATE = 2.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    d = pd.read_csv(args.csv).dropna(subset=["pose_rmsd"])
    cols = ["pose_rmsd", "fold_rmsd", "ref_iptm", "best_iptm", "af2_losses",
            "of3_loss", "loss", "ref_binder_plddt"]
    r = d[cols].rank()

    def sp(a, b):
        return r[a].corr(r[b])

    def partial(x, y, z):
        rx = r[x] - np.polyval(np.polyfit(r[z], r[x], 1), r[z])
        ry = r[y] - np.polyval(np.polyfit(r[z], r[y], 1), r[z])
        return np.corrcoef(rx, ry)[0, 1]

    n_fail = int((d.pose_rmsd > GATE).sum())
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8), facecolor=SURFACE)
    fig.suptitle(
        f"Pose self-consistency on {len(d)} hallucination designs — "
        f"{n_fail} of {len(d)} exceed {GATE:g} Å",
        fontsize=12.5, color=INK, y=0.975)

    # A. distributions, log x because the tail spans two decades
    ax = axes[0, 0]; style(ax)
    bins = np.logspace(np.log10(0.1), np.log10(max(d.pose_rmsd.max(), 20)), 44)
    ax.hist(d.fold_rmsd, bins=bins, color=S2, alpha=0.75,
            edgecolor=SURFACE, linewidth=0.5, label="fold-RMSD (binder-aligned)", zorder=2)
    ax.hist(d.pose_rmsd, bins=bins, color=S1, alpha=0.75,
            edgecolor=SURFACE, linewidth=0.5, label="pose-RMSD (target-aligned)", zorder=3)
    ax.axvline(GATE, color=CRITICAL, ls="--", lw=1.5, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("RMSD (Å), best of n samples", fontsize=9, color=INK_2)
    ax.set_ylabel("designs", fontsize=9, color=INK_2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.32)   # headroom so the legend clears the bars
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper right")
    ax.set_title("A  Pose moves where fold does not", fontsize=10, color=INK, loc="left")

    # B-D. against each label
    for ax, (col, lab) in zip(
            axes.flat[1:4],
            [("af2_losses", "AF2 ranking loss (same model as reference fold)"),
             ("of3_loss", "OpenFold3 loss — INDEPENDENT model, 6-sample mean"),
             ("ref_iptm", "reference-fold ipTM (free with one fold)")]):
        style(ax)
        ax.scatter(d[col], d.pose_rmsd, s=15, c=S1, alpha=0.55, linewidths=0, zorder=2)
        ax.axhline(GATE, color=CRITICAL, ls="--", lw=1.2, zorder=3)
        ax.set_yscale("log")
        ax.set_xlabel(lab, fontsize=8.5, color=INK_2)
        ax.set_ylabel("pose-RMSD (Å)", fontsize=9, color=INK_2)
        ax.set_title(f"{'BCD'[list(axes.flat[1:4]).index(ax)]}  vs {col}"
                     f"    ρ = {sp('pose_rmsd', col):+.2f}",
                     fontsize=10, color=INK, loc="left")

    # E. the money panel: raw vs partial against the independent label
    ax = axes[1, 1]; style(ax, grid="y")
    measures = [("pose_rmsd", "pose-RMSD\n(complex, 13× cost)"),
                ("fold_rmsd", "fold-RMSD\n(binder only)"),
                ("ref_iptm", "AF2 ipTM\n(1 fold, free)")]
    raw = [abs(sp(m, "of3_loss")) for m, _ in measures]
    par = [abs(partial(m, "of3_loss", "ref_iptm")) if m != "ref_iptm"
           else abs(partial("ref_iptm", "of3_loss", "pose_rmsd")) for m, _ in measures]
    x = np.arange(len(measures))
    ax.bar(x - 0.19, raw, 0.36, color=S1, zorder=2, label="raw |ρ|")
    ax.bar(x + 0.19, par, 0.36, color=S2, zorder=2,
           label="|ρ| after removing AF2 confidence*")
    for xi, (a, b) in enumerate(zip(raw, par)):
        ax.annotate(f"{a:.2f}", (xi - 0.19, a), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_2)
        ax.annotate(f"{b:.2f}", (xi + 0.19, b), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in measures], fontsize=8)
    ax.set_ylabel("|Spearman ρ| with OpenFold3 loss", fontsize=9, color=INK_2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    ax.set_title("E  What each measure independently buys", fontsize=10, color=INK, loc="left")
    ax.annotate("*ipTM bar controls for pose-RMSD instead", xy=(0.01, -0.30),
                xycoords="axes fraction", fontsize=7.5, color=INK_2)

    # F. redundancy between the two RMSDs
    ax = axes[1, 2]; style(ax)
    ax.scatter(d.fold_rmsd, d.pose_rmsd, s=15, c=S1, alpha=0.55, linewidths=0, zorder=2)
    lim = [0.1, max(d.pose_rmsd.max(), d.fold_rmsd.max()) * 1.2]
    ax.plot(lim, lim, color=INK_2, ls=":", lw=1, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("fold-RMSD (Å)", fontsize=9, color=INK_2)
    ax.set_ylabel("pose-RMSD (Å)", fontsize=9, color=INK_2)
    ax.set_title(f"F  pose vs fold    ρ = {sp('pose_rmsd','fold_rmsd'):+.2f}",
                 fontsize=10, color=INK, loc="left")

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    print("wrote", args.out)

    print(f"\nn = {len(d)};  pose>2Å: {n_fail}   >5Å: {int((d.pose_rmsd>5).sum())}")
    print("\nvs of3_loss (independent label):")
    for m, lab in measures:
        z = "ref_iptm" if m != "ref_iptm" else "pose_rmsd"
        print(f"  {m:16s} raw {sp(m,'of3_loss'):+.3f}   partial|{z:9s} "
              f"{partial(m,'of3_loss',z):+.3f}")
    print("\ndoes a pose gate change the picks?")
    for k in (10, 20, 50):
        top = d.nsmallest(k, "af2_losses")
        print(f"  top {k:2d} by af2_losses: {int((top.pose_rmsd>GATE).sum())} rejected by pose>{GATE:g}Å")


if __name__ == "__main__":
    main()
