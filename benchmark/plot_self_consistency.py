"""Plot structural self-consistency (scRMSD) against the other ranking metrics.

Every column plotted on the x-axes is a loss where lower is better, and scRMSD
is lower-is-better too, so a POSITIVE Spearman rho means the two metrics agree
about which designs are good.

    python hallucinate/plot_self_consistency.py \
        --csv .../self_consistency.csv --out .../self_consistency.png
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

# Palette roles (see dataviz references/palette.md)
SERIES_1 = "#2a78d6"      # categorical slot 1 — the only series here

RMSD_PASS = 2.0
PLDDT_PASS = 0.7

# (column, axis label, what "good" looks like)
COMPARISONS = [
    ("loss", "design loss (Boltz2 objective)"),
    ("af2_losses", "AF2 ranking loss"),
    ("of3_loss", "OpenFold3 ranking loss"),
    ("esmc_loss", "ESM-C loss"),
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    d = pd.read_csv(args.csv)
    scored = d["sc_rmsd"].notna()
    n_ok = int(scored.sum())

    passes = scored & (d["sc_rmsd"] < RMSD_PASS) & (d["sc_plddt"] > PLDDT_PASS)
    n_pass = int(passes.sum())

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8), facecolor=SURFACE)
    fig.suptitle(
        f"Structural self-consistency of {n_ok} DIO3 binder designs   —   "
        f"{n_pass} pass (scRMSD < {RMSD_PASS:g} Å and refold pLDDT > {PLDDT_PASS:g})",
        fontsize=12.5, color=INK, y=0.975,
    )

    # A. the distribution — the headline: how many designs are self-consistent
    ax = axes[0, 0]
    style(ax)
    vals = d.loc[scored, "sc_rmsd"]
    ax.hist(vals, bins=40, color=SERIES_1, edgecolor=SURFACE, linewidth=0.6, zorder=2)
    ax.axvline(RMSD_PASS, color=CRITICAL, linestyle="--", linewidth=1.6, zorder=3)
    ax.annotate(f"{RMSD_PASS:g} Å threshold", xy=(RMSD_PASS, ax.get_ylim()[1] * 0.92),
                xytext=(5, 0), textcoords="offset points",
                color=CRITICAL, fontsize=8.5, va="top")
    # The gate is only interesting if it actually splits the set; say how far
    # from uninformative it is rather than leaving the reader to eyeball it.
    ax.annotate(f"{(vals < 1).mean():.0%} under 1 Å\n{(vals < RMSD_PASS).mean():.0%} under {RMSD_PASS:g} Å",
                xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
                fontsize=8.5, color=INK_2)
    ax.set_xlabel("scRMSD (Å), best of n MPNN samples", fontsize=9, color=INK_2)
    ax.set_ylabel("designs", fontsize=9, color=INK_2)
    ax.set_title("A  Distribution of scRMSD", fontsize=10, color=INK, loc="left")

    # B-E. against each existing ranking metric
    for ax, (col, label) in zip(axes.flat[1:5], COMPARISONS):
        style(ax)
        if col not in d.columns:
            ax.set_visible(False)
            continue
        rho = spearman(d[col], d["sc_rmsd"])
        ax.scatter(d[col], d["sc_rmsd"], s=16, c=SERIES_1, alpha=0.55,
                   linewidths=0, zorder=2)
        ax.axhline(RMSD_PASS, color=CRITICAL, linestyle="--", linewidth=1.2, zorder=3)
        ax.set_xlabel(f"{label}  (lower = better)", fontsize=9, color=INK_2)
        ax.set_ylabel("scRMSD (Å)", fontsize=9, color=INK_2)
        ax.set_title(f"{'BCDE'[COMPARISONS.index((col, label))]}  scRMSD vs {col}"
                     f"    ρ = {rho:+.2f}", fontsize=10, color=INK, loc="left")

    # F. the value-for-money panel. ref_plddt comes free with the single
    # reference fold; if it predicts scRMSD this well, the n extra refolds are
    # buying a number one fold already gave you.
    ax = axes[1, 2]
    style(ax)
    rho = spearman(d["ref_plddt"], d["sc_rmsd"])
    rho_sc = spearman(d["sc_plddt"], d["sc_rmsd"])
    ax.scatter(d["ref_plddt"], d["sc_rmsd"], s=16, c=SERIES_1, alpha=0.55,
               linewidths=0, zorder=2)
    ax.axhline(RMSD_PASS, color=CRITICAL, linestyle="--", linewidth=1.2, zorder=3)
    ax.axvline(PLDDT_PASS, color=CRITICAL, linestyle="--", linewidth=1.2, zorder=3)
    ax.set_xlabel("reference-fold pLDDT (0-1), free with one AF2 run",
                  fontsize=9, color=INK_2)
    ax.set_ylabel("scRMSD (Å)", fontsize=9, color=INK_2)
    ax.set_title(f"F  scRMSD vs reference confidence    ρ = {rho:+.2f}",
                 fontsize=10, color=INK, loc="left")
    ax.annotate(f"vs refold pLDDT: ρ = {rho_sc:+.2f}",
                xy=(0.03, 0.06), xycoords="axes fraction",
                fontsize=8.5, color=INK_2)

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    print(f"wrote {args.out}", flush=True)

    # The numbers behind the figure, for the write-up.
    print(f"\nscored {n_ok}/{len(d)} designs; {n_pass} pass both gates")
    print(d[["sc_rmsd", "sc_rmsd_mean", "sc_plddt", "ref_plddt",
             "frac_under_2A", "best_seq_recovery"]].describe().to_string())
    print("\nSpearman rho vs scRMSD (positive = metrics agree):")
    for col, _ in COMPARISONS:
        if col in d.columns:
            print(f"  {col:14s} {spearman(d[col], d['sc_rmsd']):+.3f}")
    print(f"  {'sc_plddt':14s} {spearman(d['sc_plddt'], d['sc_rmsd']):+.3f}")

    if n_pass:
        cols = [c for c in ["sc_rmsd", "sc_plddt", "af2_losses", "of3_loss",
                            "loss", "best_seq_recovery"] if c in d.columns]
        print("\ntop 10 designs by scRMSD (passing both gates):")
        print(d[passes].nsmallest(10, "sc_rmsd")[cols].to_string())


if __name__ == "__main__":
    main()
