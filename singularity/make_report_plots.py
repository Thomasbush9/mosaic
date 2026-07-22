#!/usr/bin/env python
"""Figures for the DIO3 setup report.

Run inside the container (matplotlib lives there):

    ./singularity/mosaic-exec.sh python singularity/make_report_plots.py \
        --designs <setup>/designs --out <setup>/report/figures

Palette is the validated categorical set — slots assigned in fixed order, never
cycled. Only the first three slots are used where all pairs are on screen at
once, which is the documented all-pairs-safe cap.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Validated categorical palette (light mode), fixed order.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e5e4df", "grid.linewidth": 0.8,
})


def _finish(ax, title, sub=None):
    # Title sits above the subtitle, which sits above the axes. Both are placed
    # in axes coordinates so they cannot collide with each other or the plot.
    if sub:
        ax.set_title(title, color=INK, fontweight="600", loc="left", pad=26)
        ax.text(0, 1.015, sub, transform=ax.transAxes, color=INK2, fontsize=9,
                va="bottom", ha="left")
    else:
        ax.set_title(title, color=INK, fontweight="600", loc="left", pad=8)


def _headroom(ax, frac=0.18):
    """Leave space above the tallest mark so a legend cannot sit on the data."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * frac)


def fig_backend_cost(out: Path):
    """Per-backend cost. Magnitude across named categories -> horizontal bars,
    sorted, direct-labelled (no legend: one series, the title names it)."""
    data = [("Protenix Mini", 39.4), ("Boltz-2", 91.0), ("OpenFold3", 92.4),
            ("AF2 multimer", 109.2), ("Boltz-1", 123.8)]
    data.sort(key=lambda r: r[1])
    names = [d[0] for d in data]
    vals = [d[1] for d in data]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y = np.arange(len(names))
    ax.barh(y, vals, height=0.62, color=C1, zorder=3)
    ax.set_yticks(y, names, color=INK)
    ax.set_xlabel("seconds")
    ax.set_xlim(0, max(vals) * 1.18)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.015, yi, f"{v:.0f}s", va="center",
                color=INK2, fontsize=9)
    _finish(ax, "Optimization cost per structure backend",
            "5 soft steps, batch 2, 237-residue target + 80-residue binder, H100")
    fig.tight_layout()
    fig.savefig(out / "backend_cost.png", dpi=200)
    plt.close(fig)


def fig_msa_effect(out: Path):
    """Paired before/after on one measure -> grouped bars, two slots."""
    models = ["Boltz-2", "Boltz-1"]
    no_msa = [1, 1]
    with_msa = [8192, 8192]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(models))
    w = 0.34
    ax.bar(x - w / 2 - 0.01, no_msa, w, label="single sequence", color=C1, zorder=3)
    ax.bar(x + w / 2 + 0.01, with_msa, w, label="local ColabFold MSA", color=C2, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("MSA rows ingested (n_msa, log)")
    ax.set_xticks(x, models, color=INK)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    for xi, v in zip(x - w / 2 - 0.01, no_msa):
        ax.text(xi, v * 1.25, str(v), ha="center", color=INK2, fontsize=9)
    for xi, v in zip(x + w / 2 + 0.01, with_msa):
        ax.text(xi, v * 1.25, f"{v:,}", ha="center", color=INK2, fontsize=9)
    ax.set_ylim(0.7, 40000)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK2, fontsize=9)
    _finish(ax, "The MSA is actually consumed, not silently dropped",
            "n_msa is the row count the model ingests — the check that catches a header mismatch")
    fig.tight_layout()
    fig.savefig(out / "msa_effect.png", dpi=200)
    plt.close(fig)


def _load(pattern: str):
    rows = []
    for p in sorted(glob.glob(pattern)):
        d = json.load(open(p))
        for r in d["results"]:
            rows.append({"seed": d["seed"], "loss": r["loss"],
                         "seq": r["sequence"], "models": ",".join(d["models"])})
    return rows


def fig_campaign(designs: Path, out: Path):
    """Distribution of a measure across two conditions -> dot strip per seed.
    Every design is a point; a bar would hide the spread that matters here."""
    a = _load(str(designs / "dio3_boltz2" / "*.json"))
    b = _load(str(designs / "dio3_boltz2af2" / "*.json"))
    if not a or not b:
        print("skipping campaign figure — missing results")
        return

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for rows, colour, label, off in ((a, C1, "Boltz-2 only", -0.11),
                                     (b, C2, "Boltz-2 + AF2", 0.11)):
        xs = [r["seed"] + off for r in rows]
        ys = [r["loss"] for r in rows]
        ax.scatter(xs, ys, s=64, color=colour, label=label, zorder=3,
                   edgecolors=SURFACE, linewidths=2)
    seeds = sorted({r["seed"] for r in a})
    ax.set_xticks(seeds, [f"seed {s}" for s in seeds], color=INK)
    ax.set_ylabel("final loss (lower is better)")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    _headroom(ax, 0.16)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK2, fontsize=9,
              ncols=2)
    _finish(ax, "DIO3 binder campaign: does a second predictor change the answer?",
            "2 trajectories per seed, 100 soft + 25 sharp steps, MSA-backed target")
    fig.tight_layout()
    fig.savefig(out / "campaign_losses.png", dpi=200)
    plt.close(fig)


def fig_composition(designs: Path, out: Path):
    """Amino-acid composition of the designs vs a natural-proteome baseline.
    Sanity check: a degenerate optimizer produces a few spiked residues."""
    rows = _load(str(designs / "dio3_boltz2" / "*.json"))
    if not rows:
        return
    TOKENS = "ARNDCQEGHILKMFPSTWYV"
    # UniProt/SwissProt average composition (%), for reference only.
    natural = {"A": 8.3, "R": 5.5, "N": 4.1, "D": 5.5, "C": 1.4, "Q": 3.9,
               "E": 6.7, "G": 7.1, "H": 2.3, "I": 5.9, "L": 9.7, "K": 5.8,
               "M": 2.4, "F": 3.9, "P": 4.7, "S": 6.6, "T": 5.4, "W": 1.1,
               "Y": 2.9, "V": 6.9}
    all_seq = "".join(r["seq"] for r in rows)
    counts = {t: all_seq.count(t) / len(all_seq) * 100 for t in TOKENS}

    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    x = np.arange(len(TOKENS))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, [counts[t] for t in TOKENS], w,
           label="designed binders", color=C1, zorder=3)
    ax.bar(x + w / 2 + 0.01, [natural[t] for t in TOKENS], w,
           label="natural proteins", color=C3, zorder=3)
    ax.set_xticks(x, list(TOKENS), color=INK)
    ax.set_ylabel("% of residues")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    _headroom(ax, 0.22)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK2, fontsize=9,
              ncols=2)
    _finish(ax, "Designed binder composition vs natural proteins",
            "C is zero by construction — NoCys removes it from the alphabet entirely")
    fig.tight_layout()
    fig.savefig(out / "composition.png", dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    designs = Path(a.designs)

    fig_backend_cost(out)
    fig_msa_effect(out)
    fig_campaign(designs, out)
    fig_composition(designs, out)
    for p in sorted(out.glob("*.png")):
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
