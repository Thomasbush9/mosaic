"""Model x model agreement matrices from the tidy table, one per metric.

The diagonal of each matrix is MEASURED, not set to 1.0: it is the rank
correlation between two replicates of the same model on the same designs (AF2's
five parameter sets; a resampled diffusion trajectory for everyone else). That
is the whole point of the figure -- an off-diagonal of 0.61 means one thing if
the model agrees with itself at 0.90 and something else entirely if it agrees
with itself at 0.30.

Cells are shaded by rho, and the summary panel plots mean off-diagonal against
the diagonal so a metric where the models genuinely differ is separable from one
where they are all just noisy.
"""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (SURFACE, INK, INK_2, S1, S2, MUTED, CRITICAL, GRID,
                    style, spearman, auc, ORIENT)


MODELS = ["af2", "boltz1", "boltz2", "of3", "protenix", "esmfold2"]
LABEL = {"af2": "AF2", "boltz1": "Boltz-1", "boltz2": "Boltz-2",
         "of3": "OF3", "protenix": "Protenix", "esmfold2": "ESMFold2"}
# AF3-lineage (pairformer + diffusion) vs AF2's Evoformer + structure module.
# Panel C of the scoping report suggested agreement clusters by family; this
# marks the families so the matrices can be read against that prediction.
FAMILY = {"af2": "AF2", "boltz1": "AF3", "boltz2": "AF3",
          "of3": "AF3", "protenix": "AF3", "esmfold2": "ESM"}

def wide(df, condition, metric, replicate):
    """design_idx x model table of one metric, one replicate."""
    d = df[(df.condition == condition) & (df.metric == metric)
           & (df.replicate == replicate)]
    return d.pivot_table(index="design_idx", columns="model", values="value")


def build(df, condition, metric):
    """(matrix, models_present). Diagonal = replicate 0 vs replicate 1."""
    w0 = wide(df, condition, metric, 0)
    present = [m for m in MODELS if m in w0.columns and w0[m].notna().sum() >= 5]
    n = len(present)
    M = np.full((n, n), np.nan)
    for i, a in enumerate(present):
        for j, b in enumerate(present):
            if i == j:
                continue
            s = w0[[a, b]].dropna()
            M[i, j] = spearman(s[a].values, s[b].values)
    w1 = wide(df, condition, metric, 1)
    for i, m in enumerate(present):
        if m in w1.columns:
            s = pd.concat([w0[m].rename("r0"), w1[m].rename("r1")],
                          axis=1).dropna()
            M[i, i] = spearman(s.r0.values, s.r1.values)
    return M, present


# Every model-vs-model rho observed is positive (0.27..1.00 across both
# conditions and all 15 metrics), so these use a SEQUENTIAL map over the
# observed range. A diverging map would spend most of its range on values that
# never occur and would place its neutral midpoint at ~0.63, implying a
# "neutral" correlation that has no meaning here. The metric x metric matrices
# below keep a diverging map, because 46% of their cells really are negative.
VMIN = 0.25


def draw(ax, M, present, title):
    im = ax.imshow(M, cmap="magma_r", vmin=VMIN, vmax=1.0)
    for i in range(len(present)):
        for j in range(len(present)):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", fontsize=7,
                        color=INK_2)
                continue
            # The diagonal is data, not decoration -- box it so it reads as the
            # reference the off-diagonals should be compared against.
            if i == j:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor=INK, linewidth=1.6))
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color=INK if v < (VMIN + 1) / 2 else "#ffffff",
                    fontweight="bold" if i == j else "normal")
    lab = [LABEL[m] for m in present]
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(lab, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(lab, fontsize=7)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.grid(False)
    ax.tick_params(colors=INK_2, length=0)
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix-dir", type=Path, required=True)
    p.add_argument("--tag", default="full")
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args()

    files = sorted(args.matrix_dir.glob(f"{args.tag}_*.csv"))
    if not files:
        raise SystemExit(f"no {args.tag}_*.csv in {args.matrix_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} files, {len(df)} rows, "
          f"models={sorted(df.model.unique())}")

    summary = []
    for condition in ["complex", "monomer"]:
        sub = df[df.condition == condition]
        if sub.empty:
            continue
        metrics = sorted(sub.metric.unique())
        ncol = min(4, len(metrics))
        nrow = int(np.ceil(len(metrics) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.5 * nrow),
                                 facecolor=SURFACE, squeeze=False)
        # AF2's diagonal does not mean the same thing in both conditions, and
        # saying so on the figure is cheaper than having it misread. In complex
        # it is msa_seed (clusters resampled within one network -- the
        # like-for-like analogue of a diffusion reseed); in monomer there is no
        # MSA to resample and AF2 is bit-deterministic, so only ensemble spread
        # across the five trained networks can be measured at all.
        #
        # Carried in the suptitle rather than a separate fig.text: the monomer
        # grid is one row tall, so a second line placed by figure fraction lands
        # on top of the title.
        note = ("replicate = diffusion reseed; AF2 = MSA-cluster reseed at "
                "fixed model_idx"
                if condition == "complex" else
                "replicate = diffusion reseed; AF2 = ensemble spread over its 5 "
                "trained networks (no MSA to resample, so AF2 is deterministic here)")
        fig.suptitle(
            f"Model agreement — {condition}   (boxed diagonal = same model, "
            f"different replicate)\n{note}",
            fontsize=12.5, color=INK, y=0.995)
        im = None
        for k, metric in enumerate(metrics):
            ax = axes[k // ncol][k % ncol]
            M, present = build(sub, condition, metric)
            if not present:
                ax.axis("off")
                continue
            im = draw(ax, M, present, metric)
            off = M[~np.eye(len(present), dtype=bool)]
            dia = np.diag(M)
            summary.append({
                "condition": condition, "metric": metric,
                "mean_off_diag": np.nanmean(off),
                "mean_diag": np.nanmean(dia),
                "n_models": len(present),
            })
        for k in range(len(metrics), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        # Model names are long enough that the y labels of one column render
        # over the right edge of the column to its left at default spacing.
        fig.subplots_adjust(wspace=0.55, hspace=0.45)
        if im is not None:
            cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
            cb.set_label("Spearman ρ", fontsize=9, color=INK_2)
            cb.ax.tick_params(colors=INK_2, labelsize=8)
        out = args.outdir / f"matrices_{condition}.png"
        fig.savefig(out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
        print("wrote", out)
        plt.close(fig)

    # ---- metric x metric redundancy, within each model ---------------------
    for condition in ["complex", "monomer"]:
        sub = df[(df.condition == condition) & (df.replicate == 0)]
        if sub.empty:
            continue
        models = [m for m in MODELS if m in sub.model.unique()]
        metrics = sorted(sub.metric.unique())
        fig, axes = plt.subplots(1, len(models),
                                 figsize=(3.6 * len(models), 3.9),
                                 facecolor=SURFACE, squeeze=False)
        fig.suptitle(f"Metric redundancy within each model — {condition}\n"
                     "all metrics oriented so higher = more confident; "
                     "|ρ| near 1 means the two are one measurement",
                     fontsize=12.5, color=INK, y=1.06)
        for ax, m in zip(axes[0], models):
            w = sub[sub.model == m].pivot_table(
                index="design_idx", columns="metric", values="value")
            cols = [c for c in metrics if c in w.columns]
            M = np.full((len(cols), len(cols)), np.nan)
            for i, a in enumerate(cols):
                for j, b in enumerate(cols):
                    if i == j:
                        # A metric against itself. Unlike the model matrices,
                        # this diagonal really is 1 by construction, not a
                        # replicate measurement.
                        M[i, j] = 1.0
                        continue
                    s = w[[a, b]].dropna()
                    M[i, j] = spearman(ORIENT.get(a, 1) * s[a].values,
                                       ORIENT.get(b, 1) * s[b].values)
            ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
            for i in range(len(cols)):
                for j in range(len(cols)):
                    if not np.isnan(M[i, j]):
                        ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                                fontsize=6,
                                color=INK if abs(M[i, j]) < 0.6 else "#ffffff")
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels(cols, fontsize=6, rotation=90)
            ax.set_yticks(range(len(cols)))
            ax.set_yticklabels(cols, fontsize=6)
            ax.set_title(LABEL[m], fontsize=9.5, color=INK, loc="left")
            ax.grid(False)
            ax.tick_params(colors=INK_2, length=0)
        # Metric names are long and render outside their axes; without this the
        # y labels of one panel land on the panel to its left.
        fig.subplots_adjust(wspace=0.75)
        out = args.outdir / f"metric_redundancy_{condition}.png"
        fig.savefig(out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
        print("wrote", out)
        plt.close(fig)

    s = pd.DataFrame(summary)
    if not s.empty:
        s = s.sort_values(["condition", "mean_off_diag"])
        s.to_csv(args.outdir / "summary.csv", index=False)
        print("\nmean agreement per metric (off-diagonal vs measured diagonal):")
        print(s.to_string(index=False,
                          float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
