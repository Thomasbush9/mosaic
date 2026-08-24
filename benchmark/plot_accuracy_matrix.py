"""Model x metric accuracy against the Adaptyv experimental readout.

The agreement matrices (plot_model_matrices.py) say which models hold different
opinions. This says which opinion is right: every cell is how well one model's
one metric separates experimentally confirmed binders from non-binders.

Two readouts, both free from the same folds:
    AUC   binary binding call, Mann-Whitney rank-sum. 0.5 = chance.
    rho   Spearman against the graded strength (none < weak < medium < strong),
          which is more sensitive than the binary split.

NEGATIVE CONTROLS are computed from sequence alone and plotted in the same
matrix. Molecular weight reached AUC 0.533 on this data in the earlier scoping
pass -- beating every sequence-only model score. A co-folding metric that cannot
clear a property with no knowledge of the target has not earned its GPU time.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (SURFACE, INK, INK_2, S1, S2, MUTED, CRITICAL, GRID,
                    style, spearman, auc)

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"

LABEL = {"af2": "AF2", "boltz1": "Boltz-1", "boltz2": "Boltz-2",
         "of3": "OF3", "protenix": "Protenix", "esmfold2": "ESMFold2"}
MODELS = ["af2", "boltz1", "boltz2", "of3", "protenix", "esmfold2"]
# +1 = higher predicts binding. PAE and pTMEnergy are errors/energies.
ORIENT = {"ipsae_min": +1, "bt_ipsae": +1, "iptm": +1, "bt_iptm": +1,
          "binder_ptm": +1, "binder_plddt": +1, "complex_plddt": +1,
          "iplddt": +1, "bt_pae": -1, "tb_pae": -1, "ptm_energy": -1}

# Kyte-Doolittle-independent: average residue masses (Da), water excluded.
AA_MASS = dict(zip("ARNDCQEGHILKMFPSTWYV",
    [71.08, 156.19, 114.10, 115.09, 103.14, 128.13, 129.12, 57.05, 137.14,
     113.16, 113.16, 128.17, 131.19, 147.18, 97.12, 87.08, 101.10, 186.21,
     163.18, 99.13]))

def controls(seqs):
    """Sequence-only properties that cannot know anything about the target."""
    mw = [sum(AA_MASS.get(a, 110.0) for a in s) + 18.02 for s in seqs]
    charge = [sum(s.count(a) for a in "KR") - sum(s.count(a) for a in "DE")
              for s in seqs]
    return {"molecular weight (ctrl)": mw, "net charge (ctrl)": charge,
            "length (ctrl)": [len(s) for s in seqs]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--labelled", type=Path, required=True)
    p.add_argument("--tag", default="acc")
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args()

    files = sorted(args.scores.glob(f"{args.tag}_*.csv"))
    if not files:
        raise SystemExit(f"no {args.tag}_*.csv in {args.scores}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df.replicate == 0]
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Graded strength, joined on sequence.
    strength = {}
    for f in args.labelled.glob("*.csv"):
        for r in csv.DictReader(open(f)):
            if r.get("strength_rank") not in ("", None):
                strength[r["seq"]] = int(r["strength_rank"])

    for target_file in files:
        pass
    targets = sorted({f.stem.split("_")[-1] for f in files})
    rows_out = []

    for tgt in targets:
        sub = df[df.seq.isin(
            {r["seq"] for r in csv.DictReader(
                open(args.labelled / f"{tgt}.csv"))})]
        if sub.empty:
            continue
        metrics = [m for m in ORIENT if m in set(sub.metric)]
        models = [m for m in MODELS if m in set(sub.model)]
        M = np.full((len(models) + 3, len(metrics)), np.nan)

        for i, mdl in enumerate(models):
            for j, met in enumerate(metrics):
                d = sub[(sub.model == mdl) & (sub.metric == met)].dropna(
                    subset=["value", "label"])
                if len(d) < 20:
                    continue
                s = ORIENT[met] * d.value.values
                lab = d.label.values.astype(int)
                M[i, j] = auc(s[lab == 1], s[lab == 0])
                rows_out.append({"target": tgt, "model": mdl, "metric": met,
                                 "n": len(d), "n_binders": int(lab.sum()),
                                 "auc": M[i, j],
                                 "rho_strength": _rho_strength(d, met, strength)})

        # Controls occupy the last rows, spanning every metric column so the
        # comparison is unmissable rather than buried in a separate figure.
        anyd = sub[sub.model == models[0]].drop_duplicates("seq").dropna(
            subset=["label"])
        ctl = controls(anyd.seq.tolist())
        lab = anyd.label.values.astype(int)
        for k, (cname, cvals) in enumerate(ctl.items()):
            a = auc(np.array(cvals)[lab == 1], np.array(cvals)[lab == 0])
            M[len(models) + k, :] = a
            rows_out.append({"target": tgt, "model": cname, "metric": "—",
                             "n": len(lab), "n_binders": int(lab.sum()),
                             "auc": a, "rho_strength": np.nan})

        ylab = [LABEL[m] for m in models] + list(ctl)
        fig, ax = plt.subplots(figsize=(1.05 * len(metrics) + 3.5,
                                        0.46 * len(ylab) + 2.4),
                               facecolor=SURFACE)
        im = ax.imshow(M, cmap="RdBu_r", vmin=0.25, vmax=0.75, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                            fontsize=7.5,
                            color=INK if abs(M[i, j] - 0.5) < 0.15 else "#ffffff")
        ax.axhline(len(models) - 0.5, color=INK, lw=1.8)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(len(ylab)))
        ax.set_yticklabels(ylab, fontsize=8.5)
        n_b = int(sub.drop_duplicates("seq").label.sum())
        n_t = sub.seq.nunique()
        ax.set_title(f"{tgt} — ROC-AUC, binder vs non-binder\n"
                     f"{n_b} binders of {n_t} assayed   "
                     f"(0.5 = chance; controls below the line)",
                     fontsize=10.5, color=INK, loc="left")
        ax.grid(False); ax.tick_params(colors=INK_2, length=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("ROC-AUC", fontsize=8.5, color=INK_2)
        cb.ax.tick_params(colors=INK_2, labelsize=8)
        fig.tight_layout()
        out = args.outdir / f"accuracy_{tgt}.png"
        fig.savefig(out, dpi=165, facecolor=SURFACE, bbox_inches="tight")
        print("wrote", out)
        plt.close(fig)

    res = pd.DataFrame(rows_out)
    res.to_csv(args.outdir / "accuracy_summary.csv", index=False)
    print("\ntop cells by AUC:")
    print(res.dropna(subset=["auc"]).sort_values("auc", ascending=False)
          .head(20).to_string(index=False,
                              float_format=lambda v: f"{v:.3f}"))


def _rho_strength(d, met, strength):
    g = d[d.seq.isin(strength)]
    if len(g) < 20:
        return np.nan
    x = pd.Series(ORIENT[met] * g.value.values).rank()
    y = pd.Series([strength[s] for s in g.seq]).rank()
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


if __name__ == "__main__":
    main()
