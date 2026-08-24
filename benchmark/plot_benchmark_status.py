"""Figure for the binder-scoring benchmark scoping report.

A  What labelled data exists, per target (only targets that can support a statistic)
B  Zero-GPU baseline: which precomputed scores actually separate binders,
   with trivial physicochemical properties included as negative controls
C  Judge redundancy on our own 416 DIO3 designs -- who is an independent opinion

Everything is recomputed from source files; nothing is hardcoded.
"""

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (SURFACE, INK, INK_2, S1, S2, MUTED, CRITICAL, GRID,
                    style, spearman, auc)

csv.field_size_limit(10 ** 9)


# metric -> (orientation, family)
METRICS = {
    "boltz2_complex_iplddt": (+1, "cofold"), "boltz2_plddt": (+1, "cofold"),
    "boltz2_ipsae": (+1, "cofold"), "boltz2_lis": (+1, "cofold"),
    "boltz2_iptm": (+1, "cofold"), "boltz2_complex_pde": (-1, "cofold"),
    "boltz2_pdockq": (+1, "cofold"), "boltz2_ptm": (+1, "cofold"),
    "esmfold_plddt": (+1, "seq"),
    "proteinmpnn_score": (-1, "seq"),
    "redesigned_proteinmpnn_score": (-1, "seq"),
    "proteinmpnn_seq_recovery": (+1, "seq"),
    "molecular_weight": (+1, "control"),
    "isoelectric_point": (+1, "control"),
}
LABEL = {
    "boltz2_complex_iplddt": "Boltz-2 complex ipLDDT", "boltz2_plddt": "Boltz-2 pLDDT",
    "boltz2_ipsae": "Boltz-2 ipSAE", "boltz2_lis": "Boltz-2 LIS",
    "boltz2_iptm": "Boltz-2 ipTM", "boltz2_complex_pde": "Boltz-2 complex PDE",
    "boltz2_pdockq": "Boltz-2 pDockQ", "boltz2_ptm": "Boltz-2 pTM",
    "esmfold_plddt": "ESMFold pLDDT", "proteinmpnn_score": "ProteinMPNN score",
    "redesigned_proteinmpnn_score": "ProteinMPNN score (redesigned)",
    "proteinmpnn_seq_recovery": "ProteinMPNN seq recovery",
    "molecular_weight": "molecular weight  (control)",
    "isoelectric_point": "isoelectric point  (control)",
}

def load_pb(path):
    pairs = {}
    rows = list(csv.DictReader(open(path)))
    idcol = next(iter(rows[0].keys()))
    for r in rows:
        try:
            ev = json.loads(r["evaluations"] or "[]")
        except Exception:
            continue
        calls, comp = defaultdict(list), {}
        for e in ev:
            m, tgt, v = e.get("metric"), e.get("target"), e.get("value")
            if e.get("type") == "experimental" and m == "binding" and \
                    isinstance(v, bool) and tgt:
                calls[tgt].append(v)
            elif e.get("type") == "computational" and m in METRICS:
                try:
                    comp[m] = float(v)
                except (TypeError, ValueError):
                    pass
        for tgt, cs in calls.items():
            pairs[(r[idcol], tgt)] = {"binder": sum(cs) / len(cs) > 0.5, **comp}
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proteinbase", type=Path, required=True)
    ap.add_argument("--designs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-pairs", type=int, default=50)
    args = ap.parse_args()

    pairs = load_pb(args.proteinbase)
    by_t = defaultdict(list)
    for (_, tgt), d in pairs.items():
        by_t[tgt].append(d)
    usable = sorted(
        [t for t, ds in by_t.items()
         if len(ds) >= args.min_pairs and 0 < sum(d["binder"] for d in ds) < len(ds)],
        key=lambda t: -len(by_t[t]))

    # Panel B's metric names are long and render OUTSIDE its axes to the left,
    # so wspace has to clear them or they land on top of panel A.
    fig = plt.figure(figsize=(17.0, 6.4), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.5, 0.95], wspace=0.70)
    fig.suptitle(
        "Binder-scoring benchmark: what the existing data already answers, at zero GPU cost",
        fontsize=13.5, color=INK, y=0.97)

    # ---- A. data inventory -------------------------------------------------
    ax = fig.add_subplot(gs[0]); style(ax, grid=None)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    names = usable
    nb = [sum(d["binder"] for d in by_t[t]) for t in names]
    nn = [len(by_t[t]) - b for t, b in zip(names, nb)]
    y = np.arange(len(names))[::-1]
    ax.barh(y, nb, 0.62, color=S1, zorder=3, label="binders")
    ax.barh(y, nn, 0.62, left=nb, color=S2, zorder=3, label="non-binders")
    for yi, (b, n) in enumerate(zip(nb, nn)):
        ax.annotate(f"{b}/{b+n}", (b + n, y[yi]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("labelled protein × target pairs", fontsize=9, color=INK_2)
    ax.set_xlim(0, max(np.array(nb) + np.array(nn)) * 1.22)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower right")
    ax.set_title("A  Usable labelled data\n     (Proteinbase, ≥50 pairs, both classes)",
                 fontsize=10.5, color=INK, loc="left")

    # ---- B. zero-GPU AUC ---------------------------------------------------
    ax = fig.add_subplot(gs[1]); style(ax, grid=None)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    res = []
    for m, (sign, fam) in METRICS.items():
        aucs = []
        for t in usable:
            ds = [d for d in by_t[t] if m in d]
            if len(ds) < args.min_pairs:
                continue
            a = auc([sign * d[m] for d in ds if d["binder"]],
                    [sign * d[m] for d in ds if not d["binder"]])
            if a is not None:
                aucs.append(a)
        if aucs:
            res.append((m, fam, st.median(aucs), len(aucs)))
    res.sort(key=lambda r: r[2])
    y = np.arange(len(res))
    cols = {"cofold": S1, "seq": S2, "control": MUTED}
    ax.barh(y, [r[2] for r in res], 0.66,
            color=[cols[r[1]] for r in res], zorder=3)
    ax.axvline(0.5, color=CRITICAL, ls="--", lw=1.6, zorder=4)
    ax.annotate("chance", (0.5, len(res) - 0.3), xytext=(4, 0),
                textcoords="offset points", fontsize=8.5, color=CRITICAL)
    for yi, r in enumerate(res):
        ax.annotate(f"{r[2]:.3f}  (n={r[3]} tgt)", (r[2], yi), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[r[0]] for r in res], fontsize=8.5)
    ax.set_xlim(0.35, 0.85)
    ax.set_xlabel("median ROC-AUC across targets  (binder vs non-binder)",
                  fontsize=9, color=INK_2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cols[k]) for k in
               ("cofold", "seq", "control")]
    ax.legend(handles, ["co-folding confidence", "sequence-only model",
                        "trivial property (control)"],
              frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower right")
    ax.set_title("B  Sequence-only scores sit at chance —\n"
                 "     molecular weight beats all of them",
                 fontsize=10.5, color=INK, loc="left")

    # ---- C. judge redundancy ----------------------------------------------
    ax = fig.add_subplot(gs[2]); style(ax, grid=None)
    rows = list(csv.DictReader(open(args.designs)))
    jcols = ["loss", "af2_losses", "of3_loss", "esmc_loss"]
    jlab = ["Boltz-2\n(design)", "AF2", "OpenFold3", "ESM-C"]
    dat = {c: [float(r[c]) for r in rows] for c in jcols}
    M = np.array([[spearman(dat[a], dat[b]) for b in jcols] for a in jcols])
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(jcols)):
        for j in range(len(jcols)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=9,
                    color=INK if abs(M[i, j]) < 0.6 else "#ffffff")
    ax.set_xticks(range(len(jcols))); ax.set_xticklabels(jlab, fontsize=8)
    ax.set_yticks(range(len(jcols))); ax.set_yticklabels(jlab, fontsize=8)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Spearman ρ", fontsize=8.5, color=INK_2)
    cb.ax.tick_params(colors=INK_2, labelsize=8)
    ax.set_title(f"C  Judge redundancy on our\n     {len(rows)} DIO3 designs",
                 fontsize=10.5, color=INK, loc="left")

    fig.subplots_adjust(left=0.085, right=0.965, top=0.86, bottom=0.11)
    fig.savefig(args.out, dpi=int(__import__("os").environ.get("FIGDPI",170)), facecolor=SURFACE)
    print("wrote", args.out)
    print(f"\nusable targets: {len(usable)} -> {usable}")
    print("\nAUC (median across targets):")
    for m, fam, a, n in sorted(res, key=lambda r: -r[2]):
        print(f"  {LABEL[m]:34s} {fam:8s} {a:.3f}  (n={n} targets)")
    print("\njudge redundancy (Spearman):")
    for i, a in enumerate(jcols):
        print("  " + f"{a:12s}" + "".join(f"{M[i,j]:8.2f}" for j in range(len(jcols))))


if __name__ == "__main__":
    main()
