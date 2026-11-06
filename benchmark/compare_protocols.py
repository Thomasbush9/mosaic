"""Does the featurization protocol explain the matrix / ranking disagreement?

The agreement matrices (benchmark/matrix/full_*) scored every design with
binder_features(L, [target]) and one diffusion sample. The ranking runs that
produced dio3_cut/designs/correctly_ranked_candidates_af2.csv used
target_only_features([binder, target]) and num_samples=6 (hallucinate.py:137,
rank_designs.py:232). On the same 416 designs and the same composite the two
disagreed badly -- boltz2~of3 was +0.70 in the ranking columns and +0.40 in the
matrix.

binder_features stubs the binder's reference sidechain atoms to poly-G/UNK,
which is correct when one feature pack is reused across many sequences but
denies the all-atom models the sidechain conditioning they were trained with.
This script quantifies how much of the gap that accounts for, by rescoring the
same designs under both protocols with the exact ranking composite recorded as
a metric (`rank_composite`).

    singularity/mosaic-exec.sh python benchmark/compare_protocols.py

Reads benchmark/matrix/{bffull,tofull}_*_complex.csv, writes
benchmark/matrix_figs/protocol_comparison.{png,csv}.
"""

import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CRITICAL, GRID, INK, INK_2, MUTED, S1, S2, SURFACE, spearman, style

SETUP = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/mosaic_setup")
MATRIX = SETUP / "benchmark/matrix"
FIGS = SETUP / "benchmark/matrix_figs"
RANKED = SETUP / "dio3_cut/designs/correctly_ranked_candidates_af2.csv"

# The three models the ranking CSV actually carries a column for.
RANK_COL = {"boltz2": "loss", "of3": "of3_loss", "af2": "af2_losses"}
MODELS = ["af2", "boltz1", "boltz2", "of3", "protenix", "esmfold2"]
PROTOCOLS = {"bffull": "binder_features, 1 sample",
             "tofull": "target_only_features, 6 samples"}
METRIC = "rank_composite"


def wide(tag, metric=METRIC, replicate=0):
    """seq x model table of one metric, for one protocol."""
    cols = {}
    for m in MODELS:
        f = MATRIX / f"{tag}_{m}_complex.csv"
        if not f.exists():
            print(f"  missing {f.name}")
            continue
        d = pd.read_csv(f)
        d = d[(d.metric == metric) & (d.replicate == replicate)]
        if d.empty:
            print(f"  {f.name}: no {metric} at replicate {replicate}")
            continue
        cols[m] = d.set_index("seq").value
    return pd.DataFrame(cols)


def pairwise(df):
    out = {}
    for a, b in itertools.combinations(df.columns, 2):
        s = df[[a, b]].dropna()
        out[(a, b)] = spearman(s[a], s[b]) if len(s) >= 5 else np.nan
    return out


def diagonal(tag, metric=METRIC):
    """rep0 ~ rep1 for each model, on whatever slice carries both."""
    out = {}
    for m in MODELS:
        f = MATRIX / f"{tag}_{m}_complex.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[d.metric == metric]
        p = d.pivot_table(index="seq", columns="replicate", values="value")
        if 0 in p and 1 in p:
            s = p[[0, 1]].dropna()
            out[m] = spearman(s[0], s[1]) if len(s) >= 5 else np.nan
    return out


def main():
    FIGS.mkdir(parents=True, exist_ok=True)

    # --- the ranking run, as a third "protocol" -----------------------------
    rk = pd.read_csv(RANKED)
    # Those columns are LOSSES (lower = better); rank_composite is the SCORE.
    # Flip so every table below points the same way.
    rank_df = pd.DataFrame(
        {m: -pd.to_numeric(rk[c], errors="coerce") for m, c in RANK_COL.items()}
    ).set_index(rk.seq)

    tables = {t: wide(t) for t in PROTOCOLS}
    tables["ranking"] = rank_df
    diags = {t: diagonal(t) for t in PROTOCOLS}

    rows = []
    for a, b in itertools.combinations(MODELS, 2):
        row = {"pair": f"{a}~{b}"}
        for t in list(PROTOCOLS) + ["ranking"]:
            df = tables[t]
            if a in df.columns and b in df.columns:
                s = df[[a, b]].dropna()
                row[t] = spearman(s[a], s[b]) if len(s) >= 5 else np.nan
                row[t + "_n"] = len(s)
            else:
                row[t] = np.nan
                row[t + "_n"] = 0
        rows.append(row)
    tab = pd.DataFrame(rows)

    # --- how well does each model reproduce its own ranking column? ---------
    cross = {}
    for m, c in RANK_COL.items():
        for t in PROTOCOLS:
            df = tables[t]
            if m not in df.columns:
                continue
            s = pd.concat([df[m], rank_df[m].rename("rank")], axis=1).dropna()
            cross[(m, t)] = (spearman(s[m], s["rank"]) if len(s) >= 5
                             else np.nan, len(s))

    print("\n=== pairwise agreement on the exact ranking composite ===")
    print("(n = designs both columns cover; a partial run gives a small n and "
          "a provisional rho)")
    show = tab.copy()
    for t in list(PROTOCOLS) + ["ranking"]:
        show[t] = [f"{v:+.3f} (n={int(n)})" if not np.isnan(v) else "--"
                   for v, n in zip(tab[t], tab[t + "_n"])]
    print(show[["pair"] + list(PROTOCOLS) + ["ranking"]]
          .to_string(index=False))
    print("\n=== self-agreement (rep0~rep1), the ceiling for each protocol ===")
    for t, d in diags.items():
        print(f"  {t:8s} " + "  ".join(f"{k} {v:+.3f}" for k, v in d.items()))
    print("\n=== each model vs its OWN ranking-run column ===")
    for m in RANK_COL:
        cells = []
        for t in PROTOCOLS:
            v, n = cross.get((m, t), (float("nan"), 0))
            cells.append(f"{t} {v:+.3f} (n={n})" if not np.isnan(v)
                         else f"{t} --")
        print(f"  {m:9s} " + "   ".join(cells))

    tab.to_csv(FIGS / "protocol_comparison.csv", index=False)

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), facecolor=SURFACE)

    ax = axes[0]
    sub = tab.dropna(subset=["bffull", "tofull"]).sort_values("tofull")
    y = np.arange(len(sub))
    ax.barh(y - 0.2, sub.bffull, 0.38, color=MUTED, label=PROTOCOLS["bffull"])
    ax.barh(y + 0.2, sub.tofull, 0.38, color=S1, label=PROTOCOLS["tofull"])
    for i, v in enumerate(sub.ranking):
        if not np.isnan(v):
            ax.plot([v], [i], marker="D", ms=6, color=S2, zorder=5,
                    label="ranking run" if i == sub.ranking.first_valid_index()
                    else None)
    ax.set_yticks(y)
    ax.set_yticklabels(sub.pair, fontsize=8.5)
    ax.set_xlabel("Spearman ρ, ranking composite", fontsize=9)
    ax.set_title("A  Model agreement, by protocol", loc="left",
                 fontsize=10.5, color=INK, fontweight="bold")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    style(ax, grid="x")

    ax = axes[1]
    ms = [m for m in MODELS if m in diags["tofull"] or m in diags["bffull"]]
    x = np.arange(len(ms))
    ax.bar(x - 0.2, [diags["bffull"].get(m, np.nan) for m in ms], 0.38,
           color=MUTED, label=PROTOCOLS["bffull"])
    ax.bar(x + 0.2, [diags["tofull"].get(m, np.nan) for m in ms], 0.38,
           color=S1, label=PROTOCOLS["tofull"])
    ax.set_xticks(x)
    ax.set_xticklabels(ms, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ρ, replicate 0 vs 1", fontsize=9)
    ax.set_title("B  Self-agreement (the ceiling)", loc="left",
                 fontsize=10.5, color=INK, fontweight="bold")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    style(ax, grid="y")

    ax = axes[2]
    ms3 = list(RANK_COL)
    x = np.arange(len(ms3))
    ax.bar(x - 0.2, [cross.get((m, "bffull"), (np.nan, 0))[0] for m in ms3],
           0.38, color=MUTED, label=PROTOCOLS["bffull"])
    ax.bar(x + 0.2, [cross.get((m, "tofull"), (np.nan, 0))[0] for m in ms3],
           0.38, color=S1, label=PROTOCOLS["tofull"])
    ax.axhline(1.0, color=CRITICAL, lw=1.0, ls="--")
    ax.text(len(ms3) - 0.5, 1.005, "same measurement", ha="right", va="bottom",
            fontsize=7.5, color=CRITICAL)
    ax.set_xticks(x)
    ax.set_xticklabels(ms3, fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("ρ vs the model's own ranking column", fontsize=9)
    ax.set_title("C  Reproducing the ranking run", loc="left",
                 fontsize=10.5, color=INK, fontweight="bold")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    style(ax, grid="y")

    fig.subplots_adjust(left=0.10, right=0.975, top=0.88, bottom=0.16,
                        wspace=0.42)
    out = FIGS / "protocol_comparison.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"\nwrote {out}")
    print(f"wrote {FIGS / 'protocol_comparison.csv'}")


if __name__ == "__main__":
    main()
