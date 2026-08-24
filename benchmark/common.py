"""Shared palette and statistics for the benchmark figures.

Every plotter here previously carried its own copy of these seven colours, its
own `style()`, and its own `spearman()` / `auc()` -- four near-identical
definitions of each. They are the same functions, so they live in one place; a
palette that drifts between panels of the same report is a bug you only notice
after publishing.

Colours follow the dataviz placeholder palette: a warm off-white surface, near
black ink, and two categorical slots. Reds/blues in the matrices come from
matplotlib's RdBu_r, which is diverging and correct for a signed correlation.
"""

import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"      # page / axes background
INK = "#0b0b0b"          # primary text
INK_2 = "#52514e"        # secondary text, tick labels
S1 = "#2a78d6"           # categorical slot 1
S2 = "#eb6834"           # categorical slot 2
MUTED = "#9a9992"        # controls: deliberately recessive, not a finding
CRITICAL = "#d03b3b"     # thresholds, chance lines
GRID = "#e3e2de"


def style(ax, *, grid="both"):
    """House style: no top/right spines, muted grid behind the data.

    `grid` is "both", "x", "y", or None.
    """
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(True, axis=grid, color=GRID, linewidth=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=3)


def spearman(a, b):
    """Rank correlation. NaN rather than a crash on a constant input."""
    if len(a) < 5:
        return np.nan
    ra, rb = pd.Series(a).rank(), pd.Series(b).rank()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def auc(pos, neg):
    """Mann-Whitney U / (n_pos * n_neg). 0.5 = no discrimination.

    Ties get averaged ranks, which matters here: several scores saturate near
    1.0 on easy designs and a naive implementation would reward that.
    """
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    vals = [v for v, _ in allv]
    r, i = [0.0] * len(vals), 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        for k in range(i, j + 1):
            r[k] = (i + j) / 2 + 1
        i = j + 1
    rpos = sum(r[k] for k, (_, lab) in enumerate(allv) if lab == 1)
    return (rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def partial_spearman(x, y, z):
    """Rank correlation of x and y with z linearly removed from both."""
    rx, ry, rz = (pd.Series(v).rank().values for v in (x, y, z))
    rx = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ry = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(np.corrcoef(rx, ry)[0, 1])


# Sign convention: +1 = higher means a better/more confident prediction.
# PAE and pTMEnergy are errors, so they run the other way. Needed wherever
# metrics are compared to each other or to a label -- without it a perfect
# agreement between an error and a confidence shows up as rho = -0.9, which
# reads as disagreement.
#
# `rg` is not a quality metric at all (a radius is neither good nor bad), so
# its +1 is arbitrary and only fixes a direction for the comparison.
ORIENT = {
    "ipsae_min": +1, "bt_ipsae": +1, "iptm": +1, "bt_iptm": +1,
    "binder_ptm": +1, "binder_plddt": +1, "complex_plddt": +1, "iplddt": +1,
    "bt_pae": -1, "tb_pae": -1, "ptm_energy": -1,
    "plddt": +1, "ptm": +1, "pae": -1, "rg": +1,
}
