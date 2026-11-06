"""Summarise the matched-protocol runs: accuracy, agreement, and top-k agreement.

Three tables, written as CSV next to the figures so the report can be rebuilt
without a GPU:

  accuracy.csv      AUC per model x metric on the labelled set, ORIENT applied
  agreement.csv     model x model Spearman on the design set (global)
  topk.csv          the same restricted to each model's own top-k, plus the
                    self-agreement control

WHY TOP-K. A global Spearman over a set spanning bad -> good is dominated by
both models agreeing that the bad designs are bad. That is the easy half of the
problem and not the half anyone makes decisions on. Restricted to the top of the
ranking -- the only region a campaign actually draws from -- agreement is far
lower. Restriction also mechanically attenuates a correlation, so the honest
comparison needs a ceiling that pays the same penalty: replicate-vs-replicate
agreement for the SAME model on the SAME restricted subset. Cross << self inside
one subset is real disagreement; both falling together is just range restriction.

    python benchmark/matched_summary.py --matrix <dir> --scores <dir> \
        --labelled <dir> --outdir <dir>
"""

import argparse
import itertools
from pathlib import Path

import pandas as pd

from common import ORIENT, auc, spearman

MODELS = ["af2", "boltz1", "boltz2", "of3", "protenix", "esmfold2"]
RANK_METRIC = "rank_composite"


def oriented(df):
    """Apply the codebase sign convention so every metric reads higher = better.

    Without this a PAE (an error) that separates binders perfectly reports an
    AUC of 0.32, which reads as an inverted predictor rather than a good one.
    """
    return df.assign(value=df.value * df.metric.map(ORIENT).fillna(1))


def accuracy_table(scores: Path, labelled: Path, tag: str) -> pd.DataFrame:
    rows = []
    for f in sorted(scores.glob(f"{tag}_*.csv")):
        d = oriented(pd.read_csv(f))
        model = f.stem[len(tag) + 1:].split("_")[0]
        target = f.stem[len(tag) + 1 + len(model) + 1:]
        lab = d.drop_duplicates("design_idx").set_index("design_idx").label
        y = lab.map(lambda v: str(v).strip().lower() in ("true", "1", "yes"))
        for metric, g in d.groupby("metric"):
            s = g.set_index("design_idx").value.reindex(y.index)
            ok = s.notna()
            if y[ok].nunique() < 2:
                continue
            rows.append({
                "model": model, "target": target, "metric": metric,
                "auc": auc(s[ok & y].tolist(), s[ok & ~y].tolist()),
                "n_pos": int((ok & y).sum()), "n_neg": int((ok & ~y).sum()),
            })
    return pd.DataFrame(rows)


def _wide(matrix: Path, tag: str, metric: str, replicate=0) -> pd.DataFrame:
    out = {}
    for m in MODELS:
        f = matrix / f"{tag}_{m}_complex.csv"
        if not f.exists():
            continue
        d = oriented(pd.read_csv(f))
        d = d[(d.replicate == replicate) & (d.metric == metric)]
        if len(d):
            out[m] = d.set_index("design_idx").value
    return pd.DataFrame(out).dropna()


def agreement_table(matrix: Path, tag: str, metric: str) -> pd.DataFrame:
    w = _wide(matrix, tag, metric)
    return pd.DataFrame([
        {"a": a, "b": b, "spearman": spearman(w[a], w[b]), "n": len(w)}
        for a, b in itertools.combinations(w.columns, 2)
    ])


def topk_table(matrix: Path, tag: str, metric: str, k: int) -> pd.DataFrame:
    """Cross-model agreement in the top k, against the self-agreement ceiling.

    Everything is computed on the designs that carry a replicate (--diag-n), so
    the ceiling and the cross terms are restricted identically.
    """
    reps = {}
    for m in MODELS:
        f = matrix / f"{tag}_{m}_complex.csv"
        if not f.exists():
            continue
        d = oriented(pd.read_csv(f))
        d = d[d.metric == metric]
        p = d.pivot_table(index="design_idx", columns="replicate",
                          values="value").dropna()
        if 1 in p.columns:
            reps[m] = p
    if not reps:
        return pd.DataFrame()
    common = sorted(set.intersection(*[set(p.index) for p in reps.values()]))

    rows = []
    for m, p in reps.items():
        top = p.loc[common].nlargest(k, 0)
        rows.append({"model": m, "kind": "self", "other": m,
                     "spearman": spearman(top[0], top[1]), "k": k,
                     "n_pool": len(common)})
        for o, po in reps.items():
            if o == m:
                continue
            rows.append({"model": m, "kind": "cross", "other": o,
                         "spearman": spearman(top[0], po.loc[top.index, 0]),
                         "k": k, "n_pool": len(common)})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--labelled", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--matrix-tag", default="mfull")
    p.add_argument("--scores-tag", default="macc")
    p.add_argument("--topk", type=int, default=25)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    acc = accuracy_table(args.scores, args.labelled, args.scores_tag)
    acc.to_csv(args.outdir / "accuracy.csv", index=False)
    if len(acc):
        piv = acc.pivot_table(index="metric", columns="model", values="auc")
        print("=== ACCURACY: AUC (ORIENT applied, higher = better) ===")
        print(piv.round(3).to_string())
        print("\nbest / median per model:")
        for m in piv.columns:
            print(f"  {m:10s} best={piv[m].max():.3f} "
                  f"({piv[m].idxmax()})  median={piv[m].median():.3f}")
        print("  control bar (length, inverted) = 0.642")

    agr = agreement_table(args.matrix, args.matrix_tag, RANK_METRIC)
    agr.to_csv(args.outdir / "agreement.csv", index=False)
    print(f"\n=== AGREEMENT: {RANK_METRIC}, all designs ===")
    print(f"  mean off-diagonal = {agr.spearman.mean():.3f} "
          f"(range {agr.spearman.min():.3f}-{agr.spearman.max():.3f})")

    tk = topk_table(args.matrix, args.matrix_tag, RANK_METRIC, args.topk)
    tk.to_csv(args.outdir / "topk.csv", index=False)
    if len(tk):
        print(f"\n=== TOP-{args.topk}: cross-model agreement vs self ceiling ===")
        print(f"  {'model':10s}{'self':>8s}{'mean cross':>12s}{'gap':>8s}")
        for m in tk.model.unique():
            s = tk[(tk.model == m) & (tk.kind == "self")].spearman.iloc[0]
            c = tk[(tk.model == m) & (tk.kind == "cross")].spearman.mean()
            print(f"  {m:10s}{s:8.3f}{c:12.3f}{c - s:+8.3f}")
        print("  self pays the same range restriction as cross;")
        print("  cross << self is disagreement, not an artifact.")


if __name__ == "__main__":
    main()
