"""Zero-GPU baseline: how well do Proteinbase's PRECOMPUTED scores separate
experimentally validated binders from non-binders?

Run this before spending any GPU time. It establishes (a) that the evaluation
harness reproduces known results, and (b) a per-metric accuracy floor that any
new model has to beat to be worth running.

Reported per metric:
  AUC          Mann-Whitney rank-sum, pooled and per target
  prec@k       precision in the top k by that score -- what a campaign
               actually does (pick a plate), which AUC does not capture

Split by TARGET, never pooled-and-shuffled: per-target precision is known to
swing from 0.1 to 1.0, so a pooled number flatters every metric.
"""

import argparse
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

from common import (SURFACE, INK, INK_2, S1, S2, MUTED, CRITICAL, GRID,
                    style, spearman, auc)

csv.field_size_limit(10 ** 9)

# Sign convention: +1 means HIGHER is more likely to bind, -1 the reverse.
ORIENT = {
    "boltz2_iptm": +1, "boltz2_ipsae": +1, "boltz2_pdockq": +1,
    "boltz2_pdockq2": +1, "boltz2_ptm": +1, "boltz2_plddt": +1,
    "boltz2_complex_iplddt": +1, "boltz2_lis": +1,
    "boltz2_complex_pde": -1,          # an error -> lower is better
    "esmfold_plddt": +1,
    "proteinmpnn_score": -1,           # a negative log-likelihood in Adaptyv's export
    "redesigned_proteinmpnn_score": -1,
    "proteinmpnn_seq_recovery": +1,
    "interface_residues": +1,
    "isoelectric_point": +1, "molecular_weight": +1,   # controls: expect ~0.5
}

def precision_at_k(scored, k):
    """scored: list of (score, is_binder), higher score = predicted binder."""
    if len(scored) < k:
        return None
    top = sorted(scored, key=lambda t: -t[0])[:k]
    return sum(lab for _, lab in top) / k


def load(path):
    """-> per (protein, target): label + that protein's computational metrics."""
    rows = list(csv.DictReader(open(path)))
    idcol = rows[0].keys().__iter__().__next__()   # BOM-mangled 'id'
    pairs = {}
    for r in rows:
        pid = r[idcol]
        try:
            ev = json.loads(r["evaluations"] or "[]")
        except Exception:
            continue
        calls = defaultdict(list)
        comp = {}
        for e in ev:
            m, tgt, val = e.get("metric"), e.get("target"), e.get("value")
            if e.get("type") == "experimental" and m == "binding" and \
                    isinstance(val, bool) and tgt:
                calls[tgt].append(val)
            elif e.get("type") == "computational" and m in ORIENT:
                try:
                    comp[m] = float(val)
                except (TypeError, ValueError):
                    pass
        for tgt, cs in calls.items():
            # Replicate-level calls aggregated to one label per protein x target.
            frac = sum(cs) / len(cs)
            pairs[(pid, tgt)] = {
                "binder": frac > 0.5, "frac": frac, "n_rep": len(cs),
                "method": r.get("designMethod", ""), **comp,
            }
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--min-per-target", type=int, default=25)
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 24, 96])
    args = ap.parse_args()

    pairs = load(args.csv)
    by_t = defaultdict(list)
    for (pid, tgt), d in pairs.items():
        by_t[tgt].append(d)

    print(f"{len(pairs)} protein x target labelled pairs; "
          f"{sum(d['binder'] for d in pairs.values())} binders\n")

    targets = [t for t, ds in sorted(by_t.items())
               if len(ds) >= args.min_per_target
               and 0 < sum(d["binder"] for d in ds) < len(ds)]
    print(f"targets with >={args.min_per_target} pairs and both classes: "
          f"{len(targets)}")
    for t in targets:
        ds = by_t[t]
        print(f"   {t:28s} n={len(ds):4d}  binders={sum(d['binder'] for d in ds):4d} "
              f"({sum(d['binder'] for d in ds)/len(ds):5.1%})")

    print("\n" + "=" * 78)
    print("AUC per metric  (per target, then median across targets)")
    print("=" * 78)
    header = f"{'metric':30s} {'n_tgt':>5} {'median':>7} {'min':>6} {'max':>6}  " + \
             "  ".join(f"p@{k}" for k in args.ks)
    print(header)

    results = []
    for m, sign in ORIENT.items():
        aucs, precs = [], {k: [] for k in args.ks}
        for t in targets:
            ds = [d for d in by_t[t] if m in d]
            if len(ds) < args.min_per_target:
                continue
            pos = [sign * d[m] for d in ds if d["binder"]]
            neg = [sign * d[m] for d in ds if not d["binder"]]
            a = auc(pos, neg)
            if a is None:
                continue
            aucs.append(a)
            scored = [(sign * d[m], d["binder"]) for d in ds]
            for k in args.ks:
                p = precision_at_k(scored, k)
                if p is not None:
                    precs[k].append(p)
        if not aucs:
            continue
        row = (m, len(aucs), st.median(aucs), min(aucs), max(aucs),
               {k: (st.median(v) if v else float("nan")) for k, v in precs.items()})
        results.append(row)

    for m, n, med, lo, hi, pk in sorted(results, key=lambda r: -r[2]):
        pks = "  ".join(f"{pk[k]:.2f}" if not math.isnan(pk[k]) else "  - "
                        for k in args.ks)
        print(f"{m:30s} {n:5d} {med:7.3f} {lo:6.3f} {hi:6.3f}  {pks}")

    print("\nbase rate (what p@k must beat) per target:")
    for t in targets:
        ds = by_t[t]
        print(f"   {t:28s} {sum(d['binder'] for d in ds)/len(ds):5.1%}")


if __name__ == "__main__":
    main()
