"""Extract per-target labelled binder sets from the Proteinbase export.

Proteinbase stores one row per designed protein, with every readout buried in an
`evaluations` JSON blob and no target sequence anywhere -- the target is a NAME
("egfr"), so the construct that was actually assayed has to come from elsewhere.
This script produces the binder side only: sequence + label, per target.

Two labels are emitted, because the folds are the expensive part and both come
free from the same run:

    binder    bool, from `binding` calls aggregated over replicates (>50% = yes)
    strength  ordinal, from `binding_strength` -- a graded readout supports a
              rank correlation, which is more sensitive than a binary AUC

Replicate-level calls are aggregated to one label per (protein, target); a
protein assayed against two targets yields two rows in two files.
"""

import argparse
import collections
import csv
import json
from pathlib import Path

csv.field_size_limit(10 ** 9)

# Ascending. Anything outside this vocabulary is reported, never silently
# mapped -- a readout we cannot order is a readout we must not rank.
STRENGTH_ORDER = ["none", "weak", "medium", "moderate", "strong", "very strong"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--min-pairs", type=int, default=50)
    p.add_argument("--max-length-groups", type=int, default=20,
                   help="report coverage at this cap; the runner applies it")
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    idcol = next(iter(rows[0].keys()))
    per_target = collections.defaultdict(list)
    vocab = collections.Counter()

    for r in rows:
        try:
            ev = json.loads(r["evaluations"] or "[]")
        except Exception:
            continue
        calls = collections.defaultdict(list)
        strength = {}
        for e in ev:
            tgt, m, v = e.get("target"), e.get("metric"), e.get("value")
            if e.get("type") != "experimental" or not tgt:
                continue
            if m == "binding" and isinstance(v, bool):
                calls[tgt].append(v)
            elif m == "binding_strength" and isinstance(v, str):
                vocab[v.strip().lower()] += 1
                strength[tgt] = v.strip().lower()
        for tgt, cs in calls.items():
            s = strength.get(tgt)
            per_target[tgt].append({
                "id": r[idcol],
                "seq": r["sequence"],
                "binder": int(sum(cs) / len(cs) > 0.5),
                "n_replicates": len(cs),
                "strength": s or "",
                "strength_rank": (STRENGTH_ORDER.index(s)
                                  if s in STRENGTH_ORDER else ""),
                "design_method": r.get("designMethod", ""),
            })

    print("binding_strength vocabulary:")
    for v, c in vocab.most_common():
        mark = "" if v in STRENGTH_ORDER else "   <-- UNRANKED, not in STRENGTH_ORDER"
        print(f"  {v:16s} n={c:6d}{mark}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'target':26s} {'pairs':>6} {'binders':>8} {'%':>6} "
          f"{'lengths':>8} {'@cap':>6} {'bind@cap':>9}")
    kept = []
    for tgt, recs in sorted(per_target.items(), key=lambda kv: -len(kv[1])):
        nb = sum(r["binder"] for r in recs)
        if len(recs) < args.min_pairs or nb == 0 or nb == len(recs):
            continue
        freq = collections.Counter(len(r["seq"]) for r in recs)
        keep = {L for L, _ in freq.most_common(args.max_length_groups)}
        at_cap = [r for r in recs if len(r["seq"]) in keep]
        nb_cap = sum(r["binder"] for r in at_cap)
        if nb_cap == 0 or nb_cap == len(at_cap):
            continue
        out = args.outdir / f"{tgt}.csv"
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
            w.writeheader()
            w.writerows(recs)
        kept.append(tgt)
        print(f"{tgt:26s} {len(recs):6d} {nb:8d} {nb / len(recs):5.1%} "
              f"{len(freq):8d} {len(at_cap):6d} {nb_cap:9d}")

    print(f"\nwrote {len(kept)} target files to {args.outdir}")
    print("NOTE: target CONSTRUCTS are not in this export and must be supplied "
          "separately -- folding against the wrong domain fails silently.")


if __name__ == "__main__":
    main()
