"""Concatenate the per-shard tables model_matrix.py --num-shards writes.

design_idx is assigned before sharding, so the shards partition the design set
with no overlap and a plain concat reproduces the single-job table exactly.

    python benchmark/merge_shards.py <matrix_dir> <tag> <model> [condition]
"""

import sys
from pathlib import Path

import pandas as pd


def main():
    d = Path(sys.argv[1])
    tag, model = sys.argv[2], sys.argv[3]
    cond = sys.argv[4] if len(sys.argv) > 4 else "complex"
    parts = sorted(d.glob(f"{tag}_{model}_{cond}_shard*.csv"))
    if not parts:
        raise SystemExit(f"no {tag}_{model}_{cond}_shard*.csv in {d}")

    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    dup = df.duplicated(["design_idx", "replicate", "metric"]).sum()
    if dup:
        # Overlapping shards would double-weight designs in every correlation
        # downstream, silently. Refuse rather than average them away.
        raise SystemExit(f"{dup} duplicate (design_idx, replicate, metric) rows "
                         f"-- shards overlap; check --shard/--num-shards")
    df = df.sort_values(["replicate", "design_idx", "metric"])
    out = d / f"{tag}_{model}_{cond}.csv"
    df.to_csv(out, index=False)
    n0 = df[df.replicate == 0].design_idx.nunique()
    print(f"merged {len(parts)} shards -> {out}")
    print(f"  {len(df)} rows, {n0} designs at replicate 0, "
          f"{df.replicate.nunique()} replicates, {df.metric.nunique()} metrics")


if __name__ == "__main__":
    main()
