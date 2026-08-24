"""How much of an AF2 ipTM depends on WHICH of the five multimer models ran?

screen_proposals.py calls predict() without model_idx, so AF2 draws one of the
five parameter sets at random (models/af2.py:501). If ipTM varies strongly
across them, a headline number from a single unpinned draw is not reproducible
and cannot be compared against a run that pinned a different model.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from mosaic.common import TOKENS
from mosaic.losses.structure_prediction import IPTMLoss
from mosaic.models.af2 import AlphaFold2
from mosaic.structure_prediction import TargetChain


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--designs", type=Path, required=True)
    p.add_argument("--target-fasta", type=Path, required=True)
    p.add_argument("--target-msa", type=Path, default=None)
    p.add_argument("--no-target-msa", action="store_true",
                   help="fold the target single-sequence, as every screen.json "
                        "in this project did (target_msa: None)")
    p.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    p.add_argument("--recycling", type=int, nargs="+", default=[3, 4])
    args = p.parse_args()
    if not args.no_target_msa and args.target_msa is None:
        p.error("pass --target-msa or --no-target-msa")

    target = "".join(l.strip() for l in args.target_fasta.read_text().splitlines()
                     if not l.startswith(">"))
    d = pd.read_csv(args.designs).set_index("rank")

    af2 = AlphaFold2()
    key = jax.random.key(0)
    print(f"target {len(target)} aa; AF2 multimer models 0-4\n", flush=True)

    rows = []
    for rank in args.ranks:
        seq = d.loc[rank, "sequence"]
        reported = float(d.loc[rank, "iptm"])
        target_chain = (
            TargetChain(sequence=target, use_msa=False) if args.no_target_msa
            else TargetChain(sequence=target, use_msa=True,
                             msa_path=str(args.target_msa))
        )
        feats, _ = af2.target_only_features(chains=[
            TargetChain(sequence=seq, use_msa=False), target_chain,
        ])
        pssm = jax.nn.one_hot(jnp.array([TOKENS.index(a) for a in seq]), 20)
        for rec in args.recycling:
            vals = []
            for m in range(5):
                out = af2.model_output(PSSM=pssm, features=feats,
                                       recycling_steps=rec, model_idx=m, key=key)
                vals.append(float(-IPTMLoss()(pssm, out, key=key)[0]))
            rows.append({"rank": rank, "recycling": rec, "reported": reported,
                         **{f"model_{m}": round(v, 3) for m, v in enumerate(vals)},
                         "min": round(min(vals), 3), "max": round(max(vals), 3),
                         "spread": round(max(vals) - min(vals), 3)})
            print(f"rank {rank:3d}  recycling {rec}  reported {reported:.3f}  "
                  f"models {[round(v,3) for v in vals]}  spread {max(vals)-min(vals):.3f}",
                  flush=True)

    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False))
    print(f"\nmedian spread across the 5 models: {df['spread'].median():.3f}")


if __name__ == "__main__":
    main()
