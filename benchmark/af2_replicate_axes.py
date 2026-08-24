"""AF2's two sources of run-to-run variation, measured separately.

The agreement matrices used model_idx as AF2's replicate axis, which is NOT
comparable to what the other backends got. AF2 is fully deterministic
(models/af2.py:114-118: deterministic=True, eval_dropout=False,
resample_msa_in_recycling=False) -- same input and same model_idx returns
bit-identical output. So its "diagonal" measured disagreement between five
SEPARATELY TRAINED networks, while every diffusion backend's diagonal measured
resampling one network. Ensemble spread is the larger quantity, so AF2 was
handicapped.

This measures both axes on the same designs:

    ensemble    vary model_idx 0..4, msa_seed fixed   (5 trained networks)
    sampling    vary msa_seed 0..4, model_idx fixed   (1 network, resampled MSA)

The sampling axis is the like-for-like comparison against the diffusion models.

msa_seed only bites when the MSA is deeper than max_msa_clusters (512), because
that is what triggers cluster subsampling -- so this must run on a deep-MSA
target. DIO3 has 3032 sequences; Nipah has 64 and would show nothing.
"""

import argparse
from itertools import combinations
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import mosaic.losses.structure_prediction as sp
from mosaic.common import TOKENS
from mosaic.models.af2 import AlphaFold2
from mosaic.structure_prediction import TargetChain

METRICS = {"ipsae_min": sp.IPSAE_min(), "iptm": sp.IPTMLoss(),
           "binder_plddt": sp.PLDDTLoss()}
FIELD = {"ipsae_min": "ipsae_min", "iptm": "iptm", "binder_plddt": "plddt"}


def spearman(a, b):
    ra, rb = pd.Series(a).rank(), pd.Series(b).rank()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--designs", type=Path, required=True)
    p.add_argument("--seq-col", default="seq")
    p.add_argument("--target-fasta", type=Path, required=True)
    p.add_argument("--target-msa", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--recycling", type=int, default=3)
    args = p.parse_args()

    seqs = pd.read_csv(args.designs)[args.seq_col].astype(str).tolist()[: args.n]
    L = len(seqs[0])
    assert all(len(s) == L for s in seqs), "expects a uniform-length set"
    target = "".join(l.strip() for l in args.target_fasta.read_text().splitlines()
                     if not l.startswith(">"))
    chains = [TargetChain(sequence=target, use_msa=True,
                          msa_path=str(args.target_msa))]

    model = AlphaFold2()
    print(f"max_msa_clusters={model.max_msa_clusters}  "
          f"msa depth={sum(1 for l in open(args.target_msa) if l.startswith('>'))}",
          flush=True)
    key = jax.random.key(0)
    pssms = [jax.nn.one_hot(jnp.array([TOKENS.index(a) for a in s]), 20)
             for s in seqs]
    rows = []

    def score(features, model_idx, tag, rep):
        for i, pssm in enumerate(pssms):
            out = model.model_output(PSSM=pssm, features=features,
                                     recycling_steps=args.recycling,
                                     model_idx=model_idx, key=key)
            for name, term in METRICS.items():
                rows.append({"axis": tag, "rep": rep, "design_idx": i,
                             "metric": name,
                             "value": float(term(pssm, out, key=key)[1][FIELD[name]])})

    # --- ensemble axis: five trained networks, one featurization ------------
    feats0, _ = model.binder_features(binder_length=L, chains=chains)
    for m in range(args.reps):
        score(feats0, m % 5, "ensemble", m)
        print(f"  ensemble model_idx={m % 5} done", flush=True)

    # --- sampling axis: one network, MSA clusters resampled -----------------
    # msa_seed is a Module field consumed at featurization (af2.py:439 ->
    # make_af_features(seed=...)), so it is swapped on the module and the pack
    # rebuilt. Params are untouched, so no recompile.
    for s in range(args.reps):
        m2 = eqx.tree_at(lambda m: m.msa_seed, model, s)
        assert m2.msa_seed == s
        feats, _ = m2.binder_features(binder_length=L, chains=chains)
        score(feats, 0, "sampling", s)
        print(f"  sampling msa_seed={s} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print(f"\n{'metric':14s} {'axis':10s} {'median rho':>11} {'range':>16}")
    for name in METRICS:
        for axis in ("ensemble", "sampling"):
            w = df[(df.metric == name) & (df.axis == axis)].pivot_table(
                index="design_idx", columns="rep", values="value")
            ps = [spearman(w[a].values, w[b].values)
                  for a, b in combinations(w.columns, 2)]
            ps = [x for x in ps if not np.isnan(x)]
            if ps:
                print(f"{name:14s} {axis:10s} {np.median(ps):11.3f} "
                      f"{min(ps):7.2f}..{max(ps):.2f}")


if __name__ == "__main__":
    main()
