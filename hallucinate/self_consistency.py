"""Structural self-consistency (scRMSD) for a set of designed sequences.

For each design: fold it with AF2, redesign the sequence n times with
ProteinMPNN from that structure, refold each sample, and report the CA RMSD
back to the reference fold. This is the standard scRMSD filter -- a
structure-space check that the fold survives a design/refold round trip.

Deliberately NOT an InverseFoldingSequenceRecovery-style score: that term
carried weight 10.0 in the design objective (hallucinate/hallucinate.py), so
re-scoring candidates with it measures what the optimizer already maximised.

Run under the container on a GPU node; see singularity/self-consistency.sbatch.
Results are appended to the output CSV every FLUSH_EVERY designs so a walltime
kill still leaves usable partial data.
"""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from mosaic.common import TOKENS
from mosaic.losses.protein_mpnn import inverse_fold
from mosaic.models.af2 import AlphaFold2
from mosaic.proteinmpnn.mpnn import load_mpnn_sol
from mosaic.structure_prediction import TargetChain
from mosaic.util import calculate_rmsd

CA = 1  # atom37 index; ATOM37_NAMES = ("N", "CA", "C", "CB", "O", ...)
FLUSH_EVERY = 10

METRIC_COLS = ["sc_rmsd", "sc_rmsd_mean", "sc_plddt", "frac_under_2A",
               "ref_plddt", "best_seq", "best_seq_recovery"]


def self_consistency(af2, mpnn, seq_str, key, *, n, model_idx, recycling_steps,
                     temp, jacobi_iterations):
    k_fold, k_mpnn = jax.random.split(key, 2)

    features, _ = af2.target_only_features(
        chains=[TargetChain(sequence=seq_str, use_msa=False)],
    )

    # One-hot rather than PSSM=None so the reference fold and the refolds are
    # the same pytree structure: eqx.filter_jit then traces AF2 once, not twice.
    ref_pssm = jax.nn.one_hot(jnp.array([TOKENS.index(a) for a in seq_str]), 20)
    ref_out = af2.model_output(
        PSSM=ref_pssm, features=features, recycling_steps=recycling_steps,
        model_idx=model_idx, key=k_fold,
    )
    ref_ca = ref_out.atom37_coords[:, CA]

    seq_idx = jax.vmap(lambda ki: inverse_fold(
        mpnn, binder_length=len(seq_str), output=ref_out, temp=temp,
        key=ki, jacobi_iterations=jacobi_iterations,
    ))(jax.random.split(k_mpnn, n))
    seq_strs = ["".join(TOKENS[i] for i in row) for row in np.asarray(seq_idx)]

    # The AF2 forward is deterministic given pinned model_idx: global_config
    # .deterministic is True (models/af2.py), so dropout_rate is 0 and the key
    # only ever selected model_idx. Reusing k_fold keeps sequence the only
    # difference between reference and refold.
    rmsds, plddts = [], []
    for s in seq_idx:
        out = af2.model_output(
            PSSM=jax.nn.one_hot(s, 20), features=features,
            recycling_steps=recycling_steps, model_idx=model_idx, key=k_fold,
        )
        rmsds.append(float(calculate_rmsd(out.atom37_coords[:, CA], ref_ca)))
        plddts.append(float(out.plddt.mean()))

    rmsds, plddts = np.array(rmsds), np.array(plddts)
    best = int(rmsds.argmin())
    recovery = np.mean([a == b for a, b in zip(seq_strs[best], seq_str)])
    return {
        "sc_rmsd": float(rmsds[best]),
        "sc_rmsd_mean": float(rmsds.mean()),
        "sc_plddt": plddts[best],
        "frac_under_2A": float((rmsds < 2.0).mean()),
        "ref_plddt": float(ref_out.plddt.mean()),
        "best_seq": seq_strs[best],
        "best_seq_recovery": float(recovery),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--designs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-idx", type=int, default=0)
    p.add_argument("--recycling-steps", type=int, default=3)
    p.add_argument("--temp", type=float, default=0.1)
    p.add_argument("--jacobi-iterations", type=int, default=10)
    p.add_argument("--limit", type=int, default=None,
                   help="only process the first N designs (smoke test)")
    args = p.parse_args()

    designs = pd.read_csv(args.designs, index_col=0)
    designs = designs.loc[:, ~designs.columns.str.startswith("Unnamed")]
    if args.limit:
        designs = designs.iloc[: args.limit]
    print(f"{len(designs)} designs from {args.designs}", flush=True)

    af2 = AlphaFold2()
    mpnn = load_mpnn_sol(0.05)
    key = jax.random.key(args.seed)
    print("models loaded", flush=True)

    records, failures = [], 0
    t0 = time.time()
    for i, (idx, row) in enumerate(designs.iterrows()):
        try:
            # Common random numbers across designs: differences in scRMSD then
            # reflect the designs rather than the noise draw.
            records.append(self_consistency(
                af2, mpnn, row["seq"], key,
                n=args.n, model_idx=args.model_idx,
                recycling_steps=args.recycling_steps, temp=args.temp,
                jacobi_iterations=args.jacobi_iterations,
            ))
        except Exception as e:
            failures += 1
            print(f"row {idx} FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            records.append({c: (np.nan if c != "best_seq" else None)
                            for c in METRIC_COLS})

        done = i + 1
        if done % FLUSH_EVERY == 0 or done == len(designs):
            elapsed = time.time() - t0
            out = designs.iloc[:done].copy()
            for c in METRIC_COLS:
                out[c] = [r[c] for r in records]
            out.to_csv(args.out, index=False)
            print(f"[{done}/{len(designs)}] {elapsed/done:.1f}s/design  "
                  f"eta {(len(designs)-done)*elapsed/done/60:.0f} min  "
                  f"failures={failures}", flush=True)

    print(f"wrote {args.out}  ({len(designs)-failures}/{len(designs)} ok)",
          flush=True)


if __name__ == "__main__":
    main()
