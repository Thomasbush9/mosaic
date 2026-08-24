"""Complex-mode self-consistency: does the binding POSE survive a redesign round trip?

The monomer version (hallucinate/self_consistency.py) answers "does the binder
fold hold up", and on the DIO3 hallucination set that turned out to be a costly
restatement of reference-fold pLDDT (rho = -0.70). This asks the question
nothing in the pipeline currently asks, and the one RUN_REPORT.md identifies as
the real bottleneck: AF2's interface confidence is set by the binding pose, so
test whether the pose is REPRODUCIBLE rather than merely confident.

Per design: fold binder+target, let MPNN redesign the binder in the target's
context, refold the complex, and measure two things against the reference —

    fold_rmsd  binder CA vs binder CA, Kabsch-aligned on the binder alone
               ("did the binder's own fold hold")
    pose_rmsd  superpose on the TARGET, then binder CA vs binder CA
               ("did the binder land in the same place on the target")

pose_rmsd is the new information. A design can have fold_rmsd 0.5 A and
pose_rmsd 12 A: same fold, different dock. Refold ipTM is recorded alongside so
the geometric measure can be compared against the confidence measure already in
screen_ranked.csv.

Chain order matters and works in our favour: target_only_features([binder,
target]) puts the binder first, so inverse_fold(binder_length=L) redesigns
exactly the binder while conditioning on the target.
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
from mosaic.losses.structure_prediction import IPTMLoss
from mosaic.models.af2 import AlphaFold2
from mosaic.proteinmpnn.mpnn import load_mpnn_sol
from mosaic.structure_prediction import TargetChain
from mosaic.util import calculate_rmsd, kabsch, unaligned_rmsd

CA = 1
FLUSH_EVERY = 5

METRIC_COLS = ["pose_rmsd", "pose_rmsd_mean", "fold_rmsd", "fold_rmsd_mean",
               "ref_iptm", "best_iptm", "mean_iptm", "ref_binder_plddt",
               "best_binder_plddt", "best_seq", "best_seq_recovery"]

NAN_RECORD = {c: (None if c == "best_seq" else np.nan) for c in METRIC_COLS}


def _iptm(pssm, out, key):
    # IPTMLoss returns (-iptm, aux); af2.predict does the same negation.
    return float(-IPTMLoss()(pssm, out, key=key)[0])


def self_consistency_complex(af2, mpnn, binder_seq, target_seq, msa_path, key, *,
                             n, model_idx, recycling_steps, temp,
                             jacobi_iterations):
    L = len(binder_seq)
    k_fold, k_mpnn = jax.random.split(key, 2)

    features, _ = af2.target_only_features(chains=[
        TargetChain(sequence=binder_seq, use_msa=False),
        TargetChain(sequence=target_seq, use_msa=True, msa_path=msa_path),
    ])

    ref_pssm = jax.nn.one_hot(jnp.array([TOKENS.index(a) for a in binder_seq]), 20)
    ref_out = af2.model_output(
        PSSM=ref_pssm, features=features, recycling_steps=recycling_steps,
        model_idx=model_idx, key=k_fold,
    )
    ref_ca = ref_out.atom37_coords[:, CA]
    ref_binder, ref_target = ref_ca[:L], ref_ca[L:]

    seq_idx = jax.vmap(lambda ki: inverse_fold(
        mpnn, binder_length=L, output=ref_out, temp=temp,
        key=ki, jacobi_iterations=jacobi_iterations,
    ))(jax.random.split(k_mpnn, n))
    seq_strs = ["".join(TOKENS[i] for i in row) for row in np.asarray(seq_idx)]

    pose, fold, iptms, bplddt = [], [], [], []
    for s in seq_idx:
        pssm = jax.nn.one_hot(s, 20)
        out = af2.model_output(
            PSSM=pssm, features=features, recycling_steps=recycling_steps,
            model_idx=model_idx, key=k_fold,
        )
        ca = out.atom37_coords[:, CA]
        new_binder, new_target = ca[:L], ca[L:]

        # Pose: put the two complexes in a common frame using the TARGET only,
        # then ask how far the binder moved. kabsch solves
        # min_T ||vmap(T)(P) - Q||^2, so R,t maps refold -> reference.
        R, t = kabsch(new_target, ref_target)
        pose.append(float(unaligned_rmsd(new_binder @ R + t, ref_binder)))
        fold.append(float(calculate_rmsd(new_binder, ref_binder)))
        iptms.append(_iptm(pssm, out, k_fold))
        bplddt.append(float(out.plddt[:L].mean()))

    pose, fold = np.array(pose), np.array(fold)
    iptms, bplddt = np.array(iptms), np.array(bplddt)
    best = int(pose.argmin())
    recovery = np.mean([a == b for a, b in zip(seq_strs[best], binder_seq)])
    return {
        "pose_rmsd": float(pose[best]),
        "pose_rmsd_mean": float(pose.mean()),
        "fold_rmsd": float(fold[best]),
        "fold_rmsd_mean": float(fold.mean()),
        "ref_iptm": _iptm(ref_pssm, ref_out, k_fold),
        "best_iptm": float(iptms[best]),
        "mean_iptm": float(iptms.mean()),
        "ref_binder_plddt": float(ref_out.plddt[:L].mean()),
        "best_binder_plddt": float(bplddt[best]),
        "best_seq": seq_strs[best],
        "best_seq_recovery": float(recovery),
    }


def pick_pilot(d, n_top, n_random, seed):
    """Top-ranked designs plus a sample of the tail, so the pilot has contrast.

    Without the tail there is nothing to separate the good designs FROM, and
    the whole point is whether pose_rmsd discriminates.
    """
    d = d.sort_values("rank") if "rank" in d.columns else d
    top = d.head(n_top)
    rest = d.iloc[n_top:]
    tail = rest.sample(min(n_random, len(rest)), random_state=seed)
    out = pd.concat([top, tail])
    out["pilot_group"] = ["top"] * len(top) + ["tail"] * len(tail)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--designs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seq-col", default="sequence")
    p.add_argument("--target-fasta", type=Path, required=True)
    p.add_argument("--target-msa", type=Path, required=True)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-idx", type=int, default=0)
    p.add_argument("--recycling-steps", type=int, default=3)
    p.add_argument("--temp", type=float, default=0.1)
    p.add_argument("--jacobi-iterations", type=int, default=10)
    p.add_argument("--pilot-top", type=int, default=0,
                   help="if >0, run pilot_top top-ranked + pilot-random tail")
    p.add_argument("--pilot-random", type=int, default=30)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    target_seq = "".join(
        l.strip() for l in args.target_fasta.read_text().splitlines()
        if not l.startswith(">")
    )

    d = pd.read_csv(args.designs)
    if args.pilot_top:
        d = pick_pilot(d, args.pilot_top, args.pilot_random, args.seed)
    if args.limit:
        d = d.iloc[: args.limit]
    lengths = sorted({len(s) for s in d[args.seq_col]})
    print(f"{len(d)} designs, binder lengths {lengths}, target {len(target_seq)} aa",
          flush=True)
    if len(lengths) > 1:
        print("NOTE: mixed binder lengths -> one AF2 compile per length", flush=True)

    af2 = AlphaFold2()
    mpnn = load_mpnn_sol(0.05)
    key = jax.random.key(args.seed)
    print("models loaded", flush=True)

    records, failures = [], 0
    t0 = time.time()
    for i, (idx, row) in enumerate(d.iterrows()):
        try:
            records.append(self_consistency_complex(
                af2, mpnn, row[args.seq_col], target_seq, str(args.target_msa),
                key, n=args.n, model_idx=args.model_idx,
                recycling_steps=args.recycling_steps, temp=args.temp,
                jacobi_iterations=args.jacobi_iterations,
            ))
        except Exception:
            failures += 1
            import traceback
            print(f"row {idx} FAILED:", flush=True)
            traceback.print_exc()
            records.append(dict(NAN_RECORD))

        done = i + 1
        if done % FLUSH_EVERY == 0 or done == len(d):
            elapsed = time.time() - t0
            out = d.iloc[:done].copy()
            for c in METRIC_COLS:
                out[c] = [r[c] for r in records]
            out.to_csv(args.out, index=False)
            print(f"[{done}/{len(d)}] {elapsed/done:.1f}s/design  "
                  f"eta {(len(d)-done)*elapsed/done/60:.0f} min  failures={failures}",
                  flush=True)

    print(f"wrote {args.out}  ({len(d)-failures}/{len(d)} ok)", flush=True)

    ok = pd.read_csv(args.out)
    if "pilot_group" in ok.columns and ok["pose_rmsd"].notna().any():
        print("\npose_rmsd by group (the pilot's pass/fail condition):")
        print(ok.groupby("pilot_group")[["pose_rmsd", "fold_rmsd", "best_iptm"]]
                .describe().to_string())


if __name__ == "__main__":
    main()
