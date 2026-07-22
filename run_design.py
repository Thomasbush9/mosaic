#!/usr/bin/env python
"""Binder design entrypoint for batch/array use.

``singularity/design.sbatch`` expects this at ``/opt/mosaic/run_design.py``. The
examples are marimo notebooks (deliberately — JIT warmup is slow and the
intended workflow is interactive), so this is the non-interactive counterpart:
one target, one seed, B trajectories, results to disk.

Parallelism, since it is not obvious from the code: mosaic is single-device
throughout — there is no pmap, shard_map, jax.sharding or jax.distributed
anywhere in src/. Two axes compose instead:

  * across GPUs   — SLURM array, one GPU per task, tasks differing by --seed
  * within a GPU  — --batch B independent trajectories, vmapped by
                    batched_simplex_APGM (separate iterate, momentum and RNG key
                    each, sharing one loss object and one JIT compile)

B is what amortises the compile, which is minutes for these models.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", required=True,
                   help="target sequence, or path to a FASTA file")
    p.add_argument("--target-msa", default=None,
                   help="precomputed .a3m for the target (singularity/msa-search.sbatch). "
                        "Without it the backend queries api.colabfold.com on every "
                        "featurization; with it, no network call is made.")
    p.add_argument("--binder-length", type=int, default=80)
    p.add_argument("--seed", type=int, default=0,
                   help="what makes array tasks differ; optimizers are "
                        "deterministic given a key")
    p.add_argument("--batch", type=int, default=4,
                   help="B independent trajectories vmapped on this GPU")
    p.add_argument("--models", default="boltz2,af2",
                   help="comma-separated structure backends to co-optimize "
                        "(boltz2, boltz1, af2, of3, protenix). The point of "
                        "mosaic is that these compose into ONE differentiable "
                        "objective, so a design must satisfy all of them at "
                        "once rather than overfitting a single predictor.")
    p.add_argument("--esmc", default="biohub/ESMC-300M",
                   help="ESM-C checkpoint for the sequence-plausibility term, or "
                        "'none'. Note the lowercase repo id: the shared cache "
                        "holds models--biohub--ESMC-300M while losses/esmc.py's "
                        "alias asks for Biohub/ESMC-300M, and HF cache paths are "
                        "case-sensitive, so the alias would re-download 1.3 GB.")
    p.add_argument("--esmc-weight", type=float, default=0.5)
    p.add_argument("--soft-steps", type=int, default=100,
                   help="stage 1: optimize over the simplex interior")
    p.add_argument("--sharp-steps", type=int, default=25,
                   help="stage 2: drive toward one-hot from stage 1's solution")
    p.add_argument("--recycling-steps", type=int, default=1,
                   help="trunk passes per evaluation. Must be >=1: the trunk runs "
                        "inside jax.lax.scan(length=recycling_steps), so 0 means "
                        "the body never executes and the loss has no dependence on "
                        "the sequence — a silently zero gradient.")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--no-cys", action=argparse.BooleanOptionalAction, default=True,
                   help="forbid cysteine in the binder (default on). NoCys is a "
                        "reparameterization, not a penalty: the optimizer works "
                        "over 19 columns and NoCys splices a zero-probability C "
                        "back in, so cysteine is structurally impossible rather "
                        "than merely discouraged. Free cysteines are a liability "
                        "in a de novo binder. Use --no-no-cys to allow them.")
    return p.parse_args()


def read_target(spec: str) -> str:
    """Accept a raw sequence or a FASTA path."""
    path = Path(spec)
    if path.is_file():
        lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        seq = "".join(ln for ln in lines if not ln.startswith(">"))
    else:
        seq = spec.strip()
    return seq.upper()


def sanitize_target(seq: str) -> tuple[str, list[str]]:
    """Map residues outside mosaic's 20-letter alphabet onto their closest analog.

    TOKENS is "ARNDCQEGHILKMFPSTWYV" and every [N, 20] array in the codebase is
    in that order, so anything else has no column and would raise.

    U (selenocysteine) -> C is the substitution that matters here and is not
    merely a formality: DIO3 is a selenoprotein whose catalytic residue is Sec.
    Sec is the selenium analog of Cys and Sec->Cys mutants are a standard
    experimental construct that keeps the fold while losing most catalytic
    activity. For *structure* prediction that is the right approximation — but
    it does change the chemistry at the active site, so do not read catalytic
    conclusions off a model built this way.

    O (pyrrolysine) -> K on the same logic. B/Z/J/X are ambiguity codes; they are
    mapped to their more common member, or A for X, and reported.
    """
    from mosaic.common import TOKENS

    analog = {"U": "C", "O": "K", "B": "D", "Z": "E", "J": "L", "X": "A"}
    out, notes = [], []
    for i, c in enumerate(seq, start=1):
        if c in TOKENS:
            out.append(c)
        elif c in analog:
            out.append(analog[c])
            notes.append(f"{c}{i}->{analog[c]}")
        else:
            raise ValueError(f"residue {c!r} at position {i} has no analog")
    return "".join(out), notes


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_raw = read_target(args.target)
    target, subs = sanitize_target(target_raw)

    print(f"jax {jax.__version__} {jax.default_backend()} {jax.devices()}")
    print(f"target: {len(target)} residues")
    if subs:
        print(f"non-standard residues substituted: {', '.join(subs)}")
    print(f"binder: {args.binder_length}   batch: {args.batch}   seed: {args.seed}")
    print(f"target MSA: {args.target_msa or '(none - single sequence)'}")

    from mosaic.optimizers import batched_simplex_APGM
    from mosaic.structure_prediction import TargetChain
    import mosaic.losses.structure_prediction as sp
    from mosaic.losses.transformations import ClippedLoss, NoCys

    def build_model(name: str):
        if name == "boltz2":
            from mosaic.models.boltz2 import Boltz2
            return Boltz2()
        if name == "boltz1":
            from mosaic.models.boltz1 import Boltz1
            return Boltz1()
        if name == "af2":
            from mosaic.models.af2 import AlphaFold2
            return AlphaFold2(multimer=True)
        if name == "of3":
            from mosaic.models.of3 import OF3
            return OF3()
        if name == "protenix":
            from mosaic.models.protenix import ProtenixMini
            return ProtenixMini()
        raise SystemExit(f"unknown model {name!r}")

    names = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"structure backends: {', '.join(names)}")

    # The per-backend structural objective. Each backend contributes the same
    # geometric ask; what differs is the model's opinion about whether the
    # sequence achieves it. Weights are split evenly so adding a backend does
    # not silently inflate the structural term relative to the sequence term.
    per_model = 4.0 * sp.BinderTargetContact() + 1.0 * sp.WithinBinderContact()
    share = 1.0 / len(names)

    loss_term = None
    for name in names:
        t0 = time.time()
        model = build_model(name)
        print(f"  {name}: loaded in {time.time() - t0:.1f}s")

        # AF2's interface rejects MSAs outright (models/af2.py:391), so its chain
        # is always single-sequence even when an a3m is available for the others.
        af2_like = name == "af2"
        chain = TargetChain(
            sequence=target,
            use_msa=(args.target_msa is not None) and not af2_like,
            msa_path=None if af2_like else args.target_msa,
        )
        features, _writer = model.binder_features(args.binder_length, [chain])
        term = model.build_loss(
            loss=per_model, features=features, recycling_steps=args.recycling_steps
        )
        loss_term = share * term if loss_term is None else loss_term + share * term
        print(f"  {name}: features + loss built")

    # Sequence plausibility. Without a term like this the structure models will
    # happily accept sequences no natural protein would tolerate.
    if args.esmc.lower() != "none":
        from mosaic.losses.esmc import ESMCPseudoLikelihood, load_esmc

        t0 = time.time()
        esmc = load_esmc(args.esmc)
        # Clipped, per README: raw PLLs over-optimize to homopolymers. This is
        # the single most repeated footgun warning in the repo.
        loss_term = loss_term + args.esmc_weight * ClippedLoss(
            ESMCPseudoLikelihood(esmc), 2.0, 100.0
        )
        print(f"  esmc ({args.esmc}): loaded in {time.time() - t0:.1f}s, "
              f"clipped to [2, 100], weight {args.esmc_weight}")

    # NoCys is a reparameterization, not an additive penalty: it consumes an
    # [N, 19] sequence and splices a zero-probability cysteine column back in
    # before calling the wrapped loss. So it wraps the whole objective and the
    # optimizer's alphabet shrinks to 19 — cysteine becomes impossible rather
    # than merely expensive. The decode step below has to undo this, which is
    # what NoCys's own docstring warns about.
    n_tokens = 20
    if args.no_cys:
        loss_term = NoCys(loss_term)
        n_tokens = 19
        print("  no-cys: optimizing over 19 tokens (cysteine excluded by construction)")

    key = jax.random.key(args.seed)
    key, init_key = jax.random.split(key)
    # Start near the uniform point of the simplex with a little noise, so the B
    # trajectories diverge instead of collapsing to the same path.
    x = jax.random.uniform(
        init_key, (args.batch, args.binder_length, n_tokens), minval=0.0, maxval=1.0
    )
    x = x / x.sum(-1, keepdims=True)

    # Both stages return (final_iterate, best_iterate). Continue from the final
    # iterate — best_x is a snapshot, not a state the optimizer can resume from
    # coherently (its momentum history belongs to a different point).
    t0 = time.time()
    x, _ = batched_simplex_APGM(
        loss_function=loss_term, x=x, n_steps=args.soft_steps,
        stepsize=0.1, momentum=0.9, key=key,
    )
    print(f"soft stage ({args.soft_steps} steps): {time.time() - t0:.1f}s")

    # Sharpening: a smaller step with the iterate raised toward one-hot. Without
    # this the result is a soft sequence that does not correspond to any actual
    # protein.
    t0 = time.time()
    key, sharp_key = jax.random.split(key)
    x, best_x = batched_simplex_APGM(
        loss_function=loss_term, x=x, n_steps=args.sharp_steps,
        stepsize=0.025, momentum=0.5, key=sharp_key, scale=2.0,
    )
    print(f"sharp stage ({args.sharp_steps} steps): {time.time() - t0:.1f}s")

    # Report best_x, not the final iterate: APGM with momentum does not descend
    # monotonically, so the last step is not reliably the best one. Note the
    # optimizer's own caveat (optimizers.py:312) that best_x is tracked against a
    # loss evaluated at the extrapolated point v rather than at x, so it is a
    # close approximation of the best iterate rather than an exact one — the
    # re-evaluation below reports each design's actual loss regardless.
    x = best_x

    from mosaic.common import TOKENS
    from mosaic.optimizers import batched_eval

    keys = jax.random.split(jax.random.key(args.seed + 1), args.batch)
    values, _aux, _g = batched_eval(loss_term, x, keys)
    values = np.asarray(values)

    # Undo the NoCys reparameterization before decoding: the optimizer's 19
    # columns are not TOKENS, they are TOKENS with C removed. Reading them off
    # directly would silently shift every residue at or after index 4 (C).
    #
    # NoCys.sequence is 2D-only — it slices seq[:, :cys_idx], hardcoding axis 1 —
    # so it must be applied per trajectory, not to the [B, N, 19] batch. Handing
    # it the batched array slices the residue axis instead of the token axis and
    # fails, or worse, would quietly mangle the shape.
    def decode(xb):
        full = NoCys.sequence(xb) if args.no_cys else xb
        return "".join(TOKENS[i] for i in np.asarray(full).argmax(-1))

    results = []
    for b in range(args.batch):
        seq = decode(x[b])
        results.append({"trajectory": b, "seed": args.seed,
                        "loss": float(values[b]), "sequence": seq})
        print(f"[{b}] loss={values[b]:.4f}  {seq}")

    results.sort(key=lambda r: r["loss"])
    payload = {
        "target_length": len(target),
        "target_substitutions": subs,
        "binder_length": args.binder_length,
        "models": names, "esmc": args.esmc,
        "seed": args.seed,
        "batch": args.batch,
        "soft_steps": args.soft_steps,
        "sharp_steps": args.sharp_steps,
        "target_msa": args.target_msa,
        "results": results,
    }
    (out_dir / f"designs_seed{args.seed}.json").write_text(json.dumps(payload, indent=2))
    with (out_dir / f"designs_seed{args.seed}.fasta").open("w") as f:
        for r in results:
            f.write(f">seed{args.seed}_traj{r['trajectory']}_loss{r['loss']:.4f}\n"
                    f"{r['sequence']}\n")

    print(f"\nbest loss {results[0]['loss']:.4f} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
