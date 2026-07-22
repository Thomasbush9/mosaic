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
import jax.numpy as jnp
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
    p.add_argument("--init-fasta", default=None,
                   help="seed the optimizer from these sequences instead of "
                        "random noise — e.g. generate_boltzgen.py output. "
                        "BoltzGen cannot join the differentiable objective (it "
                        "implements no build_loss), but it can propose starting "
                        "points, which is what this consumes. One sequence per "
                        "trajectory, cycled if fewer than --batch.")
    p.add_argument("--init-noise", type=float, default=0.15,
                   help="mixing weight toward uniform when seeding from "
                        "--init-fasta. 0 gives a hard one-hot start, which has "
                        "no gradient signal off the vertex; this keeps the "
                        "iterate inside the simplex so it can still move.")
    p.add_argument("--config", default=None,
                   help="JSON campaign config (see design_config.py). When given, "
                        "it supersedes the individual flags for models, losses, "
                        "optimizer and binder settings — flags cannot express "
                        "per-model or per-loss parameters, and SLURM's --export "
                        "splits on commas so they cannot be passed that way "
                        "either. The config is also the record of what ran.")
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


def config_from_args(args) -> dict:
    """Build a config dict from flags, so both paths converge on one shape."""
    from design_config import default_config

    cfg = default_config()
    cfg["target"] = {"fasta": args.target, "msa": args.target_msa,
                     "use_msa": args.target_msa is not None}
    cfg["binder"] = {"length": args.binder_length, "no_cys": args.no_cys,
                     "init_fasta": args.init_fasta, "init_noise": args.init_noise}
    names = [m.strip() for m in args.models.replace("+", ",").split(",") if m.strip()]
    share = 1.0 / max(1, len(names))
    cfg["models"] = [
        {"name": n, "weight": share,
         "params": {"recycling_steps": args.recycling_steps}}
        for n in names
    ]
    cfg["sequence_models"] = ([] if args.esmc.lower() == "none" else
                              [{"name": "esmc", "weight": args.esmc_weight,
                                "params": {"checkpoint": args.esmc,
                                           "clip_lower": 2.0, "clip_upper": 100.0}}])
    cfg["optimizer"]["soft"]["n_steps"] = args.soft_steps
    cfg["optimizer"]["sharp"]["n_steps"] = args.sharp_steps
    cfg["run"] = {"batch": args.batch, "seed": args.seed}
    return cfg


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import design_config as dc

    if args.config:
        cfg = json.loads(Path(args.config).read_text())
        # CLI overrides that make sense per array task even with a config: the
        # seed is what distinguishes tasks, and out is set by the job script.
        cfg.setdefault("run", {})["seed"] = args.seed
        print(f"config: {args.config}")
    else:
        cfg = config_from_args(args)

    problems = dc.validate(cfg)
    if problems:
        for p_ in problems:
            print(f"config error: {p_}")
        return 2

    # Save the resolved config beside the results: this is the reproducible
    # record of what actually ran, which a command line was not.
    (out_dir / f"config_seed{cfg['run']['seed']}.json").write_text(
        json.dumps(cfg, indent=2))

    tgt, bnd = cfg["target"], cfg["binder"]
    target_raw = read_target(tgt["fasta"])
    target, subs = sanitize_target(target_raw)
    msa_path = tgt.get("msa") if tgt.get("use_msa") else None

    print(f"jax {jax.__version__} {jax.default_backend()} {jax.devices()}")
    print(f"target: {len(target)} residues")
    if subs:
        print(f"non-standard residues substituted: {', '.join(subs)}")
    print(f"binder: {bnd['length']}   batch: {cfg['run']['batch']}   "
          f"seed: {cfg['run']['seed']}")
    print(f"target MSA: {msa_path or '(none - single sequence)'}")

    from mosaic.optimizers import batched_simplex_APGM
    from mosaic.structure_prediction import TargetChain
    from mosaic.losses.transformations import NoCys

    names = [m["name"] for m in cfg["models"]]
    print(f"structure backends: {', '.join(names)}")
    print("loss terms: " + ", ".join(
        f"{l['weight']}x{l['name']}" for l in cfg["losses"]))
    _mp = cfg.get("mpnn", {}).get("terms", [])
    if _mp:
        print(f"proteinmpnn ({cfg['mpnn'].get('weights', 'soluble')}): " +
              ", ".join(f"{t['weight']}x{t['name']}" for t in _mp))
    if dc.uses_confidence(cfg):
        print("  note: a confidence term is enabled — JAX can no longer prune the "
              "structure/confidence modules, so this run is substantially slower")

    # One inner objective, evaluated against each backend's output. Each model
    # contributes its own opinion of whether the sequence achieves the same
    # geometric ask; weights are per-model and set in the config.
    inner = dc.build_inner_loss(cfg["losses"], cfg.get("mpnn"))

    loss_term = None
    for spec in cfg["models"]:
        name, params = spec["name"], (spec.get("params") or {})
        t0 = time.time()
        model = dc.build_structure_model(name, params)
        print(f"  {name}: loaded in {time.time() - t0:.1f}s")

        # AF2's interface rejects MSAs outright (models/af2.py:391), so its chain
        # is single-sequence even when an a3m is available for the others.
        accepts_msa = dc.STRUCTURE_MODELS[name]["accepts_msa"]
        chain = TargetChain(
            sequence=target,
            use_msa=(msa_path is not None) and accepts_msa,
            msa_path=msa_path if accepts_msa else None,
        )
        features, _writer = model.binder_features(bnd["length"], [chain])

        kw = {"loss": inner, "features": features,
              "recycling_steps": params.get("recycling_steps", 1)}
        # Only pass what the backend actually accepts — af2 asserts
        # sampling_steps is None, and num_samples exists only on the
        # multisample builders.
        if "sampling_steps" in params and name != "af2":
            kw["sampling_steps"] = params["sampling_steps"]
        if name == "af2" and "use_dropout" in params:
            kw["use_dropout"] = params["use_dropout"]
        if params.get("num_samples", 1) > 1 and hasattr(model, "build_multisample_loss"):
            kw["num_samples"] = params["num_samples"]
            term = model.build_multisample_loss(**kw)
        else:
            term = model.build_loss(**kw)

        w = spec.get("weight", 1.0)
        loss_term = w * term if loss_term is None else loss_term + w * term
        print(f"  {name}: features + loss built (weight {w})")

    # Sequence plausibility. Without a term like this the structure models will
    # happily accept sequences no natural protein would tolerate.
    for term in dc.build_sequence_terms(cfg.get("sequence_models", [])):
        loss_term = loss_term + term
    for spec in cfg.get("sequence_models", []):
        print(f"  sequence model {spec['name']} (weight {spec['weight']})")

    # NoCys is a reparameterization, not an additive penalty: it consumes an
    # [N, 19] sequence and splices a zero-probability cysteine column back in
    # before calling the wrapped loss. So it wraps the whole objective and the
    # optimizer's alphabet shrinks to 19 — cysteine becomes impossible rather
    # than merely expensive. The decode step below has to undo this.
    n_tokens = 20
    if bnd["no_cys"]:
        loss_term = NoCys(loss_term)
        n_tokens = 19
        print("  no-cys: optimizing over 19 tokens (cysteine excluded by construction)")

    from mosaic.common import TOKENS

    seed = cfg["run"]["seed"]
    batch = cfg["run"]["batch"]
    blen = bnd["length"]
    no_cys = bnd["no_cys"]
    init_fasta = bnd.get("init_fasta")
    init_noise = bnd.get("init_noise", 0.15)
    soft, sharp = cfg["optimizer"]["soft"], cfg["optimizer"]["sharp"]

    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)

    if init_fasta:
        # Seed from proposed sequences (e.g. BoltzGen backbones read out by
        # CoordsToToken). The optimizer then refines a plausible starting point
        # rather than searching from noise.
        seqs = []
        for ln in Path(init_fasta).read_text().splitlines():
            ln = ln.strip()
            if ln and not ln.startswith(">"):
                seqs.append(ln.upper())
        if not seqs:
            raise SystemExit(f"no sequences in {init_fasta}")
        # The optimizer's alphabet is 19 under --no-cys (C removed), so build the
        # one-hot in that alphabet, not in TOKENS, or every residue from index 4
        # onward lands on the wrong column.
        alphabet = TOKENS.replace("C", "") if no_cys else TOKENS
        rows = []
        for b in range(batch):
            # Offset by seed so array tasks refine DIFFERENT proposals rather
            # than every task starting from the same first --batch sequences.
            s = seqs[(seed * batch + b) % len(seqs)]
            if len(s) != blen:
                raise SystemExit(
                    f"--init-fasta sequence {b} is {len(s)} aa, expected "
                    f"{blen}")
            idx = [alphabet.index(c) if c in alphabet else alphabet.index("A")
                   for c in s]
            rows.append(np.eye(n_tokens, dtype=np.float32)[idx])
        seeded = jnp.asarray(np.stack(rows))
        # Blend toward uniform: a hard vertex of the simplex has no room to move
        # under a projected-gradient step.
        w = init_noise
        x = (1.0 - w) * seeded + w * (1.0 / n_tokens)
        print(f"seeded {batch} trajectories from {len(seqs)} sequences in "
              f"{init_fasta} (noise {w})")
        if no_cys and any("C" in s for s in seqs):
            print("  note: seed sequences contain C, which --no-cys forbids; "
                  "those positions were mapped to A")
    else:
        # Start near the uniform point of the simplex with a little noise, so the
        # B trajectories diverge instead of collapsing to the same path.
        x = jax.random.uniform(
            init_key, (batch, blen, n_tokens),
            minval=0.0, maxval=1.0
        )
        x = x / x.sum(-1, keepdims=True)

    # Both stages return (final_iterate, best_iterate). Continue from the final
    # iterate — best_x is a snapshot, not a state the optimizer can resume from
    # coherently (its momentum history belongs to a different point).
    t0 = time.time()
    x, _ = batched_simplex_APGM(
        loss_function=loss_term, x=x, n_steps=soft["n_steps"],
        stepsize=soft["stepsize"], momentum=soft["momentum"],
        scale=soft.get("scale", 1.0), logspace=soft.get("logspace", False),
        max_gradient_norm=soft.get("max_gradient_norm"), key=key,
    )
    print(f"soft stage ({soft['n_steps']} steps): {time.time() - t0:.1f}s")

    # Sharpening: a smaller step with the iterate raised toward one-hot. Without
    # this the result is a soft sequence that does not correspond to any actual
    # protein.
    t0 = time.time()
    key, sharp_key = jax.random.split(key)
    x, best_x = batched_simplex_APGM(
        loss_function=loss_term, x=x, n_steps=sharp["n_steps"],
        stepsize=sharp["stepsize"], momentum=sharp["momentum"],
        scale=sharp.get("scale", 2.0), logspace=sharp.get("logspace", False),
        max_gradient_norm=sharp.get("max_gradient_norm"), key=sharp_key,
    )
    print(f"sharp stage ({sharp['n_steps']} steps): {time.time() - t0:.1f}s")

    # Report best_x, not the final iterate: APGM with momentum does not descend
    # monotonically, so the last step is not reliably the best one. Note the
    # optimizer's own caveat (optimizers.py:312) that best_x is tracked against a
    # loss evaluated at the extrapolated point v rather than at x, so it is a
    # close approximation of the best iterate rather than an exact one — the
    # re-evaluation below reports each design's actual loss regardless.
    x = best_x

    from mosaic.optimizers import batched_eval

    keys = jax.random.split(jax.random.key(seed + 1), batch)
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
        full = NoCys.sequence(xb) if no_cys else xb
        return "".join(TOKENS[i] for i in np.asarray(full).argmax(-1))

    results = []
    for b in range(batch):
        seq = decode(x[b])
        results.append({"trajectory": b, "seed": seed,
                        "loss": float(values[b]), "sequence": seq})
        print(f"[{b}] loss={values[b]:.4f}  {seq}")

    results.sort(key=lambda r: r["loss"])
    payload = {
        "target_length": len(target),
        "target_substitutions": subs,
        "binder_length": blen,
        "models": names,
        "seed": seed,
        "batch": batch,
        "target_msa": msa_path,
        "config": cfg,
        "results": results,
    }
    (out_dir / f"designs_seed{seed}.json").write_text(json.dumps(payload, indent=2))
    with (out_dir / f"designs_seed{seed}.fasta").open("w") as f:
        for r in results:
            f.write(f">seed{seed}_traj{r['trajectory']}_loss{r['loss']:.4f}\n"
                    f"{r['sequence']}\n")

    print(f"\nbest loss {results[0]['loss']:.4f} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
