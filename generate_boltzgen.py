#!/usr/bin/env python
"""Generate binder backbones with BoltzGen, then read off sequences.

BoltzGen is structure-conditioned and generative: unlike Boltz/AF2/Protenix it
implements no binder_features/build_loss, so it cannot participate in the
differentiable objective. It is used *before* that objective, to propose starting
points — which is strictly better than initializing the optimizer from noise.

    ./singularity/mosaic-exec.sh python generate_boltzgen.py \
        --target-cif targets/dio3_ecd.cif --binder-length 80 \
        --num-designs 8 --out designs/dio3_boltzgen

Emits a FASTA of proposed binder sequences, plus one CIF per design. Feed the
FASTA to run_design.py --init-fasta to refine under the multi-model loss.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def helix_bundle_ss(n: int, n_helices: int = 3, loop: int = 4) -> str:
    """A three-helix bundle, the standard de novo mini-binder topology.

    BoltzGen conditions on a per-residue secondary-structure string
    (examples/boltzgen_example.py:76 uses 23H/4L/23H/3L/23H). Helices are split
    evenly over whatever length is asked for rather than hardcoding 23.
    """
    n_loops = n_helices - 1
    helix_total = n - n_loops * loop
    if helix_total < n_helices * 6:
        raise SystemExit(f"binder length {n} too short for {n_helices} helices")
    base, extra = divmod(helix_total, n_helices)
    parts = []
    for i in range(n_helices):
        parts.append("H" * (base + (1 if i < extra else 0)))
        if i < n_loops:
            parts.append("L" * loop)
    ss = "".join(parts)
    assert len(ss) == n, (len(ss), n)
    return ss


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-cif", required=True,
                   help="target STRUCTURE — BoltzGen designs against geometry, "
                        "not sequence (see predict_target.py)")
    ap.add_argument("--binder-length", type=int, default=80)
    ap.add_argument("--num-designs", type=int, default=8)
    ap.add_argument("--target-chain", default="A")
    ap.add_argument("--recycling-steps", type=int, default=3)
    ap.add_argument("--sampling-steps", type=int, default=300,
                   help="diffusion steps per design; the example uses 300")
    ap.add_argument("--step-scale", type=float, default=2.0)
    ap.add_argument("--noise-scale", type=float, default=0.88)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from mosaic.common import TOKENS
    from mosaic.models.boltzgen import (
        CoordsToToken, Sampler, load_boltzgen,
        load_features_and_structure_writer,
    )

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    target_cif = Path(a.target_cif).resolve()
    ss = helix_bundle_ss(a.binder_length)
    print(f"binder {a.binder_length} aa, secondary structure:\n  {ss}")

    t0 = time.time()
    boltzgen = load_boltzgen()
    print(f"boltzgen loaded in {time.time() - t0:.1f}s")

    # The YAML references the target by a placeholder name that is resolved
    # through `files=`, so the real path never appears in the string.
    yaml_binder = """
    entities:
      - protein:
          id: B
          sequence: {n}
          secondary_structure: {ss}

      - file:
          path: TARG.CIF

          include:
            - chain:
                id: {chain}

    structure_groups:
      - group:
          id: {chain}
          visibility: 2
    """.format(n=a.binder_length, ss=ss, chain=a.target_chain)

    features, writer = load_features_and_structure_writer(
        yaml_string=yaml_binder, files={"TARG.CIF": str(target_cif)}
    )
    print("features built")

    # Sampler.from_features runs the trunk once to precompute conditioning; the
    # returned callable then draws samples cheaply, so the trunk cost is paid
    # once rather than per design.
    t0 = time.time()
    sampler = eqx.filter_jit(
        Sampler.from_features(
            model=boltzgen, features=features,
            key=jax.random.key(a.seed), deterministic=True,
            recycling_steps=a.recycling_steps,
        )
    )
    coords2token = CoordsToToken(features)
    print(f"sampler conditioned in {time.time() - t0:.1f}s")

    records = []
    for i in range(a.num_designs):
        t0 = time.time()
        coords = sampler(
            structure_module=boltzgen.structure_module,
            num_sampling_steps=a.sampling_steps,
            step_scale=jnp.array(a.step_scale),
            noise_scale=jnp.array(a.noise_scale),
            key=jax.random.key(a.seed * 1000 + i),
        )
        tokens = np.asarray(coords2token(coords[0]))
        seq = "".join(TOKENS[t] for t in tokens)
        dt = time.time() - t0
        print(f"[{i}] {dt:5.1f}s  {seq}")
        records.append({"design": i, "sequence": seq, "seconds": round(dt, 1)})

        st = writer(coords)
        st.setup_entities()
        st.make_mmcif_document().write_file(str(out / f"boltzgen_{i:03d}.cif"))

    fasta = out / "boltzgen_designs.fasta"
    with fasta.open("w") as f:
        for r in records:
            f.write(f">boltzgen_{r['design']:03d}\n{r['sequence']}\n")
    (out / "boltzgen_designs.json").write_text(json.dumps(
        {"target_cif": str(target_cif), "binder_length": a.binder_length,
         "secondary_structure": ss, "sampling_steps": a.sampling_steps,
         "seed": a.seed, "designs": records}, indent=2))

    # Composition sanity: the coords->token map infers residues from sidechain
    # atom placement, so a degenerate sample shows up as a near-constant string.
    allseq = "".join(r["sequence"] for r in records)
    uniq = len(set(allseq))
    print(f"\n{len(records)} designs -> {fasta}")
    print(f"distinct residue types across all designs: {uniq}/20")
    if uniq < 8:
        print("WARNING: very low residue diversity — check the coords->token "
              "inference rather than trusting these sequences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
