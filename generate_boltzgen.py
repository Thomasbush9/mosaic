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
    ap.add_argument("--target-fasta", default=None)
    ap.add_argument("--target-name", default=None)
    ap.add_argument("--hotspots", default="",
                   help="1-based target residues to design against, e.g. "
                        "'95-110,143'. BoltzGen has no attractive hotspot term "
                        "(its constraints support only total_len and bond), so "
                        "this works by CROPPING what the model sees: the target "
                        "is restricted to a pocket around these residues via "
                        "the YAML's include/res_index. The binder therefore has "
                        "nowhere else to bind.")
    ap.add_argument("--hotspot-shell", type=float, default=6.0,
                   help="include target residues within this many angstrom of a "
                        "hotspot, so the model sees a coherent surface patch "
                        "rather than isolated residues. Calibrate: on the "
                        "237-residue DIO3 ECD a 6 A shell keeps 21%% of the "
                        "target, 8 A keeps 38%% and 12 A keeps 69%% — a shell "
                        "that keeps most of the protein does not constrain "
                        "anything.")
    ap.add_argument("--contact-cutoff", type=float, default=8.0,
                   help="heavy-atom distance defining an interface contact when "
                        "deriving the epitope from the generated poses")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import proposals as P

    from mosaic.common import TOKENS
    from mosaic.models.boltzgen import (
        CoordsToToken, Sampler, load_boltzgen,
        load_features_and_structure_writer,
    )

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    target_cif = Path(a.target_cif).resolve()

    # Hotspots -> a pocket, expressed as the YAML's include/res_index.
    hot1 = P.parse_positions(a.hotspots)
    include_block = ""
    pocket1: list[int] = []
    if hot1:
        pocket1 = P.pocket_residues(target_cif, hot1, a.hotspot_shell,
                                    chain_id=a.target_chain)
        # res_index is 1-based and uses '..' for ranges (schema.py:646-664).
        include_block = "\n                res_index: " + P.as_res_index(pocket1)
        print(f"hotspots: {len(hot1)} residues -> pocket of {len(pocket1)} "
              f"within {a.hotspot_shell} A")
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
                id: {chain}{include}

    structure_groups:
      - group:
          id: {chain}
          visibility: 2
    """.format(n=a.binder_length, ss=ss, chain=a.target_chain,
               include=include_block)

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
        st.make_mmcif_document().write_file(str(out / f"design_{i:03d}.cif"))

    # BoltzGen is not hotspot-conditioned: it chooses where to bind. Recovering
    # that choice is what lets the refinement stage aim at the same interface
    # instead of re-deriving a pose from a generic contact term.
    cifs = sorted(out.glob("design_*.cif"))
    epitope, notes = P.consensus_epitope(cifs, binder_length=a.binder_length,
                                         cutoff=a.contact_cutoff)

    ps = P.ProposalSet(
        name=out.name, generator="boltzgen",
        target_name=a.target_name or target_cif.stem,
        target_fasta=a.target_fasta or "", target_structure=str(target_cif),
        binder_length=a.binder_length, n_designs=a.num_designs,
        sequences=[r["sequence"] for r in records], epitope_idx=epitope,
        params={"secondary_structure": ss, "sampling_steps": a.sampling_steps,
                "step_scale": a.step_scale, "noise_scale": a.noise_scale,
                "recycling_steps": a.recycling_steps, "seed": a.seed,
                "target_chain": a.target_chain,
                "contact_cutoff": a.contact_cutoff,
                "hotspots": hot1, "pocket": pocket1,
                "hotspot_shell": a.hotspot_shell},
        notes=notes + ([f"target cropped to a {a.hotspot_shell} A pocket around "
                        f"{len(hot1)} hotspots"] if hot1 else []),
    )
    P.write(out, ps)

    allseq = "".join(r["sequence"] for r in records)
    uniq = len(set(allseq))
    print(f"\nwrote proposal set -> {out}")
    print(f"epitope derived from poses: {len(epitope)} target residues")
    for n in notes:
        print(f"  {n}")
    print(f"distinct residue types across all designs: {uniq}/20")
    if uniq < 8:
        print("WARNING: very low residue diversity — check the coords->token "
              "inference rather than trusting these sequences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
