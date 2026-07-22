#!/usr/bin/env python
"""Predict a target's structure and write it as CIF.

BoltzGen is structure-conditioned: it designs against a target *structure*, not a
sequence (examples/boltzgen_example.py reads a .cif and references it from the
binder YAML). So a generative run needs this first.

    ./singularity/mosaic-exec.sh python predict_target.py \
        --target targets/dio3_ecd.fasta --target-msa msa/dio3_ecd.a3m \
        --out targets/dio3_ecd.cif

Uses target_only_features rather than binder_features deliberately: only the
former includes real sidechain reference atoms. binder_features stubs them to
UNK/G because they are not differentiably defined for a soft sequence, so a
structure predicted that way has no sidechains — useless as a design target.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import numpy as np


def read_target(spec: str) -> str:
    p = Path(spec)
    if p.is_file():
        lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        return "".join(ln for ln in lines if not ln.startswith(">")).upper()
    return spec.strip().upper()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--target-msa", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recycling-steps", type=int, default=4,
                    help="higher than during design: this is a one-off "
                         "prediction, and the examples use 4-20 for final "
                         "folding vs 1 while optimizing")
    ap.add_argument("--sampling-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from mosaic.models.boltz2 import Boltz2
    from mosaic.structure_prediction import TargetChain
    from run_design import sanitize_target

    seq, subs = sanitize_target(read_target(a.target))
    print(f"target: {len(seq)} residues")
    if subs:
        print(f"substituted: {', '.join(subs)}")

    t0 = time.time()
    model = Boltz2()
    print(f"model loaded in {time.time() - t0:.1f}s")

    chain = TargetChain(sequence=seq, use_msa=a.target_msa is not None,
                        msa_path=a.target_msa)
    features, writer = model.target_only_features([chain])
    print("features built (target-only: real sidechain reference atoms)")

    t0 = time.time()
    pred = model.predict(features=features, writer=writer,
                         recycling_steps=a.recycling_steps,
                         sampling_steps=a.sampling_steps,
                         key=jax.random.key(a.seed))
    print(f"predicted in {time.time() - t0:.1f}s")

    # mosaic reports pLDDT on 0-1, not the conventional 0-100 (PLDDTLoss uses it
    # directly as a fraction, structure_prediction.py:442). Rescale for reporting
    # so the numbers mean what a reader expects.
    plddt = np.asarray(pred.plddt)
    if plddt.max() <= 1.0:
        plddt = plddt * 100.0
    print(f"pLDDT (0-100): mean {plddt.mean():.1f}  min {plddt.min():.1f}  "
          f"max {plddt.max():.1f}  frac>70 {(plddt > 70).mean():.2f}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.st.setup_entities()
    pred.st.make_mmcif_document().write_file(str(out))
    print(f"wrote {out}")

    # A target the model is unsure about makes a poor design scaffold; say so
    # rather than letting it quietly propagate into the generative step.
    if plddt.mean() < 70:
        print("WARNING: mean pLDDT < 70 — this target structure is low-confidence "
              "and is a weak basis for structure-conditioned design")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
