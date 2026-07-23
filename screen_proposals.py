#!/usr/bin/env python
"""Screen proposed binders: refold each with a chosen model, rank by confidence.

This is one of the two ways to combine generation with the structure models, and
the one that matches mosaic's own boltzgen_pipeline.py:

    generate candidates  ->  refold + score  ->  rank / filter

Unlike the optimize path (run_design.py, which gradient-descends a soft sequence
from a seed), this holds each proposed sequence FIXED, folds it, and reads the
confidence metrics. Nothing is optimized — the generator already chose the
sequence, and the question is only which candidates fold into a confident
complex with the target.

Any StructurePredictionModel works: all of them implement
predict(PSSM=..., features=..., writer=...) and return iptm / plddt / pae. The
model is a choice, not hardcoded — you can, and often should, screen with a
different model than the one that will optimize (e.g. generate with BoltzGen,
screen with AF2), so a candidate is not judged by the model that made it.

    ./singularity/mosaic-exec.sh python screen_proposals.py \
        --proposals designs/dio3_boltzgen --model af2 \
        --target-msa msa/dio3_ecd.a3m --out designs/dio3_boltzgen/screen_af2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--proposals", required=True,
                    help="a proposal-set directory (manifest.json + proposals.fasta)")
    ap.add_argument("--model", default="boltz2",
                    choices=["boltz2", "boltz1", "af2", "of3", "protenix"],
                    help="structural model to refold with — deliberately free, "
                         "and best chosen DIFFERENT from the generator so a "
                         "candidate is not scored by the model that made it")
    ap.add_argument("--protenix-variant", default="mini")
    ap.add_argument("--target-msa", default=None,
                    help="a3m for the target; AF2 ignores it (its wrapper rejects "
                         "MSAs), the others use it")
    ap.add_argument("--recycling-steps", type=int, default=4,
                    help="higher than during design — this is a final-quality "
                         "fold, and the examples use 4-20 for validation vs 1 "
                         "while optimizing")
    ap.add_argument("--sampling-steps", type=int, default=50)
    ap.add_argument("--write-cif", action="store_true", default=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import proposals as P
    import design_config as dc
    from run_design import sanitize_target

    import jax
    import jax.numpy as jnp
    from mosaic.common import TOKENS
    from mosaic.structure_prediction import TargetChain

    ps = P.read(Path(a.proposals))
    if ps is None:
        raise SystemExit(f"no proposal set at {a.proposals}")
    if not ps.sequences:
        raise SystemExit("proposal set has no sequences")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"jax {jax.__version__} {jax.default_backend()} {jax.devices()}")
    print(f"screening {len(ps.sequences)} proposals from {ps.name} "
          f"({ps.generator}) with {a.model}")

    # The target: prefer the proposal set's own FASTA, fall back to its name.
    target_raw = ""
    if ps.target_fasta and Path(ps.target_fasta).is_file():
        target_raw = "".join(
            ln.strip() for ln in Path(ps.target_fasta).read_text().splitlines()
            if ln.strip() and not ln.startswith(">"))
    if not target_raw:
        raise SystemExit("proposal set has no readable target FASTA")
    target, _subs = sanitize_target(target_raw)

    accepts_msa = dc.STRUCTURE_MODELS[a.model]["accepts_msa"]
    msa_path = a.target_msa if (a.target_msa and accepts_msa) else None
    chain = TargetChain(sequence=target, use_msa=msa_path is not None,
                        msa_path=msa_path)

    t0 = time.time()
    model = dc.build_structure_model(a.model, {"variant": a.protenix_variant})
    print(f"model loaded in {time.time() - t0:.1f}s")

    features, writer = model.binder_features(ps.binder_length, [chain])
    print("binder features built")

    predict_kw = {"features": features, "writer": writer,
                  "recycling_steps": a.recycling_steps}
    if a.model != "af2":
        predict_kw["sampling_steps"] = a.sampling_steps

    def one_hot(seq: str) -> jnp.ndarray:
        idx = [TOKENS.index(c) for c in seq]
        return jnp.asarray(np.eye(20, dtype=np.float32)[idx])

    results = []
    for i, seq in enumerate(ps.sequences):
        if len(seq) != ps.binder_length:
            print(f"[{i}] skipped — length {len(seq)} != {ps.binder_length}")
            continue
        t0 = time.time()
        pred = model.predict(PSSM=one_hot(seq), key=jax.random.key(i), **predict_kw)

        plddt = np.asarray(pred.plddt)
        if plddt.max() <= 1.0:            # mosaic reports pLDDT on 0-1
            plddt = plddt * 100.0
        binder_plddt = float(plddt[:ps.binder_length].mean())
        iptm = float(np.asarray(pred.iptm))
        pae = np.asarray(pred.pae)
        # Interface PAE: binder rows vs target columns. Lower is better.
        iface_pae = float(pae[:ps.binder_length, ps.binder_length:].mean())

        rec = {"design": i, "sequence": seq, "iptm": round(iptm, 4),
               "binder_plddt": round(binder_plddt, 2),
               "interface_pae": round(iface_pae, 3),
               "seconds": round(time.time() - t0, 1)}
        results.append(rec)
        print(f"[{i}] iptm {iptm:.3f}  plddt {binder_plddt:5.1f}  "
              f"ifacePAE {iface_pae:5.2f}  {seq}")

        if a.write_cif:
            pred.st.setup_entities()
            pred.st.make_mmcif_document().write_file(
                str(out / f"refold_{i:03d}.cif"))

    # Rank: high ipTM and pLDDT good, low interface PAE good. A simple,
    # transparent composite — the raw metrics are kept so you can re-rank.
    for r in results:
        r["score"] = round(r["iptm"] + r["binder_plddt"] / 100.0
                           - r["interface_pae"] / 30.0, 4)
    results.sort(key=lambda r: -r["score"])

    payload = {
        "proposal_set": ps.name, "generator": ps.generator,
        "screen_model": a.model, "target_msa": msa_path,
        "recycling_steps": a.recycling_steps,
        "score_formula": "iptm + binder_plddt/100 - interface_pae/30",
        "results": results,
    }
    (out / "screen.json").write_text(json.dumps(payload, indent=2))
    with (out / "screened.fasta").open("w") as f:
        for r in results:
            f.write(f">{ps.name}_{r['design']:03d}_iptm{r['iptm']:.3f}\n"
                    f"{r['sequence']}\n")

    print(f"\nscreened {len(results)} proposals with {a.model} -> {out}")
    if results:
        b = results[0]
        print(f"best: design {b['design']}  ipTM {b['iptm']}  "
              f"pLDDT {b['binder_plddt']}  interfacePAE {b['interface_pae']}")
    print("\nNote: ipTM/pLDDT are the folding model's own confidence, not an "
          "affinity. Screen with a model OTHER than the generator, and validate "
          "the top few before trusting them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
