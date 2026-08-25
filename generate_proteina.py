#!/usr/bin/env python
"""Generate binder backbones with Proteina, and write a proposal set.

Proteina is a flow-matching generative model. Like BoltzGen it implements no
binder_features/build_loss and cannot join a campaign objective — it proposes
starting points.

Unlike BoltzGen it is **hotspot-conditioned**: ``load_target_cond`` takes the
target residues the binder should engage, so the epitope is an input rather than
something derived from the output afterwards.

    ./singularity/mosaic-exec.sh python generate_proteina.py \
        --target-cif targets/dio3_ecd.cif --binder-length 80 \
        --num-designs 8 --hotspots 95-110 --out designs/dio3_proteina
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def parse_ranges(text: str) -> list[int]:
    """"95-110,143" -> 1-based positions."""
    out: list[int] = []
    for chunk in (text or "").replace(" ", "").split(","):
        if not chunk:
            continue
        if "-" in chunk.lstrip("-"):
            a, b = chunk.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-cif", required=True)
    ap.add_argument("--target-fasta", default=None)
    ap.add_argument("--target-name", default=None)
    ap.add_argument("--binder-length", type=int, default=80)
    ap.add_argument("--num-designs", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=32,
                    help="sample in fixed-size chunks — Proteina vmaps the whole "
                         "batch at once and memory scales with it, so a large "
                         "--num-designs OOMs the GPU. 32 is memory-safe; output "
                         "is identical, one compile.")
    ap.add_argument("--hotspots", default="",
                    help="1-based target residues the binder should engage, "
                         "e.g. '95-110,143'. Proteina conditions on these, so "
                         "unlike BoltzGen the epitope is chosen, not discovered.")
    ap.add_argument("--target-chain", default="A")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import proposals as P

    import equinox as eqx
    import gemmi
    import jax
    import jax.numpy as jnp
    from jproteina_complexa.constants import AA_3LETTER, AA_CODES
    from jproteina_complexa.flow_matching import generate
    from jproteina_complexa.hub import load_decoder, load_denoiser
    from jproteina_complexa.pdb import load_target_cond
    from jproteina_complexa.types import DecoderBatch

    out = Path(a.out)
    cif = Path(a.target_cif)
    st = gemmi.read_structure(str(cif))
    st.setup_entities()
    chain = st[0][a.target_chain] if a.target_chain in [c.name for c in st[0]] else st[0][0]
    target_seq = gemmi.one_letter_code([r.name for r in chain.get_polymer()])

    hot1 = parse_ranges(a.hotspots)
    bad = [h for h in hot1 if not (1 <= h <= len(target_seq))]
    if bad:
        raise SystemExit(f"hotspots outside 1-{len(target_seq)}: {bad}")
    hot0 = [h - 1 for h in hot1]          # load_target_cond wants 0-based
    print(f"target: {len(target_seq)} residues from {cif.name}")
    if hot0:
        print("hotspots: " + " ".join(f"{target_seq[i]}{i + 1}" for i in hot0))
    else:
        print("hotspots: none — Proteina will choose where to bind")

    t0 = time.time()
    target_cond = load_target_cond(chain, hotspots=hot0)
    denoiser, decoder = load_denoiser(), load_decoder()
    mask = jnp.ones(a.binder_length, dtype=jnp.bool_)
    print(f"loaded in {time.time() - t0:.1f}s")

    @eqx.filter_jit
    def sample_batch(key, n):
        keys = jax.random.split(key, n)
        return jax.vmap(lambda k: generate(denoiser, mask, k, target=target_cond))(keys)

    @eqx.filter_jit
    def decode_batch(bbs, lats):
        return jax.vmap(
            lambda b, l: decoder(DecoderBatch(z_latent=l, ca_coors=b, mask=mask))
        )(bbs, lats)

    # Sample in fixed-size chunks. Proteina vmaps the whole batch at once and
    # memory scales with it — 500 designs needs ~105 GiB and OOMs an 80 GB GPU.
    # A constant chunk keeps memory flat and compiles once; fold_in gives each
    # chunk a distinct key so designs stay independent across chunks.
    chunk = max(1, min(int(a.chunk), a.num_designs))
    n_chunks = (a.num_designs + chunk - 1) // chunk

    out.mkdir(parents=True, exist_ok=True)
    seqs: list[str] = []
    gi = 0
    t0 = time.time()
    for ci in range(n_chunks):
        bbs, lats = sample_batch(
            jax.random.fold_in(jax.random.PRNGKey(a.seed), ci), chunk)
        jax.block_until_ready(bbs)
        decs = decode_batch(bbs, lats)
        for bi in range(chunk):
            if gi >= a.num_designs:
                break
            dsl = jax.tree.map(lambda x: x[bi], decs)
            aatype = np.array(dsl.aatype)
            seq = "".join(AA_CODES[j] for j in aatype)
            seqs.append(seq)
            print(f"[{gi}] {seq}")

            # Write binder + target as one complex, so the epitope can be derived
            # from the pose exactly as it is for BoltzGen.
            cs = gemmi.Structure()
            model = gemmi.Model("1")
            bch = gemmi.Chain("A")
            coords = np.array(dsl.ca_coors) if hasattr(dsl, "ca_coors") \
                else np.array(bbs[bi])
            for k, (aa, xyz) in enumerate(zip(aatype, coords), start=1):
                res = gemmi.Residue()
                res.name = AA_3LETTER[AA_CODES[aa]]
                res.seqid = gemmi.SeqId(k, " ")
                at = gemmi.Atom()
                at.name, at.element = "CA", gemmi.Element("C")
                at.pos = gemmi.Position(*[float(v) for v in xyz[:3]])
                res.add_atom(at)
                bch.add_residue(res)
            model.add_chain(bch)
            tch = gemmi.Chain("B")
            tcoords = np.array(target_cond.coords)
            for k, (aa, xyz) in enumerate(zip(np.array(target_cond.seq), tcoords), start=1):
                res = gemmi.Residue()
                res.name = AA_3LETTER[AA_CODES[aa]]
                res.seqid = gemmi.SeqId(k, " ")
                at = gemmi.Atom()
                at.name, at.element = "CA", gemmi.Element("C")
                pos = xyz[1] if np.ndim(xyz) > 1 else xyz
                at.pos = gemmi.Position(*[float(v) for v in np.asarray(pos)[:3]])
                res.add_atom(at)
                tch.add_residue(res)
            model.add_chain(tch)
            cs.add_model(model)
            cs.setup_entities()
            cs.make_mmcif_document().write_file(str(out / f"design_{gi:03d}.cif"))
            gi += 1
    print(f"sampled {a.num_designs} backbones in {time.time() - t0:.1f}s "
          f"({chunk}/chunk, {n_chunks} chunks)")

    cifs = sorted(out.glob("design_*.cif"))
    if hot0:
        epitope, notes = hot0, ["epitope was specified as hotspots and conditioned on"]
    else:
        # CA-only complexes, so use a wider cutoff than the all-atom default.
        # Proteina sees the whole target and the writer numbers chain B 1..N
        # contiguously, so these indices are already full-target ones.
        epitope, notes = P.consensus_epitope(cifs, binder_length=a.binder_length,
                                             cutoff=12.0,
                                             target_length=len(target_seq))

    ps = P.ProposalSet(
        name=out.name, generator="proteina",
        target_name=a.target_name or cif.stem,
        target_fasta=a.target_fasta or "", target_structure=str(cif),
        binder_length=a.binder_length, n_designs=a.num_designs,
        sequences=seqs, epitope_idx=epitope,
        epitope_frame=P.EPITOPE_FRAME_TARGET,
        params={"hotspots": hot1, "seed": a.seed, "target_chain": a.target_chain},
        notes=notes + ["backbones are CA-only; refine before trusting geometry"],
    )
    P.write(out, ps)
    print(f"\nwrote proposal set -> {out}")
    print(f"epitope: {len(epitope)} target residues")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
