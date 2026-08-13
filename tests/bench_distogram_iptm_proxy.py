#!/usr/bin/env python
"""Characterise `DistogramIPTMProxy` with and without `epitope_idx`.

Not a pass/fail test -- the numbers you look at when deciding whether the term
is worth its weight in an objective, and the artefact to diff after each edit to
the loss. Writes `iptm_proxy_bench.json` next to the report.

Two featurisations, because they answer different questions:

  complex   `target_only_features([binder, target])` -- both real sequences, real
            sidechain reference atoms. AF2's best shot at the true complex, so
            this measures the PROXY: given a genuinely good interface, does it
            say so, and does restricting to the epitope change what it says?

  design    `binder_features(len, [target])` + a soft sequence -- the path
            `run_design.py` actually takes, where the binder is a stub with no
            sidechains. This measures the TERM AS AN OBJECTIVE: what value and
            what gradient does the optimizer see on step 0?

A loss can look excellent under `complex` and be useless under `design`.

    tests/run_iptm_proxy_tests.sh --bench
    tests/run_iptm_proxy_tests.sh --bench -- --recycling 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

KEY = jax.random.key(0)
MODEL_IDX = 0

# Two systems, deliberately.
#
#   dio3   the campaign's best design (rank01, design358) against the DIO3 ECD.
#          AF2 scores this pair ipTM 0.821 at recycling 4 (dio3_candidates/
#          screen_ranked.csv), so it is a case where the structure model really
#          does predict an interface -- the only setting in which "does the
#          proxy see the interface" is a meaningful question. Default.
#
#   7opb   IL7R + a crystallised 55-aa antagonist. A real experimental complex,
#          but AF2 without an MSA folds it at ipTM 0.10 and predicts no contacts
#          at all, so it tests the loss against garbage input. Kept because that
#          is itself worth knowing, and because it is the only in-tree structure
#          with solvent (which is how the proposals.py bug shows up).
SYSTEMS = {
    "dio3": dict(
        cif="dio3_candidates/top20_complexes/rank01_design358_iptm0.821.cif",
        target_fasta="targets/dio3_ecd.fasta",
        binder_seq="SAEEIERELREEIERLLEETREQMKGLSVEEATELAQQTMQEIDRLVDEAIERGLPLDRAIELLLEAGERLGELLGEVVE",
        note="AF2 ipTM 0.821 at recycling 4 (screen_ranked.csv rank 1)",
    ),
    "7opb": dict(cif="7opb.pdb", target_fasta=None, binder_seq=None,
                 note="crystal complex; AF2 single-sequence ipTM ~0.10"),
}
# Paths are relative to the *setup* dir (the checkout's parent), which
# run_iptm_proxy_tests.sh binds at /mnt.
ROOTS = [Path(__file__).resolve().parents[2], Path("/mnt")]


def _find(rel: str) -> Path:
    for r in ROOTS:
        if (r / rel).exists():
            return r / rel
        if (r / Path(rel).name).exists():
            return r / Path(rel).name
    raise SystemExit(f"cannot find {rel} under {[str(r) for r in ROOTS]}")


def _read_fasta(p: Path) -> str:
    return "".join(
        ln.strip() for ln in p.read_text().splitlines()
        if ln.strip() and not ln.startswith(">")
    ).upper()


# ------------------------------------------------------------------ the system


def load_system(name: str, cutoff: float = 8.0):
    import gemmi

    spec = SYSTEMS[name]
    st = gemmi.read_structure(str(_find(spec["cif"])))
    st.setup_entities()
    st.remove_ligands_and_waters()
    st.remove_hydrogens()
    chains = [c for c in st[0] if len(c) > 0]
    binder_ch, target_ch = min(chains, key=len), max(chains, key=len)

    def seq(ch):
        return gemmi.one_letter_code([r.name for r in ch]).upper()

    ns = gemmi.NeighborSearch(st, cutoff).populate()
    pos = {r.seqid.num: i for i, r in enumerate(target_ch)}
    epitope = set()
    for res in binder_ch:
        for atom in res:
            for m in ns.find_atoms(atom.pos, "\0", radius=cutoff):
                cra = m.to_cra(st[0])
                if cra.chain.name == target_ch.name and cra.residue.seqid.num in pos:
                    epitope.add(pos[cra.residue.seqid.num])
    epitope = sorted(epitope)

    # Prefer the canonical FASTA over the modelled chain: a generated CIF can be
    # missing residues, and the target sequence has to be the one the campaign
    # would featurise, or the epitope indices refer to a different numbering.
    target_seq = (
        _read_fasta(_find(spec["target_fasta"])) if spec["target_fasta"] else seq(target_ch)
    )
    if len(target_seq) != len(target_ch):
        raise SystemExit(
            f"{name}: target FASTA is {len(target_seq)} aa but the CIF chain has "
            f"{len(target_ch)} residues -- epitope indices would not line up"
        )
    binder_seq = spec["binder_seq"] or seq(binder_ch)
    if len(binder_seq) != len(binder_ch):
        raise SystemExit(f"{name}: binder length mismatch")

    ca = np.array(
        [
            [a.pos.x, a.pos.y, a.pos.z]
            if (a := r.find_atom("CA", "*")) is not None
            else [np.nan] * 3
            for r in target_ch
        ]
    )
    d = np.linalg.norm(ca - np.nanmean(ca[epitope], axis=0), axis=-1)
    away = [int(i) for i in np.argsort(-np.nan_to_num(d)) if int(i) not in set(epitope)]
    return dict(
        name=name,
        note=spec["note"],
        binder_seq=binder_seq,
        target_seq=target_seq,
        epitope=epitope,
        decoy=sorted(away[: len(epitope)]),
        core=sorted(epitope[: max(4, len(epitope) // 6)]),  # a tight hotspot
    )


# ------------------------------------------------------------------- the metrics


def decompose(output, binder_len, epitope_idx=None, contact_distance=8.0):
    """Every intermediate of Algorithm 15, not just the clipped answer.

    `margin` is the unclipped `1 - S_bar / log(n_contact_bins)`. When it is
    negative the reported proxy is 0 *and its gradient is 0* -- the term is
    inert. That distinction is invisible in the proxy alone, which is the whole
    reason this function exists.
    """
    D = output.distogram_logits[:binder_len, binder_len:]
    bins = output.distogram_bins
    m = bins < contact_distance

    log_p_full = jax.nn.log_softmax(D, axis=-1)
    p_cut = jax.nn.softmax(D, axis=-1, where=m)
    S = -(p_cut * log_p_full).sum(-1)
    p_contact = jnp.exp(jax.nn.logsumexp(log_p_full, axis=-1, where=m))

    if epitope_idx is not None:
        S = S[:, jnp.array(epitope_idx)]
        p_contact = p_contact[:, jnp.array(epitope_idx)]

    S_bar = float(-jax.lax.top_k(-S.reshape(-1), k=binder_len)[0].mean())
    log_norm = float(jnp.log(m.sum().astype(jnp.float32)))
    margin = 1.0 - S_bar / log_norm
    return dict(
        S_bar=S_bar,
        log_norm=log_norm,
        margin=margin,
        proxy=float(np.clip(margin, 0.0, 1.0)),
        clipped=bool(margin <= 0.0 or margin >= 1.0),
        pool=int(np.prod(S.shape)),
        max_p_contact=float(p_contact.max()),
        frac_pairs_contacting=float((p_contact > 0.5).mean()),
    )


def predicted_epitope(output, binder_len, cutoff=10.0):
    ca = np.asarray(output.backbone_coordinates)[:, 1]
    d = np.linalg.norm(ca[:binder_len, None] - ca[None, binder_len:], axis=-1)
    return sorted(int(j) for j in np.where(d.min(axis=0) < cutoff)[0])


def pssm(seq):
    from mosaic.common import TOKENS

    return jax.nn.one_hot(
        jnp.array([TOKENS.index(a) for a in seq], dtype=jnp.int32), len(TOKENS)
    )


# ----------------------------------------------------------------------- report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    ap.add_argument("--system", default="dio3", choices=sorted(SYSTEMS))
    ap.add_argument("--recycling", type=int, default=4,
                    help="design-path recycling. The campaign screened at 4; "
                         "run_design.py optimises at 1-3, so try both")
    ap.add_argument("--complex-recycling", type=int, default=4)
    a = ap.parse_args()

    sysd = load_system(a.system)
    n = len(sysd["binder_seq"])
    print(f"{sysd['name']}  ({sysd['note']})")
    print(f"target {len(sysd['target_seq'])} aa   binder {n} aa")
    print(f"crystal epitope ({len(sysd['epitope'])}): {sysd['epitope']}")
    print(f"decoy site      ({len(sysd['decoy'])}): {sysd['decoy']}")
    print(f"hotspot subset  ({len(sysd['core'])}): {sysd['core']}\n")

    from mosaic.models.af2 import AlphaFold2
    from mosaic.structure_prediction import TargetChain
    from mosaic.losses.structure_prediction import IPTMLoss, BinderTargetIPTM

    t0 = time.time()
    model = AlphaFold2(multimer=True)
    print(f"af2 loaded in {time.time() - t0:.1f}s")

    report = {
        "system": {k: v for k, v in sysd.items() if k != "target_seq"},
        "recycling": {"design": a.recycling, "complex": a.complex_recycling},
        "rows": [],
    }
    variants = [
        ("none", None),
        ("crystal", sysd["epitope"]),
        ("hotspot", sysd["core"]),
        ("decoy", sysd["decoy"]),
    ]

    def emit(mode, seq_name, output, recycling):
        pred = set(predicted_epitope(output, n))
        crystal = set(sysd["epitope"])
        iou = len(pred & crystal) / max(1, len(pred | crystal))
        _, a1 = IPTMLoss()(jnp.zeros((n, 20)), output, KEY)
        _, a2 = BinderTargetIPTM()(jnp.zeros((n, 20)), output, KEY)
        for vname, ep in variants:
            d = decompose(output, n, epitope_idx=ep)
            d.update(
                mode=mode, seq=seq_name, variant=vname, recycling=recycling,
                iptm=float(a1["iptm"]), bt_iptm=float(a2["bt_iptm"]),
                iou_with_crystal=iou, n_pred_contacts=len(pred),
            )
            report["rows"].append(d)

    # --- complex: both chains real, AF2's best shot at the true structure
    print("\n=== complex (target_only_features, both real chains) ===")
    feats, _ = model.target_only_features(
        [
            TargetChain(sequence=sysd["binder_seq"], use_msa=False),
            TargetChain(sequence=sysd["target_seq"], use_msa=False),
        ]
    )
    t0 = time.time()
    out = model.model_output(
        features=feats, recycling_steps=a.complex_recycling,
        model_idx=MODEL_IDX, key=KEY,
    )
    jax.block_until_ready(out.distogram_logits)
    print(f"forward: {time.time() - t0:.1f}s   mean plddt {float(out.plddt.mean()):.3f}")
    emit("complex", "crystal_pair", out, a.complex_recycling)

    # --- design: the binder is a stub, the sequence arrives as a PSSM
    print("\n=== design (binder_features + PSSM) ===")
    feats_d, _ = model.binder_features(
        n, [TargetChain(sequence=sysd["target_seq"], use_msa=False)]
    )
    rng = np.random.default_rng(0)
    seqs = {
        "real": sysd["binder_seq"],
        "scramble": "".join(rng.permutation(list(sysd["binder_seq"]))),
        "polyval": "V" * n,
    }
    for name, s in seqs.items():
        t0 = time.time()
        o = model.model_output(
            PSSM=pssm(s), features=feats_d, recycling_steps=a.recycling,
            model_idx=MODEL_IDX, key=KEY,
        )
        jax.block_until_ready(o.distogram_logits)
        print(f"forward[{name}]: {time.time() - t0:.1f}s   "
              f"plddt {float(o.plddt.mean()):.3f}")
        emit("design", name, o, a.recycling)

    # --- the table
    hdr = (f"{'mode':<8} {'seq':<13} {'epitope':<8} {'pool':>6} {'S_bar':>7} "
           f"{'margin':>8} {'proxy':>7} {'flat?':>6} {'maxPc':>7} {'iptm':>6}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in report["rows"]:
        print(f"{r['mode']:<8} {r['seq']:<13} {r['variant']:<8} {r['pool']:>6} "
              f"{r['S_bar']:7.3f} {r['margin']:8.3f} {r['proxy']:7.4f} "
              f"{'YES' if r['clipped'] else '':>6} {r['max_p_contact']:7.3f} "
              f"{r['iptm']:6.3f}")
    print(f"\nlog(n_contact_bins) = {report['rows'][0]['log_norm']:.4f}  "
          "(margin <= 0 means proxy AND gradient are exactly 0)")

    # --- the design-path gradient, which is what actually drives a campaign
    print("\n=== gradient through the design path ===")
    from mosaic.optimizers import _eval_loss_and_grad
    from mosaic.losses.structure_prediction import DistogramIPTMProxy

    x = np.asarray(pssm(sysd["binder_seq"]), dtype=np.float32)
    x = 0.9 * x + 0.1 / 20.0
    print(f"{'epitope':<9} {'compile+1st':>12} {'per step':>10} {'loss':>9} {'|grad|':>10}")
    grads, timing = {}, {}
    for vname, ep in variants:
        term = model.build_loss(
            loss=DistogramIPTMProxy(epitope_idx=ep), features=feats_d,
            recycling_steps=a.recycling,
        )
        t0 = time.time()
        (v, _), g = _eval_loss_and_grad(term, x, KEY)
        jax.block_until_ready(g)
        compile_s = time.time() - t0
        t0 = time.time()
        for i in range(3):
            (v, _), g = _eval_loss_and_grad(term, x, jax.random.fold_in(KEY, i))
        jax.block_until_ready(g)
        step_s = (time.time() - t0) / 3
        gn = float(jnp.linalg.norm(g))
        grads[vname] = np.asarray(g).ravel()
        timing[vname] = dict(compile_s=compile_s, step_s=step_s,
                             loss=float(v), grad_norm=gn)
        print(f"{vname:<9} {compile_s:12.1f}s {step_s:10.2f}s {float(v):9.4f} {gn:10.3e}")
    report["timing"] = timing

    def cos(x_, y_):
        nx, ny = np.linalg.norm(x_), np.linalg.norm(y_)
        return float(x_ @ y_ / (nx * ny)) if nx and ny else float("nan")

    print("\ngradient cosine similarity (nan == one of them is identically zero)")
    names = [v[0] for v in variants]
    print(f"{'':<9}" + "".join(f"{m:>10}" for m in names))
    cosines = {}
    for i in names:
        print(f"{i:<9}" + "".join(f"{cos(grads[i], grads[j]):10.3f}" for j in names))
        cosines[i] = {j: cos(grads[i], grads[j]) for j in names}
    report["grad_cosine"] = cosines

    outp = Path(a.out) / "iptm_proxy_bench.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
