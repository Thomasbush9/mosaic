"""Tab — generative design (BoltzGen) and target structure prediction.

Generative models are a separate stage, not an option in the campaign objective.
BoltzGen and Proteina implement no binder_features/build_loss — there is no
question to differentiate — so they cannot be optimized against. They run first,
proposing starting points that the campaign then refines via `Seed from FASTA`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import design_config as dc  # noqa: E402

import cluster  # noqa: E402
import store  # noqa: E402
from ui_helpers import params_block  # noqa: E402


def _structure_section(tgt, account: str) -> None:
    st.subheader("1. Target structure")
    st.caption(
        "BoltzGen designs against **geometry**, not sequence, so it needs a CIF. "
        "Predicted with target-only features, which include real sidechain "
        "reference atoms — binder features stub them to UNK/G because sidechains "
        "are not differentiably defined for a soft sequence, and a sidechain-free "
        "structure is a useless design target.")
    if tgt.structure:
        st.success(f"Structure exists: `{tgt.structure.name}`")
    else:
        st.info("No structure yet for this target.")
    c = st.columns(3)
    rec = c[0].number_input("recycling_steps", 1, 20, 4, 1,
                            help="Higher than during design — this is a one-off "
                                 "prediction, and the examples use 4-20 for "
                                 "final folding versus 1 while optimizing.")
    samp = c[1].number_input("sampling_steps", 5, 200, 50, 5)
    if c[2].button("Predict structure", disabled=tgt.msa is None):
        ok, msg, jid = cluster.submit_script(
            "singularity/predict-target.sbatch",
            {"TARGET_FASTA": str(tgt.fasta),
             "TARGET_MSA": str(tgt.msa) if tgt.msa else "",
             "OUT_CIF": str(store.sub("targets") / f"{tgt.name}.cif")},
            account=account)
        (st.success if ok else st.error)(f"job {jid}" if ok else "submit failed")
        st.code(msg)
    if tgt.msa is None:
        st.caption("Build an MSA first — a single-sequence structure is a weak "
                   "basis for structure-conditioned design.")


def render(cfg: dict) -> None:
    targets = store.list_targets()
    if not targets:
        st.info("No targets yet — create one in the Launch tab.")
        return

    names = [t.name for t in targets]
    cur = cfg.get("target", {}).get("fasta")
    idx = next((i for i, t in enumerate(targets) if str(t.fasta) == str(cur)), 0)
    tgt = next(t for t in targets if t.name == st.selectbox("Target", names, index=idx, key="gen_target"))
    account = cfg.get("cluster", {}).get("account", cluster.DEFAULT_ACCOUNT)

    _structure_section(tgt, account)
    st.divider()

    st.subheader("2. Generate binder backbones")
    for gname, meta in dc.GENERATIVE_MODELS.items():
        enabled = meta.get("enabled", True)
        st.markdown(f"**{meta['label']}**" + ("" if enabled else " — not wired up yet"))
        st.caption(meta["blurb"])
        if not enabled:
            st.divider()
            continue

        prev = st.session_state.get(f"gen_{gname}", {})
        params = params_block(f"g_{gname}", meta["params"], dict(prev))
        st.session_state[f"gen_{gname}"] = params

        needs = meta.get("needs_structure") and tgt.structure is None
        if needs:
            st.warning("Predict the target structure first (section 1).")

        out_dir = store.sub("designs") / f"{tgt.name}_{gname}"
        st.caption(f"Output: `{out_dir}`")

        # A secondary-structure preview, because it is the main design decision
        # here and easy to get wrong for a given binder length.
        if gname == "boltzgen":
            n = int(params.get("binder_length", 80))
            h = int(params.get("n_helices", 3))
            lp = int(params.get("loop_length", 4))
            total_helix = n - (h - 1) * lp
            if total_helix < h * 6:
                st.error(f"{n} residues is too short for {h} helices with "
                         f"{lp}-residue loops.")
            else:
                base, extra = divmod(total_helix, h)
                parts = []
                for i in range(h):
                    parts.append("H" * (base + (1 if i < extra else 0)))
                    if i < h - 1:
                        parts.append("L" * lp)
                st.code("".join(parts), language=None)

        if st.button(f"Run {meta['label']}", key=f"go_{gname}", disabled=needs):
            ok, msg, jid = cluster.submit_script(
                "singularity/boltzgen.sbatch",
                {"TARGET_CIF": str(tgt.structure) if tgt.structure else "",
                 "BINDER_LENGTH": str(params.get("binder_length", 80)),
                 "NUM_DESIGNS": str(params.get("num_designs", 8)),
                 "OUT_DIR": str(out_dir)},
                account=account)
            (st.success if ok else st.error)(f"job {jid}" if ok else "submit failed")
            st.code(msg)
        st.divider()

    # ---------------------------------------------------------- proposals
    st.subheader("3. Proposals")
    fastas = sorted(store.sub("designs").glob("*/*_designs.fasta"))
    if not fastas:
        st.info("No generated proposals yet.")
        return
    pick = st.selectbox("Proposal set", [str(p) for p in fastas], key="gen_proposal")
    p = Path(pick)
    seqs = [ln.strip() for ln in p.read_text().splitlines()
            if ln.strip() and not ln.startswith(">")]
    st.caption(f"{len(seqs)} proposals, {len(seqs[0]) if seqs else 0} residues each")
    st.dataframe([{"#": i, "length": len(s), "sequence": s}
                  for i, s in enumerate(seqs)], hide_index=True, width="stretch")

    meta_json = p.with_suffix("").with_suffix("").parent / "boltzgen_designs.json"
    if meta_json.exists():
        with st.expander("Generation settings"):
            st.json(json.loads(meta_json.read_text()))

    st.info(
        f"To refine these, paste this path into **Launch → Binder → Seed from "
        f"FASTA**:\n\n`{p}`\n\n"
        "Honest caveat: with default optimizer settings the refinement keeps "
        "only ~15% of the seed — it largely walks away and lands where it would "
        "have from noise. Shorter soft stages, or anchoring most positions, are "
        "the levers worth trying.")
