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
import proposals as P  # noqa: E402
import viewer  # noqa: E402
from ui_helpers import params_block, parse_ranges, resource_picker, show_selection  # noqa: E402


def _structure_section(tgt, res: dict) -> None:
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
            **res)
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
    res = resource_picker("gen_res", gpu=True,
                          defaults={"account": account,
                                    "partition": cfg.get("cluster", {}).get("partition", "kempner_h100")})

    _structure_section(tgt, res)
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

        hot: list[int] = []
        if meta.get("hotspots"):
            htxt = st.text_input(
                "Hotspots — target residues to engage", key=f"hot_{gname}",
                placeholder="e.g. 95-110, 143",
                help="1-based positions in the target. Proteina conditions on "
                     "these, so the epitope is chosen rather than discovered.")
            hot, herrs = parse_ranges(htxt, 1, tgt.length)
            for e in herrs:
                st.error(f"Hotspots: {e}")
            show_selection(tgt.sequence, hot, "Hotspots")
            if tgt.structure:
                with st.expander("3D — pick hotspots off the structure",
                                 expanded=bool(hot)):
                    st.caption(
                        "Click a residue to label it with its number, then type "
                        "that into the box above. py3Dmol renders in an iframe "
                        "with no channel back to Python, so clicks cannot fill "
                        "the field directly.")
                    viewer.legend([("target", viewer.C_TARGET),
                                   ("hotspots", viewer.C_HOTSPOT)])
                    viewer.show(viewer.target_view(
                        tgt.structure, hot, chain=params.get("target_chain", "A")))
            if not hot:
                st.caption(
                    "No hotspots — the generator chooses where to bind."
                    if gname == "proteina" else
                    "No hotspots — BoltzGen sees the whole target and picks a site.")
            elif gname == "boltzgen":
                st.caption(
                    "BoltzGen has no attractive hotspot term, so these work by "
                    "**cropping** the target to a pocket around them — the "
                    "binder has nowhere else to go. The job log reports how many "
                    "residues survive; if that is most of the target, reduce the "
                    "shell or the crop is not constraining anything.")

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
            exports = {
                "TARGET_CIF": str(tgt.structure) if tgt.structure else "",
                "TARGET_FASTA": str(tgt.fasta),
                "TARGET_NAME": tgt.name,
                "BINDER_LENGTH": str(params.get("binder_length", 80)),
                "NUM_DESIGNS": str(params.get("num_designs", 8)),
                "TARGET_CHAIN": str(params.get("target_chain", "A")),
                "OUT_DIR": str(out_dir),
            }
            if meta.get("hotspots"):
                exports["HOTSPOTS"] = ",".join(str(h) for h in hot)
                if "hotspot_shell" in params:
                    exports["HOTSPOT_SHELL"] = str(params["hotspot_shell"])
            script = ("singularity/proteina.sbatch" if gname == "proteina"
                      else "singularity/boltzgen.sbatch")
            ok, msg, jid = cluster.submit_script(script, exports, **res)
            (st.success if ok else st.error)(f"job {jid}" if ok else "submit failed")
            st.code(msg)
        st.divider()

    # ---------------------------------------------------------- proposals
    st.subheader("3. Proposal sets")
    st.caption(
        "Each set carries its sequences, one complex per design, and the "
        "epitope those complexes actually use — so refinement can aim at the "
        "same interface instead of re-deriving a pose from scratch.")
    sets = P.list_sets(store.sub("designs"))
    if not sets:
        st.info("No proposal sets yet. Generated sets appear here automatically.")
        return

    names = [n for n, _ in sets]
    pick = st.selectbox("Proposal set", names, index=len(names) - 1,
                        key="gen_proposal")
    ps = dict(sets)[pick]

    m = st.columns(4)
    m[0].metric("Generator", ps.generator)
    m[1].metric("Designs", ps.n_designs)
    m[2].metric("Binder length", ps.binder_length)
    m[3].metric("Epitope residues", len(ps.epitope_idx))
    for n in ps.notes:
        st.caption(n)

    if ps.epitope_idx:
        tseq = next((t.sequence for t in targets if t.name == ps.target_name), "")
        if tseq:
            show_selection(tseq, [i + 1 for i in ps.epitope_idx], "Epitope")

    st.dataframe([{"#": i, "length": len(s), "sequence": s}
                  for i, s in enumerate(ps.sequences)],
                 hide_index=True, width="stretch")


    with st.expander("Generation settings"):
        st.json(ps.params)

    # ---------------------------------------------- two routes forward
    st.divider()
    st.subheader("4. What next")
    st.caption(
        "Two ways to combine these candidates with the structure models, both "
        "legitimate. Screening matches mosaic's own generate->refold->rank "
        "pipeline; optimizing gradient-refines the sequences under a full "
        "multi-objective loss.")

    tab_screen, tab_optimize = st.tabs(["Screen & rank", "Optimize (refine)"])

    with tab_screen:
        st.markdown(
            "Refold each candidate with a chosen model and rank by confidence. "
            "Nothing is optimized — the generator picked the sequence, and the "
            "question is which ones fold into a confident complex.")
        sc = st.columns(3)
        screen_model = sc[0].selectbox(
            "Refold with", ["boltz2", "boltz1", "af2", "of3", "protenix"],
            index=2 if ps.generator == "boltzgen" else 0,
            help="Best chosen DIFFERENT from the generator, so a candidate is "
                 "not judged by the model that made it. AF2 is the default when "
                 "screening BoltzGen output.")
        screen_rec = sc[1].number_input("recycling_steps", 1, 20, 4, 1,
                                        key="screen_rec")
        use_msa_s = sc[2].checkbox("Use target MSA", value=True, key="screen_msa")
        sc2 = st.columns(3)
        inv = sc2[0].checkbox(
            "ProteinMPNN inverse-fold", value=True, key="screen_inv",
            help="Redesign the binder sequence on the folded backbone before "
                 "scoring — mosaic's own pipeline. The generator proposes a "
                 "fold; MPNN proposes the sequence best suited to it.")
        mpnn_w = sc2[1].selectbox("MPNN weights", ["soluble", "vanilla", "abmpnn"],
                                  index=0, key="screen_mpnn_w", disabled=not inv)
        if st.button("Screen this set", type="primary", key="do_screen"):
            tgt = next((t for t in targets if t.name == ps.target_name), None)
            exports = {
                "PROPOSALS": str(store.sub("designs") / pick),
                "MODEL": screen_model,
                "RECYCLING_STEPS": str(int(screen_rec)),
                "OUT_DIR": str(store.sub("designs") / f"{pick}_screen_{screen_model}"),
                "INVERSE_FOLD": "1" if inv else "0",
                "MPNN_WEIGHTS": mpnn_w,
            }
            if use_msa_s and tgt and tgt.msa:
                exports["TARGET_MSA"] = str(tgt.msa)
            ok, msg, jid = cluster.submit_script("singularity/screen.sbatch",
                                                 exports, **res)
            (st.success if ok else st.error)(f"job {jid}" if ok else "submit failed")
            st.code(msg)

    with tab_optimize:
        st.markdown(
            "Carry the sequences AND the epitope into the Launch tab and "
            "gradient-refine under a full loss — e.g. AF2 + ProteinMPNN + ESM-C. "
            "This is a real use of the optimizer: with a strong multi-objective "
            "loss the sequence is improved against several critics at once, so "
            "it will diverge from the seed, and that is the point.")
        if st.button("Use this set in Launch", type="primary", key="ship"):
            st.session_state["pending_proposal"] = {
                "name": pick,
                "fasta": str(store.sub("designs") / pick / P.FASTA),
                "epitope_idx": ps.epitope_idx,
                "binder_length": ps.binder_length,
                "target_name": ps.target_name,
                "generator": ps.generator,
            }
            st.success(f"Loaded '{pick}'. Open **Launch** — the seed FASTA, "
                       "epitope and binder length are filled in. Pick your "
                       "models and losses there (AF2 + ProteinMPNN + ESM-C is a "
                       "good refinement objective).")
