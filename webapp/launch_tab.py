"""Tab — configure and launch a design campaign.

Every parameter the pipeline accepts is exposed here. Widgets are generated from
the catalogs in design_config.py, so adding a model or loss term there makes it
appear here with no change to this file.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import design_config as dc  # noqa: E402

import cluster  # noqa: E402
import store  # noqa: E402
import viewer  # noqa: E402
from ui_helpers import params_block, parse_ranges, show_selection, widget  # noqa: E402


def _new_target_form() -> None:
    """Create a target from a pasted sequence (item: target name + sequence)."""
    with st.expander("New target — paste a sequence"):
        name = st.text_input("Name", key="nt_name",
                             help="used for the FASTA, MSA and result filenames")
        seq = st.text_area(
            "Sequence", key="nt_seq", height=140,
            help="Trim to the domain a binder can actually reach. For a membrane "
                 "protein use only the extracellular part — a transmembrane helix "
                 "with no membrane gets buried and distorts the fold.")
        clean = "".join(seq.split()).upper()
        if clean:
            from mosaic_tokens import TOKENS, ANALOGS  # local shim, see below
            bad = sorted({c for c in clean if c not in TOKENS and c not in ANALOGS})
            subs = sorted({c for c in clean if c in ANALOGS})
            st.caption(f"{len(clean)} residues")
            if subs:
                st.info("Will be substituted at run time: " + ", ".join(
                    f"{c}->{ANALOGS[c]}" for c in subs) +
                    ". U->C (selenocysteine to cysteine) preserves the fold but "
                    "changes active-site chemistry.")
            if bad:
                st.error(f"Unsupported residues: {', '.join(bad)}")
        if st.button("Create target", disabled=not (name.strip() and clean)):
            fa = store.sub("targets") / f"{name.strip()}.fasta"
            if fa.exists():
                st.error(f"{fa.name} already exists.")
            else:
                fa.write_text(f">{name.strip()}\n{clean}\n")
                st.success(f"Created {fa}")
                st.rerun()


def _msa_launcher(tgt, res: dict) -> None:
    st.warning(
        "No MSA for this target. Predictions are markedly weaker without one, "
        "and Boltz/OpenFold3/Protenix will otherwise query the public ColabFold "
        "server on every featurization.")
    if st.button("Run MSA search (20 min – 2 h)"):
        ok, msg, jid = cluster.submit_script(
            "singularity/msa-search.sbatch",
            {"TARGET_FASTA": str(tgt.fasta), "TARGET_NAME": tgt.name,
             "MSA_OUT": str(store.sub("msa"))},
            **res)
        (st.success if ok else st.error)(f"job {jid}" if ok else "submit failed")
        st.code(msg)


def _consume_pending_proposal(cfg: dict) -> str | None:
    """Apply a proposal set shipped from the Generate tab.

    Applied once, then cleared, so later edits in this tab are not overwritten
    on every rerun. Carries the epitope as well as the sequences — transferring
    only the letters is what left refinement re-deriving a binding pose.
    """
    pend = st.session_state.pop("pending_proposal", None)
    if not pend:
        return None
    cfg["binder"]["init_fasta"] = pend["fasta"]
    cfg["binder"]["length"] = int(pend["binder_length"])
    st.session_state["seed_fasta"] = pend["fasta"]
    st.session_state["epitope"] = ",".join(
        str(i + 1) for i in pend.get("epitope_idx", []))
    st.session_state["_proposal_banner"] = pend
    return pend["name"]


def render(cfg: dict) -> dict:
    cfg.setdefault("cluster", dc.default_config()["cluster"])
    cfg.setdefault("mpnn", {"weights": "soluble", "backbone_noise": 0.0, "terms": []})
    _consume_pending_proposal(cfg)

    banner = st.session_state.get("_proposal_banner")
    if banner:
        st.success(
            f"Seeded from **{banner['name']}** ({banner['generator']}): "
            f"{len(banner.get('epitope_idx', []))} epitope residues and a "
            f"{banner['binder_length']}-residue binder carried over.")

    # ------------------------------------------------------------- target
    st.subheader("1. Target")
    _new_target_form()
    targets = store.list_targets()
    if not targets:
        st.info("No targets yet — create one above.")
        return cfg

    names = [t.name for t in targets]
    cur = cfg["target"].get("fasta")
    idx = next((i for i, t in enumerate(targets) if str(t.fasta) == str(cur)), 0)
    tgt = next(t for t in targets if t.name == st.selectbox("Target", names, index=idx, key="launch_target"))

    c = st.columns(4)
    c[0].metric("Residues", tgt.length)
    c[1].metric("MSA depth", f"{tgt.msa_depth:,}" if tgt.msa_depth else "none")
    c[2].metric("Structure", "yes" if tgt.structure else "no")
    use_msa = c[3].checkbox("Use MSA", value=bool(tgt.msa), disabled=tgt.msa is None)
    if tgt.msa is None:
        _msa_launcher(tgt, {"account": cfg["cluster"]["account"],
                            "partition": cfg["cluster"]["partition"]})
    cfg["target"] = {"fasta": str(tgt.fasta),
                     "msa": str(tgt.msa) if tgt.msa else None,
                     "use_msa": bool(use_msa and tgt.msa)}
    st.code(tgt.sequence, language=None)

    # ------------------------------------------------------------- binder
    st.subheader("2. Binder")
    b = st.columns(4)
    length = b[0].number_input("Length", 10, 300, int(cfg["binder"]["length"]), 5)
    no_cys = b[1].checkbox(
        "Forbid cysteine", value=cfg["binder"]["no_cys"],
        help="A reparameterization, not a penalty: the optimizer works over 19 "
             "tokens, so cysteine is impossible rather than discouraged.")
    init_noise = b[2].slider("Seed noise", 0.0, 0.5,
                             float(cfg["binder"].get("init_noise", 0.15)), 0.05)
    if "seed_fasta" not in st.session_state:
        st.session_state["seed_fasta"] = cfg["binder"].get("init_fasta") or ""
    seed_from = st.text_input(
        "Seed from FASTA (optional)", key="seed_fasta",
        help="e.g. BoltzGen proposals from the Generate tab. Note: with default "
             "settings the optimizer keeps only ~15% of the seed.")
    cfg["binder"] = {"length": int(length), "no_cys": bool(no_cys),
                     "init_fasta": seed_from.strip() or None,
                     "init_noise": float(init_noise)}

    # -------------------------------------------------------- binding site
    st.subheader("3. Binding site")
    st.caption(
        "Where on the target should the binder land, and which binder residues "
        "should do the binding. Leave blank to let the optimizer choose.")
    bs = st.columns(2)
    with bs[0]:
        ep_txt = st.text_input(
            "Epitope — target residues", key="epitope",
            value=st.session_state.get("epitope", ""),
            placeholder="e.g. 95-110, 143, 150-158",
            help="1-based positions in the sequence shown above.")
        ep, errs = parse_ranges(ep_txt, 1, tgt.length)
        for e in errs:
            st.error(f"Epitope: {e}")
        show_selection(tgt.sequence, ep, "Epitope")
    with bs[1]:
        pt_txt = st.text_input(
            "Paratope — binder residues", key="paratope",
            value=st.session_state.get("paratope", ""),
            placeholder="e.g. 20-35",
            help="1-based positions in the designed binder.")
        pt, perrs = parse_ranges(pt_txt, 1, int(length))
        for e in perrs:
            st.error(f"Paratope: {e}")
        if pt:
            st.caption(f"Paratope — {len(pt)} of {length} binder positions")

    if tgt.structure and (ep or pt):
        with st.expander("3D — where this binds", expanded=False):
            viewer.legend([("target", viewer.C_TARGET),
                           ("epitope", viewer.C_HOTSPOT)])
            viewer.show(viewer.target_view(tgt.structure, ep, clickable=True))
            st.caption("Click a residue to label it with its number.")
    elif not tgt.structure:
        st.caption("Predict a structure in the Generate tab to pick the "
                   "epitope visually.")

    if ep:
        st.info(
            "Positions are relative to the sequence above. If you trimmed the "
            "target, they are **not** the original numbering — check the "
            "residue letters echoed above against your intended epitope.")

    # ------------------------------------------------------------- models
    st.subheader("4. Structure models")
    st.caption("Each enabled model must be satisfied. Two is a sensible ceiling.")
    selected = {m["name"]: m for m in cfg["models"]}
    new_models = []
    for name, meta in dc.STRUCTURE_MODELS.items():
        on = st.checkbox(f"**{meta['label']}**", value=name in selected, key=f"m_{name}")
        st.caption(meta["blurb"])
        if not on:
            continue
        prev = selected.get(name, {})
        wcol, _ = st.columns([1, 3])
        with wcol:
            w = st.number_input("weight", 0.0, 100.0, float(prev.get("weight", 1.0)),
                                0.25, key=f"mw_{name}")
        params = params_block(f"mp_{name}", meta["params"],
                              dict(prev.get("params") or {}))
        if not meta["accepts_msa"] and cfg["target"]["use_msa"]:
            st.info(f"{meta['label']} runs single-sequence — its wrapper rejects "
                    "MSAs, so it sees a different target than the others.")
        new_models.append({"name": name, "weight": float(w), "params": params})
        st.divider()
    cfg["models"] = new_models

    # ------------------------------------------------------------- losses
    st.subheader("5. Loss terms")
    sel_l = {l["name"]: l for l in cfg["losses"]}
    new_losses = []
    groups: dict[str, list[str]] = {}
    for lname, meta in dc.LOSS_TERMS.items():
        groups.setdefault(meta["group"], []).append(lname)

    for group, members in groups.items():
        note = (" — read the structure/confidence modules, which JAX otherwise "
                "prunes. Enabling any of these is substantially slower."
                ) if group == "Confidence" else ""
        with st.expander(f"{group}{note}", expanded=(group == "Contact")):
            for lname in members:
                meta = dc.LOSS_TERMS[lname]
                on = st.checkbox(lname, value=lname in sel_l, key=f"l_{lname}")
                st.caption(meta["blurb"])
                if not on:
                    continue
                prev = sel_l.get(lname, {})
                wcol, _ = st.columns([1, 3])
                with wcol:
                    w = st.number_input("weight", -100.0, 100.0,
                                        float(prev.get("weight", meta["default_weight"])),
                                        0.25, key=f"lw_{lname}")
                # idxlist params come from the Binding site section above, so they
                # are not duplicated here.
                params = params_block(f"lp_{lname}", meta["params"],
                                      dict(prev.get("params") or {}),
                                      skip=("paratope_idx", "epitope_idx"))
                if lname == "BinderTargetContact":
                    # Converted to 0-based here: epitope_idx indexes the target
                    # slice of the distogram (structure_prediction.py:281).
                    params["epitope_idx"] = [p - 1 for p in ep] or None
                    params["paratope_idx"] = [p - 1 for p in pt] or None
                new_losses.append({"name": lname, "weight": float(w),
                                   "params": params})
    cfg["losses"] = new_losses

    if (ep or pt) and not any(l["name"] == "BinderTargetContact" for l in new_losses):
        st.warning("A binding site is specified but BinderTargetContact is off — "
                   "nothing will use it.")

    # --------------------------------------------------------- ProteinMPNN
    st.subheader("6. ProteinMPNN (inverse folding)")
    st.caption(
        "Scores the sequence against the backbone the structure model predicted. "
        "The standard de novo filter — designs that fail it rarely express. Reads "
        "the structure module, so it carries the same cost as a confidence term.")
    mp = cfg["mpnn"]
    mcols = st.columns(2)
    with mcols[0]:
        wsel = st.selectbox("Weights", list(dc.MPNN_WEIGHTS),
                            index=list(dc.MPNN_WEIGHTS).index(mp.get("weights", "soluble")),
                            format_func=lambda k: dc.MPNN_WEIGHTS[k])
    with mcols[1]:
        bnoise = st.number_input("backbone_noise", 0.0, 1.0,
                                 float(mp.get("backbone_noise", 0.0)), 0.01)
    sel_m = {t["name"]: t for t in mp.get("terms", [])}
    new_mpnn = []
    for tname, meta in dc.MPNN_TERMS.items():
        on = st.checkbox(tname, value=tname in sel_m, key=f"mp_{tname}")
        st.caption(meta["blurb"])
        if not on:
            continue
        prev = sel_m.get(tname, {})
        wcol, _ = st.columns([1, 3])
        with wcol:
            w = st.number_input("weight", -100.0, 100.0,
                                float(prev.get("weight", meta["default_weight"])),
                                0.25, key=f"mpw_{tname}")
        params = params_block(f"mpp_{tname}", meta["params"],
                              dict(prev.get("params") or {}))
        new_mpnn.append({"name": tname, "weight": float(w), "params": params})
    cfg["mpnn"] = {"weights": wsel, "backbone_noise": float(bnoise),
                   "terms": new_mpnn}

    # ---------------------------------------------------- sequence models
    st.subheader("7. Sequence models")
    sel_s = {s["name"]: s for s in cfg.get("sequence_models", [])}
    new_seq = []
    for sname, meta in dc.SEQUENCE_MODELS.items():
        label = meta["label"] + (" — antibodies only" if meta["antibody_only"] else "")
        on = st.checkbox(label, value=sname in sel_s, key=f"s_{sname}")
        st.caption(meta["blurb"])
        if not on:
            continue
        prev = sel_s.get(sname, {})
        wcol, _ = st.columns([1, 3])
        with wcol:
            w = st.number_input("weight", 0.0, 100.0, float(prev.get("weight", 0.5)),
                                0.1, key=f"sw_{sname}")
        params = params_block(f"sp_{sname}", meta["params"],
                              dict(prev.get("params") or {}))
        new_seq.append({"name": sname, "weight": float(w), "params": params})
    cfg["sequence_models"] = new_seq

    # ---------------------------------------------------------- optimizer
    st.subheader("8. Optimizer")
    for stage in ("soft", "sharp"):
        d = cfg["optimizer"][stage]
        with st.expander(f"{stage} stage", expanded=(stage == "soft")):
            c1 = st.columns(3)
            n = c1[0].number_input("n_steps", 1, 500, int(d["n_steps"]), 5,
                                   key=f"o_{stage}_n")
            ss = c1[1].number_input("stepsize", 0.001, 1.0, float(d["stepsize"]),
                                    0.005, format="%.3f", key=f"o_{stage}_s")
            mo = c1[2].number_input("momentum", 0.0, 0.99, float(d["momentum"]),
                                    0.05, key=f"o_{stage}_m")
            c2 = st.columns(3)
            sc = c2[0].number_input("scale", 0.1, 10.0, float(d.get("scale", 1.0)),
                                    0.1, key=f"o_{stage}_sc")
            lg = c2[1].checkbox("logspace (mirror descent)",
                                value=bool(d.get("logspace", False)),
                                key=f"o_{stage}_lg")
            with c2[2]:
                mg = widget(f"o_{stage}_g", "max_gradient_norm",
                            dc.OPTIMIZER_PARAMS["max_gradient_norm"],
                            d.get("max_gradient_norm"))
            cfg["optimizer"][stage] = {
                "n_steps": int(n), "stepsize": float(ss), "momentum": float(mo),
                "scale": float(sc), "logspace": bool(lg),
                "max_gradient_norm": mg}

    # ------------------------------------------------------------ cluster
    st.subheader("9. Cluster")
    cl = cfg["cluster"]
    accts = cluster.accounts()
    parts = cluster.partitions()
    r1 = st.columns(4)
    account = r1[0].selectbox(
        "Account", accts,
        index=accts.index(cl["account"]) if cl["account"] in accts else 0,
        help="Only accounts you can actually submit under are listed — an "
             "association with MaxSubmit=0 rejects everything at submit time.")
    partition = r1[1].selectbox(
        "Partition", parts,
        index=parts.index(cl["partition"]) if cl["partition"] in parts else 0)
    n_tasks = r1[2].number_input("GPUs (array tasks)", 1, 512,
                                 int(cl.get("n_tasks", 16)), 1)
    batch = r1[3].number_input("Designs per GPU", 1, 8,
                               int(cfg["run"].get("batch", 2)), 1,
                               help="Above 4 is untested: GPU memory and a "
                                    "host-side simplex projection that scales "
                                    "with batch are both plausible ceilings.")
    r2 = st.columns(4)
    throttle = r2[0].number_input("Max concurrent (0 = none)", 0, 512,
                                  int(cl.get("throttle", 0)), 1)
    time_limit = r2[1].text_input("Time limit", cl.get("time_limit", "08:00:00"))
    mem = r2[2].text_input("Memory", cl.get("mem", "128G"))
    cpus = r2[3].number_input("CPUs per task", 1, 64, int(cl.get("cpus", 8)), 1)
    cfg["cluster"] = {"account": account, "partition": partition,
                      "n_tasks": int(n_tasks), "throttle": int(throttle),
                      "time_limit": time_limit, "mem": mem, "cpus": int(cpus)}
    cfg["run"] = {"batch": int(batch), "seed": 0}

    # --------------------------------------------------------- submission
    st.subheader("10. Review and submit")
    problems = dc.validate(cfg)
    est_min = dc.estimate_seconds(cfg) / 60.0
    m = st.columns(4)
    m[0].metric("Designs", int(n_tasks) * int(batch))
    m[1].metric("Est. per task", f"{est_min:.0f} min")
    m[2].metric("Structure module", "needed" if dc.uses_confidence(cfg) else "pruned")
    m[3].metric("Loss terms", len(cfg["losses"]) + len(cfg["mpnn"]["terms"]))

    if dc.uses_confidence(cfg):
        st.warning("A confidence or ProteinMPNN term is enabled, so JAX cannot "
                   "prune the structure module. The estimate includes that.")
    for p in problems:
        st.error(p)

    with st.expander("Resolved config (saved and run verbatim)"):
        st.json(cfg)

    default_name = f"{tgt.name}_{'_'.join(m['name'] for m in cfg['models']) or 'none'}"
    out_name = st.text_input("Campaign name", value=default_name)
    out_dir = store.sub("designs") / out_name

    if st.button("Submit campaign", type="primary", disabled=bool(problems)):
        cfg_path = store.sub("configs") / f"{out_name}_{int(time.time())}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2))
        out_dir.mkdir(parents=True, exist_ok=True)
        ok, msg, job_id = cluster.submit_campaign(
            cfg_path, out_dir, int(n_tasks), throttle=int(throttle) or None,
            partition=partition, account=account, time_limit=time_limit,
            mem=mem, cpus=int(cpus))
        if ok:
            st.success(f"Submitted job {job_id}")
            st.session_state["last_job"] = {
                "id": job_id, "n_tasks": int(n_tasks),
                "models": [m["name"] for m in cfg["models"]],
                "expect_msa": cfg["target"]["use_msa"], "out": str(out_dir)}
            st.info("Open **Monitor** and confirm from the log that it ran what "
                    "you asked — the failure modes here are silent.")
        else:
            st.error("Submission failed")
        st.code(msg)

    return cfg
