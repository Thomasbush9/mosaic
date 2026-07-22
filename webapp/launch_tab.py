"""Tab 1 — configure and launch a design campaign.

Every parameter the pipeline accepts is exposed here: per-model, per-loss-term,
both optimizer stages, and the cluster request. The widgets are generated from
the catalogs in design_config.py, so adding a loss term there makes it appear
here with no change to this file.
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
from store import SETUP, list_targets  # noqa: E402


def _widget(key: str, name: str, spec: dict, current):
    """One control, driven by the catalog's declared type."""
    t = spec["type"]
    label = name.replace("_", " ")
    if t == "int":
        return st.number_input(label, min_value=spec.get("min", 0),
                               max_value=spec.get("max", 1000),
                               value=int(current if current is not None
                                         else spec.get("default", 0)),
                               step=1, key=key)
    if t == "float":
        return st.number_input(label, min_value=float(spec.get("min", 0.0)),
                               max_value=float(spec.get("max", 1e6)),
                               value=float(current if current is not None
                                           else spec.get("default", 0.0)),
                               step=0.05, format="%.3f", key=key)
    if t == "bool":
        return st.checkbox(label, value=bool(current if current is not None
                                             else spec.get("default", False)),
                           key=key)
    if t == "choice":
        opts = spec["options"]
        cur = current if current in opts else spec.get("default", opts[0])
        return st.selectbox(label, opts, index=opts.index(cur), key=key)
    if t in ("optint", "optfloat"):
        # Optional numerics need an explicit "unset", because None is
        # meaningful — e.g. target_radius=None means "no target", not zero.
        use = st.checkbox(f"set {label}", value=current is not None, key=key + "_on")
        if not use:
            return None
        if t == "optint":
            return st.number_input(label, min_value=spec.get("min", 0),
                                   max_value=spec.get("max", 10000),
                                   value=int(current or spec.get("min", 0)),
                                   step=1, key=key)
        return st.number_input(label, min_value=float(spec.get("min", 0.0)),
                               max_value=float(spec.get("max", 1e6)),
                               value=float(current or spec.get("min", 0.0)),
                               step=0.5, format="%.2f", key=key)
    if t == "idxlist":
        raw = st.text_input(f"{label} (comma-separated indices, blank = all)",
                            value="" if not current else ",".join(map(str, current)),
                            help=spec.get("help", ""), key=key)
        raw = raw.strip()
        if not raw:
            return None
        try:
            return [int(x) for x in raw.replace(" ", "").split(",") if x]
        except ValueError:
            st.warning(f"{label}: could not parse — ignoring")
            return None
    return current


def render(cfg: dict) -> dict:
    st.subheader("1. Target")
    targets = list_targets()
    if not targets:
        st.error(f"No targets found in {SETUP / 'targets'}. Add a FASTA there first.")
        return cfg

    names = [t.name for t in targets]
    cur = cfg["target"].get("fasta")
    idx = next((i for i, t in enumerate(targets) if str(t.fasta) == str(cur)), 0)
    chosen = st.selectbox("Target", names, index=idx)
    tgt = next(t for t in targets if t.name == chosen)

    c1, c2, c3 = st.columns(3)
    c1.metric("Residues", tgt.length)
    c2.metric("MSA depth", f"{tgt.msa_depth:,}" if tgt.msa_depth else "none")
    c3.metric("Structure", "yes" if tgt.structure else "no")

    if tgt.msa is None:
        st.warning(
            "No MSA for this target. Predictions are markedly weaker without one. "
            "Build it once with `singularity/msa-search.sbatch` — 20 min to 2 h "
            "depending on length."
        )
    use_msa = st.checkbox("Use MSA", value=bool(tgt.msa), disabled=tgt.msa is None)
    cfg["target"] = {"fasta": str(tgt.fasta),
                     "msa": str(tgt.msa) if tgt.msa else None,
                     "use_msa": bool(use_msa and tgt.msa)}
    st.code(tgt.sequence[:100] + ("..." if tgt.length > 100 else ""), language=None)

    # ---------------------------------------------------------------- binder
    st.subheader("2. Binder")
    b1, b2 = st.columns(2)
    length = b1.number_input("Length", 10, 300, int(cfg["binder"]["length"]), 5)
    no_cys = b2.checkbox(
        "Forbid cysteine", value=cfg["binder"]["no_cys"],
        help="A reparameterization, not a penalty: the optimizer works over 19 "
             "tokens and cysteine becomes impossible rather than discouraged.")
    seed_from = st.text_input(
        "Seed from FASTA (optional)", value=cfg["binder"].get("init_fasta") or "",
        help="e.g. BoltzGen proposals. Note: with default settings the optimizer "
             "retains only ~15% of the seed, so this currently buys little.")
    init_noise = st.slider(
        "Seed noise", 0.0, 0.5, float(cfg["binder"].get("init_noise", 0.15)), 0.05,
        help="Blend toward uniform. At 0 the start is a hard simplex vertex with "
             "nowhere to move under a projected-gradient step.")
    cfg["binder"] = {"length": int(length), "no_cys": bool(no_cys),
                     "init_fasta": seed_from.strip() or None,
                     "init_noise": float(init_noise)}

    # ---------------------------------------------------------------- models
    st.subheader("3. Structure models")
    st.caption("Each enabled model must be satisfied by the design. Two is a "
               "sensible ceiling — three roughly triples cost for a diminishing "
               "return.")
    selected = {m["name"]: m for m in cfg["models"]}
    new_models = []
    for name, meta in dc.STRUCTURE_MODELS.items():
        on = st.checkbox(f"**{meta['label']}**", value=name in selected, key=f"m_{name}")
        st.caption(meta["blurb"])
        if not on:
            continue
        prev = selected.get(name, {})
        params = dict(prev.get("params") or {})
        cols = st.columns(len(meta["params"]) + 1)
        with cols[0]:
            w = st.number_input("weight", 0.0, 100.0,
                                float(prev.get("weight", 1.0)), 0.25,
                                key=f"mw_{name}")
        for col, (pname, pspec) in zip(cols[1:], meta["params"].items()):
            with col:
                params[pname] = _widget(f"mp_{name}_{pname}", pname, pspec,
                                        params.get(pname, pspec.get("default")))
        if not meta["accepts_msa"] and cfg["target"]["use_msa"]:
            st.info(f"{meta['label']} will run single-sequence — its wrapper "
                    "rejects MSAs, so it sees a different target than the others.")
        new_models.append({"name": name, "weight": float(w), "params": params})
        st.divider()
    cfg["models"] = new_models

    # ---------------------------------------------------------------- losses
    st.subheader("4. Loss terms")
    st.caption("The objective, evaluated against every enabled model's output.")
    sel_l = {l["name"]: l for l in cfg["losses"]}
    new_losses = []
    groups: dict[str, list[str]] = {}
    for lname, meta in dc.LOSS_TERMS.items():
        groups.setdefault(meta["group"], []).append(lname)

    for group, members in groups.items():
        note = ""
        if group == "Confidence":
            note = (" — these read the structure and confidence modules. JAX "
                    "prunes those when nothing reads them, so enabling any of "
                    "these makes the run substantially slower.")
        with st.expander(f"{group}{note}", expanded=(group == "Contact")):
            for lname in members:
                meta = dc.LOSS_TERMS[lname]
                on = st.checkbox(lname, value=lname in sel_l, key=f"l_{lname}")
                st.caption(meta["blurb"])
                if not on:
                    continue
                prev = sel_l.get(lname, {})
                params = dict(prev.get("params") or {})
                ncol = len(meta["params"]) + 1
                cols = st.columns(ncol)
                with cols[0]:
                    w = st.number_input("weight", -100.0, 100.0,
                                        float(prev.get("weight",
                                                       meta["default_weight"])),
                                        0.25, key=f"lw_{lname}")
                for col, (pname, pspec) in zip(cols[1:], meta["params"].items()):
                    with col:
                        params[pname] = _widget(f"lp_{lname}_{pname}", pname, pspec,
                                                params.get(pname,
                                                           pspec.get("default")))
                new_losses.append({"name": lname, "weight": float(w),
                                   "params": params})
    cfg["losses"] = new_losses

    # ------------------------------------------------------- sequence models
    st.subheader("5. Sequence models")
    sel_s = {s["name"]: s for s in cfg.get("sequence_models", [])}
    new_seq = []
    for sname, meta in dc.SEQUENCE_MODELS.items():
        label = meta["label"] + (" — antibodies only" if meta["antibody_only"] else "")
        on = st.checkbox(label, value=sname in sel_s, key=f"s_{sname}")
        st.caption(meta["blurb"])
        if not on:
            continue
        prev = sel_s.get(sname, {})
        params = dict(prev.get("params") or {})
        cols = st.columns(len(meta["params"]) + 1)
        with cols[0]:
            w = st.number_input("weight", 0.0, 100.0, float(prev.get("weight", 0.5)),
                                0.1, key=f"sw_{sname}")
        for col, (pname, pspec) in zip(cols[1:], meta["params"].items()):
            with col:
                params[pname] = _widget(f"sp_{sname}_{pname}", pname, pspec,
                                        params.get(pname, pspec.get("default")))
        new_seq.append({"name": sname, "weight": float(w), "params": params})
    cfg["sequence_models"] = new_seq

    # ------------------------------------------------------------- optimizer
    st.subheader("6. Optimizer")
    st.caption("Soft explores the simplex interior; sharp drives the result "
               "toward real amino acids.")
    for stage, defaults in (("soft", cfg["optimizer"]["soft"]),
                            ("sharp", cfg["optimizer"]["sharp"])):
        with st.expander(f"{stage} stage", expanded=(stage == "soft")):
            c = st.columns(3)
            n = c[0].number_input("n_steps", 1, 500, int(defaults["n_steps"]), 5,
                                  key=f"o_{stage}_n")
            ss = c[1].number_input("stepsize", 0.001, 1.0,
                                   float(defaults["stepsize"]), 0.005,
                                   format="%.3f", key=f"o_{stage}_s")
            mo = c[2].number_input("momentum", 0.0, 0.99,
                                   float(defaults["momentum"]), 0.05,
                                   key=f"o_{stage}_m")
            c2 = st.columns(3)
            sc = c2[0].number_input("scale", 0.1, 10.0,
                                    float(defaults.get("scale", 1.0)), 0.1,
                                    key=f"o_{stage}_sc")
            lg = c2[1].checkbox("logspace (mirror descent)",
                                value=bool(defaults.get("logspace", False)),
                                key=f"o_{stage}_lg")
            with c2[2]:
                mg = _widget(f"o_{stage}_g", "max_gradient_norm",
                             dc.OPTIMIZER_PARAMS["max_gradient_norm"],
                             defaults.get("max_gradient_norm"))
            cfg["optimizer"][stage] = {
                "n_steps": int(n), "stepsize": float(ss), "momentum": float(mo),
                "scale": float(sc), "logspace": bool(lg),
                "max_gradient_norm": mg,
            }

    # --------------------------------------------------------------- cluster
    st.subheader("7. Cluster")
    r1 = st.columns(4)
    n_tasks = r1[0].number_input("GPUs (array tasks)", 1, 512, 16, 1)
    batch = r1[1].number_input("Designs per GPU", 1, 8,
                               int(cfg["run"].get("batch", 2)), 1,
                               help="Above 4 is untested: GPU memory and a "
                                    "host-side simplex projection that scales "
                                    "with batch are both plausible ceilings.")
    throttle = r1[2].number_input("Max concurrent (0 = no cap)", 0, 512, 0, 1)
    partition = r1[3].selectbox("Partition", cluster.GPU_PARTITIONS, index=0)
    r2 = st.columns(3)
    time_limit = r2[0].text_input("Time limit", "08:00:00")
    mem = r2[1].text_input("Memory", "128G")
    cpus = r2[2].number_input("CPUs per task", 1, 64, 8, 1,
                              help="Part of the optimizer runs in host numpy "
                                   "every iteration.")
    cfg["run"] = {"batch": int(batch), "seed": 0}

    # ------------------------------------------------------------ submission
    st.subheader("8. Review and submit")
    problems = dc.validate(cfg)
    est_min = dc.estimate_seconds(cfg) / 60.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Designs", int(n_tasks) * int(batch))
    m2.metric("Est. per task", f"{est_min:.0f} min")
    m3.metric("Confidence terms", "yes" if dc.uses_confidence(cfg) else "no")

    if dc.uses_confidence(cfg):
        st.warning("A confidence term is enabled — the estimate above already "
                   "includes the ~3x slowdown from defeating JIT pruning.")
    for p in problems:
        st.error(p)

    with st.expander("Resolved config (this is what gets saved and run)"):
        st.json(cfg)

    out_name = st.text_input(
        "Campaign name",
        value=f"{Path(cfg['target']['fasta']).stem}_"
              f"{'_'.join(m['name'] for m in cfg['models']) or 'none'}")
    out_dir = SETUP / "designs" / out_name

    if st.button("Submit campaign", type="primary", disabled=bool(problems)):
        cfg_dir = SETUP / "configs"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = cfg_dir / f"{out_name}_{int(time.time())}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2))
        out_dir.mkdir(parents=True, exist_ok=True)

        ok, msg, job_id = cluster.submit_campaign(
            cfg_path, out_dir, int(n_tasks),
            throttle=int(throttle) or None, partition=partition,
            time_limit=time_limit, mem=mem, cpus=int(cpus))
        if ok:
            st.success(f"Submitted job {job_id}")
            st.session_state["last_job"] = {
                "id": job_id, "n_tasks": int(n_tasks),
                "models": [m["name"] for m in cfg["models"]],
                "expect_msa": cfg["target"]["use_msa"], "out": str(out_dir),
                "config": str(cfg_path),
            }
            st.info("Now open **Monitor** — the failure modes here are silent, "
                    "so confirm from the log that it ran what you asked.")
        else:
            st.error("Submission failed")
        st.code(msg)

    return cfg
