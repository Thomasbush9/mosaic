"""Tab — build and run a pipeline DAG.

Static (no drag-drop): you add nodes, pick each node's type, and configure it
with the SAME catalog-driven forms the single-stage tabs use — a generate node
offers every generator + params, a screen node every model + MPNN options, an
optimize node the full loss config. Edges are chosen from a dropdown of upstream
nodes and checked for type compatibility. The graph renders as a Mermaid diagram
coloured by job state.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import design_config as dc  # noqa: E402

import cluster  # noqa: E402
import pipeline as PL  # noqa: E402
import store  # noqa: E402
from ui_helpers import params_block  # noqa: E402


def _pipe() -> PL.Pipeline:
    key = "pipeline"
    if key not in st.session_state:
        st.session_state[key] = PL.Pipeline(name="pipeline1")
    return st.session_state[key]


def _generate_form(node: PL.Node, targets) -> None:
    tnames = [t.name for t in targets]
    cur = node.params.get("target_name")
    tgt = next((t for t in targets if t.name == cur), targets[0] if targets else None)
    if targets:
        tgt = next(t for t in targets
                   if t.name == st.selectbox("Target", tnames,
                                             index=tnames.index(tgt.name) if tgt else 0,
                                             key=f"{node.id}_tgt"))
    gen = st.selectbox("Generator", list(dc.GENERATIVE_MODELS),
                       index=list(dc.GENERATIVE_MODELS).index(
                           node.params.get("generator", "boltzgen"))
                       if node.params.get("generator") in dc.GENERATIVE_MODELS else 0,
                       format_func=lambda g: dc.GENERATIVE_MODELS[g]["label"],
                       key=f"{node.id}_gen")
    meta = dc.GENERATIVE_MODELS[gen]
    st.caption(meta["blurb"])
    p = params_block(f"{node.id}_gp", meta["params"], dict(node.params))
    hot = st.text_input("Hotspots (target residues, optional)",
                        value=node.params.get("hotspots", ""),
                        key=f"{node.id}_hot")
    node.params = {**p, "generator": gen, "hotspots": hot.strip(),
                   "target_name": tgt.name if tgt else "",
                   "target_fasta": str(tgt.fasta) if tgt else "",
                   "target_cif": str(tgt.structure) if tgt and tgt.structure else "",
                   "target_msa": str(tgt.msa) if tgt and tgt.msa else ""}
    if meta.get("needs_structure") and tgt and not tgt.structure:
        st.warning("This generator needs a target structure — predict one first.")


def _screen_form(node: PL.Node) -> None:
    c = st.columns(3)
    model = c[0].selectbox("Refold with", ["boltz2", "boltz1", "af2", "of3", "protenix"],
                           index=["boltz2", "boltz1", "af2", "of3", "protenix"].index(
                               node.params.get("model", "af2")),
                           key=f"{node.id}_m")
    inv = c[1].checkbox("ProteinMPNN inverse-fold", value=node.params.get("inverse_fold", True),
                        key=f"{node.id}_inv")
    use_msa = c[2].checkbox("Use target MSA", value=node.params.get("use_msa", True),
                            key=f"{node.id}_msa")
    rec = st.number_input("recycling_steps", 1, 20, int(node.params.get("recycling_steps", 4)),
                          key=f"{node.id}_rec")
    node.params = {**node.params, "model": model, "inverse_fold": inv,
                   "use_msa": use_msa, "recycling_steps": int(rec),
                   "mpnn_weights": node.params.get("mpnn_weights", "soluble")}


def _optimize_form(node: PL.Node, targets) -> None:
    # The optimize node carries a full design config. Rather than re-embed the
    # entire Launch form here, take a config the user built in Launch (saved in
    # session) or the default, and let them tweak the load-bearing knobs.
    cfg = node.params.get("config") or dc.default_config()
    if st.checkbox("Use the current Launch config as this node's objective",
                   key=f"{node.id}_uselaunch"):
        for k, v in st.session_state.items():
            if isinstance(k, str) and k.startswith("cfg::"):
                cfg = json.loads(json.dumps(st.session_state[k]))
                break
    names = [m["name"] for m in cfg.get("models", [])]
    st.caption("Structure models: " + ("+".join(names) or "none") +
               " · losses: " + ", ".join(l["name"] for l in cfg.get("losses", [])) +
               (" · +MPNN" if cfg.get("mpnn", {}).get("terms") else "") +
               (" · +ESM-C" if cfg.get("sequence_models") else ""))
    c = st.columns(2)
    cfg["optimizer"]["soft"]["n_steps"] = c[0].number_input(
        "soft steps", 1, 300, int(cfg["optimizer"]["soft"]["n_steps"]), key=f"{node.id}_soft")
    cfg["optimizer"]["sharp"]["n_steps"] = c[1].number_input(
        "sharp steps", 1, 200, int(cfg["optimizer"]["sharp"]["n_steps"]), key=f"{node.id}_sharp")
    cfg["binder"]["init_noise"] = st.slider(
        "seed noise", 0.0, 0.5, float(cfg["binder"].get("init_noise", 0.1)), 0.02,
        key=f"{node.id}_noise",
        help="init_fasta and epitope come from the upstream node automatically.")
    node.params = {**node.params, "config": cfg,
                   "models": cfg.get("models", [])}
    st.caption("Configure the full objective in the **Launch** tab, then check the "
               "box above to import it here.")


def render(cfg: dict) -> None:
    p = _pipe()
    targets = store.list_targets()

    st.subheader("Pipeline")
    top = st.columns([2, 1, 1])
    p.name = top[0].text_input("Name", value=p.name, key="pipe_name")
    if top[1].button("Load DAG"):
        st.session_state["_show_load"] = True
    if top[2].button("Clear"):
        st.session_state["pipeline"] = PL.Pipeline(name="pipeline1")
        st.rerun()

    if st.session_state.get("_show_load"):
        up = st.file_uploader("Pipeline JSON", type="json", key="pipe_upload")
        if up is not None:
            st.session_state["pipeline"] = PL.Pipeline.from_json(up.read().decode())
            st.session_state["_show_load"] = False
            st.rerun()

    # ---- add a node --------------------------------------------------------
    with st.expander("Add a node", expanded=not p.nodes):
        a = st.columns([1, 1, 2])
        ntype = a[0].selectbox("Type", list(PL.NODE_TYPES),
                               format_func=lambda t: PL.NODE_TYPES[t]["label"])
        st.caption(PL.NODE_TYPES[ntype]["blurb"])
        nid = a[1].text_input("Node id", value=f"{ntype}{len(p.nodes)+1}")
        spec = PL.NODE_TYPES[ntype]
        # Inputs: only upstream nodes whose output this type accepts.
        candidates = [n.id for n in p.nodes
                      if PL.NODE_TYPES[n.type]["produces"] in spec["accepts"]]
        inputs = []
        if spec["max_inputs"] > 0:
            inputs = a[2].multiselect(
                "Inputs (edges from)", candidates,
                max_selections=spec["max_inputs"], key="add_inputs")
        if st.button("Add node", disabled=not nid.strip() or any(n.id == nid for n in p.nodes)):
            p.nodes.append(PL.Node(id=nid.strip(), type=ntype, inputs=list(inputs)))
            st.rerun()

    if not p.nodes:
        st.info("Add a Generate node to begin.")
        return

    # ---- configure each node ----------------------------------------------
    st.subheader("Nodes")
    for node in list(p.nodes):
        spec = PL.NODE_TYPES[node.type]
        head = st.columns([4, 1])
        head[0].markdown(f"**{node.id}** — {spec['label']}"
                         + (f"  ←  {', '.join(node.inputs)}" if node.inputs else ""))
        if head[1].button("Remove", key=f"rm_{node.id}"):
            p.nodes = [n for n in p.nodes if n.id != node.id]
            for n in p.nodes:
                n.inputs = [i for i in n.inputs if i != node.id]
            st.rerun()
        with st.container():
            if node.type == "generate":
                _generate_form(node, targets)
            elif node.type == "screen":
                _screen_form(node)
            elif node.type == "optimize":
                _optimize_form(node, targets)
            elif node.type == "merge":
                st.caption(f"Merges: {', '.join(node.inputs) or '(pick inputs above)'}")
        st.divider()

    # ---- graph + submit ----------------------------------------------------
    st.subheader("Graph")
    jobids = st.session_state.get(f"pipe_jobids_{p.name}", {})
    node_states = PL.states(jobids) if jobids else {}
    st.markdown(f"```mermaid\n{PL.mermaid(p, node_states)}\n```")

    problems = PL.validate(p)
    for e in problems:
        st.error(e)

    with st.expander("Pipeline JSON (the reproducible record)"):
        st.download_button("Download DAG", data=p.to_json(),
                           file_name=f"{p.name}.json", mime="application/json")
        st.json(json.loads(p.to_json()))

    sc = st.columns(2)
    account = sc[0].selectbox("Account", cluster.accounts(), key="pipe_acct")
    partition = sc[1].selectbox("GPU partition", cluster.partitions(), key="pipe_part")
    if st.button("Submit pipeline", type="primary", disabled=bool(problems)):
        ok, log, jobids = PL.submit(
            p, repo=store.REPO, workdir=store.workdir(), account=account,
            gpu_partition=partition, cpu_partition="kempner_interactive")
        if ok:
            st.session_state[f"pipe_jobids_{p.name}"] = jobids
            st.success("Submitted — SLURM runs each node when its inputs are ready.")
        else:
            st.error("Submission failed")
        st.code(log)

    if jobids:
        st.caption("Node → job:  " + "   ".join(f"{k}={v}" for k, v in jobids.items()))
