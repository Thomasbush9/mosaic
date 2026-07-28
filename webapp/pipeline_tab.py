"""Tab — define a pipeline DAG node by node, then launch it.

The whole tab is one job: describe each node (its type, its edges, its params,
optionally its resources) and submit the graph. Everything else here serves
that.

Two rules shape the design.

**Parity with the Python interface.** A node is a `PL.Node` whose `params` is a
plain dict handed verbatim to `pipeline_node.py`. Anything you could express by
assigning to `node.params` in Python must be expressible here, so every node
carries a raw JSON editor beside its generated form. The forms are the
convenient path, not the only one — a catalog that does not know about a
parameter must not prevent you from setting it.

**A stored value must never be able to crash the editor.** Params come back from
data the catalog never governed (a submitted `node.json`, an uploaded DAG), so
the forms clamp and report rather than raise; see the note in `ui_helpers`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import design_config as dc  # noqa: E402

import pipeline as PL  # noqa: E402
import store  # noqa: E402
from ui_helpers import params_block, resource_picker, widget  # noqa: E402

SCREEN_MODELS = ["boltz2", "boltz1", "af2", "of3", "protenix"]

# Every widget the node editor owns is namespaced, so the whole editor's state
# can be dropped with one prefix scan. That matters because node ids are reused:
# loading a different DAG whose node happens to be called "generate1" would
# otherwise inherit the previous pipeline's values for it, since Streamlit keys
# widget state by string and prefers held state over the `value=` argument.
NS = "pn::"


def _k(node_id: str, name: str) -> str:
    return f"{NS}{node_id}::{name}"


def _forget(node_id: str | None = None) -> None:
    """Drop widget state for one node, or for the whole editor."""
    prefix = f"{NS}{node_id}::" if node_id else NS
    for key in [k for k in st.session_state
                if isinstance(k, str) and k.startswith(prefix)]:
        del st.session_state[key]


def _pipe() -> PL.Pipeline:
    key = "pipeline"
    if key not in st.session_state:
        st.session_state[key] = PL.Pipeline(name="pipeline1")
    return st.session_state[key]


def _load(p: PL.Pipeline) -> None:
    """Install a pipeline into the editor, clearing any stale widget state."""
    st.session_state["pipeline"] = p
    _forget()


# --------------------------------------------------------------------------
# Per-node forms
# --------------------------------------------------------------------------

def _target_picker(node: PL.Node, targets, key: str, current_name: str | None):
    """Shared target selector. Returns the chosen Target, or None."""
    if not targets:
        st.warning("No targets yet — create one in the Launch tab.")
        return None
    names = [t.name for t in targets]
    idx = names.index(current_name) if current_name in names else 0
    chosen = st.selectbox("Target", names, index=idx, key=_k(node.id, key))
    return next(t for t in targets if t.name == chosen)


def _generate_form(node: PL.Node, targets) -> None:
    tgt = _target_picker(node, targets, "tgt", node.params.get("target_name"))

    gens = list(dc.GENERATIVE_MODELS)
    cur_gen = node.params.get("generator")
    gen = st.selectbox("Generator", gens,
                       index=gens.index(cur_gen) if cur_gen in gens else 0,
                       format_func=lambda g: dc.GENERATIVE_MODELS[g]["label"],
                       key=_k(node.id, "gen"))
    meta = dc.GENERATIVE_MODELS[gen]
    st.caption(meta["blurb"])

    p = params_block(_k(node.id, "gp"), meta["params"], dict(node.params))
    hot = st.text_input("Hotspots (target residues, optional)",
                        value=node.params.get("hotspots", ""),
                        key=_k(node.id, "hot"))

    node.params = {**p, "generator": gen, "hotspots": hot.strip()}
    if tgt:
        # Target paths are re-derived from the selector on every rerun, so this
        # is the one group of keys the raw JSON editor cannot hold against the
        # form. Everything else it sets survives.
        node.params.update({
            "target_name": tgt.name,
            "target_fasta": str(tgt.fasta),
            "target_cif": str(tgt.structure) if tgt.structure else "",
            "target_msa": str(tgt.msa) if tgt.msa else "",
        })
        if meta.get("needs_structure") and not tgt.structure:
            st.warning("This generator designs against geometry and needs a "
                       "target structure — predict one in the Generate tab.")


def _hallucinate_form(node: PL.Node, targets) -> None:
    tgt = _target_picker(node, targets, "htgt", node.params.get("target_name"))
    st.caption("Designs binders from noise against the chosen structure "
               "model(s) — mosaic's own method run as generation. Each "
               "candidate is a full optimization, so it is far costlier per "
               "design than BoltzGen.")

    chosen = st.multiselect(
        "Fold against", SCREEN_MODELS,
        default=[m["name"] for m in node.params.get("models", [])
                 if m.get("name") in SCREEN_MODELS] or ["boltz2"],
        key=_k(node.id, "hmodels"),
        help="The structure model(s) whose confidence the hallucination "
             "maximizes. Boltz-2 is a good single choice.")

    p = params_block(_k(node.id, "hp"), dc.HALLUCINATE_PARAMS, dict(node.params))
    total = int(p.get("array", 1)) * int(p.get("num_designs", 8))
    st.caption(f"Total hallucinated: **{total}** "
               f"({p.get('array', 1)} tasks x {p.get('num_designs', 8)}).")

    node.params = {**p, "models": [{"name": m} for m in chosen]}
    if tgt:
        node.params.update({
            "target_name": tgt.name,
            "target_fasta": str(tgt.fasta),
            "target_msa": str(tgt.msa) if tgt.msa else "",
        })


def _screen_form(node: PL.Node, targets) -> None:
    cur = node.params.get("model")
    c = st.columns(3)
    model = c[0].selectbox(
        "Refold with", SCREEN_MODELS,
        index=SCREEN_MODELS.index(cur) if cur in SCREEN_MODELS else 2,
        key=_k(node.id, "m"),
        help="Best chosen DIFFERENT from the generator, so a candidate is not "
             "judged by the model that proposed it.")
    inv = c[1].checkbox("ProteinMPNN inverse-fold",
                        value=bool(node.params.get("inverse_fold", True)),
                        key=_k(node.id, "inv"))
    use_msa = c[2].checkbox("Use target MSA",
                            value=bool(node.params.get("use_msa", True)),
                            key=_k(node.id, "msa"))

    c2 = st.columns(2)
    with c2[0]:
        rec = widget(_k(node.id, "rec"), "recycling_steps",
                     {"type": "int", "default": 4, "min": 1, "max": 20},
                     node.params.get("recycling_steps", 4))
    mpnn_opts = list(dc.MPNN_WEIGHTS)
    cur_w = node.params.get("mpnn_weights")
    weights = c2[1].selectbox(
        "MPNN weights", mpnn_opts,
        index=mpnn_opts.index(cur_w) if cur_w in mpnn_opts else 0,
        format_func=lambda k: dc.MPNN_WEIGHTS[k],
        key=_k(node.id, "mw"), disabled=not inv)

    # screen reads target_msa off the node, and only the upstream generate node
    # knows which target this is. Carry it if a target is resolvable.
    msa = node.params.get("target_msa", "")
    if use_msa and not msa and targets:
        names = [t.name for t in targets]
        pick = st.selectbox("Target MSA from", ["(none)"] + names,
                            key=_k(node.id, "smsa"))
        if pick != "(none)":
            t = next(t for t in targets if t.name == pick)
            msa = str(t.msa) if t.msa else ""

    node.params = {**node.params, "model": model, "inverse_fold": bool(inv),
                   "use_msa": bool(use_msa), "recycling_steps": int(rec),
                   "mpnn_weights": weights, "target_msa": msa}


def _optimize_form(node: PL.Node, targets) -> None:
    # An optimize node carries a whole design config — the same object the
    # Launch tab builds. Rather than re-embed that entire form, import one and
    # expose the load-bearing knobs here.
    cfg = node.params.get("config") or dc.default_config()

    if st.button("Import the current Launch config as this node's objective",
                 key=_k(node.id, "uselaunch"),
                 help="Copies the config you built in the Launch tab: models, "
                      "losses, MPNN and sequence terms."):
        for k in st.session_state:
            if isinstance(k, str) and k.startswith("cfg::"):
                cfg = json.loads(json.dumps(st.session_state[k]))
                st.success("Imported.")
                break

    names = [m["name"] for m in cfg.get("models", [])]
    st.caption("Objective — models: " + ("+".join(names) or "**none**") +
               " · losses: " + (", ".join(l["name"] for l in cfg.get("losses", []))
                                or "**none**") +
               (" · +MPNN" if cfg.get("mpnn", {}).get("terms") else "") +
               (" · +ESM-C" if cfg.get("sequence_models") else ""))

    tgt = _target_picker(node, targets, "otgt",
                         next((t.name for t in targets
                               if str(t.fasta) == str((cfg.get("target") or {})
                                                      .get("fasta"))), None))
    if tgt:
        prev = cfg.get("target") or {}
        cfg["target"] = {"fasta": str(tgt.fasta),
                         "msa": str(tgt.msa) if tgt.msa else None,
                         "use_msa": bool(tgt.msa) and prev.get("use_msa", True)}

    # Step bounds come from the catalog, not from literals here. The previous
    # hard-coded 300/200 disagreed with the Launch form's 1-500, so a perfectly
    # valid imported config could not be redrawn.
    nspec = dc.OPTIMIZER_PARAMS["n_steps"]
    o = cfg.setdefault("optimizer", dc.default_config()["optimizer"])
    c = st.columns(3)
    with c[0]:
        o["soft"]["n_steps"] = widget(_k(node.id, "soft"), "soft_steps", nspec,
                                      o["soft"].get("n_steps", 100))
    with c[1]:
        o["sharp"]["n_steps"] = widget(_k(node.id, "sharp"), "sharp_steps", nspec,
                                       o["sharp"].get("n_steps", 25))
    with c[2]:
        cfg.setdefault("binder", {})["init_noise"] = widget(
            _k(node.id, "noise"), "init_noise",
            {"type": "float", "default": 0.1, "min": 0.0, "max": 0.5,
             "help": "init_fasta and the epitope come from the upstream node "
                     "automatically."},
            cfg.get("binder", {}).get("init_noise", 0.1))

    fan = params_block(
        _k(node.id, "op"),
        {"top_k": {"type": "int", "default": 0, "min": 0, "max": 5000,
                   "help": "Refine only the top-K from upstream (0 = all). A "
                           "screen ranks best-first, so this spends the "
                           "gradient stage only on its winners."},
         "array": {"type": "int", "default": 1, "min": 1, "max": 512,
                   "help": "In-node fan-out: this many array tasks, one GPU "
                           "each, refining different windows of the top-K."},
         "array_throttle": {"type": "int", "default": 0, "min": 0, "max": 512,
                            "help": "Max array tasks at once. 0 lets SLURM "
                                    "decide."}},
        dict(node.params))

    batch = int(cfg.get("run", {}).get("batch", 4))
    st.caption(f"Total optimized: **{int(fan.get('array', 1)) * batch}** "
               f"({fan.get('array', 1)} tasks x batch {batch}); seeded from top "
               f"{int(fan.get('top_k', 0)) or 'all'}.")

    node.params = {**fan, "config": cfg, "models": cfg.get("models", [])}


FORMS = {
    "generate": _generate_form,
    "hallucinate": _hallucinate_form,
    "screen": _screen_form,
    "optimize": _optimize_form,
}


# --------------------------------------------------------------------------
# Per-node chrome: edges, resources, raw params
# --------------------------------------------------------------------------

def _edge_editor(p: PL.Pipeline, node: PL.Node) -> None:
    """Edges are editable after the fact, not only at Add time."""
    spec = PL.NODE_TYPES[node.type]
    if spec["max_inputs"] == 0:
        return
    cands = [n.id for n in p.nodes
             if n.id != node.id
             and PL.NODE_TYPES[n.type]["produces"] in spec["accepts"]]
    if not cands:
        st.caption(f":orange[No upstream node produces a "
                   f"{' or '.join(spec['accepts'])} yet.]")
        node.inputs = []
        return
    node.inputs = st.multiselect(
        "Inputs (edges from)", cands,
        default=[i for i in node.inputs if i in cands],
        max_selections=spec["max_inputs"], key=_k(node.id, "edges"),
        help=f"Accepts {' or '.join(spec['accepts'])}; "
             f"{spec['min_inputs']}-{spec['max_inputs']} input(s).")


def _resource_editor(node: PL.Node) -> None:
    """Optional per-node resource override.

    Generate, screen and optimize have genuinely different shapes — one is a
    fixed diffusion pass, another is hundreds of gradient steps — so a single
    pipeline-wide walltime either wastes allocation or kills the long node.
    Blank means inherit the pipeline default.
    """
    res = dict(node.params.get("resources") or {})
    with st.expander("Resources — override the pipeline default"):
        c = st.columns(3)
        time_limit = c[0].text_input("Time (H:MM:SS)", value=res.get("time_limit", ""),
                                     placeholder="inherit",
                                     key=_k(node.id, "rtime"))
        mem = c[1].text_input("Memory", value=res.get("mem", ""),
                              placeholder="inherit", key=_k(node.id, "rmem"))
        cpus = c[2].text_input("CPUs", value=str(res.get("cpus", "")),
                               placeholder="inherit", key=_k(node.id, "rcpus"))
    over = {}
    if time_limit.strip():
        over["time_limit"] = time_limit.strip()
    if mem.strip():
        over["mem"] = mem.strip()
    if cpus.strip():
        try:
            over["cpus"] = int(cpus)
        except ValueError:
            st.error(f"{node.id}: CPUs must be a whole number, got {cpus!r}.")
    if over:
        node.params["resources"] = over
    else:
        node.params.pop("resources", None)


def _raw_params_editor(node: PL.Node) -> None:
    """The escape hatch that keeps this UI as expressive as Python.

    `node.params` is handed verbatim to pipeline_node.py, so any key the stage
    scripts read can be set here — including ones no form draws. Deliberately
    keyless: the text area is identified by its content, so it re-initialises
    when the form above changes params, and holds your edit across the rerun
    that the Apply button triggers.
    """
    with st.expander(f"Raw params for `{node.id}` — the dict pipeline_node.py receives"):
        st.caption(
            "Edit and Apply to set params the forms above do not expose. "
            "Applying replaces the whole dict and resets this node's widgets. "
            "Target paths on generate/hallucinate nodes are re-derived from "
            "the Target selector on every rerun, so set those there.")
        text = st.text_area(f"{node.id} params (JSON)",
                            value=json.dumps(node.params, indent=2, sort_keys=True),
                            height=260)
        if st.button("Apply", key=_k(node.id, "rawapply")):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as e:
                st.error(f"Not valid JSON: {e}")
                return
            if not isinstance(parsed, dict):
                st.error("params must be a JSON object, not "
                         f"{type(parsed).__name__}.")
                return
            node.params = parsed
            _forget(node.id)
            st.rerun()


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------

def _add_node(p: PL.Pipeline) -> None:
    with st.expander("Add a node", expanded=not p.nodes):
        a = st.columns([1, 1, 2])
        ntype = a[0].selectbox("Type", list(PL.NODE_TYPES),
                               format_func=lambda t: PL.NODE_TYPES[t]["label"],
                               key="add_type")
        spec = PL.NODE_TYPES[ntype]
        nid = a[1].text_input("Node id", value=f"{ntype}{len(p.nodes) + 1}",
                              key="add_id")
        cands = [n.id for n in p.nodes
                 if PL.NODE_TYPES[n.type]["produces"] in spec["accepts"]]
        inputs = []
        if spec["max_inputs"] > 0:
            inputs = a[2].multiselect("Inputs (edges from)", cands,
                                      max_selections=spec["max_inputs"],
                                      key="add_inputs")
        st.caption(spec["blurb"])
        clash = any(n.id == nid.strip() for n in p.nodes)
        if clash:
            st.error(f"A node called {nid.strip()!r} already exists.")
        if st.button("Add node", key="add_go",
                     disabled=not nid.strip() or clash):
            p.nodes.append(PL.Node(id=nid.strip(), type=ntype,
                                   inputs=list(inputs)))
            _forget(nid.strip())
            # Release the id and edge fields so they fall back to a fresh
            # default. Held, they would still contain the id just used and the
            # form would greet you with "already exists" every time you opened
            # it to add the next node.
            for stale in ("add_id", "add_inputs"):
                st.session_state.pop(stale, None)
            st.rerun()


def _past_runs() -> None:
    runs = PL.list_runs(store.workdir())
    if not runs:
        return
    with st.expander(f"Reopen a submitted pipeline ({len(runs)} on disk)"):
        st.caption("Reconstructed from each node's node.json, so a run can be "
                   "reviewed or reopened without having kept its DAG JSON.")
        pick = st.selectbox("Run", runs, index=len(runs) - 1, key="past_run")
        run_dir = store.workdir() / "pipelines" / pick
        past = PL.Pipeline.from_run_dir(run_dir)

        jobids = {}
        jf = run_dir / "jobids.json"
        if jf.exists():
            try:
                jobids = json.loads(jf.read_text())
            except (OSError, json.JSONDecodeError):
                jobids = {}
        states = {}
        if jobids and st.checkbox("Query SLURM job states", key="past_states"):
            states = PL.states(jobids)
        st.markdown(f"```mermaid\n{PL.mermaid(past, states)}\n```")

        b = st.columns(2)
        if b[0].button("Load into editor", key="load_past"):
            _load(past)
            st.rerun()
        b[1].download_button("Download DAG JSON", data=past.to_json(),
                             file_name=f"{pick}.json", mime="application/json",
                             key="dl_past")


def render(cfg: dict) -> None:
    p = _pipe()
    targets = store.list_targets()

    st.subheader("Pipeline")
    top = st.columns([2, 1, 1])
    p.name = top[0].text_input("Name", value=p.name, key="pipe_name")
    if top[1].button("Load DAG", key="pipe_load"):
        st.session_state["_show_load"] = True
    if top[2].button("Clear", key="pipe_clear"):
        _load(PL.Pipeline(name="pipeline1"))
        st.rerun()

    if st.session_state.get("_show_load"):
        up = st.file_uploader("Pipeline JSON", type="json", key="pipe_upload")
        if up is not None:
            try:
                _load(PL.Pipeline.from_json(up.read().decode()))
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                st.error(f"Not a pipeline DAG: {e}")
            else:
                st.session_state["_show_load"] = False
                st.rerun()

    _past_runs()
    _add_node(p)

    if not p.nodes:
        st.info("Add a Generate or Hallucinate node to begin — both are roots "
                "that take a target rather than an upstream node.")
        return

    # ---- configure each node ----------------------------------------------
    st.subheader("Nodes")
    for node in list(p.nodes):
        spec = PL.NODE_TYPES.get(node.type)
        head = st.columns([4, 1])
        if spec is None:
            head[0].error(f"**{node.id}** — unknown node type {node.type!r}")
        else:
            head[0].markdown(
                f"**{node.id}** — {spec['label']}"
                + (f"  <-  {', '.join(node.inputs)}" if node.inputs else ""))
        if head[1].button("Remove", key=_k(node.id, "rm")):
            p.nodes = [n for n in p.nodes if n.id != node.id]
            for n in p.nodes:
                n.inputs = [i for i in n.inputs if i != node.id]
            _forget(node.id)
            st.rerun()

        if spec is not None:
            _edge_editor(p, node)
            form = FORMS.get(node.type)
            if form is not None:
                form(node, targets)
            elif node.type == "merge":
                st.caption("Merges: " + (", ".join(node.inputs)
                                         or "(pick inputs above)"))
            if spec["gpu"] or node.type == "merge":
                _resource_editor(node)
            _raw_params_editor(node)
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
        st.download_button("Download DAG", data=p.to_json(), key="dl_dag",
                           file_name=f"{p.name}.json", mime="application/json")
        st.json(json.loads(p.to_json()))

    st.subheader("Launch")
    st.caption("Defaults for nodes that do not override them. Nodes are "
               "submitted in topological order with --dependency=afterok, so "
               "SLURM does the waiting.")
    st.markdown("**GPU nodes** (generate / hallucinate / screen / optimize)")
    gpu_res = resource_picker("pipe_gpu", gpu=True)
    st.markdown("**CPU nodes** (merge)")
    cpu_res = resource_picker("pipe_cpu", gpu=False)

    if st.button("Submit pipeline", key="pipe_submit", type="primary",
                 disabled=bool(problems)):
        ok, log, jobids = PL.submit(
            p, repo=store.REPO, workdir=store.workdir(),
            account=gpu_res["account"], gpu_partition=gpu_res["partition"],
            cpu_partition=cpu_res["partition"],
            gpu_res=gpu_res, cpu_res=cpu_res)
        if ok:
            st.session_state[f"pipe_jobids_{p.name}"] = jobids
            st.success("Submitted — SLURM runs each node when its inputs are "
                       "ready. Watch it in Monitor.")
        else:
            st.error("Submission failed — nothing was queued.")
        st.code(log)

    if jobids:
        st.caption("Node -> job:  "
                   + "   ".join(f"{k}={v}" for k, v in jobids.items()))
