#!/usr/bin/env python3
"""Define a pipeline in a file, then submit it — the webapp's graph, without the webapp.

This module and the Pipeline tab are two front ends over one engine
(`webapp/pipeline.py`): same node types, same artifact contract, same
validation, same SLURM dependency chain. Nothing here reimplements the graph.
What differs is only how you say what you want — a form you click, or a file you
edit, diff and commit.

A spec file is deliberately terse. Name a node's type and only the parameters
that differ from the catalogs in `design_config.py`; everything else is filled
exactly as the webapp's forms fill it, so a file and a clicked pipeline that
describe the same run produce the same `node.params` and therefore the same
jobs.

**Parity with the Python interface**, the same rule the node editor follows:
`node.params` is a plain dict handed verbatim to `pipeline_node.py`, so any key
you could set in Python you can set here. Unknown keys pass through untouched
and are *reported*, never dropped.

**Bounds are advisory here.** A value outside a catalog's declared range is
reported and then run as written. The webapp has to clamp — Streamlit raises on
an out-of-range widget value and takes the page down with it — but in a file you
typed the number on purpose, and silently changing it would be worse than either
crashing or obeying.

    ./pipeline_spec.py show    pipelines/dio3.yaml
    ./pipeline_spec.py dag     pipelines/dio3.yaml        # mermaid
    ./pipeline_spec.py submit  pipelines/dio3.yaml --dry-run
    ./pipeline_spec.py submit  pipelines/dio3.yaml
    ./pipeline_spec.py json    pipelines/dio3.yaml -o dio3.json   # loads in the webapp
    ./pipeline_spec.py export  dio3_2way_500b -o dio3.yaml        # a past run -> a spec
    ./pipeline_spec.py template -o my_pipeline.yaml
    ./pipeline_spec.py selftest

YAML when PyYAML is importable, JSON otherwise; the keys are identical either
way and the extension chooses. See docs/PIPELINE_FILE.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "webapp"))
sys.path.insert(0, str(REPO))

import design_config as dc  # noqa: E402
import pipeline as PL  # noqa: E402
import store  # noqa: E402

# Same list the Pipeline tab offers, and the same default: a screen is best run
# with a model DIFFERENT from the one that proposed the candidate, so it is not
# judged by its own author.
SCREEN_MODELS = ["boltz2", "boltz1", "af2", "of3", "protenix"]
SCREEN_DEFAULT = "af2"

# Keys on a node entry that describe the node rather than parameterise it.
NODE_KEYS = ("id", "type", "inputs", "label", "target", "resources", "params",
             "params_file")

_DROP = object()      # sentinel: "identical to the baseline, omit it"
_MISSING = object()


# --------------------------------------------------------------------------
# Reading and writing spec files
# --------------------------------------------------------------------------

def load(path) -> dict:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ModuleNotFoundError:
            raise SystemExit(
                f"{path.name} is YAML but PyYAML is not importable by this "
                f"interpreter ({sys.executable}). Either install it "
                "(`uv pip install pyyaml`) or write the spec as .json — the keys "
                "are identical.") from None
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: the top level must be a mapping, not "
                         f"{type(data).__name__}.")
    return data


def dumps(spec: dict, fmt: str = "yaml") -> str:
    if fmt in ("yaml", "yml"):
        try:
            import yaml
        except ModuleNotFoundError:
            fmt = "json"
        else:
            return yaml.safe_dump(spec, sort_keys=False, default_flow_style=False,
                                  width=88)
    return json.dumps(spec, indent=2)


# --------------------------------------------------------------------------
# Expansion: terse spec -> full node params
# --------------------------------------------------------------------------

def _defaults(param_spec: dict) -> dict:
    return {name: s.get("default") for name, s in param_spec.items()}


def _merge(base: Any, over: Any) -> Any:
    """Recursive merge with `over` winning.

    Lists REPLACE rather than concatenate. A list of losses is a complete
    statement of the objective, not an addition to the default one — appending
    would mean you could never turn the baseline contact terms off.
    """
    if not (isinstance(base, dict) and isinstance(over, dict)):
        return over
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if k in out else v
    return out


class _Report:
    """Errors stop a submit; warnings are things you probably meant."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.notes: list[str] = []      # e.g. which file a node was linked to

    # Deduplicated: one mistyped top-level target is one mistake, not one per
    # node that inherited it.
    def err(self, msg: str) -> None:
        if msg not in self.errors:
            self.errors.append(msg)

    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)


def _check_bounds(params: dict, param_spec: dict, where: str, rep: _Report,
                  *, strict_keys: bool = True) -> None:
    for k, v in params.items():
        s = param_spec.get(k)
        if s is None:
            if strict_keys:
                rep.warn(f"{where}: {k!r} is not a catalog parameter — passed "
                         "through to pipeline_node.py as written.")
            continue
        lo, hi = s.get("min"), s.get("max")
        if (isinstance(v, (int, float)) and not isinstance(v, bool)
                and ((lo is not None and v < lo) or (hi is not None and v > hi))):
            rep.warn(f"{where}: {k}={v} is outside the catalog range "
                     f"{lo}-{hi}. Running it as written; widen the range in "
                     "design_config.py if this is routine.")
        if s.get("type") == "choice" and s.get("options") and v not in s["options"]:
            rep.warn(f"{where}: {k}={v!r} is not one of "
                     f"{', '.join(map(str, s['options']))}.")


def _entries(value, catalog: dict, where: str, rep: _Report) -> list[dict]:
    """Normalise a model / loss / term list into [{name, weight, params}].

    Accepts every shape that reads naturally in a file:

        models: boltz2
        models: [boltz2, af2]
        models: {boltz2: 1.0, af2: 0.5}
        losses: [{name: BinderTargetContact, weight: 4.0, contact_distance: 18}]
        losses: [{name: HelixLoss, weight: 1.0, params: {max_distance: 6.0}}]
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, dict):
        value = [{"name": k, **({"weight": v} if not isinstance(v, dict) else v)}
                 for k, v in value.items()]
    out = []
    for item in value:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict) or not item.get("name"):
            rep.err(f"{where}: {item!r} is not a name or a mapping with a name.")
            continue
        name = item["name"]
        meta = catalog.get(name)
        if meta is None:
            rep.err(f"{where}: {name!r} is not in the catalog. Known: "
                    f"{', '.join(catalog)}.")
            meta = {}
        # Anything besides name/weight/params is taken as a parameter, so the
        # common case stays one line instead of a nested params block.
        given = dict(item.get("params") or {})
        given.update({k: v for k, v in item.items()
                      if k not in ("name", "weight", "params")})
        _check_bounds(given, meta.get("params", {}), f"{where}:{name}", rep)
        out.append({
            "name": name,
            "weight": float(item.get("weight", meta.get("default_weight", 1.0))),
            "params": _merge(_defaults(meta.get("params", {})), given),
        })
    return out


# --------------------------------------------------------------------------
# Linked files
#
# A node may keep its parameters in a file of their own — `params_file:` on any
# node, or `config:` given as a path on an optimize node. This is worth doing
# when a setting is SHARED (one objective reused by three pipelines, one tuned
# generator preset) or when it is large enough to bury the graph. It is not
# worth doing by default: a five-node pipeline written inline is one readable
# 37-line file, and the same pipeline split six ways is six files you have to
# open before you know what ran.
#
# Whatever you choose, the run's record is unaffected: submit() resolves
# everything and writes dag.json plus a per-node node.json, so a linked file
# that changes later cannot rewrite history.
# --------------------------------------------------------------------------

def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _link(ref, base_dir: Path | None, where: str, rep: _Report) -> dict:
    """Load a linked file: relative to the spec file first, then the workdir."""
    if not isinstance(ref, str):
        rep.err(f"{where}: a linked file must be a path, not {type(ref).__name__}.")
        return {}
    p = Path(ref).expanduser()
    roots = [None] if p.is_absolute() else [base_dir or Path.cwd(), store.workdir()]
    tried = []
    for root in roots:
        q = p if root is None else Path(root) / p
        if not q.is_file():
            tried.append(str(q))
            continue
        try:
            d = load(q)
        except Exception as exc:      # noqa: BLE001 — unreadable is unreadable
            rep.err(f"{where}: could not read {q}: {exc}")
            return {}
        if "params_file" in d:
            rep.err(f"{where}: {q} has its own params_file. Linking is one "
                    "level deep, so a chain of files cannot hide the values.")
        rep.notes.append(f"{where}: linked {q}")
        return {k: v for k, v in d.items() if k != "params_file"}
    rep.err(f"{where}: no such file {ref!r} — looked in {', '.join(tried)}.")
    return {}


def _resolve_target(value, targets: list, where: str, rep: _Report) -> dict | None:
    """A target is a name from `targets/`, or an explicit mapping of paths."""
    if not value:
        return None
    if isinstance(value, dict):
        return {"name": value.get("name", ""), "fasta": value.get("fasta", ""),
                "cif": value.get("cif", ""), "msa": value.get("msa", "")}
    t = next((t for t in targets if t.name == value), None)
    if t is None:
        # Keyed on the target, not on `where`: one mistyped top-level target is
        # one mistake, however many nodes inherited it.
        rep.err(f"target {value!r}: no such target under {store.sub('targets')} "
                f"(expects {value}.fasta). Available: "
                f"{', '.join(x.name for x in targets) or '(none)'}.")
        return None
    return {"name": t.name, "fasta": str(t.fasta),
            "cif": str(t.structure) if t.structure else "",
            "msa": str(t.msa) if t.msa else ""}


def node_params(ntype: str, given: dict, target: dict | None, where: str,
                rep: _Report) -> dict:
    """Fill one node's params from the catalogs, exactly as the forms do.

    `given` is what the file said (minus the structural keys); the return value
    is the dict `pipeline_node.py` receives.
    """
    given = dict(given)

    if ntype == "generate":
        gen = given.pop("generator", "boltzgen")
        meta = dc.GENERATIVE_MODELS.get(gen)
        if meta is None:
            rep.err(f"{where}: unknown generator {gen!r}. Known: "
                    f"{', '.join(dc.GENERATIVE_MODELS)}.")
            meta = {"params": {}}
        hot = str(given.pop("hotspots", "") or "").strip()
        _check_bounds(given, meta.get("params", {}), where, rep)
        p = {**_defaults(meta.get("params", {})), **given,
             "generator": gen, "hotspots": hot}
        if target:
            p.update({"target_name": target["name"],
                      "target_fasta": target["fasta"],
                      "target_cif": target["cif"],
                      "target_msa": target["msa"]})
            if meta.get("needs_structure") and not target["cif"]:
                rep.err(f"{where}: {gen} designs against geometry and needs a "
                        f"target STRUCTURE; {target['name']} has no .cif in "
                        f"{store.sub('targets')} — predict one first "
                        "(predict_target.py, or the Generate tab).")
        return p

    if ntype == "hallucinate":
        models = _entries(given.pop("models", ["boltz2"]), dc.STRUCTURE_MODELS,
                          f"{where}:models", rep)
        _check_bounds(given, dc.HALLUCINATE_PARAMS, where, rep)
        p = {**_defaults(dc.HALLUCINATE_PARAMS), **given,
             "models": [{"name": m["name"]} for m in models]}
        if target:
            p.update({"target_name": target["name"],
                      "target_fasta": target["fasta"],
                      "target_msa": target["msa"]})
        return p

    if ntype == "screen":
        base = {"model": SCREEN_DEFAULT, "inverse_fold": True, "use_msa": True,
                "recycling_steps": 4, "mpnn_weights": "soluble",
                "target_msa": target["msa"] if target else ""}
        p = {**base, **given}
        if p["model"] not in SCREEN_MODELS:
            rep.err(f"{where}: unknown screen model {p['model']!r}. Known: "
                    f"{', '.join(SCREEN_MODELS)}.")
        if p["mpnn_weights"] not in dc.MPNN_WEIGHTS:
            rep.err(f"{where}: unknown MPNN weights {p['mpnn_weights']!r}. "
                    f"Known: {', '.join(dc.MPNN_WEIGHTS)}.")
        return p

    if ntype == "optimize":
        cfg = _merge(dc.default_config(), given.pop("config", None) or {})
        cfg["models"] = _entries(cfg.get("models"), dc.STRUCTURE_MODELS,
                                 f"{where}:models", rep)
        cfg["losses"] = _entries(cfg.get("losses"), dc.LOSS_TERMS,
                                 f"{where}:losses", rep)
        cfg["sequence_models"] = _entries(cfg.get("sequence_models"),
                                          dc.SEQUENCE_MODELS,
                                          f"{where}:sequence_models", rep)
        mpnn = cfg.setdefault("mpnn", {})
        mpnn["terms"] = _entries(mpnn.get("terms"), dc.MPNN_TERMS,
                                 f"{where}:mpnn.terms", rep)
        if target:
            prev = cfg.get("target") or {}
            cfg["target"] = {"fasta": target["fasta"],
                             "msa": target["msa"] or None,
                             "use_msa": bool(target["msa"])
                                        and prev.get("use_msa", True)}
        # init_fasta and the epitope come from the upstream node at run time.
        fan = {"top_k": 0, "array": 1, "array_throttle": 0}
        p = {**fan, **given, "config": cfg, "models": cfg["models"]}
        for e in dc.validate(cfg):
            rep.err(f"{where}: {e}")
        return p

    # merge, and any type added to NODE_TYPES that needs no catalog.
    return given


def build(spec: dict, *, targets: list | None = None,
          base_dir: Path | None = None) -> tuple[PL.Pipeline, _Report]:
    """Terse spec -> a Pipeline the engine can validate, render and submit.

    `base_dir` is where relative `params_file` / `config` paths are resolved
    from — the spec file's own directory, so a pipelines/ directory can be
    moved or checked out anywhere and still find its parts.
    """
    rep = _Report()
    targets = store.list_targets() if targets is None else targets
    top_target = spec.get("target")
    per_type = spec.get("defaults") or {}

    entries = spec.get("nodes")
    if not isinstance(entries, list):
        rep.err("The spec needs a `nodes:` list.")
        entries = []

    nodes: list[PL.Node] = []
    for i, raw in enumerate(entries):
        if not isinstance(raw, dict):
            rep.err(f"nodes[{i}] is not a mapping.")
            continue
        nid = str(raw.get("id") or f"node{i + 1}")
        ntype = raw.get("type")
        where = nid
        if ntype not in PL.NODE_TYPES:
            rep.err(f"{where}: unknown type {ntype!r}. Known: "
                    f"{', '.join(PL.NODE_TYPES)}.")
            continue

        # Four layers, each overriding the one before: the type's `defaults`
        # block, any linked params file(s), an explicit `params:` mapping, then
        # the keys written inline on the node. A preset can therefore be linked
        # and then adjusted in place, which is the only reason splitting a
        # config out is useful rather than merely indirect.
        entry = _merge(per_type.get(ntype) or {}, raw)
        given: dict = {}
        for ref in _as_list(entry.get("params_file")):
            given = _merge(given, _link(ref, base_dir, where, rep))
        given = _merge(given, dict(entry.get("params") or {}))
        given = _merge(given, {k: v for k, v in entry.items()
                               if k not in NODE_KEYS})
        # `config:` given as a path rather than a mapping — an optimize node's
        # objective is the one part big enough to be worth its own file, and it
        # is the same object the Launch tab writes to configs/.
        if isinstance(given.get("config"), str):
            given["config"] = _link(given["config"], base_dir,
                                    f"{where}:config", rep)

        target = _resolve_target(entry.get("target", top_target), targets,
                                 where, rep)
        params = node_params(ntype, given, target, where, rep)

        res = entry.get("resources")
        if res:
            unknown = set(res) - {"time_limit", "mem", "cpus"}
            if unknown:
                rep.warn(f"{where}: resources keys {sorted(unknown)} are "
                         "ignored — only time_limit, mem and cpus are used.")
            params["resources"] = {k: v for k, v in res.items()
                                   if k in ("time_limit", "mem", "cpus")}

        # Array fan-out only works where the stage writes per-task files and
        # seeds off SLURM_ARRAY_TASK_ID — hallucinate and optimize. On the other
        # types every task would run the identical job and write the identical
        # filename in the same directory at the same time, so the node's output
        # is whatever won the race. The forms cannot express this; a file can.
        if int(params.get("array", 1) or 1) > 1 and ntype not in (
                "hallucinate", "optimize"):
            rep.err(f"{where}: array fan-out is only wired for hallucinate and "
                    f"optimize. A {ntype} node's tasks would all write the same "
                    "files in the same directory. Raise num_designs instead.")

        inputs = entry.get("inputs") or []
        if isinstance(inputs, str):
            inputs = [inputs]
        nodes.append(PL.Node(id=nid, type=ntype, params=params,
                             inputs=[str(x) for x in inputs],
                             label=str(entry.get("label") or "")))

    p = PL.Pipeline(name=str(spec.get("name") or "pipeline1"), nodes=nodes)
    rep.errors.extend(PL.validate(p))
    return p, rep


def cluster_of(spec: dict) -> dict:
    """Submit arguments, with the same defaults the Launch config carries."""
    c = spec.get("cluster") or {}
    base = dc.default_config()["cluster"]
    return {"account": c.get("account", base["account"]),
            "gpu_partition": c.get("gpu_partition", base["partition"]),
            "cpu_partition": c.get("cpu_partition", "kempner_interactive"),
            "gpu_res": c.get("gpu") or {},
            "cpu_res": c.get("cpu") or {}}


# --------------------------------------------------------------------------
# The reverse direction: a built pipeline -> a terse spec
#
# This is what makes the two front ends one system rather than two. Build a
# graph by clicking, export it to a file, commit the file. Everything equal to
# what the catalogs would have filled in is dropped, so what remains is exactly
# the set of decisions somebody made.
# --------------------------------------------------------------------------

def _prune(value, base):
    if isinstance(value, dict) and isinstance(base, dict):
        out = {}
        for k, v in value.items():
            d = _prune(v, base.get(k, _MISSING))
            if d is not _DROP:
                out[k] = d
        return out or _DROP
    return _DROP if value == base else value


def _target_name(n: PL.Node, targets: list) -> str | None:
    """Which target this node is aimed at.

    Generate and hallucinate say so outright. An optimize node does not — it
    carries the target inside its design config — and a screen node knows only
    the MSA it was handed, so recover the name by matching those paths, the same
    way the node editor does.
    """
    if n.params.get("target_name"):
        return n.params["target_name"]
    fa = ((n.params.get("config") or {}).get("target") or {}).get("fasta")
    if fa:
        return next((t.name for t in targets if str(t.fasta) == str(fa)), None)
    msa = n.params.get("target_msa")
    if msa:
        return next((t.name for t in targets
                     if t.msa and str(t.msa) == str(msa)), None)
    return None


def _simplify(entries: list, catalog: dict) -> list:
    """Collapse [{name, weight, params}] to the shortest form that re-expands.

    A term running at its catalog weight with catalog parameters is just its
    name; one that differs carries only the differences. This is most of what
    makes an exported spec readable rather than a restatement of the JSON.
    """
    out = []
    for e in entries:
        if not isinstance(e, dict) or "name" not in e:
            out.append(e)
            continue
        meta = catalog.get(e["name"], {})
        d = _prune(e.get("params") or {}, _defaults(meta.get("params", {})))
        d = {} if d is _DROP else dict(d)
        w = e.get("weight", 1.0)
        if not d and w == meta.get("default_weight", 1.0):
            out.append(e["name"])
        elif w == meta.get("default_weight", 1.0):
            out.append({"name": e["name"], **d})
        else:
            out.append({"name": e["name"], "weight": w, **d})
    return out


_ENTRY_LISTS = (("models", dc.STRUCTURE_MODELS), ("losses", dc.LOSS_TERMS),
                ("sequence_models", dc.SEQUENCE_MODELS))


def to_spec(p: PL.Pipeline) -> dict:
    """Round-trips: build(to_spec(P)) reproduces P."""
    targets = store.list_targets()
    names = {t for t in (_target_name(n, targets) for n in p.nodes) if t}
    top = names.pop() if len(names) == 1 else None
    rep = _Report()
    try:
        order = PL.topo_order(p)
    except ValueError:
        order = p.nodes

    out: dict[str, Any] = {"name": p.name}
    if top:
        out["target"] = top
    out["nodes"] = []
    for n in order:
        tname = _target_name(n, targets)
        target = _resolve_target(tname, targets, n.id, rep)
        # The baseline is this node type filled in with nothing said, except the
        # few keys that select which baseline applies.
        seed = {k: v for k, v in n.params.items() if k in ("generator",)}
        base = node_params(n.type, seed, target, n.id, _Report())
        diff = _prune(n.params, base)
        diff = {} if diff is _DROP else dict(diff)
        for k in ("target_name", "target_fasta", "target_cif", "target_msa"):
            diff.pop(k, None)
        if n.type == "hallucinate" and "models" in diff:
            diff["models"] = [m.get("name") for m in diff["models"]]
        if n.type == "optimize":
            # params["models"] is a mirror of config.models that node_params
            # rebuilds, so writing it out would only be a chance to disagree.
            diff.pop("models", None)
            cfg = diff.get("config")
            if isinstance(cfg, dict):
                for key, catalog in _ENTRY_LISTS:
                    if key in cfg:
                        cfg[key] = _simplify(cfg[key], catalog)
                if isinstance(cfg.get("mpnn"), dict) and "terms" in cfg["mpnn"]:
                    cfg["mpnn"]["terms"] = _simplify(cfg["mpnn"]["terms"],
                                                     dc.MPNN_TERMS)
        entry = {"id": n.id, "type": n.type}
        if n.inputs:
            entry["inputs"] = list(n.inputs)
        if n.label:
            entry["label"] = n.label
        if n.type == "generate":
            entry["generator"] = n.params.get("generator", "boltzgen")
            diff.pop("generator", None)
        if tname and tname != top:
            entry["target"] = tname
        if "resources" in diff:
            entry["resources"] = diff.pop("resources")
        entry.update(diff)
        out["nodes"].append(entry)
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

TEMPLATE = '''\
# A mosaic pipeline. Every node is one SLURM job; edges are file artifacts, and
# SLURM does the waiting (--dependency=afterok in topological order).
#
#   ./pipeline_spec.py show   this_file.yaml     # resolved params, per node
#   ./pipeline_spec.py submit this_file.yaml
#
# You only state what differs from the catalogs in design_config.py. Run `show`
# to see everything that was filled in for you.

name: my_pipeline

# A name under <workdir>/targets (NAME.fasta, optional NAME.cif, msa/NAME.a3m).
# Nodes may override it with their own `target:`.
target: DIO3_ECD

cluster:
  account: kempner_bsabatini_lab
  gpu_partition: kempner_h100
  cpu_partition: kempner_interactive
  gpu: {time_limit: "08:00:00", mem: 128G, cpus: 8}

# Applied to every node of that type, under the node's own settings.
defaults:
  generate:
    binder_length: 80

nodes:
  # Roots take a target rather than an upstream node.
  - id: gen_bg
    type: generate
    generator: boltzgen        # or proteina (hotspot-conditioned)
    num_designs: 200
    hotspots: "101-140"        # target residues to engage

  - id: gen_pr
    type: generate
    generator: proteina
    num_designs: 200
    hotspots: "101-140"

  # Fan-in: one candidate pool from heterogeneous generators.
  - id: pool
    type: merge
    inputs: [gen_bg, gen_pr]

  # Refold and rank. Best done with a model DIFFERENT from the generator, so a
  # candidate is not judged by its own author — hence the af2 default.
  - id: screen1
    type: screen
    inputs: [pool]
    model: af2
    inverse_fold: true

  # Gradient refinement under the full multi-objective loss. `config` is the
  # same object the Launch tab builds; anything you omit takes its default.
  - id: opt
    type: optimize
    inputs: [screen1]
    top_k: 32                  # refine only the screen's winners (0 = all)
    array: 4                   # in-node fan-out: 4 GPUs, one per array task
    resources: {time_limit: "12:00:00"}
    config:
      models: [boltz2]
      losses:
        - {name: BinderTargetContact, weight: 4.0, contact_distance: 20.0}
        - {name: WithinBinderContact, weight: 1.0}
      sequence_models:
        - {name: esmc, weight: 0.5}
      optimizer:
        soft: {n_steps: 150}
        sharp: {n_steps: 30}
      run: {batch: 4}
'''


def _fmt_node(p: PL.Pipeline, n: PL.Node, cl: dict) -> str:
    spec = PL.NODE_TYPES[n.type]
    res = PL.node_resources(n, gpu_res=cl["gpu_res"], cpu_res=cl["cpu_res"])
    part = cl["gpu_partition"] if spec["gpu"] else cl["cpu_partition"]
    arr = int(n.params.get("array", 1) or 1)
    head = f"  {n.id:<14} {n.type:<12} {PL._node_caption(n)}"
    tail = (f"{part}  {res['time_limit']}  {res['mem']}  {res['cpus']}cpu"
            + (f"  array 0-{arr - 1}" if arr > 1 else "")
            + ("  gpu:1" if spec["gpu"] else ""))
    edges = f"      <- {', '.join(n.inputs)}" if n.inputs else "      (root)"
    return f"{head}\n{edges}\n      {tail}"


def _report(rep: _Report) -> None:
    sys.stdout.flush()      # keep the report below the table it refers to
    for w in rep.warnings:
        print(f"warning: {w}", file=sys.stderr)
    for e in rep.errors:
        print(f"ERROR: {e}", file=sys.stderr)


def _open(args) -> tuple[dict, PL.Pipeline, _Report]:
    spec = load(args.file)
    p, rep = build(spec, base_dir=Path(args.file).resolve().parent)
    return spec, p, rep


def cmd_show(args) -> int:
    spec, p, rep = _open(args)
    cl = cluster_of(spec)
    order = PL.topo_order(p) if not rep.errors else p.nodes
    print(f"{p.name} — {len(p.nodes)} node(s), account {cl['account']}\n")
    for n in order:
        print(_fmt_node(p, n, cl))
    for note in rep.notes:
        print(f"  {note}")
    if args.params:
        print("\nresolved params:")
        print(json.dumps({n.id: n.params for n in p.nodes}, indent=2))
    print()
    _report(rep)
    if not rep.errors:
        print("ok — submit order: " + ", ".join(n.id for n in order))
    return 1 if rep.errors else 0


def cmd_dag(args) -> int:
    _, p, rep = _open(args)
    print(PL.mermaid(p))
    _report(rep)
    return 1 if rep.errors else 0


def cmd_json(args) -> int:
    _, p, rep = _open(args)
    _report(rep)
    text = p.to_json()
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out} — loadable in the webapp's Pipeline tab.")
    else:
        print(text)
    return 1 if rep.errors else 0


def cmd_submit(args) -> int:
    spec, p, rep = _open(args)
    cl = cluster_of(spec)
    _report(rep)
    if rep.errors:
        print("\nnothing submitted.", file=sys.stderr)
        return 1

    order = PL.topo_order(p)
    print(f"{p.name} — {len(order)} node(s) to {cl['account']}\n")
    for n in order:
        print(_fmt_node(p, n, cl))
    if args.dry_run:
        print("\n--dry-run: nothing was queued.")
        return 0

    ok, log, jobids = PL.submit(
        p, repo=store.REPO, workdir=store.workdir(), account=cl["account"],
        gpu_partition=cl["gpu_partition"], cpu_partition=cl["cpu_partition"],
        gpu_res=cl["gpu_res"], cpu_res=cl["cpu_res"])
    print()
    print(log)
    if not ok:
        print("\nsubmission failed.", file=sys.stderr)
        return 1
    # Keep the source next to the run, beside dag.json — the file you edited is
    # the most readable record of what was launched. Also write the resolved
    # form: with linked params files the source alone is not self-contained,
    # and a linked file that changes tomorrow must not be able to rewrite what
    # this run was.
    run = store.workdir() / "pipelines" / p.name
    src = Path(args.file)
    (run / f"spec{src.suffix}").write_text(src.read_text())
    (run / "spec_resolved.yaml").write_text(dumps(to_spec(p), "yaml"))
    print(f"\nsubmitted. Watch it in the Monitor tab, or:\n"
          f"  squeue -j {','.join(jobids.values())}")
    return 0


def cmd_export(args) -> int:
    """A past run, or a DAG JSON, back out as an editable spec."""
    src = Path(args.run)
    if src.suffix == ".json" and src.exists():
        p = PL.Pipeline.from_json(src.read_text())
    else:
        run_dir = store.workdir() / "pipelines" / args.run
        if not run_dir.is_dir():
            print(f"no run {args.run!r} under {store.workdir() / 'pipelines'}. "
                  f"Have: {', '.join(PL.list_runs(store.workdir())) or '(none)'}",
                  file=sys.stderr)
            return 1
        p = PL.Pipeline.from_run_dir(run_dir)
    fmt = (Path(args.out).suffix.lstrip(".") if args.out else "yaml") or "yaml"
    text = dumps(to_spec(p), fmt)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out} ({len(p.nodes)} nodes).")
    else:
        print(text)
    return 0


def cmd_template(args) -> int:
    if args.out:
        Path(args.out).write_text(TEMPLATE)
        print(f"wrote {args.out}")
    else:
        print(TEMPLATE, end="")
    return 0


def cmd_selftest(args) -> int:
    """Expansion and round-trip, with no cluster and no files required."""
    bad = []
    fake = [store.Target(name="T", fasta=Path("/w/targets/T.fasta"),
                         sequence="ACDEFGHIKLMNPQRSTVWY", msa=Path("/w/msa/T.a3m"),
                         msa_depth=1000, structure=Path("/w/targets/T.cif"))]
    spec = {
        "name": "selftest", "target": "T",
        "defaults": {"generate": {"binder_length": 75}},
        "nodes": [
            {"id": "g1", "type": "generate", "generator": "boltzgen",
             "num_designs": 500, "hotspots": "10-20"},
            {"id": "g2", "type": "generate", "generator": "proteina",
             "num_designs": 40},
            {"id": "pool", "type": "merge", "inputs": ["g1", "g2"]},
            {"id": "s1", "type": "screen", "inputs": ["pool"], "model": "boltz2"},
            {"id": "o1", "type": "optimize", "inputs": ["s1"], "top_k": 16,
             "array": 4, "resources": {"time_limit": "12:00:00"},
             "an_unknown_key": {"nested": [1, 2]},
             "config": {"models": ["boltz2", {"name": "af2", "weight": 0.5}],
                        "losses": {"BinderTargetContact": 4.0,
                                   "HelixLoss": {"weight": 2.0,
                                                 "max_distance": 7.0}},
                        "optimizer": {"soft": {"n_steps": 150}}}},
        ],
    }
    p, rep = build(spec, targets=fake)
    if rep.errors:
        bad += [f"a valid spec reported {e}" for e in rep.errors]

    g1 = p.node("g1").params
    checks = [
        (g1.get("binder_length") == 75, "defaults block did not reach g1"),
        (g1.get("num_designs") == 500, "node value did not override"),
        (g1.get("sampling_steps") == 300, "catalog default not filled"),
        (g1.get("target_cif") == "/w/targets/T.cif", "target not resolved"),
        (p.node("g2").params.get("binder_length") == 75, "defaults missed g2"),
        (p.node("s1").params.get("target_msa") == "/w/msa/T.a3m",
         "screen did not inherit the target MSA"),
    ]
    o1 = p.node("o1").params
    cfg = o1["config"]
    checks += [
        (o1.get("an_unknown_key") == {"nested": [1, 2]},
         "an unknown key was dropped instead of passed through"),
        (o1["resources"]["time_limit"] == "12:00:00", "resources not carried"),
        ([m["name"] for m in cfg["models"]] == ["boltz2", "af2"],
         "model shorthand not expanded"),
        (cfg["models"][1]["weight"] == 0.5, "model weight lost"),
        (cfg["models"][0]["params"].get("sampling_steps") == 25,
         "model params not defaulted"),
        (len(cfg["losses"]) == 2 and cfg["losses"][0]["weight"] == 4.0,
         "loss shorthand not expanded"),
        (cfg["losses"][1]["params"]["max_distance"] == 7.0,
         "inline loss param not applied"),
        (cfg["losses"][0]["params"]["contact_distance"] == 20.0,
         "loss param default not filled"),
        (cfg["optimizer"]["soft"]["n_steps"] == 150, "optimizer not merged"),
        (cfg["optimizer"]["sharp"]["n_steps"] == 25,
         "optimizer merge clobbered the other stage"),
        (cfg["target"]["fasta"] == "/w/targets/T.fasta", "optimize target unset"),
    ]
    bad += [msg for ok, msg in checks if not ok]

    # Out of range is reported, never silently changed.
    over = {"name": "x", "target": "T",
            "nodes": [{"id": "h", "type": "hallucinate", "num_designs": 9999}]}
    q, qrep = build(over, targets=fake)
    if q.node("h").params["num_designs"] != 9999:
        bad.append("an out-of-range value was altered rather than reported")
    if not any("outside the catalog range" in w for w in qrep.warnings):
        bad.append("an out-of-range value was accepted without a warning")

    # Linked files: layered under the node's own keys, resolved relative to the
    # spec, and one level deep only.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "bg_preset.yaml").write_text(
            "num_designs: 300\nbinder_length: 90\nsampling_steps: 400\n")
        (td / "objective.yaml").write_text(
            "models: [boltz2]\nlosses: [BinderTargetContact]\n"
            "optimizer:\n  soft: {n_steps: 200}\n")
        (td / "chained.yaml").write_text("params_file: bg_preset.yaml\n")
        linked = {
            "name": "linked", "target": "T",
            "nodes": [
                {"id": "g", "type": "generate", "generator": "boltzgen",
                 "params_file": "bg_preset.yaml", "num_designs": 500},
                {"id": "o", "type": "optimize", "inputs": ["g"],
                 "config": "objective.yaml"},
                {"id": "bad", "type": "merge", "inputs": ["g", "o"],
                 "params_file": "chained.yaml"},
            ],
        }
        lp, lrep = build(linked, targets=fake, base_dir=td)
        gp = lp.node("g").params
        if gp.get("binder_length") != 90:
            bad.append("params_file did not reach the node")
        if gp.get("num_designs") != 500:
            bad.append("an inline key did not override the linked file")
        if gp.get("sampling_steps") != 400:
            bad.append("params_file lost a key the node did not restate")
        ocfg = lp.node("o").params.get("config", {})
        if [m["name"] for m in ocfg.get("models", [])] != ["boltz2"]:
            bad.append("a linked optimize config was not loaded")
        if ocfg.get("optimizer", {}).get("soft", {}).get("n_steps") != 200:
            bad.append("a linked config was not merged over the defaults")
        if ocfg.get("optimizer", {}).get("sharp", {}).get("n_steps") != 25:
            bad.append("a linked config clobbered the defaults it did not set")
        if not any("one level deep" in e for e in lrep.errors):
            bad.append("a chained params_file was accepted")
        _, mrep = build(
            {"name": "m", "target": "T",
             "nodes": [{"id": "g", "type": "generate",
                        "params_file": "nope.yaml"}]},
            targets=fake, base_dir=td)
        if not any("no such file" in e for e in mrep.errors):
            bad.append("a missing linked file was not reported")

    # Round trip: export and re-expand must reproduce the same params.
    back, brep = build(to_spec(p), targets=fake)
    if brep.errors:
        bad += [f"round trip reported {e}" for e in brep.errors]
    for a in p.nodes:
        b = back.node(a.id)
        if b is None:
            bad.append(f"round trip lost node {a.id}")
        elif b.params != a.params or b.inputs != a.inputs:
            bad.append(f"round trip changed {a.id}")

    for m in bad:
        print(f"FAIL: {m}")
    print("PASS" if not bad else "FAILED")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Define a pipeline in a file and submit it.")
    ap.add_argument("--workdir", help="Working directory holding targets/, msa/, "
                                      "pipelines/ (default: the webapp's).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def with_file(name, fn, help_):
        s = sub.add_parser(name, help=help_)
        s.add_argument("file")
        s.set_defaults(fn=fn)
        return s

    s = with_file("show", cmd_show, "Resolved nodes, resources and order.")
    s.add_argument("--params", action="store_true",
                   help="also print every resolved param as JSON")
    with_file("dag", cmd_dag, "Print the graph as mermaid.")
    s = with_file("json", cmd_json, "Emit the DAG JSON the webapp loads.")
    s.add_argument("-o", "--out")
    s = with_file("submit", cmd_submit, "Queue the whole DAG.")
    s.add_argument("--dry-run", action="store_true",
                   help="print the plan; queue nothing")

    s = sub.add_parser("export", help="A submitted run (or a DAG JSON) as a spec.")
    s.add_argument("run", help="pipeline name under <workdir>/pipelines, or a .json")
    s.add_argument("-o", "--out")
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("template", help="Write a commented starter spec.")
    s.add_argument("-o", "--out")
    s.set_defaults(fn=cmd_template)

    s = sub.add_parser("selftest", help="Check expansion and round trip.")
    s.set_defaults(fn=cmd_selftest)

    args = ap.parse_args(argv)
    if args.workdir:
        store.set_workdir(args.workdir)
    else:
        # Same working directory the webapp is looking at, so a target you
        # created there is a target this can name. Falling back to store's
        # built-in default is fine, but say so rather than resolving paths
        # against a directory the user did not expect.
        try:
            import session
            store.set_workdir(session.active().path)
        except Exception as exc:      # noqa: BLE001 — any failure is the same
            print(f"note: no webapp session ({exc}); using {store.workdir()}",
                  file=sys.stderr)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
