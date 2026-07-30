"""Pipeline DAG: nodes are stages, edges are file artifacts between SLURM jobs.

This is the coarse pipeline graph, not the fine differentiable one. A node is one
job (generate / screen / optimize / merge), each with its OWN internal config —
for an optimize node that internal config is the full multi-model loss graph, the
same one the Launch tab builds. An edge is a file artifact (a proposal set, a
screen ranking, a design set), NOT a differentiable connection: nothing
back-propagates across it, because the stages are separate processes.

Execution is by SLURM dependency. Nodes are submitted in topological order, each
with --dependency=afterok on its upstream jobs' ids, so SLURM does the waiting,
fan-out and failure propagation. The output directory of every node is derived
from its id, so downstream input paths are known at submit time even though the
upstream jobs have not run yet.

Pure logic — no Streamlit, no mosaic. The DAG JSON is the reproducible record of
a whole multi-stage run, the same principle as the per-campaign config files.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# What each node type produces and what it can consume. The edge check uses
# these: an edge is legal iff the source's `produces` is in the target's
# `accepts`. This is what makes "two validators off one generator" or "merge
# several generators" express-or-reject at build time rather than at run time.
NODE_TYPES: dict[str, dict[str, Any]] = {
    "generate": {
        "label": "Generate",
        "produces": "proposal_set",
        "accepts": [],                       # root: needs a target, not a node
        "min_inputs": 0, "max_inputs": 0,
        "gpu": True,
        "blurb": "BoltzGen or Proteina propose binder backbones + sequences.",
    },
    "hallucinate": {
        "label": "Hallucinate",
        "produces": "proposal_set",         # so merge/screen consume it uniformly
        "accepts": [],                       # root: designs from noise, not a node
        "min_inputs": 0, "max_inputs": 0,
        "gpu": True,
        "blurb": "Gradient-design binders from random NOISE against a structure "
                 "model's confidence — no backbone prior, mosaic's core method "
                 "run as generation. Each design is a full optimization, so it "
                 "is far costlier per candidate than BoltzGen; the per-job count "
                 "is bounded by GPU memory (vmapped trajectories).",
    },
    "merge": {
        "label": "Merge",
        "produces": "proposal_set",
        "accepts": ["proposal_set"],
        "min_inputs": 2, "max_inputs": 16,
        "gpu": False,
        "blurb": "Combine several proposal sets into one pool (fan-in). Mixes "
                 "heterogeneous generators — BoltzGen, Proteina, Hallucinate — "
                 "into one candidate pool for a shared screen.",
    },
    "screen": {
        "label": "Screen",
        "produces": "ranked_set",
        "accepts": ["proposal_set", "design_set", "ranked_set"],
        "min_inputs": 1, "max_inputs": 1,
        "gpu": True,
        "blurb": "Refold each candidate with a chosen model, rank by confidence. "
                 "Optional ProteinMPNN inverse-fold. Two screen nodes off one "
                 "source = two independent validators.",
    },
    "optimize": {
        "label": "Optimize",
        "produces": "design_set",
        "accepts": ["proposal_set", "ranked_set"],
        "min_inputs": 1, "max_inputs": 1,
        "gpu": True,
        "blurb": "Gradient-refine the incoming sequences under a full "
                 "multi-objective loss (the Launch config).",
    },
}


@dataclass
class Node:
    id: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)   # upstream node ids
    label: str = ""


@dataclass
class Pipeline:
    name: str
    nodes: list[Node] = field(default_factory=list)

    def node(self, nid: str) -> Node | None:
        return next((n for n in self.nodes if n.id == nid), None)

    def to_json(self) -> str:
        return json.dumps({"name": self.name,
                           "nodes": [asdict(n) for n in self.nodes]}, indent=2)

    @staticmethod
    def from_json(text: str) -> "Pipeline":
        d = json.loads(text)
        return Pipeline(name=d["name"],
                        nodes=[Node(**n) for n in d.get("nodes", [])])

    @staticmethod
    def from_run_dir(run_dir) -> "Pipeline":
        """Reconstruct a submitted pipeline from its on-disk node.json shards.

        submit() writes each node's {type, params, inputs=[upstream dirs]}, and
        the upstream dir basenames are the node ids — so the whole DAG is
        recoverable even for a run made before dag.json was saved. This is what
        lets you inspect a past pipeline you never downloaded the JSON for.
        """
        run_dir = Path(run_dir)
        nodes = []
        for nj in sorted(run_dir.glob("*/node.json")):
            try:
                d = json.loads(nj.read_text())
            except Exception:
                continue
            nodes.append(Node(id=nj.parent.name, type=d.get("type", ""),
                              params=d.get("params", {}),
                              inputs=[Path(p).name for p in d.get("inputs", [])]))
        return Pipeline(name=run_dir.name, nodes=nodes)


def list_runs(workdir) -> list[str]:
    """Names of submitted pipelines on disk (dirs with node.json shards)."""
    proot = Path(workdir) / "pipelines"
    if not proot.exists():
        return []
    return sorted(p.name for p in proot.glob("*")
                  if p.is_dir() and next(iter(p.glob("*/node.json")), None))


# --------------------------------------------------------------------------
# Validation and ordering
# --------------------------------------------------------------------------

def validate(p: Pipeline) -> list[str]:
    errs: list[str] = []
    ids = [n.id for n in p.nodes]
    if len(ids) != len(set(ids)):
        errs.append("Duplicate node ids.")
    idset = set(ids)
    for n in p.nodes:
        spec = NODE_TYPES.get(n.type)
        if spec is None:
            errs.append(f"{n.id}: unknown type {n.type!r}.")
            continue
        for src in n.inputs:
            if src not in idset:
                errs.append(f"{n.id}: input {src!r} is not a node.")
                continue
            up = p.node(src)
            up_spec = NODE_TYPES.get(up.type, {})
            if up_spec.get("produces") not in spec["accepts"]:
                errs.append(
                    f"{n.id} ({n.type}) cannot take a "
                    f"{up_spec.get('produces')} from {src} ({up.type}).")
        k = len(n.inputs)
        if k < spec["min_inputs"]:
            errs.append(f"{n.id}: needs at least {spec['min_inputs']} input(s).")
        if k > spec["max_inputs"]:
            errs.append(f"{n.id}: takes at most {spec['max_inputs']} input(s).")
        errs += _node_requirements(n)
    if not _acyclic(p):
        errs.append("Pipeline has a cycle — it must be a DAG.")
    return errs


def _node_requirements(n: Node) -> list[str]:
    """What each node type must carry for pipeline_node.py to run it.

    These mirror the actual reads in pipeline_node.py rather than a general
    notion of completeness. Every one of them was previously a run-time death
    after the job had queued, waited and started: a generate node with no
    structure raises KeyError('target_cif') at pipeline_node.py:71, and an
    optimize node whose config never got a target hands run_design a --target of
    None. Catching them here costs nothing and turns a lost GPU allocation into
    a red line in the editor.
    """
    errs: list[str] = []
    p = n.params
    if n.type == "generate":
        if not p.get("target_fasta"):
            errs.append(f"{n.id}: generate needs a target.")
        if not p.get("target_cif"):
            errs.append(
                f"{n.id}: {p.get('generator', 'this generator')} designs against "
                "geometry and needs a target STRUCTURE — predict one in the "
                "Generate tab first.")
    elif n.type == "hallucinate":
        if not p.get("target_fasta"):
            errs.append(f"{n.id}: hallucinate needs a target.")
        if not p.get("models"):
            errs.append(f"{n.id}: pick at least one structure model to fold against.")
    elif n.type == "optimize":
        cfg = p.get("config") or {}
        if not (cfg.get("target") or {}).get("fasta"):
            errs.append(
                f"{n.id}: the optimize objective has no target — pick one on the "
                "node, or import a Launch config that has one.")
        if not cfg.get("models"):
            errs.append(f"{n.id}: the optimize objective has no structure model.")
        if not cfg.get("losses"):
            errs.append(f"{n.id}: the optimize objective has no loss terms.")
    return errs


def _acyclic(p: Pipeline) -> bool:
    try:
        topo_order(p)
        return True
    except ValueError:
        return False


def topo_order(p: Pipeline) -> list[Node]:
    """Kahn's algorithm; raises ValueError on a cycle."""
    incoming = {n.id: set(n.inputs) for n in p.nodes}
    ready = [n for n in p.nodes if not incoming[n.id]]
    order, seen = [], set()
    while ready:
        n = ready.pop(0)
        order.append(n)
        seen.add(n.id)
        for m in p.nodes:
            if n.id in incoming[m.id] and m.id not in seen:
                incoming[m.id].discard(n.id)
                if not incoming[m.id] and m not in ready:
                    ready.append(m)
    if len(order) != len(p.nodes):
        raise ValueError("cycle")
    return order


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

_STATE_CLASS = {"COMPLETED": "done", "RUNNING": "run", "PENDING": "run",
                "FAILED": "fail", "CANCELLED": "fail", "TIMEOUT": "fail",
                "OUT_OF_MEMORY": "fail"}


def mermaid(p: Pipeline, states: dict[str, str] | None = None) -> str:
    """A flowchart of the DAG, nodes coloured by SLURM state."""
    states = states or {}
    lines = ["flowchart TD"]
    for n in p.nodes:
        spec = NODE_TYPES.get(n.type, {})
        label = n.label or n.id
        sub = _node_caption(n)
        # Mermaid renders <br/> but not <small>; keep the caption plain.
        text = f"{label}<br/>{sub}" if sub else label
        lines.append(f'    {n.id}["{text}"]')
    for n in p.nodes:
        for src in n.inputs:
            up = p.node(src)
            art = NODE_TYPES.get(up.type, {}).get("produces", "")
            lines.append(f"    {src} -->|{art}| {n.id}")
    # State classes.
    lines += [
        "    classDef done fill:#1baf7a,stroke:#199e70,color:#fff",
        "    classDef run fill:#eda100,stroke:#c98500,color:#fff",
        "    classDef fail fill:#e34948,stroke:#c53030,color:#fff",
        "    classDef idle fill:#2a78d6,stroke:#2060b0,color:#fff",
    ]
    for n in p.nodes:
        cls = _STATE_CLASS.get(states.get(n.id, ""), "idle")
        lines.append(f"    class {n.id} {cls}")
    return "\n".join(lines)


def _node_caption(n: Node) -> str:
    pr = n.params
    if n.type == "generate":
        return f"{pr.get('generator', '?')} · {pr.get('num_designs', '?')}×"
    if n.type == "screen":
        mp = "+MPNN" if pr.get("inverse_fold", True) else ""
        return f"{pr.get('model', '?')}{mp}"
    if n.type == "hallucinate":
        models = "+".join(m.get("name", "") for m in pr.get("models", []))
        total = int(pr.get("array", 1) or 1) * int(pr.get("num_designs", 0) or 0)
        return f"{models or 'loss'} · {total or '?'}× from noise"
    if n.type == "optimize":
        models = "+".join(m.get("name", "") for m in pr.get("models", []))
        tk = pr.get("top_k")
        arr = int(pr.get("array", 1) or 1)
        return (models or "loss") + (f" · ×{arr}" if arr > 1 else "") \
            + (f" · top {tk}" if tk else "")
    return ""


# --------------------------------------------------------------------------
# Execution — SLURM dependency chain
# --------------------------------------------------------------------------

def _out_dir(root: Path, p: Pipeline, n: Node) -> Path:
    return root / "pipelines" / p.name / n.id


def node_resources(n: Node, *, gpu_res: dict | None = None,
                   cpu_res: dict | None = None) -> dict:
    """Walltime, memory and CPUs for one node, resolved in three layers.

    The built-in default for the node's kind, then the pipeline-wide setting,
    then the node's own `params["resources"]`. The per-node layer exists because
    the stages have genuinely different shapes — a generate node is one diffusion
    pass, an optimize node is hundreds of gradient steps — so a single pipeline
    walltime either wastes allocation or kills the long node.

    Exposed rather than inlined in submit() so a dry run can print exactly what
    a real submit would ask for, instead of a second copy of the same rules.
    """
    gpu = NODE_TYPES.get(n.type, {}).get("gpu", True)
    res = {"cpus": 8, "mem": "96G", "time_limit": "06:00:00"} if gpu \
        else {"cpus": 2, "mem": "16G", "time_limit": "01:00:00"}
    for layer in ((gpu_res if gpu else cpu_res) or {},
                  n.params.get("resources") or {}):
        res.update({k: v for k, v in layer.items()
                    if k in ("cpus", "mem", "time_limit") and v})
    return res


def submit(p: Pipeline, *, repo: Path, workdir: Path, account: str,
           gpu_partition: str, cpu_partition: str,
           gpu_res: dict | None = None,
           cpu_res: dict | None = None) -> tuple[bool, str, dict[str, str]]:
    """Submit the whole DAG. Returns (ok, log, {node_id: job_id}).

    Each node runs pipeline_node.py, which assembles its inputs from the upstream
    output directories at run time (they exist by then — the dependency
    guarantees it) and dispatches to the right underlying stage.
    """
    errs = validate(p)
    if errs:
        return False, "cannot submit:\n" + "\n".join(errs), {}

    order = topo_order(p)
    jobids: dict[str, str] = {}
    log = []
    for n in order:
        out = _out_dir(workdir, p, n)
        out.mkdir(parents=True, exist_ok=True)
        # Per-node param file — avoids --export comma issues entirely.
        (out / "node.json").write_text(json.dumps(
            {"type": n.type, "params": n.params,
             "inputs": [str(_out_dir(workdir, p, p.node(s))) for s in n.inputs]},
            indent=2))

        spec = NODE_TYPES[n.type]
        dep = ""
        up_ids = [jobids[s] for s in n.inputs if s in jobids]
        if up_ids:
            dep = f"--dependency=afterok:{':'.join(up_ids)}"

        part = gpu_partition if spec["gpu"] else cpu_partition
        cmd = ["sbatch", "--parsable",
               f"--job-name=pipe-{p.name}-{n.id}",
               f"--account={account}", f"--partition={part}"]
        res = node_resources(n, gpu_res=gpu_res, cpu_res=cpu_res)
        if spec["gpu"]:
            cmd.append("--gres=gpu:1")
        cmd += [f"--cpus-per-task={res['cpus']}",
                f"--mem={res['mem']}",
                f"--time={res['time_limit']}"]
        # In-node fan-out: a node with array>1 becomes a SLURM array, one GPU
        # task per index. Each task seeds off SLURM_ARRAY_TASK_ID and writes its
        # own designs_seed{i}.json; the downstream node gathers by globbing them
        # (screen/merge already read every designs_seed*.json). afterok on an
        # array's parent id waits for ALL tasks, so the dependency chain still
        # holds with no extra node. This is how hallucinate/optimize scale past
        # one GPU's batch without adding a merge node to the graph.
        arr = int(n.params.get("array", 1) or 1)
        outfmt = "job-%j.out"
        if arr > 1:
            thr = int(n.params.get("array_throttle", 0) or 0)
            cmd.append(f"--array=0-{arr - 1}" + (f"%{thr}" if thr else ""))
            outfmt = "job-%A_%a.out"       # %j collides across array tasks
        if dep:
            cmd.append(dep)
        cmd += [f"--output={out}/{outfmt}",
                f"--export=ALL,NODE_DIR={out}",
                "singularity/pipeline-node.sbatch"]

        r = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
        if r.returncode != 0:
            log.append(f"{n.id}: SUBMIT FAILED\n{r.stderr.strip()}")
            return False, "\n".join(log), jobids
        jobids[n.id] = r.stdout.strip().split(";")[0]
        log.append(f"{n.id} ({n.type}) -> job {jobids[n.id]}"
                   + (f"  after {','.join(up_ids)}" if up_ids else ""))
    run_root = _out_dir(workdir, p, order[0]).parent
    (run_root / "jobids.json").write_text(json.dumps(jobids, indent=2))
    # Save the whole DAG so it can be reopened later without the node.json shards.
    (run_root / "dag.json").write_text(p.to_json())
    return True, "\n".join(log), jobids


def states(jobids: dict[str, str]) -> dict[str, str]:
    """Current SLURM state per node."""
    out = {}
    for nid, jid in jobids.items():
        r = subprocess.run(
            ["sacct", "-j", str(jid), "-n", "-P", "--format=State"],
            capture_output=True, text=True)
        st = ""
        for line in r.stdout.strip().splitlines():
            st = line.split("|")[0].split()[0] if line.strip() else st
            break
        out[nid] = st
    return out
