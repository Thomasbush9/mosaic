#!/usr/bin/env python
"""Execute one pipeline node: assemble inputs, dispatch to the right stage.

Reads NODE_DIR/node.json ({type, params, inputs=[upstream dirs]}), and because it
runs only after its SLURM dependency is satisfied, the upstream outputs exist.
This is where the artifact contract between node types lives — everything the
pipeline engine promised on the edges is honoured here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0:
        raise SystemExit(f"stage failed ({r.returncode}): {' '.join(cmd)}")


def _exec(script_args: list[str]) -> None:
    """Run a python stage inside the container via the wrapper."""
    _run([str(REPO / "singularity" / "mosaic-exec.sh"), "python"] + script_args)


def main() -> int:
    node_dir = Path(os.environ["NODE_DIR"])
    node = json.loads((node_dir / "node.json").read_text())
    ntype, params, inputs = node["type"], node["params"], node["inputs"]
    print(f"== node {node_dir.name}: {ntype} ==", flush=True)
    sys.path.insert(0, str(REPO))
    import proposals as P

    if ntype == "generate":
        gen = params.get("generator", "boltzgen")
        script = ("generate_proteina.py" if gen == "proteina"
                  else "generate_boltzgen.py")
        args = [str(REPO / script),
                "--target-cif", params["target_cif"],
                "--target-fasta", params.get("target_fasta", ""),
                "--target-name", params.get("target_name", ""),
                "--binder-length", str(params.get("binder_length", 80)),
                "--num-designs", str(params.get("num_designs", 8)),
                "--out", str(node_dir)]
        if params.get("hotspots"):
            args += ["--hotspots", params["hotspots"]]
            if gen == "boltzgen":
                args += ["--hotspot-shell", str(params.get("hotspot_shell", 6.0))]
        _exec(args)

    elif ntype == "merge":
        # Combine upstream proposal sets. Light file work — done here rather than
        # in a stage script.
        merged = P.merge_sets([Path(d) for d in inputs], node_dir, node_dir.name)
        print(f"merged {len(inputs)} sets -> {merged.n_designs} designs")

    elif ntype == "screen":
        src = Path(inputs[0])
        # A screen can point at an optimize node (design_set), whose output is a
        # per-seed designs FASTA, not a proposal set. Wrap it so screen_proposals
        # sees a manifest — the artifact-contract adapter promised by the engine.
        if not (src / "manifest.json").exists():
            _wrap_designs_as_proposal_set(src, node_dir / "input_set", P)
            src = node_dir / "input_set"
        args = [str(REPO / "screen_proposals.py"),
                "--proposals", str(src),
                "--model", params.get("model", "boltz2"),
                "--recycling-steps", str(params.get("recycling_steps", 4)),
                "--out", str(node_dir)]
        if params.get("use_msa") and params.get("target_msa"):
            args += ["--target-msa", params["target_msa"]]
        args += (["--inverse-fold"] if params.get("inverse_fold", True)
                 else ["--no-inverse-fold"])
        args += ["--mpnn-weights", params.get("mpnn_weights", "soluble")]
        _exec(args)

    elif ntype == "optimize":
        src = Path(inputs[0])
        # Seed FASTA: a screen node ranks into screened.fasta; a proposal/merge
        # node has proposals.fasta.
        seed = (src / "screened.fasta")
        if not seed.exists():
            seed = src / "proposals.fasta"
        cfg = dict(params.get("config", {}))
        # Carry the epitope from the upstream manifest into the contact loss, so
        # the optimizer aims at the interface the generator chose.
        man = src / "manifest.json"
        ep = json.loads(man.read_text()).get("epitope_idx", []) if man.exists() else []
        cfg.setdefault("binder", {})["init_fasta"] = str(seed)
        for l in cfg.get("losses", []):
            if l["name"] == "BinderTargetContact" and ep:
                l.setdefault("params", {})["epitope_idx"] = ep
        cfg_path = node_dir / "config.json"
        cfg_path.write_text(json.dumps(cfg, indent=2))
        n_tasks = int(params.get("array", 1))
        # Optimize itself may be an array; here we run the single-task form and
        # let the pipeline node be the unit. (Array fan-out within a node is a
        # future extension.)
        _exec([str(REPO / "run_design.py"), "--config", str(cfg_path),
               "--target", cfg["target"]["fasta"], "--seed", "0",
               "--out", str(node_dir)])
    else:
        raise SystemExit(f"unknown node type {ntype!r}")

    (node_dir / ".done").write_text("ok\n")
    print(f"== node {node_dir.name} done ==")
    return 0


def _wrap_designs_as_proposal_set(design_dir: Path, out: Path, P) -> None:
    import glob
    seqs = []
    for f in sorted(glob.glob(str(design_dir / "designs_seed*.json"))):
        d = json.loads(Path(f).read_text())
        seqs += [r["sequence"] for r in d.get("results", [])]
    if not seqs:
        raise SystemExit(f"no designs to screen in {design_dir}")
    # Reach back to the campaign's recorded config for target + length.
    cfg = {}
    for f in sorted(glob.glob(str(design_dir / "config_seed*.json"))):
        cfg = json.loads(Path(f).read_text()); break
    tgt = cfg.get("target", {})
    ps = P.ProposalSet(
        name=out.name, generator="optimize",
        target_name="target", target_fasta=tgt.get("fasta", ""),
        target_structure=None,
        binder_length=cfg.get("binder", {}).get("length", len(seqs[0])),
        n_designs=len(seqs), sequences=seqs, epitope_idx=[], params={}, notes=[])
    P.write(out, ps)


if __name__ == "__main__":
    raise SystemExit(main())
