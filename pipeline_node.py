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

    # Target paths must be absolute: the stage scripts run inside the container
    # where the host cwd does not apply, so a relative path is read as a raw
    # sequence (run_design) or simply not found (gemmi). node_dir is
    # workdir/pipelines/<name>/<node>, so workdir is parents[2] and relative
    # paths in a hand-edited DAG resolve against it. The webapp already emits
    # absolute paths; this just makes hand-authored DAGs behave the same.
    workdir = node_dir.parents[2] if len(node_dir.parents) >= 3 else node_dir.parent

    def _abs(p):
        if not p:
            return p
        q = Path(p)
        return str(q if q.is_absolute() else (workdir / q))

    for _k in ("target_cif", "target_fasta", "target_msa"):
        if params.get(_k):
            params[_k] = _abs(params[_k])

    # In-node fan-out: when this node was submitted as a SLURM array, each task
    # runs the same stage with a distinct seed and writes its own
    # designs_seed{task}.json. run_design offsets its seed*batch window into the
    # seed FASTA, so array tasks explore different noise / refine different
    # candidates. Outside an array this is just 0.
    task_seed = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

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
                "--seed", str(task_seed),
                "--out", str(node_dir)]
        # The generator's own knobs. The UI has always drawn these and a spec
        # file can set them, but none of them used to reach the stage script: a
        # node asking for 600 sampling steps ran the default 300 and said
        # nothing about it. This table is the honest statement of what a
        # generate node forwards — only flags the chosen script actually
        # accepts, since an unknown one is an argparse error at run time.
        FORWARD = {
            "boltzgen": {"target_chain": "--target-chain",
                         "n_helices": "--n-helices",
                         "loop_length": "--loop-length",
                         "recycling_steps": "--recycling-steps",
                         "sampling_steps": "--sampling-steps",
                         "step_scale": "--step-scale",
                         "noise_scale": "--noise-scale",
                         "contact_cutoff": "--contact-cutoff"},
            "proteina": {"target_chain": "--target-chain",
                         "chunk": "--chunk"},
        }
        for key, flag in FORWARD.get(gen, {}).items():
            if params.get(key) is not None:
                args += [flag, str(params[key])]
        if params.get("hotspots"):
            args += ["--hotspots", params["hotspots"]]
            if gen == "boltzgen":
                args += ["--hotspot-shell", str(params.get("hotspot_shell", 6.0))]
        _exec(args)

    elif ntype == "hallucinate":
        # mosaic's own method run as a generator: optimize from NOISE (no
        # --init-fasta, so run_design starts near the uniform simplex point)
        # against the chosen structure model(s), then expose the result as a
        # proposal_set so a merge/screen downstream consumes it exactly like a
        # BoltzGen or Proteina set. num_designs is the vmapped --batch: it runs
        # in a single GPU job, so it is bounded by GPU memory, not thousands.
        models = "+".join(m.get("name", "") for m in params.get("models", [])) \
            or "boltz2"
        args = [str(REPO / "run_design.py"),
                "--target", params["target_fasta"],
                "--binder-length", str(params.get("binder_length", 80)),
                "--models", models,
                "--batch", str(params.get("num_designs", 8)),
                "--soft-steps", str(params.get("soft_steps", 60)),
                "--sharp-steps", str(params.get("sharp_steps", 25)),
                "--seed", str(task_seed), "--out", str(node_dir)]
        if params.get("target_msa"):
            args += ["--target-msa", params["target_msa"]]
        _exec(args)
        # Do NOT materialize the proposal_set here: under array fan-out this
        # branch runs in every task, so writing the shared manifest would race.
        # The consumer (merge/screen) gathers all designs_seed*.json instead —
        # merge wraps a raw design dir below, screen already does. No epitope is
        # attached: hallucination picked the site itself.
        print(f"hallucinated batch (seed {task_seed}) -> designs_seed{task_seed}.json")

    elif ntype == "merge":
        # Combine upstream proposal sets. Light file work — done here rather than
        # in a stage script. An input may be a materialized proposal_set (has a
        # manifest) or a raw design dir from an arrayed/hallucinate node (only
        # designs_seed*.json); wrap the latter once, here, where it runs single-
        # task, so merge_sets sees a manifest for every part.
        parts = []
        for i, d in enumerate(inputs):
            d = Path(d)
            if not (d / "manifest.json").exists():
                w = node_dir / f"part_{i}"
                _wrap_designs_as_proposal_set(d, w, P)
                d = w
            parts.append(d)
        merged = P.merge_sets(parts, node_dir, node_dir.name)
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
        # Seed FASTA: a screen node ranks into screened.fasta (best first); a
        # proposal/merge node has proposals.fasta.
        seed = (src / "screened.fasta")
        if not seed.exists():
            seed = src / "proposals.fasta"
        # Optionally refine only the top-K ranked candidates: screened.fasta is
        # best-first, so truncating it keeps the funnel's winners and spends the
        # expensive gradient stage only on them.
        top_k = int(params.get("top_k", 0) or 0)
        if top_k > 0 and seed.exists():
            kept = _head_fasta(seed, top_k)
            seed = node_dir / "seed_topk.fasta"
            seed.write_text(kept)
            print(f"optimize: seeding from top {top_k} of {src.name}")
        cfg = dict(params.get("config", {}))
        tsec = cfg.get("target")
        if isinstance(tsec, dict):
            if tsec.get("fasta"):
                tsec["fasta"] = _abs(tsec["fasta"])
            if tsec.get("msa"):
                tsec["msa"] = _abs(tsec["msa"])
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
        # One array task's share of the work. Fan-out is the submitter's job
        # (params["array"] becomes --array), so all that is needed here is to
        # run with this task's seed: run_design offsets seed*batch into the seed
        # FASTA, so each task refines a different window of the top-K and writes
        # its own designs_seed<i>.json.
        _exec([str(REPO / "run_design.py"), "--config", str(cfg_path),
               "--target", cfg["target"]["fasta"], "--seed", str(task_seed),
               "--out", str(node_dir)])
    else:
        raise SystemExit(f"unknown node type {ntype!r}")

    (node_dir / ".done").write_text("ok\n")
    print(f"== node {node_dir.name} done ==")
    return 0


def _head_fasta(path: Path, k: int) -> str:
    """The first k records of a FASTA (records are header+sequence pairs)."""
    out, seen = [], 0
    for ln in path.read_text().splitlines():
        if ln.startswith(">"):
            if seen >= k:
                break
            seen += 1
        out.append(ln)
    return "\n".join(out) + "\n"


def _wrap_designs_as_proposal_set(design_dir: Path, out: Path, P) -> None:
    import glob
    seqs = []
    cfg = {}
    for f in sorted(glob.glob(str(design_dir / "designs_seed*.json"))):
        d = json.loads(Path(f).read_text())
        seqs += [r["sequence"] for r in d.get("results", [])]
        # run_design embeds its resolved config in each designs_seed*.json (it
        # writes no separate config_seed file), so take target + length from the
        # first shard. This is what lets a hallucinated set — which merges the
        # array's designs_seed{0..N}.json — carry a real target and length.
        if not cfg:
            cfg = d.get("config", {})
    if not seqs:
        raise SystemExit(f"no designs to screen in {design_dir}")
    # A campaign that DID record a separate config file still wins if present.
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
