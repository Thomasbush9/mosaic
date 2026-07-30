# Pipeline spec files

One file per multi-stage run: a DAG of SLURM jobs, submitted with
`--dependency=afterok` so SLURM does the waiting. Full format reference in
`docs/PIPELINE_FILE.md`.

```bash
PY=webapp/.venv/bin/python                     # a bare python3 here is 3.6

$PY pipeline_spec.py show   pipelines/dio3_2gen.yaml           # resolved nodes
$PY pipeline_spec.py show   pipelines/dio3_2gen.yaml --params  # every value
$PY pipeline_spec.py submit pipelines/dio3_2gen.yaml --dry-run
$PY pipeline_spec.py submit pipelines/dio3_2gen.yaml
```

## What is here

| File | |
|---|---|
| `dio3_2gen.yaml` | **two generators → merge → optimize.** The graph; parameters are linked from `presets/`. |
| `presets/boltzgen_dio3.yaml` | every BoltzGen knob, annotated |
| `presets/proteina_dio3.yaml` | the second, independent generator |
| `presets/objective_dio3_consensus.yaml` | the optimize node's design config — Boltz-2 + AF2 over distogram-only losses |
| `dio3_2way_500b.yaml` | the earlier five-node run, exported from `pipelines/dio3_2way_500b` with `pipeline_spec.py export` |

## The split layout, and when it earns its keep

`dio3_2gen.yaml` holds only the graph — node types, edges, fan-out, walltimes —
and each node links its parameters:

```yaml
  - id: gen_boltzgen
    type: generate
    generator: boltzgen
    params_file: presets/boltzgen_dio3.yaml

  - id: optimize
    type: optimize
    inputs: [pool]
    config: presets/objective_dio3_consensus.yaml
    array: 32
```

Override order is `defaults:` → linked file → `params:` → keys written inline on
the node, so a preset can be linked and then adjusted in place:

```yaml
  - id: gen_variant
    type: generate
    generator: boltzgen
    params_file: presets/boltzgen_dio3.yaml
    noise_scale: 1.1          # everything else from the preset
```

That last override is the whole reason to split. Without it you would need a
near-copy of the preset per variation. Paths resolve relative to this directory
first, then to the working directory; linking is one level deep.

Splitting is worth it when a setting is genuinely shared or large enough to bury
the graph — an objective reused across campaigns, a generator preset you have
tuned. It is not worth it by default: `dio3_2way_500b.yaml` is 37 readable lines
inline, and splitting it six ways would mean six files to open before you know
what ran.

`submit` writes both the spec as submitted and `spec_resolved.yaml`, the fully
inlined equivalent, next to `dag.json` — so retuning a preset later cannot
rewrite what a past run was.

## How `dio3_2gen` is sized

The three numbers are not independent, and getting this wrong is silent.

`run_design.py:316` gives array task *i* the seed-FASTA window
`[i*batch, (i+1)*batch)`, cycling if the pool is short. So the optimize node
consumes exactly `array × batch` = 32 × 4 = **128** candidates — and the pool is
64 + 64 = **128**. Every candidate is refined exactly once.

Generate more than that without a screen node and the surplus is dropped in
generator order, not by quality; generate fewer and some are refined twice. This
is also why `top_k: 0` on the optimize node: `top_k` truncates the seed FASTA
from the front, which only means "the best" when something upstream ranked it.
A merge node's order is "BoltzGen's, then Proteina's".

To scale up, insert a screen (the commented block at the foot of
`dio3_2gen.yaml`) so ranking exists, then raise the generators and use `top_k`.

## Cost, and why the objective is confidence-free

`design_config.estimate_seconds` puts the optimize node at **1.7 h per array
task** at batch 4 — 32 tasks, throttled to 8 concurrent.

Every loss in the objective reads the distogram only. Under JIT, JAX prunes the
structure and confidence modules when nothing reads them; adding a single pLDDT,
PAE, ipTM or ProteinMPNN term defeats that and takes the same node from 1.7 h to
**5.0 h** per task. Confidence belongs in a screen, where it is paid once per
candidate instead of once per gradient step.

## The one decision left open

Neither generator has hotspots set. Both then choose their own binding surface,
write the epitope they chose into the node manifest, and `pipeline_node.py` feeds
that epitope into the optimize stage's `BinderTargetContact` — so the funnel is
coherent as it stands.

To aim at a chosen surface, set `hotspots` in **both** presets, in 1-based ECD
positions: the target is DIO3 residues 68–304 (`U170C`), so ECD 1 = R68, and
UniProt *N* is ECD *N−67*. Then calibrate `hotspot_shell` — on this 237-residue
target 6 Å keeps 21% of it, 8 Å keeps 38%, 12 Å keeps 69%, and a shell that keeps
most of the protein constrains nothing.
