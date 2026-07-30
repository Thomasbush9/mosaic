# Pipelines as files

A pipeline is a DAG of SLURM jobs: generate, hallucinate, merge, screen,
optimize. You can build one by clicking in the webapp's Pipeline tab, or by
writing a file and submitting it from a shell.

Both go through the same engine — `webapp/pipeline.py` — so they produce the
same nodes, the same validation, the same `--dependency=afterok` chain and the
same `dag.json`. Nothing about a pipeline is available only to one of them.

```bash
PY=webapp/.venv/bin/python          # or: module load python/3.13.12-fasrc01

$PY pipeline_spec.py template -o pipelines/dio3.yaml   # a commented starter
$PY pipeline_spec.py show     pipelines/dio3.yaml      # what it resolves to
$PY pipeline_spec.py submit   pipelines/dio3.yaml --dry-run
$PY pipeline_spec.py submit   pipelines/dio3.yaml
```

The repo's own Python is fine too (`uv run python pipeline_spec.py …`). What
will *not* work is a bare `python3` on a login node — that is 3.6.

## Why a file as well as a UI

Different jobs, not redundancy.

The UI is better when you are deciding: it lists what exists, shows the cost
estimate, and refuses an illegal edge as you draw it. A file is better once you
have decided: it diffs, it reviews, it lives in git next to the code that ran,
and it does not need a browser, a tunnel or a surviving login-node process.

They meet in the middle. `export` turns any submitted run into a spec file, and
`json` turns a spec file into a DAG the Pipeline tab loads:

```bash
$PY pipeline_spec.py export dio3_2way_500b -o pipelines/dio3.yaml   # run  -> file
$PY pipeline_spec.py json   pipelines/dio3.yaml -o dio3.json        # file -> webapp
```

## The format

YAML if PyYAML is importable, JSON otherwise; identical keys, the extension
decides. A real 500-design two-generator campaign is 37 lines:

```yaml
name: dio3_2way_500b
target: dio3_ecd            # <workdir>/targets/dio3_ecd.fasta (+ .cif, msa/*.a3m)

nodes:
  - id: gen_boltzgen
    type: generate
    generator: boltzgen
    num_designs: 500

  - id: gen_proteina
    type: generate
    generator: proteina
    num_designs: 500

  - id: pool
    type: merge
    inputs: [gen_boltzgen, gen_proteina]

  - id: screen
    type: screen
    inputs: [pool]
    recycling_steps: 3

  - id: optimize
    type: optimize
    inputs: [screen]
    top_k: 100
    array: 25
    array_throttle: 8
    config:
      models: [af2]
      losses: [BinderTargetContact, WithinBinderContact]
      optimizer:
        soft: {n_steps: 60}
      run: {batch: 4}
```

**You state only what differs from the catalogs in `design_config.py`.**
Everything else is filled in exactly as the webapp's forms fill it. `show`
prints the resolved node; `show --params` prints every value that was filled in
on your behalf.

### Keys

Top level: `name`, `target`, `cluster`, `defaults`, `nodes`.

A node entry: `id`, `type`, `inputs`, and then its parameters inline. Four keys
are structural rather than parameters:

| Key | Meaning |
|---|---|
| `target` | overrides the top-level target for this node |
| `resources` | `time_limit` / `mem` / `cpus` for this node only |
| `label` | display name in the graph |
| `params` | an explicit parameter dict, merged with the inline keys |
| `params_file` | a file (or list of files) of parameters — see below |

`defaults:` holds per-type settings applied under every node of that type:

```yaml
defaults:
  generate:
    binder_length: 75      # both generators, unless the node says otherwise
```

`cluster:` sets the account and the fallback resources:

```yaml
cluster:
  account: kempner_bsabatini_lab
  gpu_partition: kempner_h100
  cpu_partition: kempner_interactive
  gpu: {time_limit: "08:00:00", mem: 128G, cpus: 8}
  cpu: {time_limit: "01:00:00", mem: 16G, cpus: 2}
```

### Shorthands for models and loss terms

Anywhere a list of named things appears — `models`, `losses`,
`sequence_models`, `mpnn.terms` — all of these mean something sensible:

```yaml
models: boltz2                                  # one, at its catalog weight
models: [boltz2, af2]
models: {boltz2: 1.0, af2: 0.5}                 # name: weight
losses:
  - BinderTargetContact                         # catalog weight and parameters
  - {name: HelixLoss, weight: 2.0}              # weight only
  - {name: WithinBinderContact, min_sequence_separation: 12}   # inline params
  - {name: PLDDTLoss, params: {}}               # explicit params block
```

A bare name means "at its catalog weight, with its catalog parameters", which is
why an exported spec is so much shorter than the JSON it came from.

## Splitting a node's config into its own file

Any node can keep its parameters in a separate file, and an optimize node's
`config` can be a path:

```yaml
name: dio3_split
target: dio3_ecd

nodes:
  - id: gen_a
    type: generate
    generator: boltzgen
    params_file: presets/boltzgen_dio3.yaml

  - id: gen_b
    type: generate
    generator: boltzgen
    params_file: presets/boltzgen_dio3.yaml
    noise_scale: 1.1               # same preset, one knob changed

  - id: opt
    type: optimize
    inputs: [screen]
    top_k: 100
    config: presets/objective_af2.yaml
```

`presets/boltzgen_dio3.yaml` is just the parameter mapping — the keys you would
otherwise have written inline:

```yaml
num_designs: 500
binder_length: 75
n_helices: 4
loop_length: 5
sampling_steps: 400
noise_scale: 0.9
hotspots: "101-140"
hotspot_shell: 8.0
```

Four layers, each overriding the one before: `defaults:` for the type, then the
linked file(s), then an explicit `params:` mapping, then the keys written inline
on the node. **That last override is the point** — link a tuned preset and adjust
one value in place, as `gen_b` does above. `params_file` also accepts a list, so
several presets can compose in order.

Paths resolve relative to the spec file first, then to the working directory, so
a `pipelines/` directory can be checked out anywhere and still find its parts.
Linking is one level deep: a linked file may not itself link, because a chain of
files is a good way to lose track of what a number actually is.

`pipelines/dio3_2gen.yaml` is a worked example of this layout: two generators
into a merge into a gradient stage, with all three parameter sets in
`pipelines/presets/`. `pipelines/README.md` explains how its numbers were
chosen.

### When to split, and when not to

Worth splitting: a setting genuinely **shared** — one objective reused by three
pipelines, one generator preset you have tuned and trust — or one large enough to
bury the graph. An optimize node's `config` is the usual candidate, and it is
the same object the Launch tab writes to `configs/`.

Not worth splitting by default. The DIO3 funnel written inline is one readable
37-line file; the same pipeline split six ways is six files you must open before
you know what ran. Indirection is a cost you should be buying reuse with.

Either way the **record is unaffected**: `submit` resolves everything and writes
`dag.json`, a per-node `node.json`, the spec as you submitted it, and
`spec_resolved.yaml` — the fully inlined equivalent. A linked file that changes
next week cannot rewrite what this run was.

## Two rules

**Parity with the Python interface.** `node.params` is a plain dict handed
verbatim to `pipeline_node.py`, so anything you could set by assigning to
`node.params` in Python you can set here. A key no catalog knows is *reported
and passed through*, never dropped — the same guarantee the node editor's raw
JSON box gives.

**Bounds are advisory.** A value outside a catalog's declared range is reported
and then run as written:

```
warning: gen1: num_designs=9999 is outside the catalog range 1-5000.
         Running it as written; widen the range in design_config.py if this is routine.
```

The webapp has to *clamp* instead — Streamlit raises on an out-of-range widget
value and takes the whole page down with it — but in a file you typed the number
on purpose, and quietly changing it would be worse than either obeying or
refusing.

Errors are a different matter and stop a submit: an unknown node type, an
illegal edge, a cycle, a target that does not exist, an objective with no loss
terms, a generate node with no target structure. `show` prints them all at once
rather than stopping at the first.

## What `submit` does

Validates, prints the plan, then queues each node in topological order with
`--dependency=afterok` on its upstream jobs. SLURM does the waiting, the fan-out
and the failure propagation; no process needs to stay alive. Each node's output
directory is derived from its id, so downstream paths are known at submit time
even though the upstream jobs have not run.

`--dry-run` prints the same plan — resolved parameters, partition, walltime,
memory, array width per node — and queues nothing.

After a successful submit, `<workdir>/pipelines/<name>/` holds `dag.json`,
`jobids.json`, a `node.json` per node, and a copy of the spec file you
submitted. Watch it in the Monitor tab, or with `squeue`.

## Array fan-out

`array: N` on a **hallucinate** or **optimize** node runs N SLURM array tasks,
one GPU each, seeding off `SLURM_ARRAY_TASK_ID` and writing its own
`designs_seed<i>.json`. The downstream node globs them, and `afterok` on an
array's parent job waits for all of it — so a node scales past one GPU with no
extra node in the graph. `array_throttle` caps how many run at once.

It is refused on the other types. A generate or screen array would run the
identical job N times and write the identical filename in the same directory at
the same time; raise `num_designs` instead.

## Checking a change without a cluster

```bash
webapp/.venv/bin/python pipeline_spec.py selftest
```

Expansion, shorthands, defaults layering, linked files and their override order,
pass-through of unknown keys, out-of-range reporting, and the export/re-import
round trip — all against a synthetic target, so it needs no cluster.
