# The design webapp

A browser UI for configuring, launching, monitoring and exploring binder design
campaigns. Modelled on ProtForge's webapp — same shape, same SSH-forwarding
access pattern.

**Nothing heavy runs in the app.** It reads files and queues SLURM jobs; every
model evaluation happens in a batch job on a GPU node. It never imports mosaic
and never runs a model, which is what keeps it light enough for a login node —
though see **Access** below for why a compute node is still the better home.

One page renders per interaction. Streamlit re-executes the whole script on
every widget change, and `st.tabs` is client-side only, so six tabs meant six
file scans and eight SLURM queries per keystroke — and an exception anywhere
replaced all six with a traceback. The sidebar renders only the page you are on,
inside an error boundary.

---

## Access

**Preferred — run it on a compute node.** A login node holds every user to a
shared 8 GiB / 1-core cgroup that covers all their processes at once, so the app
competes with your own shells and editors and cannot be relied on to survive.
The batch job gets its own cores:

```bash
# on the login node, from the repo root
sbatch singularity/webapp.sbatch
cat logs/webapp-<jobid>.out     # prints the exact one-hop ssh command
```

The job holds the app for its walltime (8 h by default — raise `--time` in the
sbatch file for a longer session), and `scancel <jobid>` stops it.

**Quick look — run it on the login node:**

```bash
bash webapp/connect.sh
```

It forwards a port, creates the virtual environment on first use, starts the app
**detached** and waits for it to answer, then drops you in a shell on the login
node. Leaving that shell closes the tunnel; the app keeps running, and
`kill $(cat logs/webapp.pid)` stops it.

> Expect roughly a minute before it answers on first start. The venv is ~8,500
> files on Lustre and a login node gives you one core, so `import streamlit`
> and `import pandas` alone account for about a minute of that.

Manual equivalent, if you prefer:

```bash
# on the login node
module load python/3.13.12-fasrc01
uv venv webapp/.venv --python 3.13
uv pip install --python webapp/.venv -r webapp/requirements.txt
setsid nohup bash -c 'echo $$ > logs/webapp.pid
    exec webapp/.venv/bin/streamlit run webapp/app.py \
        --server.port 8502 --server.address 127.0.0.1 --server.headless true' \
    > logs/webapp.log 2>&1 < /dev/null &

# on your laptop
ssh -L 8502:localhost:8502 <user>@holylogin06.rc.fas.harvard.edu
```

Note the `setsid nohup … &`. Started in the foreground the server takes SIGINT
from Ctrl-C and SIGHUP when the connection drops, which looks exactly like a
crash: the app disappears with nothing in the log.

The pidfile is written from inside the detached process rather than from `$!`,
because `setsid` forks when it needs a new session leader — `$!` would name a
wrapper that has already exited, and `kill $(cat logs/webapp.pid)` would report
"no such process" while the server carried on.

---

## Launch

Eight sections, exposing every parameter the pipeline accepts.

1. **Target** — pick from `targets/`, with residue count, MSA depth and whether
   a structure exists. Warns when no MSA is present.
2. **Binder** — length, cysteine policy, optional seeding from a FASTA
   (e.g. BoltzGen proposals) with a seed-noise control.
3. **Structure models** — enable any of Boltz-2, Boltz-1, AF2, OpenFold3,
   Protenix, each with its own weight **and its own parameters**
   (`recycling_steps`, `sampling_steps`, `num_samples`, `use_dropout`, Protenix
   variant). Selecting AF2 tells you it will run single-sequence, because its
   wrapper rejects MSAs.
4. **Loss terms** — all 17, grouped Contact / Shape / Interface / Confidence,
   each with weight and its own parameters (contact distances, sequence
   separation, paratope and epitope indices, target radii…).
5. **Sequence models** — ESM-C with checkpoint and clip bounds; AbLang and
   AbLang2 marked antibody-only.
6. **Optimizer** — soft and sharp stages independently: `n_steps`, `stepsize`,
   `momentum`, `scale`, `logspace`, `max_gradient_norm`.
7. **Cluster** — array size, designs per GPU, concurrency cap, partition, time,
   memory, CPUs.
8. **Review** — live cost estimate, validation errors, the resolved config, and
   submit.

### Two things the UI tells you that are easy to miss

**Confidence losses are expensive.** pLDDT, PAE, ipTM and ipSAE read the
structure and confidence modules. Under JIT, JAX *prunes* those modules when
nothing reads them — which is why a contact-only objective is fast. Enabling any
confidence term defeats that pruning; the estimate reflects roughly a 3×
slowdown.

**`recycling_steps` must be ≥ 1.** Validation refuses 0. The trunk runs inside
`jax.lax.scan(length=recycling_steps)`, so at 0 the body never executes and the
loss stops depending on the sequence — a believable loss with an exactly zero
gradient.

---

## Pipeline

Build a multi-stage DAG — generate, hallucinate, merge, screen, optimize — and
submit the whole thing. Nodes are submitted in topological order with
`--dependency=afterok` on their upstream jobs, so SLURM does the waiting,
fan-out and failure propagation. Each node's output directory is derived from
its id, so downstream input paths are known at submit time even though the
upstream jobs have not run yet.

Per node you can set:

- **Type and id**, at Add time.
- **Edges** — a dropdown of upstream nodes whose output this type accepts, and
  editable afterwards, not only when the node is created. An edge is legal iff
  the source's `produces` is in the target's `accepts`, so "two validators off
  one generator" or "merge several generators" express-or-reject as you build.
- **Params** — the same catalog-driven forms the single-stage pages use: a
  generate node offers every generator and its parameters, a screen node every
  model plus the MPNN options, an optimize node a full design config imported
  from Launch.
- **Resources** — an optional per-node override of walltime, memory and CPUs.
  The stages have genuinely different shapes (one diffusion pass versus hundreds
  of gradient steps), so one pipeline-wide walltime either wastes allocation or
  kills the long node. Blank means inherit.
- **Raw params (JSON)** — the dict `pipeline_node.py` actually receives.

### The raw params editor

Every node has one, and it is the point rather than an afterthought. A node's
`params` is a plain dict passed verbatim to `pipeline_node.py`, so anything you
could express by assigning to `node.params` in Python must be expressible here:
keys no form draws, values outside a catalog's declared range, whole nested
structures. Applying replaces the dict and resets that node's widgets so the
forms redraw from what you wrote.

The one exception is the `target_*` paths on generate and hallucinate nodes,
which are re-derived from the node's Target selector on every rerun. Set those
there.

### Ranges are advisory, not load-bearing

The catalogs' `min`/`max` guard against typos. They are **not** a contract with
your saved data: a run that really did generate 500 candidates per node produced
a `node.json` that a form capping the field at 200 could not draw, and Streamlit
raises rather than coerces in that situation. Forms now clamp and tell you what
they did, so a stored value can never take the app down — but if the clamp is
wrong, the number in `design_config.py` is what to change.

### Reopening past runs

`submit()` writes each node's `{type, params, inputs}` to `node.json` under
`<workdir>/pipelines/<name>/<node>/`, plus a `dag.json` and `jobids.json` for
the run as a whole. **Reopen a submitted pipeline** reconstructs the graph from
those shards, so a run can be reviewed, coloured by live SLURM state, or loaded
back into the editor without having kept its DAG JSON.

---

## Monitor

Queue table, progress across the array, log viewer, cancel — and the part that
matters most, **verification**.

Every failure mode this pipeline has produced was *silent*: the job completes,
writes plausible output, and only the log reveals it ran one model instead of
two, or single-sequence instead of MSA-backed. So the tab greps the log for what
the job **says** it did and shows it as pass/fail against what you requested:

- `structure backends:` — must match the models you selected
- `target MSA:` — must be a path, not `(none - single sequence)`
- `n_msa` — must be in the thousands, not `1`
- `loss terms:` and `seeded from` when relevant

The submit command is not evidence. The log is.

---

## Results

Ranked designs with filtering by top-N or loss threshold, FASTA download, and
statistics: loss distribution, spread across seeds, composition versus natural
amino-acid frequencies, and mean pairwise identity.

**Checks come before results, deliberately.** The tab leads with problems rather
than a ranked list:

- degenerate composition (any residue above 25%) — the optimizer gamed the
  objective
- unexpected cysteines — a decode bug, not a design choice
- mean pairwise identity above 80% — the run collapsed to one answer
- mixed model settings in one directory — losses are not comparable

It also shows the config each campaign ran with, so a result is traceable to its
settings.

### What the loss means

A weighted heuristic over contact geometry, compactness and sequence
plausibility. **Not** an affinity, a Kd, or a probability of binding. Comparable
only within one model setting — each extra backend adds a critic, so a two-model
run reads higher at equal design quality.

---

## How configs work

The app writes a JSON config to `mosaic_setup/configs/` and passes SLURM only
its **path**. That is deliberate: `--export` is a comma-separated `KEY=VALUE`
list, so any value containing a comma is silently truncated. A model list passed
that way once turned an eight-GPU `boltz2` vs `boltz2+af2` comparison into
`boltz2` vs `boltz2`, with no error and no symptom except a suspicious timing.

A path contains no commas, and the config carries everything else. It is also
copied next to the results as `config_seed<N>.json`, so every campaign records
exactly what produced it.

Configs are plain JSON — editable by hand, loadable through the sidebar,
runnable without the app:

```bash
sbatch --array=0-15 \
  --export=ALL,DESIGN_CONFIG=/path/to/config.json,OUT_DIR=/path/to/out \
  singularity/campaign.sbatch
```

---

## Architecture

Following ProtForge: pure logic separate from UI, so it can be tested and reused.

| File | Role |
|---|---|
| `app.py` | bootstrap, sidebar navigation, page dispatch, error boundary |
| `store.py` | pure — targets, MSAs, campaigns, designs, statistics |
| `cluster.py` | pure — sbatch / squeue / sacct / scancel, log verification |
| `pipeline.py` | pure — the DAG: node types, validation, topo order, submit |
| `session.py` | pure — projects (a named working directory) |
| `launch_tab.py` | Launch UI, generated from the catalogs |
| `generate_tab.py` | Generate UI — structure prediction and proposal sets |
| `pipeline_tab.py` | Pipeline UI — node editor and submit |
| `monitor_tab.py` | Monitor UI |
| `results_tab.py` | Results UI |
| `docs_tab.py` | renders this file and its siblings in the browser |
| `ui_helpers.py` | shared widgets, including the range reconciliation above |
| `viewer.py` | py3Dmol structure views |
| `smoke_test.py` | headless checks — see below |
| `../design_config.py` | the catalogs and config builders (shared with `run_design.py`) |

### Testing

```bash
webapp/.venv/bin/python webapp/smoke_test.py
```

Runs the real app in-process via Streamlit's `AppTest`: every page renders, a
saved pipeline reopens, a graph can be built and rewired, and an out-of-range
stored value clamps and reports rather than raising. Exit code 0 means all four
passed. Worth running before and after any change to the forms — two of those
checks exist because the corresponding bugs reached a user.

`design_config.py` is importable **without mosaic**, so the app can build its UI
host-side while `run_design.py` uses the same catalogs inside the container.
Adding a loss term there makes it appear in the UI with no change to the app.
