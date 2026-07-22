# The design webapp

A browser UI for configuring, launching, monitoring and exploring binder design
campaigns. Modelled on ProtForge's webapp — same shape, same SSH-forwarding
access pattern.

**Nothing heavy runs in the app.** It reads files and queues SLURM jobs; every
model evaluation happens in a batch job on a GPU node. That is what makes it
safe to run on a login node, where FASRC's arbiter kills CPU-heavy processes.

---

## Access

From your laptop:

```bash
bash webapp/connect.sh
```

It forwards a port, creates the virtual environment on first use, starts the
app, and tells you to open <http://localhost:8502>. Ctrl-C disconnects; the app
keeps running on the cluster.

Manual equivalent, if you prefer:

```bash
# on the login node
module load python/3.13.12-fasrc01
uv venv webapp/.venv --python 3.13
uv pip install --python webapp/.venv -r webapp/requirements.txt
webapp/.venv/bin/streamlit run webapp/app.py --server.port 8502 --server.address 127.0.0.1

# on your laptop
ssh -L 8502:localhost:8502 <user>@holylogin06.rc.fas.harvard.edu
```

---

## Tab 1 — Launch

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

## Tab 2 — Monitor

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

## Tab 3 — Results

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
| `app.py` | bootstrap, sidebar, tab dispatch |
| `store.py` | pure — targets, MSAs, campaigns, designs, statistics |
| `cluster.py` | pure — sbatch / squeue / sacct / scancel, log verification |
| `launch_tab.py` | Launch UI, generated from the catalogs |
| `monitor_tab.py` | Monitor UI |
| `results_tab.py` | Results UI |
| `../design_config.py` | the catalogs and config builders (shared with `run_design.py`) |

`design_config.py` is importable **without mosaic**, so the app can build its UI
host-side while `run_design.py` uses the same catalogs inside the container.
Adding a loss term there makes it appear in the UI with no change to the app.
