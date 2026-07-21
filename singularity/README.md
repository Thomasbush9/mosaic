# Running mosaic under Singularity on SLURM

Cluster packaging for [`escalante-bio/mosaic`](https://github.com/escalante-bio/mosaic).
Nothing here changes the library itself — it builds an image, arranges the
weight caches, and checks the environment before a job wastes a GPU allocation.

| File | Purpose |
| --- | --- |
| `mosaic.def` | Singularity definition. Code + dependencies, no weights. |
| `mosaic-exec.sh` | Runs a command in the container with the caches bound in. |
| `design.sbatch` | Job-array template for a design sweep. |
| `selftest.py` | Environment check. Runs at build time and on a compute node. |

## Why weights are not in the image

They are bound at runtime instead, for three reasons: the image stays around
15 GB rather than 45 GB+; one weight tree serves every user instead of each
carrying a copy; and refreshing a checkpoint doesn't mean rebuilding.

The layout the image expects:

```
/weights      read-only, shared      model checkpoints
  /weights/hf                        HF_HOME — ESM-C, ESMFold2, ESM2
/cache        writable, per-user     MSA cache, OpenDDE derived templates
/jax_cache    writable, shared       compiled XLA kernels
/work         writable, per-user     designs, logs
```

`/jax_cache` has to be that exact path — `examples/proteina.py:16` and
`examples/promera_design.py:26` hardcode it as an absolute path. Outside a
container that's unusable on a cluster; inside one it's just a bind target.

## Build

Needs internet and roughly 40 GB of free space. Run it on a login node.

```bash
cd <repo root>
singularity build --fakeroot singularity/mosaic.sif singularity/mosaic.def
```

Then extract the generated lockfile and commit it to this fork:

```bash
singularity exec singularity/mosaic.sif cat /opt/mosaic/uv.lock > uv.lock
git add -f uv.lock          # upstream gitignores it
```

This matters more than usual. Nine dependencies are git-sourced with no version
pinning, so without a committed lock two builds a month apart can silently
differ. Once it's committed, add it to `%files` and switch `%post` to
`uv sync --frozen --group jax-cuda` for reproducible rebuilds.

## How the caches get in

Each shared cache is bound **directly onto the path mosaic already looks for**,
inside a writable per-user `HOME` on scratch:

```
$MOSAIC_WEIGHTS/boltz      ──►  ~/.boltz              (read-only)
$MOSAIC_WEIGHTS/alphafold  ──►  ~/.alphafold          (read-only)
$MOSAIC_WEIGHTS/protenix   ──►  ~/.protenix           (read-only)
$MOSAIC_WEIGHTS/hf         ──►  ~/.cache/huggingface  (read-only)
$MOSAIC_SCRATCH/jax_cache  ──►  /jax_cache            (writable, shared)
$MOSAIC_SCRATCH/out        ──►  /work                 (writable)
                                ~/.cache/mosaic       (writable, plain dir)
```

Two consequences worth knowing.

**No symlinks.** A link created on the login node encodes a *host* path, which
does not exist inside the container — it would dangle in the only namespace
that ever dereferences it. Binding onto the real location has no such
host/container ambiguity.

**No environment redirection.** Because the caches land on mosaic's own
defaults, `HF_HOME`, `MOSAIC_MSA_CACHE` and `MOSAIC_OPENDDE_TEMPLATE_CACHE` are
all unnecessary and the image deliberately leaves them unset. This is also the
only arrangement that works for Protenix, whose `losses/protenix.py:23` sets
`PROTENIX_DATA_ROOT_DIR` unconditionally at import and would overwrite anything
you exported.

Note the shared tree uses plain names (`boltz/`, `hf/`), not dotfiles — it does
not have to mimic a home directory, since nothing resolves through it.

## First run

```bash
./singularity/mosaic-exec.sh python /opt/mosaic/singularity/selftest.py
```

The wrapper creates the mount points (Singularity cannot mount onto a
destination that does not exist, and overlay auto-creation is not enabled on
every cluster), assembles the binds, and execs. Pass `--show` to print the
resolved command without running it — worth doing once to see what it builds.

Configure with `MOSAIC_SIF`, `MOSAIC_WEIGHTS`, `MOSAIC_SCRATCH`.

`--nv` is applied by the wrapper and is mandatory: without it JAX falls back to
CPU and the job takes days instead of hours while appearing to work.

A cache whose source directory is absent is **skipped rather than mounted
empty**, so mosaic falls back to downloading it. `selftest.py` flags the
opposite case — a mount point that exists but is empty, meaning a bind silently
did not happen and mosaic is about to re-download tens of GB into scratch.

## Submitting a sweep

```bash
mkdir -p logs && sbatch singularity/design.sbatch
```

`logs/` must exist before `sbatch` — SLURM opens `--output` before the script
runs. Task 0 runs `selftest.py` and writes a flag under `$MOSAIC_SCRATCH`;
tasks 1–N wait on that flag so a broken GPU/weight setup cannot burn the
array. Override the (not-yet-landed) entrypoint with `DESIGN_SCRIPT=...`.

## Reusing the existing cluster caches

The `cache_models/` tree from the structure-prediction pipeline already has the
`hub/` + `xet/` layout `huggingface_hub` expects, so binding it at `/weights/hf`
covers ESM-C 300M/600M/**6B** and ESMFold2 + ESMFold2-Fast. ESM-C 6B alone is
24 GB, which is the single largest download in the dependency set.

**Watch the capitalization.** mosaic is inconsistent about the org name:
`src/mosaic/losses/esmc.py:25-27` requests `Biohub/` with a capital B, while
`src/mosaic/models/esmfold2.py` and `examples/promera_design.py:135` use
lowercase `biohub/`. The existing cache directories are lowercase
(`models--biohub--ESMC-6B`), and `huggingface_hub` derives the cache folder name
verbatim from the string it is given. So:

```python
load_esmc("biohub/ESMC-6B")   # hits the existing cache
load_esmc("esmc_6b")          # resolves to Biohub/… and re-downloads 24 GB
```

`losses/esmc.py:49` does `_CHECKPOINTS.get(model_name, model_name)`, so passing
the explicit lowercase repo id works with no patch. Correcting the dict to
lowercase is a one-line fix worth sending upstream.

## Still to resolve

- `cache_models/ckpt_root` — if this is Boltz's cache (`boltz1_conf.ckpt`,
  `boltz2_conf.ckpt`, `ccd.pkl`, `mols/`) it drops straight in, and the path is
  a real argument (`load_boltz(checkpoint_path=…)`), so no `HOME` trick needed.
- `of3-p2-145k.pt` / `of3-p2-155k.pt` — mosaic reaches OpenFold3 through
  `jopenfold3`, whose `OpenFold3.load()` does its own checkpoint discovery.
  Whether these are the layout it expects needs checking against that package.
- The ESMFold2 examples use `ESMFold2ExperimentalFast` and
  `ESMFold2ExperimentalFast2025` — the `-Experimental-*` repos, not the base
  `ESMFold2` / `ESMFold2-Fast` currently cached. Either fetch those two or point
  the examples at the base loaders.

Still to fetch regardless: AlphaFold2 params (3.5 GB tar, needs ~9 GB free while
unpacking), Protenix, BoltzGen, OpenDDE, AbLang.

## Resource shape

One GPU per job. There is no `pmap`, `shard_map`, `jax.sharding` or
`jax.distributed` anywhere in the source, so additional GPUs cannot be used —
scale with a job array throttled to the per-user cap (`--array=0-63%8`).

H100 (80 GB) is sufficient unless ESM-C 6B is in the objective; the torch→JAX
converter upcasts to fp32, so 6B parameters cost 24 GB of weights before any
activation. Use H200 (141 GB) for that case.

Budget `--cpus-per-task=8` (part of the optimizer runs in host numpy every
iteration) and `--mem=128G` (loading a model holds the torch checkpoint and the
converted JAX copy at the same time).
