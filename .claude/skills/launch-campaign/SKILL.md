---
name: launch-campaign
description: Launch a binder-design campaign on the cluster — pick models and step counts, submit the SLURM array, and verify the job is actually doing what was asked. Use when the user wants to "run a campaign", "design binders against X", "launch designs", or tune and relaunch an existing one.
---

# Launch a design campaign

## Prerequisites

`targets/<name>.fasta` and `msa/<name>.a3m` must exist — if not, use the
`new-target` skill first. Absolute paths throughout.

## Submit

```bash
cd <repo> && mkdir -p logs
sbatch --array=0-15 \
  --export=ALL,TARGET_FASTA=<abs>,TARGET_MSA=<abs>,BINDER_LENGTH=80,\
MODELS=boltz2+af2,BATCH=2,SOFT_STEPS=100,SHARP_STEPS=25,OUT_DIR=<abs> \
  singularity/campaign.sbatch
```

Total designs = array size × `BATCH`. One task ≈ 15 min for two models.

## The three traps — check every time

**1. `MODELS` must use `+`, never a comma.** SLURM's `--export` is itself a
comma-separated `KEY=VALUE` list, so `MODELS=boltz2,af2` silently becomes
`MODELS=boltz2` and the stray `af2` is discarded. No error. This once invalidated
an entire 8-GPU comparison — the only symptom was that adding a whole model cost
no extra time.

**2. `INIT_FASTA`, if seeding, must exist** — `campaign.sbatch` errors on a
missing file rather than silently falling back to random initialization.

**3. Verify after launching, do not trust the submit line:**

```bash
sleep 90
grep -E "structure backends|target MSA|seeding from" logs/campaign-<jobid>_0.out
```

`structure backends:` must list every model requested. `target MSA:` must show a
path, not `(none - single sequence)`. Report a mismatch immediately and cancel
rather than letting the array run.

## Choosing settings

| Goal | Settings |
|---|---|
| Fast iteration on config | `MODELS=protenix`, `SOFT_STEPS=50` |
| Standard campaign | `MODELS=boltz2`, 100/25 |
| Robust to model quirks | `MODELS=boltz2+of3` or `boltz2+af2` |
| More candidates | larger `--array`, not larger `BATCH` |

`BATCH` above 4 is untested — GPU memory and a host-side simplex projection that
scales with B are both plausible ceilings. Prefer more array tasks.

Adding a model roughly adds its cost: Boltz-2 alone ran 100 steps in 373 s,
Boltz-2+AF2 in 630 s. Per-model costs for 5 steps: Protenix 39 s, Boltz-2 91 s,
OpenFold3 92 s, AF2 109 s, Boltz-1 124 s.

Note AF2 never receives an MSA — its wrapper rejects them — so in a mixed run its
view of the target is single-sequence. Mention this if the user includes `af2`.

## Iterating without rebuilding

`campaign.sbatch` runs `run_design.py` from the repo, so editing the objective
takes effect immediately. To edit mosaic's own source too:

```bash
MOSAIC_DEV_SRC=<repo>/src <repo>/singularity/mosaic-exec.sh python ...
```

Say clearly that a run with `MOSAIC_DEV_SRC` set is **not reproducible from the
image alone** — rebuild (`sbatch singularity/build.sbatch`, ~10 min) before a
campaign whose results are meant to be kept.

## Rules

- Never run the design on a login node.
- `--account=kempner_bsabatini_lab` — `bsabatini_lab` has `MaxSubmit=0` and
  cannot submit anything.
- GPU partitions reject jobs with no GPU request; CPU-only work goes to
  `kempner_interactive`.
