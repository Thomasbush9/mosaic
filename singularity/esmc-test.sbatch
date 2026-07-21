#!/bin/bash
#
# First real model load: ESM-C 300M from the existing shared HuggingFace cache,
# through the full torch -> esmjfold2 -> Equinox path, then one loss evaluation.
#
#     mkdir -p logs && sbatch singularity/esmc-test.sbatch
#
# HF_HUB_OFFLINE=1 is the point of this test as much as the load is: it turns a
# cache miss into an immediate error instead of a silent multi-GB download into
# scratch, which is exactly how "it worked but took 40 minutes" happens.
#
# NOTE ON THE MODEL ID: losses/esmc.py maps the alias `esmc_300m` to
# `Biohub/ESMC-300M`, but the shared cache was populated under the lowercase id
# and HF cache paths are case-sensitive (models--biohub--ESMC-300M on disk vs
# models--Biohub--ESMC-300M for the alias). The raw id is passed explicitly
# here; load_esmc() accepts one via `_CHECKPOINTS.get(model_name, model_name)`.
#
#SBATCH --job-name=mosaic-esmc
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/esmc-test-%j.out
#SBATCH --error=logs/esmc-test-%j.out

set -euo pipefail

export MOSAIC_SIF="${MOSAIC_SIF:-/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup/images/mosaic.sif}"
export MOSAIC_WEIGHTS="${MOSAIC_WEIGHTS:-/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup/weights}"
export MOSAIC_SCRATCH="${MOSAIC_SCRATCH:-/n/netscratch/bsabatini_lab/Everyone/$USER/mosaic}"

# Forwarded into the container by Singularity (SINGULARITYENV_ prefix is stripped).
export SINGULARITYENV_HF_HUB_OFFLINE=1

MOSAIC_REPO="${MOSAIC_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
HERE="$MOSAIC_REPO/singularity"
if [[ ! -x "$HERE/mosaic-exec.sh" ]]; then
    echo "error: $HERE/mosaic-exec.sh not found — set MOSAIC_REPO to the repo root" >&2
    exit 66
fi

echo "── host ──"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo
echo "── load ESM-C 300M + one loss evaluation ──"
"$HERE/mosaic-exec.sh" python -c '
import os, time
import jax, jax.numpy as jnp, numpy as np

print("HF_HUB_OFFLINE =", os.environ.get("HF_HUB_OFFLINE"))
print("jax backend:", jax.default_backend(), jax.devices())

from mosaic.common import TOKENS
from mosaic.losses.esmc import ESMCPseudoLikelihood, load_esmc

t0 = time.time()
model = load_esmc("biohub/ESMC-300M")
print(f"loaded ESM-C 300M in {time.time()-t0:.1f}s -> {type(model).__name__}")

# Ubiquitin N-terminal fragment: a real sequence, so the PLL should be sane
# rather than the near-uniform value random residues would give.
seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQ"
idx = np.array([TOKENS.index(c) for c in seq])
soft = jnp.asarray(np.eye(20, dtype=np.float32)[idx])
print("sequence length:", soft.shape)

loss = ESMCPseudoLikelihood(model)
t0 = time.time()
value, aux = loss(soft, key=jax.random.key(0))
value = float(value)
print(f"first call (incl. JIT trace): {time.time()-t0:.1f}s")
print("loss:", value, " aux:", {k: float(v) for k, v in aux.items()})

assert np.isfinite(value), "loss is not finite"
# -log p per residue over 20 AAs: uniform would be ~3.0, a real sequence a
# well-trained model recognises should score clearly better than chance.
assert 0.0 < value < 3.0, f"PLL loss {value} outside plausible range — suspect a bad load"
print("ESMC LOAD + LOSS OK")
'
echo
echo "ALL ESM-C CHECKS PASSED"
