#!/bin/bash
#
# Minimal GPU smoke test for the mosaic container. Confirms the one thing the
# build-time %test cannot: that `singularity exec --nv` actually hands the host
# driver to JAX inside the image.
#
#     mkdir -p logs && sbatch singularity/nv-test.sbatch
#
# No model weights are needed. selftest.py treats missing weight caches as
# warnings, not failures, so this passes on a bare cluster and still fails hard
# if --nv is broken (check_gpu is fatal at runtime).
#
# This is a driver check, not a workload — hence 1 CPU / 8G / 5 min. The one
# matmul is deliberate and is not a benchmark: enumerating a device only proves
# the driver loaded, so the kernel has to actually run and return a correct
# result to rule out a broken CUDA library path. It takes well under a second.
#
#SBATCH --job-name=mosaic-nv
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/nv-test-%j.out
#SBATCH --error=logs/nv-test-%j.out

set -euo pipefail

export MOSAIC_SIF="${MOSAIC_SIF:-/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup/images/mosaic.sif}"
export MOSAIC_WEIGHTS="${MOSAIC_WEIGHTS:-/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup/weights}"
export MOSAIC_SCRATCH="${MOSAIC_SCRATCH:-/n/netscratch/bsabatini_lab/Everyone/$USER/mosaic}"

# NOT $(dirname $BASH_SOURCE): SLURM copies the batch script into its spool
# directory before running it, so BASH_SOURCE points at /var/slurmd/spool/... and
# mosaic-exec.sh is not next to it. SLURM_SUBMIT_DIR is the repo root when
# submitted as `sbatch singularity/nv-test.sbatch` from there.
MOSAIC_REPO="${MOSAIC_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
HERE="$MOSAIC_REPO/singularity"
if [[ ! -x "$HERE/mosaic-exec.sh" ]]; then
    echo "error: $HERE/mosaic-exec.sh not found — set MOSAIC_REPO to the repo root" >&2
    exit 66
fi

echo "── host ──"
echo "node: $(hostname)   gpus: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo
echo "── container selftest (runtime mode) ──"
"$HERE/mosaic-exec.sh" python /opt/mosaic/singularity/selftest.py

echo
echo "── jax device + real matmul on device ──"
# selftest reports the backend; this proves a kernel actually executes and the
# result comes back off the GPU, which a mis-set LD_LIBRARY_PATH would not.
"$HERE/mosaic-exec.sh" python -c '
import jax, jax.numpy as jnp
print("jax", jax.__version__, "backend:", jax.default_backend())
print("devices:", jax.devices())
assert jax.default_backend() == "gpu", "JAX did not see the GPU — --nv is not working"
x = jnp.ones((4096, 4096))
y = (x @ x).block_until_ready()
print("matmul ok, device:", y.device, "sum:", float(y.sum()))
'

echo
echo "ALL GPU CHECKS PASSED"
