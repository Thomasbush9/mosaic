#!/bin/bash
#
# Inventory audit: for every backend the benchmark plan names, answer two
# separate questions that are easy to conflate --
#
#   IMPORTS   the wrapper module loads (code present, deps satisfied)
#   WEIGHTS   the checkpoint it will reach for is on disk and bound
#
# A wrapper can import fine and still have no weights (it would silently try to
# download into scratch), and weights can be present for a wrapper whose deps
# are broken. Both are reported per model.
#
#     mkdir -p logs && sbatch singularity/model-audit.sbatch
#
#SBATCH --job-name=mosaic-audit
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=logs/audit-%j.out
#SBATCH --error=logs/audit-%j.out

set -euo pipefail
SETUP=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/mosaic_setup
# images/ did not move with the rest of the tree: the SIFs now
# live one level up, in binder_design/images/.
IMAGES=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/images
export MOSAIC_SIF="${MOSAIC_SIF:-$IMAGES/mosaic.sif}"
export MOSAIC_WEIGHTS="${MOSAIC_WEIGHTS:-$SETUP/weights}"
export MOSAIC_SCRATCH="${MOSAIC_SCRATCH:-/n/netscratch/bsabatini_lab/Everyone/$USER/mosaic}"
export SINGULARITYENV_HF_HUB_OFFLINE=1

REPO="${MOSAIC_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
export MOSAIC_DEV_SRC="${MOSAIC_DEV_SRC-$REPO/src}"

nvidia-smi --query-gpu=name --format=csv,noheader || true
exec "$REPO/singularity/mosaic-exec.sh" python "$REPO/singularity/model_audit.py"
