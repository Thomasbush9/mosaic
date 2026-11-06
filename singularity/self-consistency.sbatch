#!/bin/bash
#
# scRMSD self-consistency scoring for a design set, on one GPU.
#
#     mkdir -p logs && sbatch singularity/self-consistency.sbatch
#
# Configure via environment:
#     DESIGNS  OUT  N_SAMPLES  SEED  RECYCLING  TEMP  LIMIT
#
# Each design costs N_SAMPLES+1 AF2 forward passes, so wall-clock scales
# linearly in both the design count and N_SAMPLES. The script writes the
# output CSV every 10 designs, so hitting the walltime still leaves usable
# partial results rather than nothing.
#
#SBATCH --job-name=mosaic-scrmsd
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/scrmsd-%j.out
#SBATCH --error=logs/scrmsd-%j.out

set -euo pipefail

SETUP=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/mosaic_setup
# images/ did not move with the rest of the tree: the SIFs now
# live one level up, in binder_design/images/.
IMAGES=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/images

export MOSAIC_SIF="${MOSAIC_SIF:-$IMAGES/mosaic.sif}"
export MOSAIC_WEIGHTS="${MOSAIC_WEIGHTS:-$SETUP/weights}"
export MOSAIC_SCRATCH="${MOSAIC_SCRATCH:-/n/netscratch/bsabatini_lab/Everyone/$USER/mosaic}"

# SLURM copies the batch script into its spool directory, so BASH_SOURCE points
# there and not at the repo. Derive the repo from the submit directory instead,
# as campaign.sbatch does.
REPO="${MOSAIC_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
HERE="$REPO/singularity"

# Match the notebook workflow: run the working tree's sources, not the copy
# baked into the image, so edits under src/mosaic take effect without a rebuild.
export MOSAIC_DEV_SRC="${MOSAIC_DEV_SRC-$REPO/src}"
export SINGULARITYENV_JAX_COMPILATION_CACHE_DIR=/jax_cache

DESIGNS="${DESIGNS:-$SETUP/dio3_cut/designs/correctly_ranked_candidates_af2.csv}"
OUT="${OUT:-$SETUP/dio3_cut/designs/self_consistency.csv}"
N_SAMPLES="${N_SAMPLES:-10}"
SEED="${SEED:-42}"
RECYCLING="${RECYCLING:-3}"
TEMP="${TEMP:-0.1}"

args=(
    --designs "$DESIGNS"
    --out "$OUT"
    --n "$N_SAMPLES"
    --seed "$SEED"
    --recycling-steps "$RECYCLING"
    --temp "$TEMP"
)
[[ -n "${LIMIT:-}" ]] && args+=(--limit "$LIMIT")

echo "node:    $(hostname -f)"
echo "designs: $DESIGNS"
echo "out:     $OUT"
echo "samples: $N_SAMPLES per design"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

exec "$HERE/mosaic-exec.sh" python "$REPO/hallucinate/self_consistency.py" "${args[@]}"
