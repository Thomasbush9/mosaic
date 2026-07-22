#!/bin/bash
#
# Generate binder backbones with BoltzGen against a target structure.
#
# BoltzGen is generative, not a scoring model: it implements no build_loss and
# cannot appear in MODELS=. It proposes starting points, which run_design.py
# consumes via --init-fasta.
#
#     TARGET_CIF=... OUT_DIR=... NUM_DESIGNS=8 sbatch --export=ALL singularity/boltzgen.sbatch
#
#SBATCH --job-name=mosaic-boltzgen
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/boltzgen-%j.out
#SBATCH --error=logs/boltzgen-%j.out
set -euo pipefail
SETUP=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup
export MOSAIC_SIF="${MOSAIC_SIF:-$SETUP/images/mosaic.sif}"
export MOSAIC_WEIGHTS="${MOSAIC_WEIGHTS:-$SETUP/weights}"
export MOSAIC_SCRATCH="${MOSAIC_SCRATCH:-/n/netscratch/bsabatini_lab/Everyone/$USER/mosaic}"
REPO="${MOSAIC_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
"$REPO/singularity/mosaic-exec.sh" python "$REPO/generate_boltzgen.py" \
  --target-cif "${TARGET_CIF:-$SETUP/targets/dio3_ecd.cif}" \
  --binder-length "${BINDER_LENGTH:-80}" \
  --num-designs "${NUM_DESIGNS:-8}" \
  --out "${OUT_DIR:-$SETUP/designs/boltzgen}"
