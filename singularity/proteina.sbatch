#!/bin/bash
#
# Generate binder backbones with Proteina and write a proposal set.
#
# Proteina is hotspot-conditioned: HOTSPOTS names the target residues the binder
# should engage (1-based), so the epitope is an input rather than something
# derived from the poses afterwards as it is for BoltzGen.
#
#SBATCH --job-name=mosaic-proteina
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/proteina-%j.out
#SBATCH --error=logs/proteina-%j.out
set -euo pipefail
SETUP=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup
export MOSAIC_SIF="${MOSAIC_SIF:-$SETUP/images/mosaic.sif}"
export MOSAIC_WEIGHTS="${MOSAIC_WEIGHTS:-$SETUP/weights}"
export MOSAIC_SCRATCH="${MOSAIC_SCRATCH:-/n/netscratch/bsabatini_lab/Everyone/$USER/mosaic}"
REPO="${MOSAIC_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
"$REPO/singularity/mosaic-exec.sh" python "$REPO/generate_proteina.py" \
  --target-cif "${TARGET_CIF:?set TARGET_CIF}" \
  --target-fasta "${TARGET_FASTA:-}" \
  --target-name "${TARGET_NAME:-}" \
  --binder-length "${BINDER_LENGTH:-80}" \
  --num-designs "${NUM_DESIGNS:-8}" \
  --hotspots "${HOTSPOTS:-}" \
  --target-chain "${TARGET_CHAIN:-A}" \
  --out "${OUT_DIR:?set OUT_DIR}"
