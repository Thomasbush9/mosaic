#!/bin/bash
#
# Screen a proposal set: refold each candidate with a chosen structural model
# and rank by confidence. mosaic's generate->refold->rank pipeline, model-
# agnostic (any of boltz2/boltz1/af2/of3/protenix).
#
#     sbatch --export=ALL,PROPOSALS=<dir>,MODEL=af2,OUT_DIR=<dir> singularity/screen.sbatch
#
#SBATCH --job-name=mosaic-screen
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/screen-%j.out
#SBATCH --error=logs/screen-%j.out
set -euo pipefail
SETUP=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup
export MOSAIC_SIF="${MOSAIC_SIF:-$SETUP/images/mosaic.sif}"
export MOSAIC_WEIGHTS="${MOSAIC_WEIGHTS:-$SETUP/weights}"
export MOSAIC_SCRATCH="${MOSAIC_SCRATCH:-/n/netscratch/bsabatini_lab/Everyone/$USER/mosaic}"
REPO="${MOSAIC_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
MSA_ARG=(); [[ -n "${TARGET_MSA:-}" && -f "${TARGET_MSA:-}" ]] && MSA_ARG=(--target-msa "$TARGET_MSA")
"$REPO/singularity/mosaic-exec.sh" python "$REPO/screen_proposals.py" \
  --proposals "${PROPOSALS:?set PROPOSALS}" \
  --model "${MODEL:-boltz2}" \
  "${MSA_ARG[@]}" \
  --recycling-steps "${RECYCLING_STEPS:-4}" \
  --out "${OUT_DIR:?set OUT_DIR}"
