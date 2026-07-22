#!/bin/bash
#
# Fold a target and write CIF — the input BoltzGen needs (it designs against
# geometry, not sequence). Not required for scoring-model campaigns.
#
#     TARGET_FASTA=... TARGET_MSA=... OUT_CIF=... sbatch --export=ALL singularity/predict-target.sbatch
#
#SBATCH --job-name=mosaic-predict
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/predict-%j.out
#SBATCH --error=logs/predict-%j.out
set -euo pipefail
SETUP=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup
export MOSAIC_SIF="${MOSAIC_SIF:-$SETUP/images/mosaic.sif}"
export MOSAIC_WEIGHTS="${MOSAIC_WEIGHTS:-$SETUP/weights}"
export MOSAIC_SCRATCH="${MOSAIC_SCRATCH:-/n/netscratch/bsabatini_lab/Everyone/$USER/mosaic}"
REPO="${MOSAIC_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
TARGET_FASTA="${TARGET_FASTA:-$SETUP/targets/dio3_ecd.fasta}"
TARGET_MSA="${TARGET_MSA:-$SETUP/msa/dio3_ecd.a3m}"
OUT_CIF="${OUT_CIF:-$SETUP/targets/dio3_ecd.cif}"
MSA_ARG=(); [[ -f "$TARGET_MSA" ]] && MSA_ARG=(--target-msa "$TARGET_MSA")
"$REPO/singularity/mosaic-exec.sh" python "$REPO/predict_target.py" \
  --target "$TARGET_FASTA" "${MSA_ARG[@]}" --out "$OUT_CIF"
