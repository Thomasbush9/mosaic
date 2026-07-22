#!/bin/bash
#
# Binder design campaign: a SLURM array, one GPU per task, seeds differing.
#
#     mkdir -p logs
#     sbatch --array=0-15%8 singularity/campaign.sbatch
#
# Configure via environment:
#     TARGET_FASTA  TARGET_MSA  BINDER_LENGTH  MODELS  BATCH
#     SOFT_STEPS  SHARP_STEPS  OUT_DIR
#
# WHY AN ARRAY AND NOT MULTI-GPU: mosaic is single-device throughout. There is
# no pmap, shard_map, jax.sharding or jax.distributed anywhere in src/ — verified
# by grep, and stated in singularity/README.md. Extra GPUs in one process are
# simply idle. Scaling therefore has two composing axes:
#
#     across GPUs   this array, one GPU per task, differing by --seed
#     within a GPU  --batch B trajectories vmapped by batched_simplex_APGM
#
# so N tasks x B trajectories = N*B designs. B also amortises the JIT compile,
# which is minutes per backend.
#
# This supersedes design.sbatch, which cannot run here: it requests
# `--gres=gpu:h100:1` (rejected on FASRC — the GRES name is
# nvidia_h100_80gb_hbm3, and `gpu:1` is what works), sets no partition or
# account, and defaults to /shared and /scratch paths that do not exist.
#
# run_design.py is invoked from the repo rather than /opt/mosaic inside the
# image, so editing the objective does not require a 10-minute rebuild.
# /n/holylfs06 is auto-bound inside the container, so the path resolves.
#
#SBATCH --job-name=mosaic-campaign
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/campaign-%A_%a.out
#SBATCH --error=logs/campaign-%A_%a.out

set -euo pipefail

SETUP=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup

export MOSAIC_SIF="${MOSAIC_SIF:-$SETUP/images/mosaic.sif}"
export MOSAIC_WEIGHTS="${MOSAIC_WEIGHTS:-$SETUP/weights}"
export MOSAIC_SCRATCH="${MOSAIC_SCRATCH:-/n/netscratch/bsabatini_lab/Everyone/$USER/mosaic}"

TARGET_FASTA="${TARGET_FASTA:-$SETUP/targets/dio3_ecd.fasta}"
TARGET_MSA="${TARGET_MSA:-$SETUP/msa/dio3_ecd.a3m}"
BINDER_LENGTH="${BINDER_LENGTH:-80}"
MODELS="${MODELS:-boltz2,af2}"
BATCH="${BATCH:-2}"
SOFT_STEPS="${SOFT_STEPS:-100}"
SHARP_STEPS="${SHARP_STEPS:-25}"
OUT_DIR="${OUT_DIR:-$SETUP/designs/dio3}"

MOSAIC_REPO="${MOSAIC_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
HERE="$MOSAIC_REPO/singularity"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

mkdir -p "$OUT_DIR"

echo "── task $TASK_ID on $(hostname) ──"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Only pass --target-msa if the a3m exists; otherwise the backends fall back to
# single-sequence rather than failing, and the log says which happened.
MSA_ARG=()
if [[ -f "$TARGET_MSA" ]]; then
    MSA_ARG=(--target-msa "$TARGET_MSA")
else
    echo "note: $TARGET_MSA missing — running single-sequence" >&2
fi

"$HERE/mosaic-exec.sh" python "$MOSAIC_REPO/run_design.py" \
    --target "$TARGET_FASTA" \
    "${MSA_ARG[@]}" \
    --binder-length "$BINDER_LENGTH" \
    --models "$MODELS" \
    --batch "$BATCH" \
    --seed "$TASK_ID" \
    --soft-steps "$SOFT_STEPS" \
    --sharp-steps "$SHARP_STEPS" \
    --out "$OUT_DIR"

echo "task $TASK_ID done"
