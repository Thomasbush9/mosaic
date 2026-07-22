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
# CAUTION with multiple models: SLURM's --export takes a comma-separated list of
# KEY=VALUE pairs, so
#     --export=ALL,MODELS=boltz2,af2        # WRONG — silently runs boltz2 only
# parses as MODELS=boltz2 plus a stray `af2` that is discarded. Nothing errors;
# the job just quietly runs one backend. Use either
#     --export=ALL,MODELS=boltz2+af2        # '+' is accepted as a separator
# or set it in the environment and let --export=ALL carry it:
#     MODELS=boltz2,af2 sbatch --export=ALL singularity/campaign.sbatch
# The log line "structure backends: ..." reports what was actually used — check
# it rather than trusting the submit command.
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

# INIT_FASTA seeds the optimizer from proposed sequences (generate_boltzgen.py)
# instead of noise. Fail loudly if it is set but missing: silently falling back
# to random initialization would make a seeded run indistinguishable from an
# unseeded one in the results.
INIT_ARG=()
if [[ -n "${INIT_FASTA:-}" ]]; then
    if [[ ! -f "$INIT_FASTA" ]]; then
        echo "error: INIT_FASTA=$INIT_FASTA does not exist" >&2
        exit 66
    fi
    INIT_ARG=(--init-fasta "$INIT_FASTA")
    echo "seeding from $INIT_FASTA"
fi

# DESIGN_CONFIG is the webapp's path: a JSON file carrying models, per-model and
# per-loss parameters, and both optimizer stages. It travels as a FILE PATH
# rather than as exported variables because --export splits on commas, so any
# value containing one is silently truncated — the failure that once turned a
# boltz2-vs-boltz2+af2 comparison into boltz2-vs-boltz2.
if [[ -n "${DESIGN_CONFIG:-}" ]]; then
    if [[ ! -f "$DESIGN_CONFIG" ]]; then
        echo "error: DESIGN_CONFIG=$DESIGN_CONFIG does not exist" >&2
        exit 66
    fi
    echo "config: $DESIGN_CONFIG"
    "$HERE/mosaic-exec.sh" python "$MOSAIC_REPO/run_design.py" \
        --config "$DESIGN_CONFIG" \
        --target "$TARGET_FASTA" \
        --seed "$TASK_ID" \
        --out "$OUT_DIR"
else
    "$HERE/mosaic-exec.sh" python "$MOSAIC_REPO/run_design.py" \
        --target "$TARGET_FASTA" \
        "${MSA_ARG[@]}" \
        "${INIT_ARG[@]}" \
        --binder-length "$BINDER_LENGTH" \
        --models "$MODELS" \
        --batch "$BATCH" \
        --seed "$TASK_ID" \
        --soft-steps "$SOFT_STEPS" \
        --sharp-steps "$SHARP_STEPS" \
        --out "$OUT_DIR"
fi

echo "task $TASK_ID done"
