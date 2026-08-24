#!/bin/bash
#
# Accuracy run: score Adaptyv's EXPERIMENTALLY LABELLED binders with every model
# and correlate each metric against the readout.
#
# This is the complement to model-matrix.sbatch. That one folds our own
# unlabelled DIO3 designs and measures whether the models AGREE; this one folds
# binders with a known answer and measures whether they are RIGHT. Complex only
# -- a monomer fold cannot be checked against a binding assay.
#
# No design happens here. The binders are Adaptyv's, already synthesised and
# assayed; we only refold them against the target and read metrics off.
#
#   TARGETS="nipah-glycoprotein-g" sbatch singularity/model-matrix-labelled.sbatch
#   TARGETS="nipah-glycoprotein-g egfr il7r pd-l1" sbatch ...
#
# Array size must match 6 models x N targets. Set --array on the command line
# when TARGETS is not the full four, e.g. --array=0-5%10 for one target.
#
# Sizing is from `jobstats 41448343_2` on the DIO3 run: 8.4 GB RAM of 128 GB
# requested, 9.6% of 8 cores, 24 min of a 10 h limit. GPU memory read 96% but
# that is JAX preallocating (mosaic.def:206 sets
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95), not real demand. Time is kept high
# because these targets are larger than DIO3 (Nipah 413 aa vs 201, so ~3x the
# per-fold cost at N^2) and carry ~20 length groups instead of 1 -- each length
# is its own JIT compile, which for the slower backends outweighs the folds.
#
#SBATCH --job-name=mosaic-labelled
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-23%10
#SBATCH --output=logs/labelled-%A_%a.out
#SBATCH --error=logs/labelled-%A_%a.out

set -euo pipefail

SETUP=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup
export MOSAIC_SIF="${MOSAIC_SIF:-$SETUP/images/mosaic.sif}"
export MOSAIC_WEIGHTS="${MOSAIC_WEIGHTS:-$SETUP/weights}"
export MOSAIC_SCRATCH="${MOSAIC_SCRATCH:-/n/netscratch/bsabatini_lab/Everyone/$USER/mosaic}"
export SINGULARITYENV_JAX_COMPILATION_CACHE_DIR=/jax_cache
export SINGULARITYENV_HF_HUB_OFFLINE=1

REPO="${MOSAIC_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
export MOSAIC_DEV_SRC="${MOSAIC_DEV_SRC-$REPO/src}"

LABELLED="${LABELLED:-$SETUP/benchmark/labelled}"
TARGETDIR="${TARGETDIR:-$SETUP/benchmark/targets}"
MSADIR="${MSADIR:-$SETUP/msa}"
OUTDIR="${OUTDIR:-$SETUP/benchmark/labelled_scores}"
TAG="${TAG:-acc}"
# Replicates default to 1: the diagonal was already measured on the DIO3 run and
# is a property of the model, not of the design set. Raise it only to re-check.
REPLICATES="${REPLICATES:-1}"
DIAG_N="${DIAG_N:-0}"
RECYCLING="${RECYCLING:-3}"
MAX_LEN_GROUPS="${MAX_LEN_GROUPS:-20}"
SEED="${SEED:-42}"

MODELS=(af2 boltz1 boltz2 of3 protenix esmfold2)
read -r -a TGTS <<< "${TARGETS:-nipah-glycoprotein-g egfr il7r pd-l1}"

i=${SLURM_ARRAY_TASK_ID:-0}
NT=${#TGTS[@]}
MODEL=${MODELS[$((i / NT))]}
TARGET=${TGTS[$((i % NT))]}

DESIGNS="$LABELLED/${TARGET}.csv"
TARGET_FASTA="$TARGETDIR/${TARGET}.fasta"
TARGET_MSA="$MSADIR/${TARGET}.a3m"

# Fail loudly and early rather than silently folding against a missing chain or
# a single-sequence target. An absent MSA is not a warning here: AF2 agrees with
# ITSELF at rho 0.30 without one (dio3_candidates/af2_iptm_model_spread.csv), so
# a no-MSA cell is noise wearing the costume of a measurement.
for f in "$DESIGNS" "$TARGET_FASTA" "$TARGET_MSA"; do
    [[ -s "$f" ]] || { echo "error: missing or empty: $f" >&2; exit 66; }
done

mkdir -p "$OUTDIR"
OUT="$OUTDIR/${TAG}_${MODEL}_${TARGET}.csv"

args=(
    --model "$MODEL" --condition complex
    --designs "$DESIGNS" --seq-col seq --label-col binder
    --target-fasta "$TARGET_FASTA" --target-msa "$TARGET_MSA"
    --out "$OUT" --replicates "$REPLICATES" --diag-n "$DIAG_N"
    --recycling "$RECYCLING" --seed "$SEED"
    --max-length-groups "$MAX_LEN_GROUPS"
)
[[ -n "${LIMIT:-}" ]] && args+=(--limit "$LIMIT")

echo "task $i -> model=$MODEL target=$TARGET"
echo "node:   $(hostname -f)"
echo "out:    $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

exec "$REPO/singularity/mosaic-exec.sh" \
    python "$REPO/benchmark/model_matrix.py" "${args[@]}"
