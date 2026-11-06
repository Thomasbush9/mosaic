#!/bin/bash
#
# Model x model agreement matrices: one array task per (model, condition).
#
#   smoke test (2 designs, 1 replicate -- proves all 12 combos load and fold):
#       mkdir -p logs
#       LIMIT=2 REPLICATES=1 TAG=smoke sbatch singularity/model-matrix.sbatch
#
#   full run (original protocol):
#       sbatch singularity/model-matrix.sbatch
#
#   ranking-matched protocol (real binder sidechains, 6 diffusion samples),
#   complex only -- the run that is comparable to the af2/of3 ranking columns:
#       FEATURIZATION=target_only NUM_SAMPLES=6 ONLY_CONDITION=complex \
#           TAG=tofull sbatch --array=0-5%3 singularity/model-matrix.sbatch
#
# 6 models x 2 conditions = 12 tasks, capped at 5 concurrent (%5).
#
# One task per (model, condition) rather than per (model, condition, replicate):
# JIT compile of a structure model costs minutes and dominates a monomer fold,
# so the replicate loop belongs INSIDE a task where it reuses the compiled
# function. Splitting it across tasks would pay the compile 3x for nothing.
#
#SBATCH --job-name=mosaic-matrix
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-11%5
#SBATCH --output=logs/matrix-%A_%a.out
#SBATCH --error=logs/matrix-%A_%a.out

set -euo pipefail

SETUP=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/mosaic_setup
# images/ did not move with the rest of the tree: the SIFs now
# live one level up, in binder_design/images/.
IMAGES=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/images
export MOSAIC_SIF="${MOSAIC_SIF:-$IMAGES/mosaic.sif}"
export MOSAIC_WEIGHTS="${MOSAIC_WEIGHTS:-$SETUP/weights}"
export MOSAIC_SCRATCH="${MOSAIC_SCRATCH:-/n/netscratch/bsabatini_lab/Everyone/$USER/mosaic}"
export SINGULARITYENV_JAX_COMPILATION_CACHE_DIR=/jax_cache
export SINGULARITYENV_HF_HUB_OFFLINE=1

REPO="${MOSAIC_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
export MOSAIC_DEV_SRC="${MOSAIC_DEV_SRC-$REPO/src}"

# The hallucination set and its target. dio3_cut (201 aa) is an internal
# fragment of dio3_ecd starting at offset 36 -- NOT interchangeable with it, and
# nothing downstream would warn you. The MSA is the one hallucinate.py:39 used.
DESIGNS="${DESIGNS:-$SETUP/dio3_cut/designs/self_consistency.csv}"
SEQ_COL="${SEQ_COL:-seq}"
TARGET_FASTA="${TARGET_FASTA:-$SETUP/dio3_cut/dio3_cut.fasta}"
TARGET_MSA="${TARGET_MSA:-$SETUP/dio3_cut/msa_out/sequences/dio3_cut/msa/DIO3.a3m}"
OUTDIR="${OUTDIR:-$SETUP/benchmark/matrix}"
TAG="${TAG:-full}"
REPLICATES="${REPLICATES:-3}"
DIAG_N="${DIAG_N:-100}"
RECYCLING="${RECYCLING:-3}"
SEED="${SEED:-42}"
# Protocol. binder/1 is the original matrix; target_only/6 matches what the
# ranking runs actually did (hallucinate.py:137, rank_designs.py:232) and is the
# fair comparison against dio3_cut/designs/correctly_ranked_candidates_af2.csv.
FEATURIZATION="${FEATURIZATION:-binder}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
# Pinned, not left to each backend's own default -- those range from 2 (Protenix
# mini) to 25 (Boltz), which silently made every run before 2026-08-24
# incomparable. RECYCLING counts TRUNK PASSES; model_matrix.py subtracts the
# offset OF3 and ESMFold2 add internally. Both are echoed in the PROTOCOL line
# of every log, which is the thing to diff when comparing two runs.
SAMPLING_STEPS="${SAMPLING_STEPS:-25}"

# XLA preallocates XLA_PYTHON_CLIENT_MEM_FRACTION of the card (0.95, set in
# mosaic.def). CUBIN modules and command buffers are allocated OUTSIDE that pool,
# so a run that recompiles per design (--featurization target_only) exhausts the
# remaining sliver long before the pool itself is full. It surfaces as "Failed to
# load in-memory CUBIN" or "instantiate command buffer", NOT as a normal JAX OOM,
# and the per-design except records NaN -- so the run completes while quietly
# dropping the designs that needed the most memory, which are the longest
# binders. A length-correlated dropout in one model's column is exactly what a
# model comparison cannot absorb. Protenix hits this on the 434-design labelled
# set (20 length groups, 413 aa target); XLA_MEM_FRACTION=0.70 gives it room.
# Purely an allocator setting -- it changes no number the run produces.
[[ -n "${XLA_MEM_FRACTION:-}" ]] && \
    export SINGULARITYENV_XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_MEM_FRACTION"

MODELS=(af2 boltz1 boltz2 of3 protenix esmfold2)
CONDITIONS=(complex monomer)

i=${SLURM_ARRAY_TASK_ID:-0}
if [[ -n "${ONLY_MODEL:-}" ]]; then
    # One model -> the array index is the SHARD index, not a model index.
    MODEL=$ONLY_MODEL
    CONDITION=${ONLY_CONDITION:-complex}
elif [[ -n "${ONLY_CONDITION:-}" ]]; then
    # One condition -> submit --array=0-5, one task per model.
    MODEL=${MODELS[$i]}
    CONDITION=$ONLY_CONDITION
else
    MODEL=${MODELS[$((i / 2))]}
    CONDITION=${CONDITIONS[$((i % 2))]}
fi

mkdir -p "$OUTDIR"
OUT="$OUTDIR/${TAG}_${MODEL}_${CONDITION}.csv"
# NUM_SHARDS: one model split across --array=0-(N-1), one task per slice of the
# design set. Cuts wall time at the cost of N model loads and N JIT compiles.
# Merge with benchmark/merge_shards.py; design_idx is global so concat suffices.
[[ -n "${NUM_SHARDS:-}" ]] && OUT="${OUT%.csv}_shard${i}.csv"

args=(
    --model "$MODEL" --condition "$CONDITION"
    --designs "$DESIGNS" --seq-col "$SEQ_COL"
    --target-fasta "$TARGET_FASTA" --target-msa "$TARGET_MSA"
    --out "$OUT" --replicates "$REPLICATES" --diag-n "$DIAG_N"
    --recycling "$RECYCLING" --seed "$SEED"
    --featurization "$FEATURIZATION" --num-samples "$NUM_SAMPLES"
    --sampling-steps "$SAMPLING_STEPS"
)
[[ -n "${LIMIT:-}" ]] && args+=(--limit "$LIMIT")
[[ -n "${AF2_REPLICATE:-}" ]] && args+=(--af2-replicate "$AF2_REPLICATE")
[[ -n "${AF2_MODEL_IDX:-}" ]] && args+=(--af2-model-idx "$AF2_MODEL_IDX")
[[ -n "${CLEAR_CACHE_EVERY:-}" ]] && args+=(--clear-cache-every "$CLEAR_CACHE_EVERY")
[[ -n "${NUM_SHARDS:-}" ]] && args+=(--shard "$i" --num-shards "$NUM_SHARDS")

echo "task $i -> model=$MODEL condition=$CONDITION"
echo "protocol: featurization=$FEATURIZATION num_samples=$NUM_SAMPLES"
echo "          trunk_passes=$RECYCLING sampling_steps=$SAMPLING_STEPS"
echo "          target_msa=$TARGET_MSA"
echo "node:   $(hostname -f)"
echo "out:    $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

exec "$REPO/singularity/mosaic-exec.sh" \
    python "$REPO/benchmark/model_matrix.py" "${args[@]}"
