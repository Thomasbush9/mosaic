#!/bin/bash
#
# Complex-mode self-consistency: does the binding POSE survive an MPNN redesign
# round trip? See hallucinate/self_consistency_complex.py for what it measures.
#
#     mkdir -p logs && sbatch singularity/self-consistency-complex.sbatch
#
# Measured cost (H100, 80 aa binder + 201 aa target, recycling 3): 1.68 s/fold,
# 13.4x the single-chain-no-MSA fold. Each design is N_SAMPLES+1 folds, so the
# 50-design pilot is ~10 min and all 1000 BoltzGen candidates at n=10 is ~5 h.
#
# CAUTION — target must match what the designs were made against:
#   BoltzGen screen (dio3_candidates/screen_ranked.csv) -> targets/dio3_ecd.fasta (237 aa)
#   hallucination set (dio3_cut/designs/*)              -> dio3_cut/dio3_cut.fasta (201 aa)
# The cut is an internal fragment of the ECD starting at offset 36; they are NOT
# interchangeable and nothing downstream will warn you.
#
#SBATCH --job-name=mosaic-scpose
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/scpose-%j.out
#SBATCH --error=logs/scpose-%j.out

set -euo pipefail

SETUP=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/mosaic_setup
# images/ did not move with the rest of the tree: the SIFs now
# live one level up, in binder_design/images/.
IMAGES=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design/images

export MOSAIC_SIF="${MOSAIC_SIF:-$IMAGES/mosaic.sif}"
export MOSAIC_WEIGHTS="${MOSAIC_WEIGHTS:-$SETUP/weights}"
export MOSAIC_SCRATCH="${MOSAIC_SCRATCH:-/n/netscratch/bsabatini_lab/Everyone/$USER/mosaic}"
export SINGULARITYENV_JAX_COMPILATION_CACHE_DIR=/jax_cache

REPO="${MOSAIC_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
export MOSAIC_DEV_SRC="${MOSAIC_DEV_SRC-$REPO/src}"

DESIGNS="${DESIGNS:-$SETUP/dio3_candidates/screen_ranked.csv}"
SEQ_COL="${SEQ_COL:-sequence}"
TARGET_FASTA="${TARGET_FASTA:-$SETUP/targets/dio3_ecd.fasta}"
TARGET_MSA="${TARGET_MSA:-$SETUP/msa/dio3_ecd.a3m}"
OUT="${OUT:-$SETUP/dio3_candidates/self_consistency_pose_pilot.csv}"
N_SAMPLES="${N_SAMPLES:-5}"
SEED="${SEED:-42}"
PILOT_TOP="${PILOT_TOP:-20}"
PILOT_RANDOM="${PILOT_RANDOM:-30}"
# Pin the AF2 parameter set. To compare against a label produced elsewhere, use
# the model THAT run drew: af2_ranking.py hands AlphaFoldLoss a fixed
# key=jax.random.key(42) for every design, which resolves to model_idx 4.
MODEL_IDX="${MODEL_IDX:-0}"
RECYCLING="${RECYCLING:-3}"

args=(
    --designs "$DESIGNS" --seq-col "$SEQ_COL"
    --target-fasta "$TARGET_FASTA" --target-msa "$TARGET_MSA"
    --out "$OUT" --n "$N_SAMPLES" --seed "$SEED"
    --model-idx "$MODEL_IDX" --recycling-steps "$RECYCLING"
    --pilot-top "$PILOT_TOP" --pilot-random "$PILOT_RANDOM"
)
[[ -n "${LIMIT:-}" ]] && args+=(--limit "$LIMIT")

echo "node:   $(hostname -f)"
echo "target: $TARGET_FASTA"
echo "out:    $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

exec "$REPO/singularity/mosaic-exec.sh" \
    python "$REPO/hallucinate/self_consistency_complex.py" "${args[@]}"
