#!/bin/bash
#
# Hallucinate DIO3 binders as a SLURM array: one GPU per task, each task
# designing sequences in a loop until its time budget runs out.
#
#     cd <repo root>
#     mkdir -p logs
#     sbatch --array=0-15%8 hallucinate/launch_hallucinations.slurm
#
# Total designs = tasks x (MAX_RUNTIME / time-per-design). At roughly 8 minutes
# per design, a 16-task array with MAX_RUNTIME=7.5 gives on the order of 900.
#
# Configure via the environment:
#     MAX_RUNTIME  hours each task keeps designing (default 7.5, under the 8h wall)
#     N_DESIGNS    stop each task after this many designs (0 = time-bounded)
#     SAVE_DIR     where designs_<task>.txt are written
#
#     MAX_RUNTIME=3 SAVE_DIR=/path/out sbatch --array=0-7 hallucinate/launch_hallucinations.slurm
#
# WHY AN ARRAY AND NOT MULTI-GPU: mosaic is single-device throughout — no pmap,
# shard_map or jax.sharding anywhere in src/ — so a second GPU in one process
# would sit idle. Scale with tasks, and throttle with %N to stay inside the
# per-user cap.
#
# --gres=gpu:1 rather than --gpus-per-node=1: on FASRC the GRES name is
# nvidia_h100_80gb_hbm3, and gpu:1 is the form that reliably allocates.
#
#SBATCH --job-name=dio3-hallucinate
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/hallucinate-%A_%a.out
#SBATCH --error=logs/hallucinate-%A_%a.out

set -euo pipefail

SETUP=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup

export MOSAIC_SIF="${MOSAIC_SIF:-$SETUP/images/mosaic.sif}"
export MOSAIC_WEIGHTS="${MOSAIC_WEIGHTS:-$SETUP/weights}"
export MOSAIC_SCRATCH="${MOSAIC_SCRATCH:-/n/netscratch/bsabatini_lab/Everyone/$USER/mosaic}"

REPO="${MOSAIC_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"

# Share compiled XLA kernels across every task in the array. Without this each
# task pays the first-iteration compile (minutes) on its own; with it, only the
# first task to get there does. /jax_cache is bound writable by mosaic-exec.sh.
export SINGULARITYENV_JAX_COMPILATION_CACHE_DIR=/jax_cache

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SAVE_DIR="${SAVE_DIR:-$SETUP/dio3_cut/designs}"

echo "task $TASK_ID on $(hostname -s) -> $SAVE_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

exec "$REPO/singularity/mosaic-exec.sh" python "$REPO/hallucinate/hallucinate.py" \
    --array_id "$TASK_ID" \
    --max_runtime "${MAX_RUNTIME:-7.5}" \
    --n_designs "${N_DESIGNS:-0}" \
    --save_dir "$SAVE_DIR"
