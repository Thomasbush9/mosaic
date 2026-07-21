#!/bin/bash
#
# Batch design sweep: N independent trajectories, throttled to the per-user GPU
# cap. Submit with:  sbatch singularity/design.sbatch
#
# Requires logs/ to already exist (SLURM opens --output before this script runs):
#     mkdir -p logs && sbatch singularity/design.sbatch
#
# DESIGN_SCRIPT defaults to /opt/mosaic/run_design.py — add that entrypoint to
# the repo (and mosaic.def %files) when ready, or override:
#     DESIGN_SCRIPT=/opt/mosaic/examples/your_script.py sbatch ...
#
#SBATCH --job-name=mosaic
#SBATCH --array=0-63%8          # 64 trajectories, 8 at a time (per-user cap)
#SBATCH --gres=gpu:h100:1       # 1 GPU/task — no multi-GPU support exists
#SBATCH --cpus-per-task=8       # host numpy runs every optimizer iteration
#SBATCH --mem=128G              # torch ckpt + fp32 + JAX copy are co-resident
#SBATCH --time=24:00:00         # optimizers have no checkpoint/resume
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.out

set -euo pipefail

# ── configuration ────────────────────────────────────────────────────────────
export MOSAIC_SIF="${MOSAIC_SIF:-/shared/mosaic/mosaic.sif}"
export MOSAIC_WEIGHTS="${MOSAIC_WEIGHTS:-/shared/mosaic-weights}"
export MOSAIC_SCRATCH="${MOSAIC_SCRATCH:-/scratch/$USER/mosaic}"

TARGET="${TARGET:-/work/PDL1.pdb}"
BINDER_LENGTH="${BINDER_LENGTH:-80}"
DESIGN_SCRIPT="${DESIGN_SCRIPT:-/opt/mosaic/run_design.py}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$MOSAIC_SCRATCH"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
PREFLIGHT_OK="$MOSAIC_SCRATCH/preflight.${JOB_ID}.ok"
PREFLIGHT_FAIL="$MOSAIC_SCRATCH/preflight.${JOB_ID}.fail"

# ── array barrier: task 0 preflight must pass before anyone runs ─────────────
# Without this, tasks 1–N start in parallel with task 0 and burn GPU hours on a
# broken --nv / empty-weight setup. Flag files are per array job id.
if [[ "$TASK_ID" == "0" ]]; then
    rm -f "$PREFLIGHT_OK" "$PREFLIGHT_FAIL"
    echo "── preflight ──"
    if ! "$HERE/mosaic-exec.sh" python /opt/mosaic/singularity/selftest.py; then
        touch "$PREFLIGHT_FAIL"
        echo "error: preflight failed — aborting array" >&2
        exit 1
    fi
    touch "$PREFLIGHT_OK"
    echo "───────────────"
else
    # ~30 min ceiling; task 0 may be queued behind the %8 throttle.
    for _ in $(seq 1 360); do
        if [[ -f "$PREFLIGHT_OK" ]]; then
            break
        fi
        if [[ -f "$PREFLIGHT_FAIL" ]]; then
            echo "error: preflight failed (task 0) — skipping" >&2
            exit 1
        fi
        sleep 5
    done
    if [[ ! -f "$PREFLIGHT_OK" ]]; then
        echo "error: timed out waiting for task 0 preflight" >&2
        exit 1
    fi
fi

# ── the run ──────────────────────────────────────────────────────────────────
# --seed is what makes the 64 trajectories differ (optimizers are deterministic
# given a key). DESIGN_SCRIPT is the future CLI entrypoint; override until then.
exec "$HERE/mosaic-exec.sh" \
    python "$DESIGN_SCRIPT" \
        --target "$TARGET" \
        --binder-length "$BINDER_LENGTH" \
        --seed "$TASK_ID" \
        --out "/work/design_${TASK_ID}"