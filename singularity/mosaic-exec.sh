#!/usr/bin/env bash
#
# Run a command inside the mosaic container with the weight caches bound into
# place.
#
# Each cache is bound directly onto the path mosaic already looks for, inside a
# writable per-user HOME. No symlinks anywhere: a link created on the host
# encodes a host path, which does not exist in the container's namespace, so it
# would dangle in the only place it is ever dereferenced.
#
# That also means no environment overrides are needed. mosaic's own defaults
# (~/.boltz, ~/.alphafold, ~/.protenix, ~/.cache/huggingface,
# ~/.cache/mosaic/...) all resolve correctly once HOME is a writable directory
# with the right things mounted into it.
#
# Layout expected in $MOSAIC_WEIGHTS (plain names, not dotfiles — the shared
# tree does not have to mimic a home directory):
#
#     boltz/       -> ~/.boltz              Boltz-1/2, BoltzGen, ccd.pkl, mols/
#     alphafold/   -> ~/.alphafold          AF2 params
#     protenix/    -> ~/.protenix           Protenix weights + reference data
#     hf/          -> ~/.cache/huggingface  ESM-C, ESMFold2, ESM2
#
# Usage:
#     ./mosaic-exec.sh python /opt/mosaic/singularity/selftest.py
#     ./mosaic-exec.sh --show python run_design.py --seed 3   # print, don't run
#
# Override any of these from the environment or your sbatch script:
#     MOSAIC_SIF MOSAIC_WEIGHTS MOSAIC_SCRATCH MOSAIC_GPU_FLAG
#
set -euo pipefail

: "${MOSAIC_SIF:=/shared/mosaic/mosaic.sif}"
: "${MOSAIC_WEIGHTS:=/shared/mosaic-weights}"
: "${MOSAIC_SCRATCH:=/scratch/$USER/mosaic}"
: "${MOSAIC_GPU_FLAG:=--nv}"

SHOW_ONLY=0
if [[ "${1:-}" == "--show" ]]; then
    SHOW_ONLY=1
    shift
fi

if [[ $# -eq 0 ]]; then
    echo "usage: $0 [--show] <command> [args...]" >&2
    exit 64
fi

if [[ ! -f "$MOSAIC_SIF" ]]; then
    echo "error: image not found: $MOSAIC_SIF" >&2
    echo "       set MOSAIC_SIF, or build it (see singularity/README.md)" >&2
    exit 66
fi

CHOME="$MOSAIC_SCRATCH/home"

# Bind destinations must already exist. $CHOME is itself a host directory that
# becomes the container HOME, so creating these on the host creates them inside
# the container as well — no reliance on Apptainer's overlay auto-creation,
# which is not enabled on every cluster.
mkdir -p "$CHOME/.cache"
mkdir -p "$CHOME/.cache/mosaic/msa"
mkdir -p "$CHOME/.cache/mosaic/opendde_templates"
mkdir -p "$MOSAIC_SCRATCH/jax_cache"
mkdir -p "$MOSAIC_SCRATCH/out"

binds=()

# Bind a shared cache read-only onto its home location, but only if the source
# is actually populated. A missing source is skipped rather than mounted empty,
# so mosaic falls back to downloading it instead of failing confusingly.
bind_cache() {
    local src="$MOSAIC_WEIGHTS/$1" dest="$CHOME/$2"
    if [[ -d "$src" ]]; then
        mkdir -p "$dest"
        binds+=(-B "$src:$dest:ro")
    else
        echo "note: $src not present — skipping bind for ~/$2" >&2
    fi
}

bind_cache boltz     .boltz
bind_cache alphafold .alphafold
bind_cache protenix  .protenix
bind_cache hf        .cache/huggingface

# /jax_cache is bound, not defaulted: examples/proteina.py:16 and
# examples/promera_design.py:26 hardcode that absolute path. Shared and
# writable, so array tasks reuse each other's compiled kernels instead of each
# paying several minutes of first-iteration compilation.
binds+=(-B "$MOSAIC_SCRATCH/jax_cache:/jax_cache")
binds+=(-B "$MOSAIC_SCRATCH/out:/work")

# Singularity inherits the host environment, so a host-side HF_HOME (set in
# ~/.bashrc to keep weights off the home quota) would follow us in and point at
# a host path instead of the bind above. mosaic.def deliberately leaves HF_HOME
# unset precisely so the bind is what decides; pin it to the container-side path
# here so the result does not depend on the caller's shell.
export SINGULARITYENV_HF_HOME="$CHOME/.cache/huggingface"

cmd=(singularity exec $MOSAIC_GPU_FLAG -H "$CHOME" "${binds[@]}" "$MOSAIC_SIF" "$@")

if [[ $SHOW_ONLY -eq 1 ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    exit 0
fi

exec "${cmd[@]}"
