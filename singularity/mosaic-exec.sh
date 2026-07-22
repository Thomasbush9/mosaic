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

# jopenfold3 reads ~/.openfold3/jax/of3.{skeleton.pkl,eqx} — a plain default
# argument to OpenFold3.load(), with no env var to redirect it. jopenfold3
# converts the upstream torch checkpoint to equinox on first call; the
# conversion is done once by fetch-weights.sbatch and bound read-only here.
bind_cache openfold3 .openfold3

# AbLang/AbLang2 are the awkward case: they cache inside their own package
# directory rather than under HOME, so these bind onto site-packages paths that
# mosaic.def pre-creates (they cannot be created here — the image is read-only).
# Done this way, ablang.pretrained("heavy") finds amodel.pt and never attempts
# the download that would fail on the read-only filesystem.
# MOSAIC_DEV_SRC=<repo>/src binds the working tree's source over the image's
# copy, so edits to src/mosaic take effect immediately instead of costing a
# ~10 min rebuild. This works because mosaic is an editable install: the venv
# ships _editable_impl_mosaic.pth pointing at /opt/mosaic/src, so whatever is
# mounted there is what gets imported.
#
# Off by default and announced loudly when on, because a run with it set is NOT
# reproducible from the .sif alone — the image no longer determines the code.
# Use it while iterating; rebuild before a campaign you intend to keep.
if [[ -n "${MOSAIC_DEV_SRC:-}" ]]; then
    if [[ ! -d "$MOSAIC_DEV_SRC/mosaic" ]]; then
        echo "error: MOSAIC_DEV_SRC=$MOSAIC_DEV_SRC has no mosaic/ subdirectory" >&2
        exit 65
    fi
    binds+=(-B "$MOSAIC_DEV_SRC:/opt/mosaic/src:ro")
    echo "note: DEV SOURCE from $MOSAIC_DEV_SRC overrides the image's src" >&2
fi

SITE=/opt/mosaic/.venv/lib/python3.12/site-packages
bind_pkg_weights() {
    local src="$MOSAIC_WEIGHTS/$1" dest="$2"
    if [[ -d "$src" ]]; then
        binds+=(-B "$src:$dest:ro")
    else
        echo "note: $src not present — skipping bind for $dest" >&2
    fi
}
bind_pkg_weights ablang/heavy  "$SITE/ablang/model-weights-heavy"
bind_pkg_weights ablang/light  "$SITE/ablang/model-weights-light"
bind_pkg_weights ablang2/paired "$SITE/ablang2/model-weights-ablang2-paired"

# jproteina_complexa.hub hardcodes ~/.cache/jproteina_complexa as a default
# argument too (hub.py:8), so it needs the same treatment.
bind_cache jproteina .cache/jproteina_complexa

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

# The FASRC hosts export RHEL CA paths (SSL_CERT_FILE / CURL_CA_BUNDLE ->
# /etc/ssl/certs/ca-bundle.crt) which do not exist in this Ubuntu container,
# where the bundle is ca-certificates.crt. Singularity inherits them, so every
# HTTPS request through `requests` dies with
#   OSError: Could not find a suitable TLS CA certificate bundle
# That breaks weight downloads and, at run time, the ColabFold MSA fetches that
# Boltz/OF3/Protenix make during featurization. Point them at the container's
# own bundle.
export SINGULARITYENV_SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export SINGULARITYENV_SSL_CERT_DIR=/etc/ssl/certs
export SINGULARITYENV_CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

cmd=(singularity exec $MOSAIC_GPU_FLAG -H "$CHOME" "${binds[@]}" "$MOSAIC_SIF" "$@")

if [[ $SHOW_ONLY -eq 1 ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    exit 0
fi

exec "${cmd[@]}"
