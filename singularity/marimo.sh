#!/usr/bin/env bash
#
# Launch a marimo notebook server inside the mosaic container, on a GPU node,
# reachable from your laptop over an ssh tunnel.
#
#     # on a login node
#     salloc -p kempner_h100 -A kempner_bsabatini_lab --gres=gpu:1 \
#            -c 8 --mem=128G -t 6:00:00
#     # once the allocation lands (you get a shell ON the GPU node)
#     cd /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup/mosaic
#     ./singularity/marimo.sh examples/example_notebook.py
#     # the banner prints the exact ssh command to run on your laptop
#
# Or non-interactively, holding a node for the job's walltime:
#
#     mkdir -p logs && sbatch singularity/marimo.sbatch
#     # read logs/marimo-<jobid>.out for the ssh command and the URL
#
# Everything a notebook needs comes from mosaic-exec.sh: the same read-only
# weight binds the batch jobs use (~/.boltz, ~/.alphafold, ~/.protenix,
# ~/.openfold3, ~/.cache/huggingface, AbLang's in-package caches), the writable
# per-user HOME on scratch, and the shared /jax_cache. So a notebook session hits
# exactly the caches a campaign does and downloads nothing.
#
# The notebook FILE is read from the repo on /n/holylfs06 (auto-bound inside the
# container, and writable), not from the image's /opt/mosaic/examples copy, which
# is read-only — marimo could not save there.
#
# Configure with: MOSAIC_PORT MOSAIC_TOKEN MOSAIC_REPO MOSAIC_DEV_SRC
#                 MOSAIC_SIF MOSAIC_WEIGHTS MOSAIC_SCRATCH
set -euo pipefail

SETUP=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup

export MOSAIC_SIF="${MOSAIC_SIF:-$SETUP/images/mosaic.sif}"
export MOSAIC_WEIGHTS="${MOSAIC_WEIGHTS:-$SETUP/weights}"
export MOSAIC_SCRATCH="${MOSAIC_SCRATCH:-/n/netscratch/bsabatini_lab/Everyone/$USER/mosaic}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${MOSAIC_REPO:-$(cd "$HERE/.." && pwd)}"

# A marimo server is a long-lived process that accumulates CPU, and a notebook
# running a structure model would peg a core for minutes at a time. That is
# precisely what FASRC's process arbiter kills on a login node — the job dies
# silently and presents to the browser as a dropped connection with nothing in
# the log. Refuse rather than reproduce that.
if [[ -z "${SLURM_JOB_ID:-}" && "${MOSAIC_ALLOW_LOGIN_NODE:-0}" != "1" ]]; then
    cat >&2 <<'ERR'
error: no SLURM allocation — this looks like a login node.

Get a GPU node first, then re-run this from the allocation's shell:

    salloc -p kempner_h100 -A kempner_bsabatini_lab --gres=gpu:1 \
           -c 8 --mem=128G -t 6:00:00

(kempner_interactive also has GPUs, but they are 20 GB A100 MIG slices —
fine for small Boltz/AF2 work, not for ESM-C 6B or large complexes.)

Set MOSAIC_ALLOW_LOGIN_NODE=1 to override, at your own risk.
ERR
    exit 78
fi

NOTEBOOK="${1:-$REPO/examples/example_notebook.py}"
[[ "$NOTEBOOK" = /* ]] || NOTEBOOK="$REPO/$NOTEBOOK"

if [[ "$NOTEBOOK" == /opt/mosaic/* ]]; then
    echo "error: $NOTEBOOK is inside the read-only image — marimo cannot save there." >&2
    echo "       use the repo copy: $REPO/examples/..." >&2
    exit 65
fi
if [[ ! -e "$NOTEBOOK" ]]; then
    # marimo creates a new notebook from a nonexistent NAME, which is a fine way
    # to start a fresh one — but only if the directory exists and is writable.
    NBDIR="$(dirname "$NOTEBOOK")"
    [[ -d "$NBDIR" ]] || { echo "error: no such directory: $NBDIR" >&2; exit 66; }
    [[ -w "$NBDIR" ]] || { echo "error: not writable: $NBDIR" >&2; exit 66; }
    echo "note: $NOTEBOOK does not exist — marimo will create it." >&2
elif [[ ! -w "$NOTEBOOK" ]]; then
    echo "error: not writable: $NOTEBOOK" >&2
    exit 66
fi

# Compute nodes are shared, so the default port may already belong to someone
# else's server. Walk forward until one is free instead of failing at bind time.
port_free() {
    if command -v ss >/dev/null 2>&1; then
        [[ -z "$(ss -Htln "sport = :$1" 2>/dev/null)" ]]
    else
        ! (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
    fi
}
PORT="${MOSAIC_PORT:-2718}"
for _ in $(seq 1 25); do
    port_free "$PORT" && break
    PORT=$((PORT + 1))
done
if ! port_free "$PORT"; then
    echo "error: no free port near ${MOSAIC_PORT:-2718} — set MOSAIC_PORT." >&2
    exit 69
fi

# Bound to 127.0.0.1, but a compute node is shared and any other user's process
# on it can reach that address. marimo's edit mode executes arbitrary code as
# whoever started it, so the token is not optional here.
TOKEN="${MOSAIC_TOKEN:-$(head -c 18 /dev/urandom | base64 | tr -dc 'A-Za-z0-9')}"

# Persistent XLA cache without touching the notebook. Only examples/proteina.py
# and examples/promera_design.py call jax.config.update("jax_compilation_cache_dir",
# ...) themselves; this gives every other notebook the same shared /jax_cache
# that mosaic-exec.sh binds, so a re-run after a kernel restart skips the
# multi-minute compile.
export SINGULARITYENV_JAX_COMPILATION_CACHE_DIR=/jax_cache

if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)"
    [[ -n "$GPUS" ]] && echo "GPU: $GPUS" || echo "WARNING: nvidia-smi reports no GPU — JAX will fall back to CPU." >&2
else
    echo "WARNING: no nvidia-smi on this host — JAX will fall back to CPU." >&2
fi

if [[ -n "${MOSAIC_DEV_SRC:-}" ]]; then
    echo "dev source: $MOSAIC_DEV_SRC (overrides the image's src/mosaic)"
else
    echo "hint: MOSAIC_DEV_SRC=$REPO/src to edit src/mosaic without rebuilding the image"
fi

NODE="$(hostname -f)"
LOGIN="${MOSAIC_LOGIN_NODE:-holylogin06.rc.fas.harvard.edu}"
cat <<MSG

================================================================
  mosaic + marimo — running in the container on ${NODE}

  notebook: ${NOTEBOOK}

  From your laptop, forward the port through the login node:

      ssh -J ${USER}@${LOGIN} -L ${PORT}:localhost:${PORT} ${USER}@${NODE}

  Then open:

      http://localhost:${PORT}/?access_token=${TOKEN}

  Ctrl-C here stops the server. ${SLURM_JOB_ID:+scancel ${SLURM_JOB_ID} releases the node.}
================================================================

MSG

cd "$REPO"
exec "$HERE/mosaic-exec.sh" marimo edit "$NOTEBOOK" \
    --headless \
    --host 127.0.0.1 \
    --port "$PORT" \
    --token-password "$TOKEN" \
    --skip-update-check
