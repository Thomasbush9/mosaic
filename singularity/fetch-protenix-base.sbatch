#!/bin/bash
#SBATCH --job-name=protenix-base-fetch
#SBATCH --partition=kempner_interactive
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/protenix-base-fetch-%j.out
#SBATCH --error=logs/protenix-base-fetch-%j.out
set -euo pipefail

ROOT=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/binder_design
SIF="${MOSAIC_SIF:-$ROOT/images/mosaic.sif}"
W="${MOSAIC_WEIGHTS:-$ROOT/mosaic_setup/weights}"
SCRATCH="${MOSAIC_SCRATCH:-/n/netscratch/bsabatini_lab/Everyone/$USER/mosaic}"
SCRIPT="$ROOT/mosaic_setup/mosaic/singularity/fetch-protenix-base.py"
CHOME="$SCRATCH/protenix-base-fetch-home-$SLURM_JOB_ID"
mkdir -p "$CHOME/.cache" "$CHOME/.protenix"
export SINGULARITYENV_PROTENIX_DATA_ROOT_DIR="$CHOME/.protenix"
export SINGULARITYENV_JAX_PLATFORMS=cpu
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=""
export SINGULARITYENV_SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export SINGULARITYENV_SSL_CERT_DIR=/etc/ssl/certs
export SINGULARITYENV_CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SINGULARITYENV_REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SINGULARITYENV_ACQUISITION_SIF="$SIF"
# Only the Protenix cache is writable; no other weight caches are mounted.
singularity exec -H "$CHOME" -B "$W/protenix:$CHOME/.protenix" \
    -B "$SCRIPT:/opt/fetch-protenix-base.py:ro" "$SIF" \
    python -u /opt/fetch-protenix-base.py
