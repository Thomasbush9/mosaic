#!/usr/bin/env bash
#
# Run the DistogramIPTMProxy tests inside the mosaic container, against the
# WORKING TREE rather than the image's copy of the code -- so an edit to
# src/mosaic/losses/structure_prediction.py takes effect without a rebuild.
#
#   tests/run_iptm_proxy_tests.sh              # toy tier only (seconds, no GPU)
#   tests/run_iptm_proxy_tests.sh --real       # + AF2 on 7opb (GPU, ~10 min)
#   tests/run_iptm_proxy_tests.sh --bench      # comparison report -> tests/out/
#
# Anything after `--` is passed straight to pytest.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETUP="$(dirname "$REPO")"

export MOSAIC_SIF="${MOSAIC_SIF:-$SETUP/images/mosaic.sif}"
export MOSAIC_WEIGHTS="${MOSAIC_WEIGHTS:-$SETUP/weights}"
export MOSAIC_SCRATCH="${MOSAIC_SCRATCH:-/n/netscratch/bsabatini_lab/Everyone/$USER/mosaic}"
export MOSAIC_DEV_SRC="${MOSAIC_DEV_SRC:-$REPO/src}"
# The image has no tests/ and Apptainer overlay auto-creation is not enabled
# here, so the mount point has to be one that already exists: /mnt. The whole
# setup dir goes there, not just the checkout, because the real-data tier reads
# dio3_candidates/ and targets/ from alongside it.
export SINGULARITY_BIND="${SINGULARITY_BIND:+$SINGULARITY_BIND,}$SETUP:/mnt"

mode=toy
case "${1:-}" in
    --real)  mode=real;  shift ;;
    --bench) mode=bench; shift ;;
    --toy)   mode=toy;   shift ;;
esac
[[ "${1:-}" == "--" ]] && shift

cd "$REPO"
case "$mode" in
toy)
    exec ./singularity/mosaic-exec.sh \
        env JAX_PLATFORMS=cpu PYTHONPATH=/mnt/mosaic/tests:/mnt/mosaic \
        /opt/mosaic/.venv/bin/python -m pytest \
        /mnt/mosaic/tests/test_distogram_iptm_proxy.py -v --no-header "$@"
    ;;
real)
    exec ./singularity/mosaic-exec.sh \
        env PYTHONPATH=/mnt/mosaic/tests:/mnt/mosaic \
        /opt/mosaic/.venv/bin/python -m pytest \
        /mnt/mosaic/tests/test_distogram_iptm_proxy_real.py -v --no-header -s -m slow "$@"
    ;;
bench)
    mkdir -p "$REPO/tests/out"
    exec ./singularity/mosaic-exec.sh \
        env PYTHONPATH=/mnt/mosaic/tests:/mnt/mosaic \
        /opt/mosaic/.venv/bin/python /mnt/mosaic/tests/bench_distogram_iptm_proxy.py \
        --out /mnt/mosaic/tests/out "$@"
    ;;
esac
