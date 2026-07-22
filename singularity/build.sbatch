#!/bin/bash
#
# Build mosaic.sif on a compute node.
#
#     mkdir -p logs && sbatch singularity/build.sbatch
#
# WHY NOT A LOGIN NODE: the build itself succeeds there, but the final
# `mksquashfs` step is CPU- and memory-heavy and gets killed by FASRC's login
# node process arbiter — surfacing as an unhelpful
# `while creating squashfs: create command failed: signal: killed`, with the
# preceding %post and %test output all green.
#
# This is safe here even though %post needs network access (nine git-sourced
# dependencies, PyPI, download.pytorch.org): Kempner/FASRC compute nodes have
# working outbound HTTPS — verified against huggingface.co (200) and
# api.colabfold.com from holygpu8a11104. No proxy variables are needed.
#
# SINGULARITY_TMPDIR must be node-local: fakeroot extracts a full container
# rootfs and Lustre handles that badly.
#
# PARTITION/ACCOUNT: the bsabatini_lab association has MaxJobs=0 / MaxSubmit=0,
# so every job must go through kempner_bsabatini_lab. The kempner_h100 /
# kempner_h200 partitions reject jobs that do not request a GPU, and a build
# needs none — kempner_interactive is the one that takes CPU-only work
# (MaxTime 8h, 64 cores/node).
#
#SBATCH --job-name=mosaic-build
#SBATCH --partition=kempner_interactive
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/build-%j.out
#SBATCH --error=logs/build-%j.out

set -euo pipefail

MOSAIC_REPO="${MOSAIC_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
IMAGE="${IMAGE:-/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup/images/mosaic.sif}"

export SINGULARITY_TMPDIR="/tmp/$USER/sing_tmp"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup/sing_cache}"
mkdir -p "$SINGULARITY_TMPDIR"

echo "node:      $(hostname)"
echo "repo:      $MOSAIC_REPO"
echo "image:     $IMAGE"
echo "tmpdir:    $SINGULARITY_TMPDIR ($(df -h /tmp | tail -1 | awk '{print $4}') free)"
echo

cd "$MOSAIC_REPO"

# Build to a temporary name and move into place only on success, so a failed
# build never destroys a working image (--force deletes the destination).
STAGED="${IMAGE}.building.$$"
singularity build --fakeroot "$STAGED" singularity/mosaic.def
mv -f "$STAGED" "$IMAGE"

echo
echo "build complete: $IMAGE"
ls -lh "$IMAGE"
