#!/bin/bash
#
# Hold a GPU node and run a marimo notebook server on it, for the job's walltime.
#
#     mkdir -p logs
#     sbatch singularity/marimo.sbatch                       # example_notebook.py
#     NOTEBOOK=examples/of3.py sbatch singularity/marimo.sbatch
#     # then read logs/marimo-<jobid>.out for the ssh command and the URL
#
# This is the same thing as running singularity/marimo.sh inside an salloc shell,
# with two differences worth knowing: the server survives your laptop sleeping or
# the tunnel dropping (reconnect with the same ssh command), and it dies at the
# walltime below rather than when you close the terminal.
#
# Unlike webapp.sbatch this needs a real GPU — a notebook that calls Boltz/AF2
# runs the model in this process. kempner_h100 rather than kempner_interactive:
# the interactive partition hands out 20 GB A100 MIG slices, which is too small
# for ESM-C 6B (24 GB of weights alone) and tight for large complexes.
#
#SBATCH --job-name=mosaic-marimo
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/marimo-%j.out
#SBATCH --error=logs/marimo-%j.out

set -euo pipefail

REPO="${MOSAIC_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

exec "$REPO/singularity/marimo.sh" "${NOTEBOOK:-examples/example_notebook.py}"
