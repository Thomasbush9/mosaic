#!/bin/bash
#
# Generic pipeline node. The DAG engine (webapp/pipeline.py) submits one of these
# per node with NODE_DIR set and a SLURM dependency on its upstream nodes.
# pipeline_node.py reads NODE_DIR/node.json and dispatches to the right stage.
#
# Resources (partition, gpu, time) are set by the submitter per node type — this
# script only carries the sane defaults for a direct/manual run.
#
#SBATCH --job-name=pipeline-node
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
REPO="${MOSAIC_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
SETUP=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup
export MOSAIC_SIF="${MOSAIC_SIF:-$SETUP/images/mosaic.sif}"
export MOSAIC_WEIGHTS="${MOSAIC_WEIGHTS:-$SETUP/weights}"
export MOSAIC_SCRATCH="${MOSAIC_SCRATCH:-/n/netscratch/bsabatini_lab/Everyone/$USER/mosaic}"
# pipeline_node.py runs on the host (it only orchestrates + does light file work),
# and calls into the container via mosaic-exec.sh for the GPU stages.
python3 "$REPO/pipeline_node.py"
