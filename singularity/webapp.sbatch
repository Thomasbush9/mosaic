#!/bin/bash
#
# Run the design webapp on a COMPUTE node.
#
#     sbatch singularity/webapp.sbatch
#     # then read logs/webapp-<jobid>.out for the exact ssh command
#
# Why not a login node: the app is light, but a long-lived process accumulating
# CPU on a shared login node is what FASRC's arbiter kills — the same policy
# that kills a login-node `singularity build` at the mksquashfs step. A login
# node run works and then silently disappears, which presents as a connection
# error with nothing in the log.
#
# CPU-only, so kempner_interactive: kempner_h100/h200 reject jobs with no GPU.
# The app still never runs a model — it queues SLURM jobs like before.
#
#SBATCH --job-name=mosaic-webapp
#SBATCH --partition=kempner_interactive
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/webapp-%j.out
#SBATCH --error=logs/webapp-%j.out

set -euo pipefail

PORT="${MOSAIC_PORT:-8502}"
REPO="${MOSAIC_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"

if [[ ! -x webapp/.venv/bin/streamlit ]]; then
    echo "error: webapp/.venv missing — create it once on a login node:" >&2
    echo "  module load python/3.13.12-fasrc01" >&2
    echo "  uv venv webapp/.venv --python 3.13" >&2
    echo "  uv pip install --python webapp/.venv -r webapp/requirements.txt" >&2
    exit 66
fi

NODE=$(hostname -f)
cat <<MSG

================================================================
  mosaic webapp — running on a compute node

  From your laptop, forward through the login node in one hop:

      ssh -J ${USER}@holylogin06.rc.fas.harvard.edu \\
          -L ${PORT}:localhost:${PORT} ${USER}@${NODE}

  Then open:  http://localhost:${PORT}

  This job holds the app for its walltime. scancel ${SLURM_JOB_ID} to stop it.
================================================================

MSG

exec webapp/.venv/bin/streamlit run webapp/app.py \
    --server.port "${PORT}" --server.address 127.0.0.1 \
    --server.headless true --browser.gatherUsageStats false
