#!/usr/bin/env bash
# Open the mosaic design webapp from your laptop.
#
#   bash webapp/connect.sh
#
# Forwards a local port to the login node and starts the app there if it is not
# already running. The app only queues SLURM jobs, so it is safe on a login node.
set -euo pipefail

PORT="${MOSAIC_PORT:-8502}"
REPO_DEFAULT=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup/mosaic

read -r -p "Cluster username: " USER_
read -r -p "Login node [holylogin06.rc.fas.harvard.edu]: " HOST_
HOST_="${HOST_:-holylogin06.rc.fas.harvard.edu}"
read -r -p "mosaic repo on cluster [$REPO_DEFAULT]: " REPO_
REPO_="${REPO_:-$REPO_DEFAULT}"

echo
echo "Forwarding localhost:${PORT} -> ${USER_}@${HOST_}"
echo "Once connected open: http://localhost:${PORT}"
echo "Ctrl+C disconnects; the app keeps running on the cluster."
echo

ssh -L "${PORT}:localhost:${PORT}" -t "${USER_}@${HOST_}" "
  set -e
  cd '${REPO_}'
  if ! curl -s -o /dev/null localhost:${PORT}; then
    if [ ! -x webapp/.venv/bin/streamlit ]; then
      echo 'Creating webapp venv (first run only)...'
      source /etc/profile.d/lmod.sh
      module load python/3.13.12-fasrc01
      \$HOME/.local/bin/uv venv webapp/.venv --python 3.13
      \$HOME/.local/bin/uv pip install --python webapp/.venv -r webapp/requirements.txt
    fi
    echo 'Starting app...'
    webapp/.venv/bin/streamlit run webapp/app.py \
      --server.port ${PORT} --server.address 127.0.0.1 \
      --server.headless true
  else
    echo 'App already running.'
    exec bash
  fi
"
