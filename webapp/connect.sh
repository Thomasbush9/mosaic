#!/usr/bin/env bash
# Open the mosaic design webapp from your laptop.
#
#   bash webapp/connect.sh
#
# Forwards a local port to the login node and starts the app there if it is not
# already running. The app only queues SLURM jobs, so it is light enough for a
# login node.
#
# For anything longer than a quick look, prefer the compute-node route:
#
#   ssh <user>@holylogin06.rc.fas.harvard.edu
#   cd <repo> && sbatch singularity/webapp.sbatch
#   # the job's log prints the exact one-hop ssh command
#
# A login node holds every user to a shared 8 GiB / 1-core cgroup covering all
# their processes; the batch job gets its own cores and cannot be squeezed by
# whatever else you have open.
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
echo
echo "You get a shell on the login node. Exiting it closes the tunnel; the app"
echo "itself is started detached and keeps running. To stop it:"
echo "  kill \$(cat ${REPO_}/logs/webapp.pid)"
echo

# The app is started with setsid+nohup and backgrounded, NOT in the foreground
# of this session. Run in the foreground it would take SIGINT from Ctrl-C and
# SIGHUP from a dropped connection — which is what made "the app keeps running
# on the cluster" false for as long as this script has existed, and presented as
# the app mysteriously vanishing whenever a laptop slept.
ssh -L "${PORT}:localhost:${PORT}" -t "${USER_}@${HOST_}" "
  set -e
  cd '${REPO_}'
  if curl -s -o /dev/null --max-time 2 localhost:${PORT}; then
    echo 'App already running.'
  else
    if [ ! -x webapp/.venv/bin/streamlit ]; then
      echo 'Creating webapp venv (first run only)...'
      source /etc/profile.d/lmod.sh
      module load python/3.13.12-fasrc01
      \$HOME/.local/bin/uv venv webapp/.venv --python 3.13
      \$HOME/.local/bin/uv pip install --python webapp/.venv -r webapp/requirements.txt
    fi
    mkdir -p logs
    echo 'Starting app (detached)...'
    # The pidfile is written from INSIDE the detached process, not from \$! out
    # here: setsid forks when it needs a new session leader, so \$! is the pid of
    # a wrapper that exits immediately and the recorded pid would be dead on
    # arrival. \$\$ inside the inner shell survives the exec, so it is the
    # server's own pid.
    setsid nohup bash -c '
      echo \$\$ > logs/webapp.pid
      exec webapp/.venv/bin/streamlit run webapp/app.py \
        --server.port ${PORT} --server.address 127.0.0.1 \
        --server.headless true --browser.gatherUsageStats false
    ' > logs/webapp.log 2>&1 < /dev/null &

    # Imports alone take about a minute here: the venv is ~8500 files on Lustre
    # and a login node gives you one core. Poll rather than guess at a sleep.
    printf 'Waiting for it to come up'
    for _ in \$(seq 1 60); do
      if curl -s -o /dev/null --max-time 2 localhost:${PORT}; then
        echo ' ready.'
        break
      fi
      printf '.'
      sleep 2
    done
    if ! curl -s -o /dev/null --max-time 2 localhost:${PORT}; then
      echo
      echo 'Still not answering after 2 minutes. Last lines of logs/webapp.log:'
      tail -20 logs/webapp.log
    fi
  fi
  exec bash
"
