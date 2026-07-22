#!/bin/bash
#
# Verify that a staged weight cache is actually bound into the container and
# loadable — not merely present on the host.
#
#     mkdir -p logs && sbatch singularity/weights-test.sbatch
#     MODEL=af2 sbatch singularity/weights-test.sbatch    # pick one explicitly
#
# The failure this exists to catch: mosaic-exec.sh creates each bind
# destination before mounting, so a skipped or empty bind leaves a real but
# empty directory and the model quietly re-downloads tens of GB into scratch
# instead of erroring. selftest.py flags empty mount points; this goes further
# and actually deserialises the weights.
#
#SBATCH --job-name=mosaic-weights
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/weights-test-%j.out
#SBATCH --error=logs/weights-test-%j.out

set -euo pipefail

export MOSAIC_SIF="${MOSAIC_SIF:-/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup/images/mosaic.sif}"
export MOSAIC_WEIGHTS="${MOSAIC_WEIGHTS:-/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup/weights}"
export MOSAIC_SCRATCH="${MOSAIC_SCRATCH:-/n/netscratch/bsabatini_lab/Everyone/$USER/mosaic}"

# Any download attempt means a bind failed. Fail loudly instead of quietly
# pulling 5 GB into scratch.
export SINGULARITYENV_HF_HUB_OFFLINE=1

MODEL="${MODEL:-af2}"

MOSAIC_REPO="${MOSAIC_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
HERE="$MOSAIC_REPO/singularity"
if [[ ! -x "$HERE/mosaic-exec.sh" ]]; then
    echo "error: $HERE/mosaic-exec.sh not found — set MOSAIC_REPO to the repo root" >&2
    exit 66
fi

echo "── environment ──"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "MODEL=$MODEL"

echo
echo "── selftest (runtime) ──"
"$HERE/mosaic-exec.sh" python /opt/mosaic/singularity/selftest.py

echo
echo "── load $MODEL ──"
MODEL="$MODEL" "$HERE/mosaic-exec.sh" python -c '
import os, time, pathlib
model = os.environ["MODEL"]

if model == "af2":
    p = pathlib.Path("~/.alphafold/params").expanduser()
    files = sorted(f.name for f in p.glob("*.npz"))
    print(f"{p} -> {len(files)} npz files")
    assert files, "no params found — bind failed"

    # load_af2 re-downloads if ~/.alphafold/params is absent. It is present, so
    # reaching the load means the bind worked; a download here would be the bug.
    from mosaic.models.af2 import load_af2
    t0 = time.time()
    af2 = load_af2(multimer=True)
    print(f"loaded AF2 multimer in {time.time()-t0:.1f}s -> {type(af2).__name__}")

elif model == "boltz2":
    from mosaic.losses.boltz2 import load_boltz2
    t0 = time.time()
    m = load_boltz2()
    print(f"loaded Boltz-2 in {time.time()-t0:.1f}s -> {type(m).__name__}")

else:
    raise SystemExit(f"unknown MODEL={model}")

print(f"{model.upper()} WEIGHTS OK")
'
echo
echo "ALL WEIGHT CHECKS PASSED"
