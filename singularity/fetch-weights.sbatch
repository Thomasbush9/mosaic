#!/bin/bash
#
# Download a backend's weights into the shared weights tree.
#
#     mkdir -p logs
#     sbatch --export=ALL,MODEL=protenix singularity/fetch-weights.sbatch
#
# MODEL is one of: protenix, boltz1, boltzgen, proteina, of3
#
# This does NOT use mosaic-exec.sh, which binds every weight cache read-only —
# correct for running, wrong for fetching. It runs singularity directly with the
# one target directory bound read-write, so a download can only ever write where
# it is supposed to.
#
# Each backend fetches to a hardcoded location under $HOME (~/.protenix,
# ~/.boltz, ~/.openfold3, ~/.cache/jproteina_complexa). Rather than fight that,
# HOME is a staging directory and the real destinations are bound onto those
# paths — the same trick mosaic-exec.sh uses at run time.
#
# CPU-only: these are downloads and, for of3, a torch->equinox conversion. No
# GPU is requested, so this runs on kempner_interactive (kempner_h100/h200
# reject jobs without a GPU, and bsabatini_lab cannot submit at all).
#
#SBATCH --job-name=mosaic-fetch
#SBATCH --partition=kempner_interactive
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fetch-%j.out
#SBATCH --error=logs/fetch-%j.out

set -euo pipefail

MODEL="${MODEL:?set MODEL=protenix|boltz1|boltzgen|proteina|of3}"

SIF="${MOSAIC_SIF:-/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup/images/mosaic.sif}"
W="${MOSAIC_WEIGHTS:-/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup/weights}"
SCRATCH="${MOSAIC_SCRATCH:-/n/netscratch/bsabatini_lab/Everyone/$USER/mosaic}"
SHARED_HF=/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/singularity_dev/images/cache_models

CHOME="$SCRATCH/fetch_home"
mkdir -p "$CHOME/.cache"
mkdir -p "$W/boltz" "$W/protenix" "$W/openfold3" "$W/jproteina" "$W/ablang"

# Destinations exist before mounting: Singularity will not create them.
mkdir -p "$CHOME/.protenix" "$CHOME/.boltz" "$CHOME/.openfold3" \
         "$CHOME/.cache/jproteina_complexa" "$CHOME/.cache/huggingface"

binds=(
  -B "$W/protenix:$CHOME/.protenix"
  -B "$W/boltz:$CHOME/.boltz"
  -B "$W/openfold3:$CHOME/.openfold3"
  -B "$W/jproteina:$CHOME/.cache/jproteina_complexa"
  -B "$SHARED_HF:$CHOME/.cache/huggingface"
)

echo "── fetching $MODEL ──"
echo "node:  $(hostname)"
echo "dest:  $W"
df -h "$W" | tail -1
echo

# of3 reuses the .pt checkpoints already in the shared cache via OPENFOLD_CACHE,
# whose ckpt_root indirection file already points there — so only the
# torch->equinox conversion runs, not a multi-GB S3 download.
export SINGULARITYENV_OPENFOLD_CACHE="$CHOME/.cache/huggingface"
export SINGULARITYENV_MODEL="$MODEL"

# See mosaic-exec.sh: the host exports RHEL CA paths that do not exist in this
# Ubuntu container, so every `requests` HTTPS call fails until these are fixed.
export SINGULARITYENV_SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export SINGULARITYENV_SSL_CERT_DIR=/etc/ssl/certs
export SINGULARITYENV_CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

singularity exec -H "$CHOME" "${binds[@]}" "$SIF" python -c '
import os, time, pathlib
model = os.environ["MODEL"]
t0 = time.time()

if model == "protenix":
    # losses/protenix.py sets PROTENIX_DATA_ROOT_DIR to ~/.protenix at import,
    # unconditionally, so importing it is how we point the fetch at the bind.
    import mosaic.losses.protenix  # noqa: F401
    from protenix.backend import load_model
    name = os.environ.get("PROTENIX_MODEL", "protenix_mini_default_v0.5.0")
    print("fetching", name)
    load_model(name)

elif model == "boltz1":
    from boltz import main as boltz_main
    cache = pathlib.Path("~/.boltz").expanduser()
    print("download_boltz1 ->", cache)
    boltz_main.download_boltz1(cache)

elif model == "boltzgen":
    # There is no standalone downloader — load_boltzgen() fetches on demand
    # (models/boltzgen.py:43-68) and then re-saves each checkpoint stripped of
    # its validators, so this both downloads and exercises the load path.
    # It shells out to wget and does NOT check the return code, so a failed
    # fetch leaves a truncated file that only surfaces much later as an opaque
    # unpickling error — hence the size check at the end of this script.
    from mosaic.models.boltzgen import load_boltzgen
    load_boltzgen()

elif model == "proteina":
    from jproteina_complexa.hub import load_denoiser, load_decoder
    print("fetching denoiser + decoder")
    load_denoiser(); load_decoder()

elif model == "of3":
    from jopenfold3.model import OpenFold3
    print("OpenFold3.load() — converts torch ckpt to equinox on first call")
    OpenFold3.load()

else:
    raise SystemExit(f"unknown MODEL={model}")

print(f"done in {time.time()-t0:.1f}s")
' || { echo "FETCH FAILED for $MODEL" >&2; exit 1; }

echo
echo "── resulting tree ──"
case "$MODEL" in
  protenix)  ls -lh "$W/protenix" | head -20 ;;
  boltz1|boltzgen) ls -lhL "$W/boltz" | head -20 ;;
  proteina)  ls -lh "$W/jproteina" | head -20 ;;
  of3)       ls -lhR "$W/openfold3" | head -20 ;;
esac

# Truncated-checkpoint guard: the failure mode selftest.py warns about.
echo
echo "── size check (anything under 100 MiB is suspect) ──"
find -L "$W" -name '*.ckpt' -o -name '*.eqx' -o -name '*.pt' 2>/dev/null | while read -r f; do
    sz=$(stat -Lc %s "$f")
    printf '%10.1f MiB  %s%s\n' "$(echo "$sz/1048576" | bc -l)" "$f" \
        "$([ "$sz" -lt 104857600 ] && echo '   <-- SUSPECT' || true)"
done

echo
echo "FETCH OK: $MODEL"
