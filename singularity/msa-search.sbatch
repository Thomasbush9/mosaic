#!/bin/bash
#
# Generate a ColabFold MSA for one target sequence, locally.
#
#     mkdir -p logs
#     TARGET_FASTA=/path/to/target.fasta sbatch singularity/msa-search.sbatch
#     # or, from a bare sequence:
#     TARGET_SEQ=MQIFVKT... TARGET_NAME=ubq sbatch singularity/msa-search.sbatch
#
# Output: $MSA_OUT/<name>.a3m, to be passed as TargetChain(msa_path=...).
#
# PROVENANCE: the singularity invocation is copied from ProtForge
# (containers/test/msa_test_image.sh and workflow/rules/msa.smk::run_colabfold_search)
# and adapted for the single-target case. ProtForge itself is not modified — it
# is a separate working library. The container and database below are shared
# read-only assets, consumed in place:
#
#   msa.sif  — ColabFold + GPU mmseqs2
#   mmseq2_db — colabfold_envdb_202108, already built with GPU_READY markers
#
# The same-path binds (host:host) are deliberate and come from ProtForge:
# colabfold_search records paths internally, so the fasta, DB and output must
# resolve identically inside and outside the container.
#
# Only the TARGET needs an MSA. Every binder_features() implementation hardcodes
# use_msa=False for the binder — a de novo binder has no homologs — so this is
# run once per target, not per design trajectory.
#
#SBATCH --job-name=mosaic-msa
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/msa-%j.out
#SBATCH --error=logs/msa-%j.out

set -euo pipefail

MSA_IMAGE="${MSA_IMAGE:-/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/singularity_dev/images/msa.sif}"
MSA_DB="${MSA_DB:-/n/holylfs06/LABS/kempner_shared/Everyone/workflow/boltz/mmseq2_db}"
MSA_OUT="${MSA_OUT:-/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup/msa}"
THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "$MSA_OUT"

# Accept either a FASTA path or a raw sequence.
if [[ -n "${TARGET_FASTA:-}" ]]; then
    FASTA="$TARGET_FASTA"
    NAME="${TARGET_NAME:-$(basename "${FASTA%.*}")}"
else
    : "${TARGET_SEQ:?set TARGET_FASTA=<file> or TARGET_SEQ=<sequence>}"
    NAME="${TARGET_NAME:?set TARGET_NAME when using TARGET_SEQ}"
    FASTA="$MSA_OUT/${NAME}.fasta"
    printf '>%s\n%s\n' "$NAME" "$TARGET_SEQ" > "$FASTA"
fi

WORK="$MSA_OUT/${NAME}_search"
rm -rf "$WORK"; mkdir -p "$WORK"

echo "── colabfold_search ──"
echo "node:   $(hostname)"
echo "fasta:  $FASTA"
echo "db:     $MSA_DB"
echo "out:    $WORK"
nvidia-smi --query-gpu=name --format=csv,noheader

singularity exec --nv \
    -B "$FASTA:$FASTA" \
    -B "$MSA_DB:$MSA_DB" \
    -B "$WORK:$WORK" \
    "$MSA_IMAGE" \
    colabfold_search "$FASTA" "$MSA_DB" "$WORK" --gpu 1 --thread "$THREADS"

# colabfold_search names outputs by its own internal index, not by the FASTA
# header, so pick up whatever .a3m it produced rather than guessing the name.
A3M="$(find "$WORK" -maxdepth 1 -name '*.a3m' -print -quit)"
[[ -n "$A3M" ]] || { echo "error: colabfold_search produced no .a3m" >&2; exit 1; }

FINAL="$MSA_OUT/${NAME}.a3m"
mv -f "$A3M" "$FINAL"
rm -rf "$WORK"

# An a3m with a single unique sequence means no homologs were found. That is a
# legitimate result for a designed or orphan sequence, but silently proceeding
# with it looks like an MSA when it is not — so say so. (Check adapted from
# ProtForge's organize_msa_outputs.py::is_empty_msa.)
python3 - "$FINAL" <<'PY'
import sys
from pathlib import Path
lines = [l.strip() for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
seqs = {l for l in lines if not l.startswith(">")}
n = len(seqs)
print(f"unique sequences in MSA: {n}")
if n <= 1:
    print("WARNING: MSA contains only the query — no homologs found. "
          "Prediction will be effectively single-sequence.")
PY

echo
ls -lh "$FINAL"
echo "MSA OK: $FINAL"
echo
echo "Use it with:  TargetChain(sequence=..., msa_path='$FINAL')"
