#!/bin/bash
#
# Functional smoke test: run a backend forward AND differentiate through it.
#
#     mkdir -p logs
#     MODEL=af2    sbatch --export=ALL,MODEL=af2    singularity/model-smoke.sbatch
#     MODEL=boltz2 sbatch --export=ALL,MODEL=boltz2 singularity/model-smoke.sbatch
#
# weights-test.sbatch only proves a checkpoint deserialises. That is not the
# same as the model working: a design run needs a forward pass over a soft
# sequence and a gradient flowing back to it. This tests both, which is what
# actually breaks when a backend is subtly mismatched to its weights.
#
# use_msa=False throughout, deliberately — it keeps the test self-contained
# (no api.colabfold.com round trip) and AF2 rejects MSAs outright anyway
# (models/af2.py:391).
#
# recycling_steps=0 and a distogram-only loss keep this fast: under JIT, JAX
# prunes the structure and confidence modules when no loss term reads them.
#
#SBATCH --job-name=mosaic-smoke
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.out

set -euo pipefail

export MOSAIC_SIF="${MOSAIC_SIF:-/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup/images/mosaic.sif}"
export MOSAIC_WEIGHTS="${MOSAIC_WEIGHTS:-/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup/weights}"
export MOSAIC_SCRATCH="${MOSAIC_SCRATCH:-/n/netscratch/bsabatini_lab/Everyone/$USER/mosaic}"
export SINGULARITYENV_HF_HUB_OFFLINE=1

MODEL="${MODEL:-af2}"

MOSAIC_REPO="${MOSAIC_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
HERE="$MOSAIC_REPO/singularity"
if [[ ! -x "$HERE/mosaic-exec.sh" ]]; then
    echo "error: $HERE/mosaic-exec.sh not found — set MOSAIC_REPO to the repo root" >&2
    exit 66
fi

echo "── $MODEL forward + gradient ──"
nvidia-smi --query-gpu=name --format=csv,noheader

MODEL="$MODEL" SINGULARITYENV_MSA_PATH="${MSA_PATH:-}" SINGULARITYENV_HAIKU_SHIM="${HAIKU_SHIM:-0}" "$HERE/mosaic-exec.sh" python -c '
import os, time
import numpy as np, jax, jax.numpy as jnp

model_name = os.environ["MODEL"]
print("jax", jax.__version__, jax.default_backend(), jax.devices())

# EXPERIMENTAL (HAIKU_SHIM=1): dm-haiku 0.0.16 is the newest release and still
# reads 15 names that JAX removed in 0.11 — it declares jax>=0.8.1 with no upper
# bound, so uv resolves a combination that cannot work. 13 of the 15 moved to
# jax.extend.core unchanged; this re-exposes those on jax.core before haiku
# imports. cur_sublevel and thread_local_state have no replacement and are left
# alone — if AF2 touches them this shim will not save it, which is exactly what
# this run is measuring.
if os.environ.get("HAIKU_SHIM") == "1":
    import jax.core
    from jax.extend import core as _jc
    restored = []
    for _n in ("ClosedJaxpr", "DropVar", "Jaxpr", "JaxprEqn", "Literal",
               "Primitive", "TraceTag", "Var", "get_opaque_trace_state",
               "jaxpr_as_fun", "primitives", "set_current_trace",
               "take_current_trace"):
        if hasattr(_jc, _n):
            setattr(jax.core, _n, getattr(_jc, _n))
            restored.append(_n)
    print(f"haiku shim: restored {len(restored)} names onto jax.core")

from mosaic.common import TOKENS
from mosaic.structure_prediction import TargetChain
import mosaic.losses.structure_prediction as sp

# Short real target (ubiquitin N-terminal fragment) keeps the complex small.
TARGET = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQ"
BINDER_LEN = 16

# AbLang / AbLang2 are sequence models: LossTerms taking a soft sequence
# directly, with no structure features. They still get the full forward+gradient
# treatment, just without binder_features. Loading them at all is the real test
# — both cache inside their own read-only site-packages directory, so this only
# works because mosaic.def pre-creates the mount points and mosaic-exec.sh binds
# populated weights over them.
if model_name in ("ablang", "ablang2"):
    # A real VH sequence; these are antibody-specific models and would be
    # meaningless on a generic protein.
    VH = ("EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKG"
          "RFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDLGYYFDYWGQGTLVTVSS")
    idx = np.array([TOKENS.index(c) for c in VH])
    soft = jnp.asarray(np.eye(len(TOKENS), dtype=np.float32)[idx])
    key = jax.random.key(0)

    t0 = time.time()
    if model_name == "ablang":
        from mosaic.losses.ablang import AbLangPseudoLikelihood, load_ablang
        m, tok = load_ablang("heavy")
        term = AbLangPseudoLikelihood(m, tok)
    else:
        from mosaic.losses.ablang2 import Ablang2PseudoLikelihood, load_ablang2
        m, tok = load_ablang2()
        # heavy_len == len(seq) means heavy-only, per the class docstring.
        term = Ablang2PseudoLikelihood(m, tok, len(VH), None)
    print(f"loaded in {time.time()-t0:.1f}s")

    from mosaic.optimizers import _eval_loss_and_grad
    t0 = time.time()
    (value, aux), grad = _eval_loss_and_grad(term, soft, key)
    value = float(value); grad = np.asarray(grad)
    print(f"loss+grad (incl. JIT trace): {time.time()-t0:.1f}s")
    print("loss:", value)
    print("grad shape:", grad.shape, "finite:", bool(np.isfinite(grad).all()),
          "absmax:", float(np.abs(grad).max()))
    assert np.isfinite(value), "non-finite loss"
    assert np.isfinite(grad).all(), "gradient contains NaN/Inf"
    assert np.abs(grad).max() > 0, "gradient is identically zero"
    print(f"{model_name.upper()} FORWARD+GRAD OK")
    raise SystemExit(0)

# Proteina and BoltzGen are not StructurePredictionModel implementations — they
# are generative and have no binder_features/build_loss — so they get a load
# check rather than the forward+gradient path below.
if model_name in ("proteina", "boltzgen"):
    t0 = time.time()
    if model_name == "proteina":
        from jproteina_complexa.hub import load_denoiser, load_decoder
        den, dec = load_denoiser(), load_decoder()
        print(f"loaded in {time.time()-t0:.1f}s -> {type(den).__name__}, {type(dec).__name__}")
    else:
        from mosaic.models.boltzgen import load_boltzgen
        bg = load_boltzgen()
        print(f"loaded in {time.time()-t0:.1f}s -> {type(bg).__name__}")
    print(f"{model_name.upper()} LOAD OK")
    raise SystemExit(0)

t0 = time.time()
if model_name == "af2":
    from mosaic.models.af2 import AlphaFold2
    model = AlphaFold2(multimer=True)
elif model_name == "boltz1":
    from mosaic.models.boltz1 import Boltz1
    model = Boltz1()
elif model_name == "boltz2":
    from mosaic.models.boltz2 import Boltz2
    model = Boltz2()
elif model_name == "protenix":
    # Mini is the smallest checkpoint and the cheapest thing that still
    # exercises the whole protenij load + featurize + forward path.
    from mosaic.models.protenix import ProtenixMini
    model = ProtenixMini()
elif model_name == "of3":
    from mosaic.models.of3 import OF3
    model = OF3()
else:
    raise SystemExit(f"unknown MODEL={model_name}")
print(f"model loaded in {time.time()-t0:.1f}s")

# MSA_PATH=<a3m> switches the target chain from single-sequence to a locally
# computed MSA (singularity/msa-search.sbatch). Without it the test stays
# self-contained and makes no network call. Note only the TARGET gets an MSA —
# binder_features always builds the binder with use_msa=False.
msa_path = os.environ.get("MSA_PATH") or None
target_chain = (
    TargetChain(sequence=TARGET, use_msa=True, msa_path=msa_path)
    if msa_path
    else TargetChain(sequence=TARGET, use_msa=False)
)
print("target MSA:", msa_path or "(none — single sequence)")

features, _ = model.binder_features(BINDER_LEN, [target_chain])
print("features built")

# recycling_steps must be >= 1. The trunk runs inside
# jax.lax.scan(..., length=recycling_steps) (models/af2.py:480), so 0 means the
# body never executes: AF2 then indexes a zero-length output axis
# (IndexError: index is out of bounds for axis 0 with size 0) and Boltz-2
# returns a loss with no data dependence on the sequence, giving a silently
# all-zero gradient. 1 is the default for a reason.
loss_term = model.build_loss(
    loss=1.0 * sp.BinderTargetContact() + 1.0 * sp.WithinBinderContact(),
    features=features,
    recycling_steps=1,
)

# Start from the uniform point of the simplex — the natural init for the
# continuous relaxation, and it exercises soft (non-one-hot) inputs.
soft = jnp.full((BINDER_LEN, len(TOKENS)), 1.0 / len(TOKENS))
key = jax.random.key(0)

# Use mosaic own evaluator rather than calling the term eagerly. It is what
# every optimizer calls, and the difference is load-bearing: it passes the loss
# as an ARGUMENT to eqx.filter_jit (optimizers.py:60-63), so numpy leaves inside
# the features become JAX arrays. Called eagerly, Protenix dies on
# features["restype"].at[...] because featurize() returns numpy and .at is
# JAX-only. It also applies the simplex tangent-space projection
# (g - g.mean(-1)) that simplex_APGM depends on, so the gradient checked here is
# the gradient a design run actually steps with.
from mosaic.optimizers import _eval_loss_and_grad

t0 = time.time()
(value, aux), grad = _eval_loss_and_grad(loss_term, soft, key)
value = float(value)
grad = np.asarray(grad)
print(f"loss+grad (incl. JIT trace): {time.time()-t0:.1f}s")
print("loss:", value)
# aux is not a uniform shape across backends: a dict for AF2/Boltz, a list for
# Protenix and OF3 (which return one entry per sample). Nor are the values all
# scalars — some terms report per-residue arrays. Map over it as a pytree rather
# than assuming .items().
def _fmt(v):
    a = np.asarray(v)
    return float(a) if a.ndim == 0 else f"array{a.shape}"
print("aux:", jax.tree_util.tree_map(_fmt, aux))
assert np.isfinite(value), "forward produced a non-finite loss"

print("grad shape:", grad.shape, "finite:", bool(np.isfinite(grad).all()),
      "absmax:", float(np.abs(grad).max()))
assert np.isfinite(grad).all(), "gradient contains NaN/Inf"
assert np.abs(grad).max() > 0, "gradient is identically zero — nothing reaches the sequence"
# _eval_loss_and_grad projects onto the simplex tangent space, so each row must
# sum to ~0. If this drifts, mirror descent in simplex_APGM will walk off the
# simplex.
row_sums = np.abs(grad.sum(axis=-1)).max()
print("max |row sum| after tangent projection:", float(row_sums))
assert row_sums < 1e-4, f"gradient rows do not sum to zero ({row_sums})"

print(f"{model_name.upper()} FORWARD+GRAD OK")
'
echo
echo "SMOKE TEST PASSED"
