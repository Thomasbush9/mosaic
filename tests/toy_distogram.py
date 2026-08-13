"""Synthetic `StructureModelOutput`s with a planted, known answer.

The distogram losses read nothing but `distogram_logits` and `distogram_bins`,
so a hand-built output pins down their arithmetic exactly — no model, no GPU,
no weights. Everything else in `StructureModelOutput` is filled with correctly
shaped zeros so the module still constructs.

Used by `test_distogram_iptm_proxy.py` and `bench_distogram_iptm_proxy.py`.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from mosaic.losses.structure_prediction import StructureModelOutput, PAE_BINS

# Boltz-ish geometry: 64 bins spanning 2-22 A. `contact_distance=8.0` (the
# DistogramIPTMProxy default) then puts ~19 bins inside the contact region.
N_BINS = 64
BINS = np.linspace(2.0, 22.0, N_BINS)
NEAR_BIN = 5  # ~3.6 A  -- inside any sane cutoff
FAR_BIN = 55  # ~19.4 A -- outside it


def planted_logits(
    binder_len: int,
    target_len: int,
    contact_cols,
    sharpness: float = 8.0,
    bg: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Full [L, L, B] distogram logits with contacts planted at `contact_cols`.

    `contact_cols` are 0-based indices *into the target block*, i.e. the same
    convention `DistogramIPTMProxy.epitope_idx` uses. Binder->target pairs whose
    target column is in that set get their mass on `NEAR_BIN`; every other pair
    gets it on `FAR_BIN`. `sharpness` is the peak logit: large means the model is
    confident, ~0 means it is uniform over bins.
    """
    rng = rng or np.random.default_rng(0)
    L = binder_len + target_len
    logits = bg * rng.standard_normal((L, L, N_BINS))

    peak = np.full(target_len, FAR_BIN)
    peak[np.asarray(list(contact_cols), dtype=int)] = NEAR_BIN
    # binder rows x target cols, and the symmetric block, so the array looks
    # like something a real model could have emitted.
    for j, b in enumerate(peak):
        logits[:binder_len, binder_len + j, b] += sharpness
        logits[binder_len + j, :binder_len, b] += sharpness
    return logits


def graded_logits(
    binder_len: int,
    target_len: int,
    contact_cols,
    row_sharpness,
    far_sharpness: float = 8.0,
) -> np.ndarray:
    """Like `planted_logits`, but each binder residue contacts with its own
    confidence, given by `row_sharpness[i]`.

    Needed for anything about the bottom-k selection: with a uniform interface
    every contacting pair has the same score, so which pairs the top_k picks --
    and therefore how the pool size matters -- is invisible.
    """
    row_sharpness = np.asarray(row_sharpness, dtype=float)
    assert row_sharpness.shape == (binder_len,)
    L = binder_len + target_len
    logits = np.zeros((L, L, N_BINS))
    cols = set(int(c) for c in contact_cols)
    for j in range(target_len):
        col = binder_len + j
        if j in cols:
            logits[:binder_len, col, NEAR_BIN] += row_sharpness
            logits[col, :binder_len, NEAR_BIN] += row_sharpness
        else:
            logits[:binder_len, col, FAR_BIN] += far_sharpness
            logits[col, :binder_len, FAR_BIN] += far_sharpness
    return logits


def toy_output(logits: np.ndarray, binder_len: int) -> StructureModelOutput:
    """Wrap raw distogram logits in an otherwise-inert StructureModelOutput."""
    L = logits.shape[0]
    return StructureModelOutput(
        distogram_logits=jnp.asarray(logits, dtype=jnp.float32),
        distogram_bins=jnp.asarray(BINS, dtype=jnp.float32),
        plddt=jnp.zeros(L),
        pae=jnp.zeros((L, L)),
        pae_logits=jnp.zeros((L, L, len(PAE_BINS))),
        pae_bins=jnp.asarray(PAE_BINS, dtype=jnp.float32),
        structure_coordinates=jnp.zeros((L, 37, 3)),
        backbone_coordinates=jnp.zeros((L, 4, 3)),
        full_sequence=jnp.zeros((L, 20)),
        asym_id=jnp.concatenate([jnp.zeros(binder_len), jnp.ones(L - binder_len)]),
        residue_idx=jnp.arange(L),
        atom37_coords=jnp.zeros((L, 37, 3)),
        atom37_mask=jnp.zeros((L, 37)),
    )


def restrict_to_columns(logits: np.ndarray, binder_len: int, cols) -> np.ndarray:
    """A complex in which the target chain *is* `cols` and nothing else.

    Lets us state the epitope contract as an identity: masking columns inside
    the loss must equal deleting them from the input.
    """
    cols = np.asarray(list(cols), dtype=int)
    keep = np.concatenate([np.arange(binder_len), binder_len + cols])
    return logits[np.ix_(keep, keep)]


def reference_proxy(
    logits: np.ndarray,
    binder_len: int,
    contact_distance: float = 8.0,
    epitope_idx=None,
    bins: np.ndarray = BINS,
) -> float:
    """Independent NumPy transcription of Algorithm 15 (ESM2 supp. A.3.3).

    Written from the docstring rather than from the JAX code, so that agreement
    is evidence about the algorithm and not just about the two implementations
    sharing a bug.
    """
    D = np.asarray(logits, dtype=np.float64)[:binder_len, binder_len:]  # [N, Lt, B]
    if epitope_idx is not None:
        D = D[:, np.asarray(list(epitope_idx), dtype=int)]

    m = bins < contact_distance
    n_contact_bins = int(m.sum())

    log_p_full = D - _logsumexp(D, axis=-1, keepdims=True)
    masked = np.where(m, D, -np.inf)
    log_p_cut = masked - _logsumexp(masked, axis=-1, keepdims=True)
    p_cut = np.where(m, np.exp(log_p_cut), 0.0)

    S = -(p_cut * log_p_full).sum(-1)
    flat = np.sort(S.reshape(-1))
    S_bar = flat[:binder_len].mean()
    return float(np.clip(1.0 - S_bar / np.log(n_contact_bins), 0.0, 1.0))


def _logsumexp(x, axis, keepdims=False):
    mx = np.max(x, axis=axis, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    out = mx + np.log(np.exp(x - mx).sum(axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)
