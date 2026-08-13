"""Toy-data tests for `DistogramIPTMProxy`, with and without `epitope_idx`.

Tier 1 of two. No model, no weights, no GPU: the loss reads only the distogram,
so a synthetic distogram with contacts planted at known columns pins the
arithmetic down exactly and runs in seconds. Tier 2
(`test_distogram_iptm_proxy_real.py`) checks the same claims against AF2 on a
real complex, where they can fail for reasons a toy cannot show.

The tests are grouped by what they are protecting:

  1. the unchanged path      -- epitope_idx=None must still be Algorithm 15
  2. the contract            -- masking columns == deleting them
  3. the point of the change -- the epitope actually steers the objective
  4. plumbing                -- jit, gradients, dtypes
  5. edges and footguns      -- empty / out-of-range / duplicate indices, and
                                how the value drifts with epitope size

Run:  uv run pytest tests/test_distogram_iptm_proxy.py -v
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mosaic.losses.structure_prediction import DistogramIPTMProxy
from toy_distogram import (
    BINS,
    planted_logits,
    graded_logits,
    toy_output,
    restrict_to_columns,
    reference_proxy,
)

BINDER_LEN = 12
TARGET_LEN = 40
EPITOPE = [3, 4, 5, 6, 7, 8]  # "true" site
DECOY = [30, 31, 32, 33, 34, 35]  # far side of the target
KEY = jax.random.key(0)


def proxy(output, binder_len=BINDER_LEN, **kw) -> float:
    """The reported proxy (aux), which is -loss."""
    loss = DistogramIPTMProxy(**kw)
    seq = jnp.zeros((binder_len, 20))
    v, aux = loss(seq, output, KEY)
    assert float(v) == pytest.approx(-float(aux["distogram_iptm"]), abs=1e-6)
    return float(aux["distogram_iptm"])


@pytest.fixture(scope="module")
def planted():
    """Logits with real contacts at EPITOPE and nowhere else."""
    return planted_logits(BINDER_LEN, TARGET_LEN, EPITOPE, sharpness=8.0)


@pytest.fixture(scope="module")
def wrong_site():
    """A decoy binder: confidently in contact, but at the wrong place."""
    return planted_logits(BINDER_LEN, TARGET_LEN, DECOY, sharpness=8.0)


# ---------------------------------------------------------------- 1. unchanged


@pytest.mark.parametrize("sharpness", [0.0, 2.0, 8.0])
@pytest.mark.parametrize("cutoff", [6.0, 8.0, 12.0])
def test_no_epitope_matches_reference_implementation(sharpness, cutoff):
    """epitope_idx=None must reproduce Algorithm 15 bit-for-bit (to float32).

    This is the regression guard: whatever the epitope branch does, the old
    behaviour has to survive untouched.
    """
    logits = planted_logits(BINDER_LEN, TARGET_LEN, EPITOPE, sharpness=sharpness)
    got = proxy(toy_output(logits, BINDER_LEN), contact_distance=cutoff)
    want = reference_proxy(logits, BINDER_LEN, contact_distance=cutoff)
    assert got == pytest.approx(want, abs=1e-5)


def test_epitope_branch_matches_reference_implementation(planted):
    got = proxy(toy_output(planted, BINDER_LEN), epitope_idx=EPITOPE)
    want = reference_proxy(planted, BINDER_LEN, epitope_idx=EPITOPE)
    assert got == pytest.approx(want, abs=1e-5)


def test_full_epitope_is_identical_to_none(planted):
    """Naming every target residue must be a no-op."""
    out = toy_output(planted, BINDER_LEN)
    assert proxy(out, epitope_idx=list(range(TARGET_LEN))) == pytest.approx(
        proxy(out), abs=1e-6
    )


# ----------------------------------------------------------------- 2. contract


def test_masking_columns_equals_deleting_them(planted):
    """The semantics of `epitope_idx`, stated as an identity.

    Restricting to an epitope must be indistinguishable from having been handed
    a complex whose target chain is only those residues. If this ever fails, the
    indexing is off by something -- a 1-based list, an index into the full
    complex instead of the target block, a transposed block.
    """
    full = proxy(toy_output(planted, BINDER_LEN), epitope_idx=EPITOPE)
    sliced = restrict_to_columns(planted, BINDER_LEN, EPITOPE)
    assert full == pytest.approx(proxy(toy_output(sliced, BINDER_LEN)), abs=1e-6)


def test_indices_are_zero_based_into_the_target_block(planted):
    """Convention check, shared with `BinderTargetContact.epitope_idx` and with
    `proposals.epitope_from_complex` (which returns 0-based target indices).

    Off-by-one would still "work" numerically, so we detect it by planting a
    single contact column and asking which index recovers it.
    """
    single = planted_logits(BINDER_LEN, TARGET_LEN, [7], sharpness=8.0)
    out = toy_output(single, BINDER_LEN)
    at_7 = proxy(out, epitope_idx=[7])
    at_8 = proxy(out, epitope_idx=[8])  # 1-based reading of the same residue
    assert at_7 > 0.9, "index 7 should land on the planted contact"
    assert at_8 < 0.2, "index 8 must NOT -- indices are 0-based, target-relative"


def test_permuting_the_epitope_does_not_change_the_value(planted):
    out = toy_output(planted, BINDER_LEN)
    shuffled = list(np.random.default_rng(1).permutation(EPITOPE))
    assert proxy(out, epitope_idx=[int(i) for i in shuffled]) == pytest.approx(
        proxy(out, epitope_idx=EPITOPE), abs=1e-6
    )


# -------------------------------------------------------------- 3. the point


def test_epitope_restriction_reports_the_planted_site(planted):
    """Right site scores high, wrong site scores low."""
    out = toy_output(planted, BINDER_LEN)
    assert proxy(out, epitope_idx=EPITOPE) > 0.9
    assert proxy(out, epitope_idx=DECOY) < 0.2


def test_global_proxy_cannot_tell_the_sites_apart(planted, wrong_site):
    """Why the change is worth making.

    A binder docked at the wrong site is, to the unrestricted proxy, exactly as
    good as one docked at the right site -- it takes the k best pairs from
    anywhere on the target. Restricting to the epitope is what breaks the tie.
    """
    right, wrong = toy_output(planted, BINDER_LEN), toy_output(wrong_site, BINDER_LEN)

    assert proxy(right) == pytest.approx(proxy(wrong), abs=0.05)  # blind

    assert proxy(right, epitope_idx=EPITOPE) > 0.9  # discriminating
    assert proxy(wrong, epitope_idx=EPITOPE) < 0.2


def test_proxy_increases_with_contact_confidence():
    """Monotone in the thing it claims to measure."""
    vals = [
        proxy(
            toy_output(
                planted_logits(BINDER_LEN, TARGET_LEN, EPITOPE, sharpness=s), BINDER_LEN
            ),
            epitope_idx=EPITOPE,
        )
        for s in (0.0, 1.0, 2.0, 4.0, 8.0)
    ]
    assert vals == sorted(vals), vals
    assert vals[-1] - vals[0] > 0.5, "the range should be usable, not marginal"


def test_proxy_stays_in_unit_interval():
    """Includes the degenerate inputs -- clip() is the only thing keeping the
    ratio in range once S_bar exceeds log(n_contact_bins)."""
    for s in (-20.0, 0.0, 50.0):
        for ep in (None, EPITOPE, [0]):
            v = proxy(
                toy_output(
                    planted_logits(BINDER_LEN, TARGET_LEN, EPITOPE, sharpness=s),
                    BINDER_LEN,
                ),
                epitope_idx=ep,
            )
            assert 0.0 <= v <= 1.0


# ----------------------------------------------------------------- 4. plumbing


def test_gradient_reaches_only_the_epitope_columns(planted):
    """Mechanism, not just value: the columns outside the epitope must receive
    exactly zero gradient, and the ones inside must receive some.

    This is the assertion that the restriction actually changes what the
    optimizer is pushed toward, rather than only what gets reported.
    """
    binder_len = BINDER_LEN

    def f(logits, ep):
        loss = DistogramIPTMProxy(epitope_idx=ep)
        v, _ = loss(jnp.zeros((binder_len, 20)), toy_output(logits, binder_len), KEY)
        return v

    # sharpness 4.0 is inside the band where the proxy is neither clipped at 0
    # nor saturated at 1 -- see test_proxy_has_a_dead_zone_at_both_ends.
    logits = jnp.asarray(
        planted_logits(BINDER_LEN, TARGET_LEN, EPITOPE, sharpness=4.0), jnp.float32
    )
    g = np.asarray(jax.grad(f)(logits, EPITOPE))[:binder_len, binder_len:]
    per_col = np.abs(g).sum(axis=(0, 2))

    outside = np.setdiff1d(np.arange(TARGET_LEN), EPITOPE)
    assert np.all(per_col[outside] == 0.0), "gradient leaked outside the epitope"
    assert per_col[EPITOPE].sum() > 0.0, "no gradient inside the epitope"

    g_all = np.asarray(jax.grad(f)(logits, None))[:binder_len, binder_len:]
    assert np.abs(g_all).sum(axis=(0, 2))[outside].sum() >= 0.0  # sanity: defined
    assert np.all(np.isfinite(g_all))


def test_jit_matches_eager(planted):
    """`_eval_loss_and_grad` wraps everything in `eqx.filter_jit`; a python
    `list[int]` field has to survive that as a static leaf."""
    import equinox as eqx

    out = toy_output(planted, BINDER_LEN)
    seq = jnp.zeros((BINDER_LEN, 20))

    for ep in (None, EPITOPE):
        loss = DistogramIPTMProxy(epitope_idx=ep)
        eager = float(loss(seq, out, KEY)[1]["distogram_iptm"])
        jitted = float(
            eqx.filter_jit(lambda m, s, o, k: m(s, o, k))(loss, seq, out, KEY)[1][
                "distogram_iptm"
            ]
        )
        assert eager == pytest.approx(jitted, abs=1e-6)


def test_accepts_numpy_and_jax_index_arrays(planted):
    """Configs, manifests and `epitope_from_complex` hand this thing a list, but
    anything array-like should behave the same."""
    out = toy_output(planted, BINDER_LEN)
    want = proxy(out, epitope_idx=EPITOPE)
    for ep in (np.array(EPITOPE), jnp.array(EPITOPE), tuple(EPITOPE)):
        assert proxy(out, epitope_idx=ep) == pytest.approx(want, abs=1e-6)


# --------------------------------------------------- 5. edges and known sharp bits


def test_empty_epitope_is_rejected_loudly(planted):
    """An empty list must not silently mean "no restriction".

    `pipeline_node.py` guards this for BinderTargetContact (`if ... and ep`),
    but a hand-written config can still produce `epitope_idx: []`. Whatever the
    behaviour is, it must not be a quiet wrong answer -- record it here.
    """
    out = toy_output(planted, BINDER_LEN)
    with pytest.raises(Exception):
        proxy(out, epitope_idx=[])


def test_out_of_range_indices_are_silently_clamped(planted):
    """XFAIL-style documentation of a real footgun.

    JAX gather clamps out-of-bounds indices instead of raising, so a 1-based
    epitope, or one indexed into the full complex rather than the target block,
    produces a plausible number computed from the wrong residues. Nothing in the
    loss can catch that -- validation has to happen at config-parse time.
    """
    out = toy_output(planted, BINDER_LEN)
    bogus = [TARGET_LEN + 100, TARGET_LEN + 101]
    v = proxy(out, epitope_idx=bogus)
    assert np.isfinite(v)  # no error, no warning
    assert v == pytest.approx(proxy(out, epitope_idx=[TARGET_LEN - 1]), abs=1e-6), (
        "clamped to the last column, as documented"
    )


def test_proxy_has_a_dead_zone_at_both_ends():
    """`clip(., 0, 1)` means the loss has exactly zero gradient outside a narrow
    band of contact confidence.

    Pre-existing (the clip is not new), but it interacts badly with the epitope:
    restricting to a site can only ever *raise* S_bar, since it removes the
    freely-chosen best pairs from the pool. So a run that had signal without an
    epitope can land in the flat zone with one, and then contributes nothing but
    a constant to the objective -- silently, since the reported value (0.0) is
    also what a genuinely bad interface reports.

    If this ever starts failing because someone swapped clip for a softplus or
    a SoftClip, that is an improvement: delete the test.
    """

    def grad_norm(sharpness, ep):
        def f(logits):
            loss = DistogramIPTMProxy(epitope_idx=ep)
            return loss(
                jnp.zeros((BINDER_LEN, 20)), toy_output(logits, BINDER_LEN), KEY
            )[0]

        logits = jnp.asarray(
            planted_logits(BINDER_LEN, TARGET_LEN, EPITOPE, sharpness=sharpness),
            jnp.float32,
        )
        return float(jnp.abs(jax.grad(f)(logits)).sum())

    for ep in (None, EPITOPE):
        assert grad_norm(2.0, ep) == 0.0, "clipped at 0: no gradient (weak contacts)"
        assert grad_norm(4.0, ep) > 0.1, "live band"
        assert grad_norm(14.0, ep) < 1e-3, "saturated at 1: gradient has vanished"


def test_duplicate_indices_reweight_the_pool():
    """Duplicates are not deduplicated: repeating a residue lets the bottom-k
    take the same good pair twice, which inflates the score.

    `proposals.consensus_epitope` dedupes, but a hand-written config or a
    concatenated epitope from `merge` need not."""
    logits = graded_logits(
        BINDER_LEN, TARGET_LEN, EPITOPE, np.linspace(6.0, 3.5, BINDER_LEN)
    )
    out = toy_output(logits, BINDER_LEN)
    once = proxy(out, epitope_idx=EPITOPE)
    twice = proxy(out, epitope_idx=EPITOPE + EPITOPE)
    assert twice > once + 0.01, (once, twice)


def test_k_is_not_rescaled_with_the_epitope():
    """The value drifts with |epitope|, holding the structure fixed.

    `k = binder_len` is the paper's "minibinder" preset, chosen against the full
    N x (L-N) pool. Restricting to |E| columns shrinks the pool to N x |E| but
    leaves k alone, so the same prediction scores differently depending on how
    many residues were listed -- and at |E| == 1 the bottom-k degenerates into a
    plain mean over every binder position, i.e. the loss silently switches from
    "some k pairs must contact" to "every residue must contact".

    Not a bug, but it means epitope runs are not comparable to non-epitope runs,
    nor to each other across epitope sizes, and that the weight you want on this
    term depends on |E|. Asserted so it is a decision rather than an accident.
    """
    # Every binder residue contacts, but with a different confidence, so which
    # pairs the bottom-k picks is what decides the value.
    logits = graded_logits(
        BINDER_LEN, TARGET_LEN, EPITOPE, np.linspace(6.0, 3.5, BINDER_LEN)
    )
    out = toy_output(logits, BINDER_LEN)
    values = {k: proxy(out, epitope_idx=EPITOPE[:k]) for k in (1, 2, 3, 6)}
    assert values[6] > values[1] + 0.05, values  # bigger pool == easier

    # |E| == 1: pool size == binder_len == k, so bottom-k is the whole pool and
    # the proxy is a plain mean over binder positions.
    assert values[1] == pytest.approx(
        reference_proxy(logits, BINDER_LEN, epitope_idx=[EPITOPE[0]]), abs=1e-5
    )


def test_bins_below_cutoff_set_the_normaliser():
    """`log(n_contact_bins)` is computed from the bins, not hardcoded -- so a
    backend with different bin edges must still land in [0, 1]."""
    logits = planted_logits(BINDER_LEN, TARGET_LEN, EPITOPE, sharpness=8.0)
    out = toy_output(logits, BINDER_LEN)
    for cutoff in (3.0, 8.0, 21.0):
        v = proxy(out, contact_distance=cutoff, epitope_idx=EPITOPE)
        assert 0.0 <= v <= 1.0, (cutoff, v)
    assert float((BINS < 8.0).sum()) > 1  # normaliser is not log(1) == 0
