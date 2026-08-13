"""Real-data tests for `DistogramIPTMProxy`, with and without `epitope_idx`.

Tier 2 of two. Same claims as `test_distogram_iptm_proxy.py`, but against AF2
forward passes, where they can fail for reasons a planted distogram cannot show:
the target block is not block-diagonal-clean, contact probability lives wherever
AF2 puts it, and the loss has to survive `eqx.filter_jit` +
`filter_value_and_grad` through the whole trunk.

System (default `dio3`): the campaign's rank-1 design against the DIO3 ECD.
AF2 scores that pair ipTM 0.821 at recycling 4, so it is a case where the
structure model genuinely predicts an interface -- the only setting in which
"does the proxy see the epitope" is a meaningful question. `MOSAIC_TEST_SYSTEM=
7opb` switches to the IL7R crystal complex, where AF2 (single-sequence) folds at
ipTM 0.10 and the proxy has nothing to read; useful as a negative control.

MOST ASSERTIONS ARE ON `margin`, NOT `proxy`.
    `proxy = clip(margin, 0, 1)` and on real AF2 output the margin is negative
    in every condition we have measured, so the proxy is a constant 0 and any
    test written against it compares 0 to 0 and passes vacuously. `margin` is
    the same quantity before the clip; ordering claims stated on it are real.
    `test_proxy_is_usable_as_an_objective` is the one that asserts on `proxy`,
    and it is the one that is expected to fail today.

Run:  tests/run_iptm_proxy_tests.sh --real
"""

from __future__ import annotations

import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from mosaic.losses.structure_prediction import (
    DistogramIPTMProxy,
    BinderTargetIPTM,
    IPTMLoss,
)
from bench_distogram_iptm_proxy import load_system, decompose, predicted_epitope, pssm

pytestmark = pytest.mark.slow

SYSTEM = os.environ.get("MOSAIC_TEST_SYSTEM", "dio3")
RECYCLING = int(os.environ.get("MOSAIC_TEST_RECYCLING", "4"))
MODEL_IDX = 0  # AlphaFoldLoss samples one of 5 per key; pin it so runs compare
KEY = jax.random.key(0)


# --------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def system():
    s = load_system(SYSTEM)
    print(f"\n{s['name']}: target {len(s['target_seq'])} aa, "
          f"binder {len(s['binder_seq'])} aa  ({s['note']})")
    print(f"epitope ({len(s['epitope'])} res, 0-based): {s['epitope']}")
    print(f"decoy   ({len(s['decoy'])} res): {s['decoy']}")
    assert not set(s["epitope"]) & set(s["decoy"])
    return s


@pytest.fixture(scope="module")
def af2():
    from mosaic.models.af2 import AlphaFold2

    t0 = time.time()
    model = AlphaFold2(multimer=True)
    print(f"af2 loaded in {time.time() - t0:.1f}s")
    return model


@pytest.fixture(scope="module")
def design_features(af2, system):
    """The design path: binder is a stub, sequence arrives as a PSSM."""
    from mosaic.structure_prediction import TargetChain

    features, _ = af2.binder_features(
        len(system["binder_seq"]),
        [TargetChain(sequence=system["target_seq"], use_msa=False)],
    )
    return features


@pytest.fixture(scope="module")
def outputs(af2, design_features, system):
    """One AF2 forward per sequence, cached, plus the real-complex prediction.

    Every value-level claim below reads off these, so the module costs four
    forward passes rather than one per assertion.
    """
    from mosaic.structure_prediction import TargetChain

    n = len(system["binder_seq"])
    out = {}

    complex_features, _ = af2.target_only_features(
        [
            TargetChain(sequence=system["binder_seq"], use_msa=False),
            TargetChain(sequence=system["target_seq"], use_msa=False),
        ]
    )
    t0 = time.time()
    out["complex"] = af2.model_output(
        features=complex_features, recycling_steps=RECYCLING,
        model_idx=MODEL_IDX, key=KEY,
    )
    jax.block_until_ready(out["complex"].distogram_logits)
    print(f"  forward[complex]: {time.time() - t0:.1f}s")

    rng = np.random.default_rng(0)
    real = system["binder_seq"]
    for name, s in (
        ("real", real),
        ("scramble", "".join(rng.permutation(list(real)))),
        ("polyval", "V" * n),
    ):
        t0 = time.time()
        o = af2.model_output(
            PSSM=pssm(s), features=design_features, recycling_steps=RECYCLING,
            model_idx=MODEL_IDX, key=KEY,
        )
        jax.block_until_ready(o.distogram_logits)
        print(f"  forward[{name}]: {time.time() - t0:.1f}s  "
              f"plddt {float(o.plddt.mean()):.3f}")
        out[name] = o
    return out


def _proxy(output, binder_len, **kw) -> float:
    loss = DistogramIPTMProxy(**kw)
    _v, aux = loss(jnp.zeros((binder_len, 20)), output, KEY)
    return float(aux["distogram_iptm"])


# ------------------------------------------- 0. where epitopes come from here,
#                                                and how that goes wrong


@pytest.mark.skipif(SYSTEM != "7opb", reason="needs a structure with solvent")
def test_proposals_epitope_is_unsafe_on_solvated_structures(system):
    """The producer side of `epitope_idx`, which the loss cannot defend against.

    `proposals.epitope_from_complex` enumerates *all* residues of the target
    chain, waters and heteroatoms included, so its indices are positions in that
    list rather than in the target sequence. 7opb carries 90 solvent residues on
    chain A, so the epitope it returns runs past the end of the chain.

    Feeding those into `DistogramIPTMProxy(epitope_idx=...)` does not raise --
    JAX gather clamps -- so the loss would optimise against the last target
    column repeated and report a perfectly ordinary number. Generated CIFs have
    no solvent, which is why this has not bitten the pipeline yet.
    """
    import proposals
    from bench_distogram_iptm_proxy import _find, SYSTEMS

    raw = proposals.epitope_from_complex(
        _find(SYSTEMS["7opb"]["cif"]), binder_length=len(system["binder_seq"]),
        cutoff=8.0,
    )
    n_target = len(system["target_seq"])
    bad = [i for i in raw if i >= n_target]
    print(f"\nepitope_from_complex: {len(raw)} indices, {len(bad)} >= "
          f"len(target)={n_target}: {bad}")
    assert bad, "solvent no longer leaks in -- if proposals.py was fixed, delete this"


# ------------------------------------------------------- 1. the change is inert
#                                                             where it should be


def test_none_matches_the_reference_implementation(outputs, system):
    """The untouched path, on real distograms this time."""
    from toy_distogram import reference_proxy

    n = len(system["binder_seq"])
    for name, o in outputs.items():
        want = reference_proxy(
            np.asarray(o.distogram_logits), n, bins=np.asarray(o.distogram_bins)
        )
        assert _proxy(o, n) == pytest.approx(want, abs=2e-4), name


def test_full_epitope_is_identical_to_none(outputs, system):
    n, t = len(system["binder_seq"]), len(system["target_seq"])
    for name, o in outputs.items():
        a = decompose(o, n, epitope_idx=list(range(t)))["margin"]
        b = decompose(o, n)["margin"]
        assert a == pytest.approx(b, abs=1e-4), (name, a, b)


# -------------------------------------------------------------- 2. it separates


def test_restricting_can_only_lower_the_score(outputs, system):
    """A mathematical invariant, and the reason the epitope is not free.

    The bottom-k of a subset is elementwise >= the bottom-k of the superset, so
    S_bar can only rise and the margin can only fall when you restrict. Any
    epitope therefore moves the term *toward* the clip at 0, and a smaller
    epitope moves it further. Consequence for tuning: the weight that worked for
    a global run is wrong for an epitope run, and weights are not transferable
    between epitopes of different sizes either.
    """
    n = len(system["binder_seq"])
    for name, o in outputs.items():
        full = decompose(o, n)["margin"]
        margins = {
            key: decompose(o, n, epitope_idx=system[key])["margin"]
            for key in ("epitope", "core", "decoy")
        }
        for key, m in margins.items():
            assert m <= full + 1e-4, (name, key, m, full)
        # ...and the tighter the epitope, the further it falls
        assert margins["core"] <= margins["epitope"] + 1e-4, (name, margins)


def test_epitope_restriction_tracks_where_af2_docked(outputs, system):
    """The load-bearing claim, stated against AF2's own prediction.

    Whatever site AF2 chose, the score restricted to that site must beat the
    score restricted to the opposite face. If this fails, the column indexing is
    wrong -- transposed, off-by-one, or indexing the complex rather than the
    target block.
    """
    n = len(system["binder_seq"])
    o = outputs["real"]
    pred = predicted_epitope(o, n)
    print(f"\nAF2-predicted epitope ({len(pred)} res): {pred}")
    assert pred, "AF2 placed the binder nowhere near the target"

    far = [j for j in system["decoy"] if j not in pred]
    assert far, "decoy site lies entirely inside the predicted epitope"

    here = decompose(o, n, epitope_idx=pred)["margin"]
    there = decompose(o, n, epitope_idx=far)["margin"]
    print(f"margin @ predicted site {here:.4f}   @ far site {there:.4f}")
    assert here > there, (here, there)


def test_real_binder_beats_a_scramble(outputs, system):
    """Sanity on the signal itself: if this cannot separate a real binder from
    its own shuffled sequence, nothing downstream of it means anything."""
    n = len(system["binder_seq"])
    for label, ep in (("global", None), ("epitope", system["epitope"])):
        vals = {
            k: decompose(outputs[k], n, epitope_idx=ep)["margin"]
            for k in ("real", "scramble", "polyval")
        }
        print(f"\nmargin[{label}]: " + "  ".join(f"{k}={v:+.4f}" for k, v in vals.items()))
        assert vals["real"] > vals["polyval"], (label, vals)
        assert vals["real"] > vals["scramble"], (label, vals)


def test_the_reported_proxy_barely_leaves_the_floor(outputs, system):
    """How much of [0, 1] the term actually uses, on a design that works.

    Not an ordering claim -- a range claim. On the rank-1 DIO3 design, which AF2
    folds at ipTM 0.79 with a maximum binder-target contact probability of 0.93,
    the unrestricted proxy reports 0.036. That is the good case. Everything
    worse than it reports exactly 0.

    So the term is a hinge that only lifts off the floor for interfaces that are
    already excellent, and carries no gradient for anything below that. Whatever
    weight it is given, it does nothing for the first half of an optimisation.
    """
    n = len(system["binder_seq"])
    good = _proxy(outputs["real"], n)
    print(f"\nproxy on the ipTM-0.79 design: {good:.4f}  "
          f"(scramble {_proxy(outputs['scramble'], n):.4f}, "
          f"poly-V {_proxy(outputs['polyval'], n):.4f})")
    assert good > 0.0, "even the good case is on the floor"
    assert good < 0.2, (
        f"proxy reached {good:.3f} -- if the dynamic range improved, retune the "
        "bound in this test rather than deleting it"
    )


def test_proxy_separates_a_real_binder_from_a_scramble(outputs, system):
    """Unrestricted, the reported number does carry signal at the top end."""
    n = len(system["binder_seq"])
    assert _proxy(outputs["real"], n) > _proxy(outputs["scramble"], n)


@pytest.mark.xfail(
    reason="under a tight epitope every candidate clips to proxy 0, so the term "
           "becomes a constant -- see test_restricting_can_only_lower_the_score",
    strict=False,
)
def test_proxy_still_discriminates_under_a_tight_epitope(outputs, system):
    """The failure that matters, stated where it actually happens.

    A hotspot-sized epitope is the whole point of the feature: aim the design at
    a few specific residues. But restricting can only raise S_bar, and a
    hotspot raises it past `log(n_contact_bins)` even for the good design. Then
    a real binder, a scramble and poly-V all report exactly 0.0000 and the term
    is a constant -- silently, since 0 is also what a genuinely bad interface
    reports.

    Flips to XPASS once the clip is replaced by something with a tail (SoftClip,
    softplus, or reporting the margin), which is the change this suite argues
    for. The epitope machinery itself is fine: the same comparison on `margin`
    passes (see test_real_binder_beats_a_scramble).
    """
    n = len(system["binder_seq"])
    core = system["core"]
    vals = {k: _proxy(outputs[k], n, epitope_idx=core) for k in ("real", "scramble")}
    print(f"\nproxy under a {len(core)}-residue epitope: {vals}")
    assert vals["real"] > vals["scramble"] + 1e-4, vals


def test_reports_the_full_picture(outputs, system):
    """Reporting, not gatekeeping: the numbers to look at when deciding whether
    the term earns its weight."""
    n = len(system["binder_seq"])
    crystal = set(system["epitope"])
    variants = [("none", None), ("epitope", system["epitope"]),
                ("core", system["core"]), ("decoy", system["decoy"])]
    print(f"\n{'seq':<10} {'variant':<8} {'pool':>6} {'S_bar':>7} {'margin':>8} "
          f"{'proxy':>7} {'maxPc':>7} {'iptm':>7} {'IoU':>5}")
    for name, o in outputs.items():
        pred = set(predicted_epitope(o, n))
        iou = len(pred & crystal) / max(1, len(pred | crystal))
        _, ai = IPTMLoss()(jnp.zeros((n, 20)), o, KEY)
        for vname, ep in variants:
            d = decompose(o, n, epitope_idx=ep)
            print(f"{name:<10} {vname:<8} {d['pool']:>6} {d['S_bar']:7.3f} "
                  f"{d['margin']:+8.3f} {d['proxy']:7.4f} {d['max_p_contact']:7.3f} "
                  f"{float(ai['iptm']):7.3f} {iou:5.2f}")


# ------------------------------------------------ 3. it survives the design path


def test_gradient_through_af2_is_finite_and_costs_the_same(af2, design_features, system):
    """`build_loss` -> `eqx.filter_jit` -> `filter_value_and_grad` through the
    AF2 trunk, exactly as `run_design.py` does it.

    A `list[int]` field has to survive that as a static leaf, the gather must not
    produce NaNs, and the extra index must not change the per-step cost. Whether
    the gradient is non-zero is a separate question -- see
    `test_proxy_is_usable_as_an_objective`.
    """
    from mosaic.optimizers import _eval_loss_and_grad

    x = np.asarray(pssm(system["binder_seq"]), dtype=np.float32)
    x = 0.9 * x + 0.1 / 20.0  # off the vertex, as run_design.py seeds it

    timings = {}
    for label, ep in (("global", None), ("epitope", system["epitope"]),
                      ("core", system["core"])):
        term = af2.build_loss(
            loss=DistogramIPTMProxy(epitope_idx=ep),
            features=design_features,
            recycling_steps=RECYCLING,
        )
        t0 = time.time()
        (v, _aux), g = _eval_loss_and_grad(term, x, KEY)
        jax.block_until_ready(g)
        compile_s = time.time() - t0

        t0 = time.time()
        for i in range(3):
            (v, _aux), g = _eval_loss_and_grad(term, x, jax.random.fold_in(KEY, i))
        jax.block_until_ready(g)
        step_s = (time.time() - t0) / 3

        g = np.asarray(g)
        assert g.shape == x.shape
        assert np.all(np.isfinite(g)), f"{label}: non-finite gradient"
        assert np.isfinite(float(v))
        timings[label] = (compile_s, step_s, float(v), float(np.abs(g).max()))

    print(f"\n{'variant':<9} {'compile+1st':>12} {'per step':>10} {'loss':>9} {'max|g|':>10}")
    for k, (c, s, v, gm) in timings.items():
        print(f"{k:<9} {c:12.1f}s {s:10.2f}s {v:9.4f} {gm:10.3e}")

    steps = sorted(t[1] for t in timings.values())
    assert steps[-1] < 2.0 * steps[0] + 0.5, (
        f"the epitope variant changed the per-step cost: {timings}"
    )


@pytest.fixture(scope="module")
def design_gradients(af2, design_features, system):
    """One grad through AF2 per epitope variant, cached (they are not cheap)."""
    from mosaic.optimizers import _eval_loss_and_grad

    x = np.asarray(pssm(system["binder_seq"]), dtype=np.float32)
    x = 0.9 * x + 0.1 / 20.0  # off the vertex, as run_design.py seeds it

    grads = {}
    for label, ep in (("global", None), ("epitope", system["epitope"]),
                      ("core", system["core"]), ("decoy", system["decoy"])):
        term = af2.build_loss(
            loss=DistogramIPTMProxy(epitope_idx=ep),
            features=design_features, recycling_steps=RECYCLING,
        )
        (v, _a), g = _eval_loss_and_grad(term, x, KEY)
        grads[label] = (float(v), np.asarray(g).ravel())
        print(f"  grad[{label}]: loss {float(v):+.4f}  "
              f"|g| {float(np.linalg.norm(g)):.3e}")
    return grads


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else float("nan")


def test_epitope_changes_the_gradient_direction(design_gradients):
    """The restriction reaches the optimizer, not just the logged number.

    Where the term is off the clip, aiming it at the epitope must rotate the
    gradient away from the unrestricted one -- but not to the point of being
    unrelated, since the epitope is where the binder already is.
    """
    c = _cos(design_gradients["global"][1], design_gradients["epitope"][1])
    print(f"\ncos(global, epitope) = {c:.3f}")
    assert np.isfinite(c), "one of the gradients is identically zero"
    assert c < 0.995, "the epitope did not change the search direction at all"
    assert c > 0.0, "the epitope reversed the search direction -- indexing bug?"


@pytest.mark.xfail(
    reason="a decoy epitope clips the term flat, so its gradient is identically "
           "zero and there is no direction to compare",
    strict=False,
)
def test_a_decoy_epitope_pushes_somewhere_else(design_gradients):
    """What a working epitope term would do, and does not.

    Pointing the loss at the far face of the target should push the design
    toward that face. Instead the term clips to 0 there, the gradient is exactly
    zero, and the objective silently loses the term altogether -- an optimizer
    aimed at a site the binder does not currently occupy gets no help finding it,
    which is the situation the epitope feature exists to fix.
    """
    c = _cos(design_gradients["epitope"][1], design_gradients["decoy"][1])
    print(f"\ncos(epitope, decoy) = {c:.3f}")
    assert np.isfinite(c), "the decoy gradient is identically zero"
    assert c < 0.99
