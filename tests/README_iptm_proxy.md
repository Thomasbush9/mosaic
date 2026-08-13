# Testing `DistogramIPTMProxy(epitope_idx=...)`

Test suite for the `epitope_idx` addition to `DistogramIPTMProxy` in
`src/mosaic/losses/structure_prediction.py`. Two tiers plus a bench, so a change
to the loss can be checked in seconds and then confirmed against a real model.

```
tests/run_iptm_proxy_tests.sh            # toy tier, CPU, ~20 s
tests/run_iptm_proxy_tests.sh --real     # AF2 on the DIO3 rank-1 design, ~15 min
tests/run_iptm_proxy_tests.sh --bench    # comparison report -> tests/out/, ~15 min
```

The runner binds the working tree over the image's `src` (`MOSAIC_DEV_SRC`), so
edits to the loss take effect with no container rebuild.

| file | what it is |
|---|---|
| `toy_distogram.py` | synthetic `StructureModelOutput`s + an independent NumPy transcription of Algorithm 15 |
| `test_distogram_iptm_proxy.py` | 27 toy tests: contract, indexing convention, gradient localisation, edges |
| `test_distogram_iptm_proxy_real.py` | AF2 forward passes; ordering, gradient plumbing, cost |
| `bench_distogram_iptm_proxy.py` | not pass/fail — the diagnostic table and `iptm_proxy_bench.json` to diff between edits |

## What is asserted where

The toy tier pins the **arithmetic**: `epitope_idx=None` still reproduces
Algorithm 15 exactly; restricting to columns is identical to having been handed
a complex with only those target residues; indices are 0-based into the target
block (matching `BinderTargetContact` and `proposals.epitope_from_complex`);
gradient reaches the epitope columns and *exactly zero* elsewhere.

The real tier pins the **behaviour under a model**: ordering claims, that a
`list[int]` field survives `eqx.filter_jit`, that the gather adds no measurable
cost, and that the restriction rotates the gradient rather than only changing
the logged number.

Most real-tier assertions are on `margin` (the unclipped
`1 − S_bar / log n_contact_bins`), not on `proxy`. See the findings below: on
real output the proxy is usually a constant 0, so tests written against it
compare 0 to 0 and pass vacuously.

## Choice of system

`dio3` (default) is the campaign's rank-1 design (`design358`) against the DIO3
ECD — AF2 folds that pair at ipTM 0.79–0.82 with a maximum binder→target contact
probability of 0.93. It is the only in-tree case where the structure model
genuinely predicts an interface, which is the precondition for "does the proxy
see the epitope" to be a meaningful question.

`7opb` (IL7R + a crystallised 55-aa antagonist) is a *negative* control:
single-sequence AF2 folds it at ipTM 0.10 and predicts no contacts anywhere, so
every variant reads 0 and the system discriminates nothing. Worth keeping, and
worth not mistaking for a test of the loss.

For AF2 specifically, `target_only_features([binder, target])` and
`binder_features(len, [target])` + the real PSSM produce **identical** distograms
— AF2 multimer takes no template input, so the "real complex" and "design path"
featurisations coincide. That will not hold for Boltz/OF3/Protenix, which do
consume reference atoms.

## Findings

Numbers from `tests/out/iptm_proxy_bench.json`, AF2 multimer, model 0,
recycling 4, binder 80 aa, target 237 aa. `log(n_contact_bins) = 2.944`.

| sequence | epitope | pool | S_bar | margin | **proxy** | ‖grad‖ |
|---|---|---:|---:|---:|---:|---:|
| rank-1 design (ipTM 0.79) | none | 18960 | 2.839 | +0.036 | **0.0357** | 5.7e-2 |
| rank-1 design | crystal (47 res) | 3760 | 2.878 | +0.023 | **0.0225** | 5.9e-2 |
| rank-1 design | hotspot (7 res) | 560 | 6.017 | −1.044 | **0.0000** | **0** |
| rank-1 design | decoy (47 res) | 3760 | 6.657 | −1.261 | **0.0000** | **0** |
| scramble | none | 18960 | 5.321 | −0.807 | **0.0000** | — |
| poly-V | none | 18960 | 9.482 | −2.220 | **0.0000** | — |

**1. The epitope mechanism itself is correct.** Indexing convention matches the
rest of the codebase, restriction is exactly equivalent to deleting the other
columns, gradient reaches the named columns and nothing else, and the
restriction rotates the design-path gradient
(`cos(global, epitope) = 0.893`) rather than merely relabelling it. All 27 toy
tests pass. Nothing here needs fixing.

**2. It costs nothing.** 15.01 s/step with an epitope vs 15.07 s without —
within noise. Each distinct `epitope_idx` triggers a fresh ~90 s AF2 compile,
because a `list[int]` is a static field; that is correct and unavoidable, but it
means sweeping epitopes in one session pays a compile per value.

**3. `clip(·, 0, 1)` makes the term inert over most of its range.** This is
inherited from upstream, not from the epitope change: the proxy is exactly 0
with an exactly zero gradient whenever `S_bar > log n_contact_bins`. On the
*best* design we have — validated at ipTM 0.82, contact probability 0.93 — the
unrestricted proxy reads **0.036**, i.e. it uses 3.6 % of its nominal [0, 1]
range. A scramble and poly-V both read 0.0000, indistinguishable from each
other and from a correct design aimed at the wrong site.

**4. The epitope makes (3) much worse, and provably so.** Restricting to a
subset can only *raise* `S_bar` — the bottom-k of a subset is elementwise ≥ the
bottom-k of the superset — so an epitope always moves the term toward the clip,
and a smaller epitope moves it further. Measured: 47 residues costs 0.04 nats,
7 residues costs 3.2 nats and puts the term firmly on the floor. A hotspot-sized
epitope, which is the main reason to want this feature, silently reduces the
term to a constant that contributes nothing to the objective.

The same restriction pointed at a site the binder does *not* currently occupy
(the decoy) is also flat — so the term cannot pull a design toward a new
epitope, only reward one that is already there.

**5. `k = binder_len` is not rescaled with the pool.** It is the paper's
"minibinder" preset, chosen against the full `N × (L−N)` pool. With an epitope
the pool becomes `N × |E|`, so the same prediction scores differently depending
on how many residues were listed, and at `|E| == 1` the bottom-k degenerates
into a plain mean over every binder position — the loss quietly switches from
"some k pairs must contact" to "every residue must contact". Practical
consequence: a weight tuned for a global run is wrong for an epitope run, and
weights do not transfer between epitopes of different sizes.

**6. Out-of-range indices are silently clamped.** JAX gather does not raise, so
a 1-based epitope, or one indexed into the full complex rather than the target
block, yields a plausible number computed from the wrong residues. Worth
validating at config-parse time (`design_config.py`) rather than in the loss.

This is not hypothetical: `proposals.epitope_from_complex` enumerates *all*
residues of the target chain including waters and heteroatoms, so on any
structure with solvent it returns indices past the end of the chain. 7opb (90
solvent residues on chain A) yields 12 such indices out of 59. Generated CIFs
have no solvent, which is why the pipeline has not hit this yet — it would hit
it the first time someone points a campaign at a PDB entry.

## Suggestions

In rough order of value:

1. **Replace the hard clip.** `SoftClip` (already in `losses/transformations.py`)
   or a softplus keeps a gradient below the threshold. `test_proxy_still_
   discriminates_under_a_tight_epitope` and `test_a_decoy_epitope_pushes_
   somewhere_else` are marked xfail and flip to XPASS when this is done — they
   are the acceptance test for the fix.
2. **Report the margin in `aux` alongside the proxy.** One extra key, and the
   difference between "interface is bad" and "term is off the clip" stops being
   invisible in the logs.
3. **Decide what `k` means under an epitope** — either `k = binder_len` as now
   (documented as "every restriction is stricter"), or scaled to the pool.
   `test_k_is_not_rescaled_with_the_epitope` pins whichever you choose.
4. **Validate `epitope_idx` where it enters** — reject empty lists, indices ≥
   target length, and negatives in `design_config.py`, and dedupe.
5. **Expose `epitope_idx` in `design_config.LOSS_TERMS["DistogramIPTMProxy"]`
   and in `pipeline_node.py`.** Right now only `BinderTargetContact` gets the
   epitope carried over from the upstream manifest, so the new parameter is
   unreachable from a config or a pipeline run.
