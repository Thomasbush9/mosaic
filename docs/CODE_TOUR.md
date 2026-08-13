# A reading guide to `src/mosaic`

**Who this is for:** someone who wants to understand or modify the library, not
someone who wants to run a design campaign. If you want to *run* things, read
`docs/MANUAL.md` and `docs/MODELS.md` instead and never open `src/`.

`src/mosaic` is ~30k lines, but roughly 12k of that is vendored AlphaFold you
will probably never read, and another 8k is per-backend plumbing that is
repetitive by design. The load-bearing ideas live in about **900 lines across
five files**. This guide names those files, puts them in order, and tells you
what you can safely skip.

---

## The one-paragraph mental model

A design problem is a *soft sequence*: an `[N, 20]` array where every row is a
probability distribution over amino acids. A `LossTerm` is a differentiable
function from that array to a scalar. You add loss terms together with ordinary
`+` and `*`, hand the sum to an optimizer, and the optimizer walks the soft
sequence downhill while keeping each row on the probability simplex. Everything
in `src/mosaic` is either a `LossTerm`, something that *builds* a `LossTerm`, or
an optimizer that consumes one.

---

## Tier 1 — the core (read these, in this order, ~500 lines)

Read these four files start to finish. After them the rest of the codebase is
mostly pattern-matching.

| # | File | Lines | Why |
|---|---|---|---|
| 1 | `common.py` | 76 | The whole abstraction, and it really is this small |
| 2 | `losses/structure_prediction.py` | 719 | Read only the top ~120 lines now (see below) |
| 3 | `structure_prediction.py` | 101 | The five-method contract every backend implements |
| 4 | `optimizers.py` | 676 | Read `_eval_loss_and_grad` + `simplex_APGM` (~150 lines) |

**1. `common.py`** — `TOKENS`, `LossTerm`, `LinearCombination`. `LossTerm`
overloads `__add__` / `__rmul__` / `__neg__` / `__sub__`, so `4 * ContactLoss()
+ 0.3 * HelixLoss()` produces a `LinearCombination`, which is itself an
`eqx.Module` and therefore a JAX pytree you can differentiate through. This is
the single most important file in the repo and takes five minutes.

`TOKENS = "ARNDCQEGHILKMFPSTWYV"` is the canonical column order for every
`[N, 20]` array in the codebase. Do not reorder it. Model wrappers translate to
and from their own alphabets internally, and getting that conversion wrong is
the classic silent bug when adding a backend.

**2. `losses/structure_prediction.py`** — for now, read only the
`StructureModelOutput` definition at the top. This is the model-agnostic view
that *every* backend must populate: distogram logits and bins, pLDDT, PAE
(values and logits), backbone and atom37 coordinates, `full_sequence`,
`asym_id`, `residue_idx`. It's the reason a loss written once works across all
ten backends. Come back for the loss terms themselves in Tier 3.

**3. `structure_prediction.py`** (the top-level one — note it is a *different
file* from `losses/structure_prediction.py`, an unfortunate naming collision)
— defines `TargetChain`, `StructurePrediction`, and the
`StructurePredictionModel` ABC with its five methods:

- `target_only_features(chains)` — predict an existing complex; includes real
  sidechain reference atoms.
- `binder_features(binder_length, chains)` — for design; sidechain reference
  atoms are **stubbed to UNK/G** because they aren't differentiably defined for
  a soft sequence. Predictions from binder features therefore have no
  sidechains. This split is load-bearing; don't try to unify the two.
- `predict` / `model_output` / `build_loss`.

**4. `optimizers.py`** — read `_eval_loss_and_grad` first. It's short and it
explains a convention you must preserve: the gradient has its row-mean
subtracted (`g - g.mean(axis=-1, keepdims=True)`), which projects onto the
tangent space of the simplex. It also converts `x` to a plain numpy float32
array before the jitted call, so that tuning the step size doesn't retrigger
compilation. Then read `simplex_APGM` — accelerated proximal gradient, with
`logspace=True` switching it to the entropic / mirror-descent variant.

Everything else in the file is a variation you can read on demand:
`batched_simplex_APGM` (vmapped over B designs, what the cluster runs),
`gradient_MCMC` (discrete, Plug-and-Play style), `batch_greedy_descent`
(1-hop hillclimb), `biohub_optimizer` (temperature-annealed logit-space descent
with best-of-tail selection, used for ESMFold2).

---

## Tier 2 — one backend, end to end (~370 lines)

Now trace a single model from features to loss. Use **Boltz-1**, which is the
smallest complete implementation:

1. `models/boltz1.py` (149 lines) — the canonical wrapper. All five interface
   methods, nothing extra, featurization delegated elsewhere. The whole pattern
   fits on one screen.
2. `losses/boltz.py` (537 lines) — where the actual work lives:
   `set_binder_sequence`, the trunk/forward split, the `StructureWriter`, and
   `Boltz1Loss`.

Read the pair together. The division of labour — thin interface class in
`models/`, heavy featurization and forward pass in `losses/` — is the
convention across the whole directory.

Then, if you want a second: `models/boltz2.py` adds template injection and
multi-sample losses, and `models/of3.py` or `models/opendde.py` show what
realistic full featurization looks like (of3 spends ~430 lines on template and
MSA features before the class even starts).

**A performance fact worth internalizing here:** under JIT, JAX prunes any
computation nothing reads from. A loss that touches only the distogram never
runs the confidence head or the structure module, and is dramatically cheaper.
The `*_trunk` / `*_forward_from_trunk` split in each backend file exists to make
this explicit. The tier table in the next section tells you which losses cost
what.

---

## Tier 3 — the loss library (read on demand, by need)

Go back to `losses/structure_prediction.py` and read the loss terms. They sort
into three cost tiers:

**Cheap — distogram / trunk only.** No confidence head, no structure module.
`WithinBinderContact`, `BinderTargetContact`, `ESMFoldInterContact`,
`HelixLoss`, `ESMFoldGlobularity`, `DistogramRadiusOfGyration`,
`MAERadiusOfGyration`, `DistogramCE`, `DistogramIPTMProxy`.

**Mid — needs the confidence module.** `PLDDTLoss`, `WithinBinderPAE`,
`BinderTargetPAE`, `TargetBinderPAE`, `IPTMLoss`, `BinderTargetIPTM`,
`BinderPTMLoss`, `BinderTargetIPSAE`, `TargetBinderIPSAE`, `IPSAE_min`,
`pTMEnergy`.

**Expensive — needs sampled atom coordinates.** `ActualRadiusOfGyration`, and
everything in `protein_mpnn.py`, `proteina.py`, and `atom37.py`.

### `losses/transformations.py` — read this early, it is not optional

263 lines of wrappers that turn raw losses into ones that actually behave.
Skipping this file is the most common way to waste a week of compute.

- `SoftClip` / `ClippedLoss` — bound a loss from below. **Raw pseudo-likelihoods
  over-optimize to homopolymers without this.** The README says so repeatedly
  and it is true. `ClippedLoss(..., 2, 100)` is the standard incantation.
- `NoCys` — reparameterize to 19 amino acids, pinning cysteine to zero.
- `SetPositions` — optimize only selected positions against a fixed wildtype;
  `from_sequence()` treats `'X'` as variable.
- `FixedPositionsPenalty` — softer alternative to `SetPositions`, an L2 pull
  toward target residues.
- `ClippedGradient` / `NormedGradient` — control a term's gradient magnitude so
  terms combine in a stable ratio regardless of raw scale.
- `RandomChoice` — evaluate one random member of an ensemble per step, no
  recompiles.

Note that `NoCys` and `SetPositions` reparameterize, so after optimizing you
must call `loss.sequence(...)` to recover the real `[N, 20]` sequence.

### The rest of `losses/`, by category

**Sequence-model likelihoods** (no structure): `esm.py` (`ESM2PseudoLikelihood`),
`esmc.py` (`ESMCPseudoLikelihood`, `ESMCPseudoPerplexity`), `ablang.py` /
`ablang2.py` (antibody LMs; ablang2 is paired heavy/light), `trigram.py`
(`TrigramLL`, `UnigramExcess`, `BigramExcess` — n-gram priors, no neural net),
`stability.py` (`StabilityModel`, an ESM-C backbone plus a regression head).

**Inverse folding** (needs coordinates): `protein_mpnn.py`
(`FixedStructureInverseFoldingLL`, `InverseFoldingSequenceRecovery` are the two
you'll actually use), `proteina.py` (`ProteinaInverseFoldingRecovery`).

**Per-backend structure losses:** `boltz.py`, `boltz2.py`, `of3.py`,
`protenix.py`, `opendde.py`, `promera.py`, `esmfold2.py`. Each pairs a
`set_binder_sequence` + trunk/forward split with a `LossTerm` wrapping a
`LinearCombination` of the model-agnostic terms. Read one, and you've read all
of them.

**Helpers, no loss terms:** `atom37.py` (canonical AF2 atom37 layout, used by
every backend), `confidence_metrics.py` (pLDDT / PAE / pTM from raw head
logits).

---

## Tier 4 — the model zoo

Ten backends in `src/mosaic/models/`. Most are variations on Boltz-1. Two are
not predictors at all:

| File | Backend package | Interface |
|---|---|---|
| `boltz1.py` | `joltz` + `boltz` | Full — **canonical example** |
| `boltz2.py` | `joltz` + `boltz` | Full + `build_multisample_loss` |
| `protenix.py` | `protenij` | Full + multisample; several checkpoint factories |
| `of3.py` | `jopenfold3` | Full + multisample; heavy featurization |
| `opendde.py` | `jopendde` | Full + multisample; mirrors `of3.py`, cleaner |
| `esmfold2.py` | `esmjfold2` | Full; ~10 factory functions, best module docstring |
| `promera.py` | `jpromera` (+ `tinyprot`) | Full; compact, featurizes inline |
| `af2.py` | **vendored** `mosaic.alphafold` on haiku | Full, but see below |
| `boltzgen.py` | `joltzgen` | **Not a predictor** — a conditional generator |
| `proteina.py` | `jproteina-complexa` | **Not a predictor** — beam search over a sampler |

Three notes:

- **`af2.py` is a plumbing outlier.** It satisfies the interface, but it runs on
  haiku against the vendored AlphaFold in `src/mosaic/alphafold/`, so it threads
  `hk.Params` around and defines its own `AlphaFoldLoss`. Don't read it to learn
  the pattern; the haiku layer obscures it.
- **`boltzgen.py` and `proteina.py` don't implement `StructurePredictionModel`.**
  They generate or search rather than score. `boltzgen.py` carries a top-of-file
  comment from its author calling its data loading "terrible" — treat that as
  accurate. This scoring/generative split is explained well in `docs/MODELS.md`.
- **`CLAUDE.md` is stale here.** It lists seven backends; there are ten.
  `esmfold2.py`, `opendde.py`, and `promera.py` are missing from it (they *are*
  in `README.md`), as are the sister packages `esmjfold2`, `jopendde`,
  `jpromera`, and the `tinyprot` dependency.

---

## What to skip

- **`src/mosaic/alphafold/`** (~12k lines) — vendored DeepMind AF2 under its
  original Apache license. Read it only if you are debugging AF2 specifically.
  `modules.py` alone is 2029 lines.
- **`src/mosaic/proteinmpnn/mpnn.py`** (606 lines) — a faithful JAX port of
  ProteinMPNN. You want the three loaders at the bottom (`load_mpnn`,
  `load_mpnn_sol`, `load_abmpnn`) and nothing else, unless you're changing the
  architecture. `torch_mpnn.py` is the reference implementation kept for
  comparison.
- **`stability_model/train.py`** — a training script, not part of the inference
  path. Useful as a template if you want to add a similar predictor head.
- **`util.py`, `notebook_utils.py`, `msa.py`** — small and self-explanatory.
  Read `msa.py`'s docstring though: it explains why you want
  `TargetChain.msa_path` set (without it, Boltz and OpenFold3 re-query the
  public ColabFold server on *every* featurization — 64 times for a
  64-trajectory array).

---

## How the library is actually driven

`src/` is a library; nothing in it is an entry point. The callers live at the
repo root and are worth skimming to see the intended usage:

- **`examples/*.py`** — marimo notebooks, not scripts. Run with
  `uv run marimo edit examples/example_notebook.py`. They're notebooks because
  the first JIT call on a structure model can take minutes, so an interactive
  session is the intended workflow. `example_notebook.py` and
  `boltz_notebook.py` are the two general ones.
- **`design_config.py`** — the translation layer from campaign config to mosaic
  objects. Its builder functions (from line ~424) are the most compact real
  example of composing a multi-model loss.
- **`run_design.py`** — what the cluster actually executes: `TargetChain` →
  loss → `batched_simplex_APGM`.
- **`pipeline_spec.py`, `pipeline_node.py`, `webapp/`** — orchestration above
  the library. Documented in `docs/PIPELINE_FILE.md` and `docs/WEBAPP.md`.

---

## Conventions to know before you write code

- **No torch in the hot path.** Load from PyTorch via `from_pretrained`, convert
  to Equinox with a `from_torch` helper, then JIT the Equinox module. Never call
  torch inside a `LossTerm.__call__`.
- **Loss terms are pytrees.** Store hyperparameters as arrays, not Python
  floats, if they should be traced — `eqx.field(converter=jnp.array)`. See
  `SoftClip` for the pattern.
- **Clip aggressive losses.** See the `transformations.py` section above.
- **Mark slow tests.** `pytest` runs with `-m 'not slow'` by default; anything
  touching a full structure model should be `@pytest.mark.slow`. There is
  currently exactly one test file (`tests/test_ablang2_loss.py`) and no CI, so
  the examples are the de facto integration tests.
- **Never run any of this on a login node.** Submit with `sbatch`; see
  `singularity/`.

---

## Suggested first session

If you have two hours:

1. `common.py` in full (5 min).
2. `StructureModelOutput` in `losses/structure_prediction.py` (10 min).
3. `structure_prediction.py` in full (10 min).
4. `_eval_loss_and_grad` and `simplex_APGM` in `optimizers.py` (20 min).
5. `losses/transformations.py` in full (25 min).
6. `models/boltz1.py`, then skim `losses/boltz.py` (40 min).

That's the whole framework. Everything after it is one more backend or one more
loss, and both follow templates you'll have already seen.
