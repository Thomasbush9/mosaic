# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`mosaic` is a JAX-based framework for **multi-objective protein design via continuous relaxation of sequence space**. It reimplements a zoo of protein structure/sequence models (Boltz-1/2, AlphaFold2, OpenFold3, Protenix, ProteinMPNN, ESM2/C, AbLang, stability, trigram, Proteina-Complexa, BoltzGen) behind a single JAX-native interface so they can be composed into one differentiable loss and optimized jointly.

Read `README.md` for the motivation and worked examples before making non-trivial changes — the design rationale (continuous relaxation over $\Delta_c^n$, mirror descent / proximal gradient, target-only vs binder features) is documented there and not duplicated here.

## Environment and commands

- Python is pinned to `==3.12.*`. Dependency manager is `uv`.
- Install: `uv sync --group jax-cuda` (or `jax-cpu` / `jax-tpu`). The three JAX groups are declared as **conflicting** in `pyproject.toml` — pick one. Structure prediction needs GPU/TPU JAX in practice.
- `pyproject.toml` declares `required-environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]`. On macOS many deps (`protenix`, `boltz`, `boltzgen`, `jopenfold3`, `joltzgen`, `jproteina-complexa`) won't resolve — expect to develop / run on a Linux x86_64 box.
- Run a notebook: `uv run marimo edit examples/example_notebook.py`. Examples (`examples/*.py`) are **marimo notebooks**, not plain scripts; that's by design (JIT warmup is slow, so interactive sessions are the intended workflow).
- Tests: `uv run pytest`. By default `addopts = "-m 'not slow'"` excludes anything marked `@pytest.mark.slow`. Run one: `uv run pytest tests/test_ablang2_loss.py::test_name`. Include slow: `uv run pytest -m slow` or `uv run pytest -m ""`.
- Lint / typecheck: `ruff` and `ty` are in the `dev` group (`uv run ruff check .`, `uv run ty check`). There is no CI config in-repo.

Several deps are git-sourced sister repos under `escalante-bio/*` and `nboyd/joltz` (see `[tool.uv.sources]`). Updating them requires `uv lock --upgrade-package <name>`.

## Big-picture architecture

The whole framework is built around two abstractions in `src/mosaic/common.py`:

- **`LossTerm`** (eqx.Module): a JIT-compatible callable pytree with signature `(soft_sequence: [N,20], *, key) -> (scalar, aux_dict)`.
- **`LinearCombination`**: weighted sum of `LossTerm`s. `LossTerm` overloads `__add__`, `__rmul__`, `__neg__`, `__sub__` so users compose losses with plain arithmetic (`4 * BinderTargetContact() + 0.3 * HelixLoss() + ...`) and the result is itself a pytree that JAX can differentiate end-to-end.

Everything else is either (a) a `LossTerm` subclass or (b) a model wrapper that *produces* `LossTerm`s.

### Structure prediction interface (`src/mosaic/structure_prediction.py`)

All structure models implement `StructurePredictionModel` with four methods:

- `target_only_features(chains)` → `(features, writer)` — for predicting an existing complex; includes real sidechain reference atoms.
- `binder_features(binder_length, chains)` → `(features, writer)` — for design; sidechain reference atom positions are **stubbed to UNK/G** because they're not differentiably defined for a soft sequence. Predictions made with binder features therefore have no sidechains (see the WARNING in README.md). This split is load-bearing — don't try to unify them.
- `predict(...)` → `StructurePrediction` (contains `st`, `plddt`, `pae`, `iptm`, `model_output`).
- `build_loss(loss=..., features=..., recycling_steps=...)` → a `LossTerm` that, when called on a soft sequence, runs the model forward and evaluates the inner loss against the resulting `StructureModelOutput`.

`StructureModelOutput` (in `src/mosaic/losses/structure_prediction.py`) is the **model-agnostic** view every backend must populate: distogram logits/bins, plddt, pae, atom37 coords/mask, backbone coords, full_sequence, asym_id, residue_idx. New structure-prediction losses should consume `StructureModelOutput` and they will then work across all backends.

Under JIT, JAX prunes structure-module / confidence-module computation if no loss term reads from it — trunk-only losses are much faster than confidence-module losses. Keep this in mind when adding losses.

### Layout

- `src/mosaic/models/` — thin wrappers around each backend (`af2.py`, `boltz1.py`, `boltz2.py`, `of3.py`, `protenix.py`, `boltzgen.py`, `proteina.py`). The actual weights/forward come from sister packages (`joltz`, `jopenfold3`, `protenij`, `joltzgen`, `jproteina-complexa`).
- `src/mosaic/losses/` — loss terms grouped by source: `structure_prediction.py` (model-agnostic, e.g. `BinderTargetContact`, `WithinBinderContact`, `DistogramRadiusOfGyration`, `HelixLoss`, `PLDDTLoss`), then per-backend specifics (`boltz.py`, `boltz2.py`, `of3.py`, `protenix.py`, `proteina.py`), and per-sequence-model (`esm.py`, `esmc.py`, `ablang.py`, `ablang2.py`, `protein_mpnn.py`, `stability.py`, `trigram.py`). `transformations.py` is critical — `ClippedLoss`, `SoftClip`, `NoCys`, `SetPositions`, `FixedPositionsPenalty`, `ClippedGradient`, `NormedGradient` wrap other `LossTerm`s and are heavily used in practice (raw PLLs over-optimize to homopolymers without clipping).
- `src/mosaic/proteinmpnn/` — JAX port of ProteinMPNN with bundled weights in `weights/` (vanilla, soluble, AbMPNN). `InverseFoldingSequenceRecovery` and `FixedStructureInverseFoldingLL` are the main entry points.
- `src/mosaic/alphafold/` — vendored DeepMind AF2 code (under its original Apache license, see `LICENSE`).
- `src/mosaic/optimizers.py` — `simplex_APGM` (accelerated proximal gradient on the simplex; supports `logspace=True` for entropic / mirror descent variant), `batched_simplex_APGM` (vmapped), `gradient_MCMC` (discrete optimization à la Plug-&-Play). The `_eval_loss_and_grad` helper standardizes inputs to avoid recompilation when tuning step size/momentum.
- `src/mosaic/stability_model/train.py` — small regression head on ESM embeddings trained on the megascale ΔG dataset; useful as a reference for adding similar predictors.
- `src/mosaic/common.py` — `TOKENS = "ARNDCQEGHILKMFPSTWYV"` (canonical 20-AA ordering, used everywhere; do not reorder), `LossTerm`, `LinearCombination`.

### TOKENS ordering

`TOKENS = "ARNDCQEGHILKMFPSTWYV"`. This is the canonical column order for every `[N, 20]` soft-sequence array in the codebase. Several model wrappers translate this to/from their own internal alphabets — when adding a new model, audit the alphabet conversion carefully.

## Conventions that matter

- **No torch in the hot path.** Models are loaded from PyTorch (`from_pretrained`) and then converted to Equinox modules via `from_torch` helpers from the sister `esmj`, `esm2quinox`, `jablang`, etc. The converted JAX model is what gets JIT-traced. When adding a new pretrained model, follow this pattern; don't call torch inside a `LossTerm.__call__`.
- **First JIT call is slow** (sometimes minutes for structure models). Examples are notebooks for this reason. When adding tests that exercise full models, mark them `@pytest.mark.slow`.
- **Loss terms are pytrees.** Store arrays as fields, not Python floats, if they should be traced. Use `eqx.field(converter=jnp.array)` for scalar hyperparameters that need to be JAX arrays (see e.g. `SoftClip` in `transformations.py`).
- **Gradient cleanup.** `_eval_loss_and_grad` subtracts the row-mean from the gradient (`g - g.mean(axis=-1, keepdims=True)`) — this is the projection onto the tangent space of the simplex and is needed for `simplex_APGM` to behave. New optimizers operating on soft sequences should preserve this convention or document why they don't.
- **Clip aggressive losses.** Wrap PLLs / inverse-folding likelihoods in `ClippedLoss(..., 2, 100)` or `SoftClip` — README calls this out repeatedly and it's the most common footgun.
