# The models, in plain terms

Companion to `MANUAL.md`. What each model is, what it is good at, what it costs,
and which settings actually change its behaviour.

---

## Two kinds of model

This is the distinction that explains most of the design of the pipeline.

**Scoring models** take a sequence and predict a structure. Because that
prediction is differentiable, you can ask "which direction should I nudge residue
17 to make this bind better?" and get an answer. These are the models you
optimize *against*. All of §1 below.

**Generative models** produce structures directly by denoising. There is no
question to differentiate — they do not score a sequence you hand them. They are
used to *propose* candidates, not to refine them. §2 below.

Practically: only scoring models can go in `MODELS=`. Asking for
`MODELS=boltzgen` is not a configuration you have got wrong — it is not a thing
that exists.

---

## 1. Scoring models (usable in `MODELS=`)

### `boltz2` — Boltz-2

**Default choice.** Current-generation co-folding model: it predicts the target
and binder together as one complex, which is what you want when the question is
whether they touch.

- **Uses your MSA:** yes, up to 8192 alignment rows.
- **Cost:** ~91 s for 5 optimization steps (237-residue target, 80-residue
  binder, 2 designs).
- **Weights:** 2.2 GB, shared from the Kempner lab store.
- Good general accuracy on protein–protein interfaces. If you run one model,
  run this one.

### `boltz1` — Boltz-1

Previous generation of the same lineage. Slowest of the set (~124 s / 5 steps)
and generally superseded by Boltz-2. Keep it for comparison when you want to know
whether a result depends on model generation, not for production campaigns.

Uses your MSA. Also needs a chemical-component dictionary (`ccd.pkl`), which is
already staged.

### `af2` — AlphaFold2-multimer

The familiar one, and useful precisely *because* it is a different lineage from
Boltz — a design that satisfies both is less likely to be exploiting a quirk of
either.

- **Cannot use MSAs in this pipeline.** mosaic's AF2 interface rejects them
  outright. If you include `af2`, its view of your target is single-sequence
  while the other models get the full alignment. This is a property of the
  wrapper, not something you can switch on.
- **Cost:** ~109 s / 5 steps.
- **Weights:** 5.2 GB (all five multimer models are loaded and one is chosen per
  evaluation).
- Note it required a compatibility patch to run at all here — the upstream
  `dm-haiku` release is incompatible with our JAX version. That is handled inside
  the container; you should never see it.

### `of3` — OpenFold3

Open re-implementation in the AlphaFold3 lineage. Another genuinely independent
opinion, which is its main value in a consensus objective.

- **Uses your MSA:** yes.
- **Cost:** ~92 s / 5 steps — comparable to Boltz-2.
- **Weights:** 2.2 GB.

### `protenix` — Protenix Mini

**Fastest by a wide margin** — ~39 s / 5 steps, roughly 3× quicker than anything
else. Use it while you are still deciding what you want: binder length, loss
weights, how many steps. Then confirm with Boltz-2.

- **Uses your MSA:** yes.
- **Weights:** 512 MB (the "Mini" checkpoint; larger Protenix variants exist and
  are not staged).

### Sequence models (always on, not in `MODELS=`)

**ESM-C** is a protein language model. It does not predict structure — it judges
whether a sequence *looks like a real protein*. Without it, structure models
happily accept sequences no organism would tolerate.

It is added automatically with weight 0.5 and **clipped**. Clipping matters: an
unclipped language-model score is trivially maximized by repeating one favourable
residue, and you get homopolymers. The clip caps how much the objective can be
improved this way.

**AbLang / AbLang2** are the antibody-specific equivalents, for VH/VL sequences.
Staged and working, but not wired into `run_design.py` — they are for antibody
campaigns, which need a different setup than a de novo mini-binder.

---

## 2. Generative models (not usable in `MODELS=`)

### `boltzgen` — BoltzGen

Generates binder **backbones** against a target structure by diffusion, then
reads off which amino acid each position looks like from where the sidechain
atoms landed.

- **Needs a target *structure* (CIF), not a sequence.** Run `predict_target.py`
  first.
- **You specify the fold.** A secondary-structure string like
  `24×H, 4×L, 24×H, 4×L, 24×H` requests a three-helix bundle — the standard de
  novo mini-binder architecture.
- **Very fast:** ~50 s once to set up, then ~2.5 s per design.
- **Output quality is visibly good** — the sequences look like designed proteins
  in a way that optimizer output often does not.

See `MANUAL.md` §5 for the honest status of combining it with the optimizer
(short version: it does not currently help, and why).

### `proteina` — Proteina

Also generative (a flow-matching backbone model). Staged and loads correctly, but
not yet wired into any workflow here. Listed so you know it exists and why it is
not an option in `MODELS=`.

---

## 3. Parameters, and what they actually do

### Parameters you set

| Parameter | Meaning | Sensible range |
|---|---|---|
| `BINDER_LENGTH` | residues in the designed binder | 60–100 |
| `MODELS` | scoring models that must agree (`+`-separated) | 1–2 |
| `SOFT_STEPS` | optimization while the sequence is blurry | 50–200 |
| `SHARP_STEPS` | forcing it into real amino acids | 20–50 |
| `BATCH` | independent designs per GPU | 2–4 |
| `--array=0-N` | number of GPUs | as many as you want |

### Parameters set for you (and why)

**`recycling_steps = 1`.** Structure models refine their own prediction by
feeding it back in. More passes are more accurate but proportionally slower, so
design uses 1 and final validation uses 4–20.

*This must never be 0.* At 0 the model's main computation is skipped entirely and
the objective stops depending on the sequence at all — producing a believable
loss with a gradient of exactly zero. It looks like it is working. It is
optimizing nothing.

**`sampling_steps` (~25).** Diffusion steps used to generate coordinates. Only
matters for losses that read 3D positions; the contact-based objective here is
computed from the distance map, so it is largely irrelevant.

**`num_samples`.** How many structures to average per evaluation. More is more
robust, and multiplies both memory and time. Left at defaults — robustness here
comes from running many independent designs instead.

**ESM-C weight 0.5, clipped to [2, 100].** See above. Raising the weight pushes
toward more natural sequences at some cost to predicted binding; this is the
first knob to reach for if your designs look chemically implausible.

---

## 4. Cost, measured

Seconds for 5 optimization steps, 2 designs, 237-residue target + 80-residue
binder, on one H100:

| Model | Time |
|---|---|
| Protenix Mini | 39 s |
| Boltz-2 | 91 s |
| OpenFold3 | 92 s |
| AF2 multimer | 109 s |
| Boltz-1 | 124 s |

Costs add when you combine models: Boltz-2 alone ran 100 steps in 373 s;
Boltz-2 + AF2 took 630 s.

A full campaign task — 100 soft + 25 sharp steps, 2 designs, two models — takes
about **15 minutes on one GPU**. Thirty-two candidates from a 16-GPU array is
roughly a quarter of an hour of wall time.

---

## 5. Choosing a combination

| Situation | Use |
|---|---|
| Trying settings, want fast feedback | `protenix` |
| Standard campaign | `boltz2` |
| Want robustness against model quirks | `boltz2+of3` or `boltz2+af2` |
| Checking a result is not model-specific | rerun the winner through a different single model |
| Antibody rather than de novo binder | AbLang/AbLang2 — needs a different setup, ask |

Two models is a reasonable ceiling. Three roughly triples the cost for a
diminishing return, and it is usually better to spend that GPU time on more
independent seeds.
