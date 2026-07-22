# Designing binders with mosaic — a practical manual

**Who this is for:** you know proteins. You do not need to know JAX, SLURM, or
containers. Everything here is copy-paste plus editing a few numbers.

**What this does:** you give it a target protein sequence. It designs short
proteins ("binders") predicted to stick to that target, by asking several
structure-prediction models at once and gradually rewriting a candidate sequence
until they all agree it binds.

**What this does *not* do:** it does not tell you a binder works. Everything it
produces is a hypothesis that needs folding validation and then a wet-lab assay.

---

## 0. The one-paragraph version

A binder starts as a *blurry* sequence — every position is a mixture of all 20
amino acids at once. Structure predictors are asked "does this bind?", and
because the whole thing is differentiable, we learn which direction to nudge each
position to improve the answer. Repeat a few hundred times, then sharpen the
blur into real amino acids. Because you run many independent attempts from
different random starts, you get a population of candidates, not one answer.

---

## 1. Before you start

You need three things on disk. Two of them are one-time-per-target.

| Thing | Command | Time | Reuse |
|---|---|---|---|
| Target FASTA | you write it | — | forever |
| Target MSA | `msa-search.sbatch` | 20 min – 2 h | forever |
| Target structure | `predict_target.py` | ~3 min | forever, and only needed for BoltzGen |

Everything runs on the cluster through `sbatch`. **Never run these on the login
node** — they get killed, and it slows the machine down for everyone.

### 1.1 Write your target FASTA

Put the sequence in `targets/yourtarget.fasta`:

```
>MYTARGET
MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQ...
```

**Trim it to the domain a binder can actually reach.** This matters more than
anything else you will do. For a membrane protein, use only the extracellular
part. Including a transmembrane helix gives the model a greasy segment with no
membrane to sit in, and it will fold it into the core and distort everything
around it. UniProt's "Topology" section tells you the boundaries.

Worked example — human DIO3 (UniProt P55073) is 304 residues:

| Residues | Region | Include? |
|---|---|---|
| 1–44 | cytoplasmic | no |
| 45–67 | transmembrane | **no** |
| 68–304 | extracellular | **yes** |

So `targets/dio3_ecd.fasta` contains residues 68–304 only.

**Non-standard amino acids.** The models only know the standard 20. If your
sequence has `U` (selenocysteine), `O` (pyrrolysine), or ambiguity codes
(`B`,`Z`,`J`,`X`), they are automatically swapped for the closest standard
residue and the substitution is printed. `U → C` is the usual one and is
chemically reasonable — selenocysteine is the selenium version of cysteine — but
**it does change the chemistry at that position**, so do not draw conclusions
about catalysis from a model built this way.

### 1.2 Build the MSA (once per target)

An MSA is a pile of evolutionarily related sequences. Structure predictors are
much more accurate with one. This searches a local database, so nothing is sent
to an external server.

```bash
cd /n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup/mosaic
mkdir -p logs
sbatch --export=ALL,TARGET_FASTA=/full/path/targets/yourtarget.fasta,TARGET_NAME=yourtarget \
       singularity/msa-search.sbatch
```

Result: `mosaic_setup/msa/yourtarget.a3m`.

Time scales steeply with length — a 40-residue query took 22 minutes; a
237-residue one took nearly 2 hours. Check it found something:

```bash
grep "unique sequences" logs/msa-<jobid>.out
```

A few thousand is healthy. If it says **"only the query — no homologs found"**,
your target has no known relatives and predictions will be much less reliable.
That is a real biological finding, not an error.

### 1.3 Predict the target structure (only if using BoltzGen)

```bash
sbatch singularity/predict-target.sbatch    # see §5 for the BoltzGen route
```

Check `pLDDT: mean` in the log. Above 70 is confident, below 50 means the model
does not know what your target looks like — treat downstream designs with
suspicion.

---

## 2. Launch a design campaign

```bash
sbatch --array=0-15 \
  --export=ALL,\
TARGET_FASTA=/full/path/targets/yourtarget.fasta,\
TARGET_MSA=/full/path/msa/yourtarget.a3m,\
BINDER_LENGTH=80,\
MODELS=boltz2+af2,\
BATCH=2,\
SOFT_STEPS=100,\
SHARP_STEPS=25,\
OUT_DIR=/full/path/designs/yourtarget \
  singularity/campaign.sbatch
```

That gives **16 tasks × 2 designs = 32 candidates**, one GPU each, about 15
minutes per task.

> **Two traps that produce wrong results silently.**
>
> 1. **Use `+` between model names, never a comma.** `MODELS=boltz2,af2` is
>    silently truncated to `boltz2` by the scheduler and you get a single-model
>    run that looks fine. This cost us a whole 8-GPU experiment.
> 2. **Always use full absolute paths.** Relative paths resolve differently
>    inside the container.
>
> After launching, confirm what actually ran:
> ```bash
> grep "structure backends" logs/campaign-<jobid>_0.out
> ```
> It must list every model you asked for.

---

## 3. The knobs that matter

Ordered by how much they change your result.

### `BINDER_LENGTH` — how long a binder to design
Typical 60–100. Shorter is easier to make and cheaper to compute; longer gives
more surface to bind with. 80 is a reasonable default.

### `MODELS` — which structure predictors must agree
This is the central idea. Asking one model produces designs that exploit that
model's blind spots. Asking two means a design must satisfy both.

| Name | What it is | Speed | Notes |
|---|---|---|---|
| `boltz2` | Boltz-2, current-generation co-folding | fast | best all-round default |
| `boltz1` | Boltz-1, previous generation | slowest | mainly for comparison |
| `af2` | AlphaFold2-multimer | medium | **cannot use MSAs here** — see below |
| `of3` | OpenFold3 | medium | independent lineage, good for consensus |
| `protenix` | Protenix Mini | **fastest** (3×) | best for quick iteration |

Recommended: `boltz2` alone while exploring, `boltz2+af2` or `boltz2+of3` for
real campaigns. More models = proportionally more time (adding AF2 took the soft
stage from 373 s to 630 s).

**AF2 caveat:** mosaic's AF2 interface does not accept MSAs. If you include
`af2`, its view of the target is single-sequence even though the others get the
full MSA. That is a property of the wrapper, not a setting.

### `SOFT_STEPS` / `SHARP_STEPS` — how long to optimize
`SOFT_STEPS` explores while the sequence is still blurry; `SHARP_STEPS` forces it
into real amino acids. 100/25 is the default. Fewer than ~50 soft steps and the
design barely leaves its random start. More than ~200 gives diminishing returns.

### `BATCH` — designs per GPU
Independent attempts running side by side on one GPU. They share the expensive
one-time compilation, so `BATCH=2` is much cheaper than two separate jobs.
2–4 is tested; higher is untested and may exhaust GPU memory.

### `--array=0-N` — how many GPUs
Each task is one GPU with a different random start. Total designs = tasks ×
`BATCH`. Add `%8` (e.g. `--array=0-63%8`) to cap how many run at once.

### Cysteine
Off by default. Free cysteines in a small binder tend to form wrong disulfides
and aggregate, so the alphabet excludes them entirely — cysteine is not
penalized, it is impossible. Pass `--no-no-cys` to `run_design.py` if you
genuinely want them.

---

## 4. Reading your results

Each task writes to `OUT_DIR`:

- `designs_seed<N>.fasta` — sequences, best first
- `designs_seed<N>.json` — the same plus loss values and settings used

Rank everything:

```bash
cd /full/path/designs/yourtarget
python3 -c "
import json,glob
rows=[]
for p in glob.glob('*.json'):
    d=json.load(open(p))
    rows += [(r['loss'], r['sequence'], d['seed']) for r in d['results']]
for loss,seq,seed in sorted(rows)[:10]:
    print(f'{loss:7.3f}  seed{seed}  {seq}')
"
```

### What the loss means

**Lower is better, and that is nearly all it means.** It is a weighted sum of
"is the binder touching the target", "is the binder compact", and "is this a
plausible sequence". It is *not* an affinity, a Kd, or a probability.

**Never compare losses between runs with different `MODELS`.** A two-model run
must satisfy two critics, so it will show a higher number than a one-model run on
the same design quality. Comparing them is meaningless.

### Sanity checks before you get excited

1. **Composition.** Real proteins are a mix. A design that is 40% valine has
   gamed the objective. Compare against natural frequencies.
2. **Spread.** If every design across every seed is nearly identical, the
   optimizer collapsed and you have one answer, not thirty.
3. **Fold them properly.** Design uses 1 recycling step for speed. Re-fold your
   top candidates at higher settings before believing anything.

---

## 5. The BoltzGen route (optional)

Two different kinds of model live here:

- **Scoring models** (Boltz, AF2, OpenFold3, Protenix) answer "does this
  sequence bind?" and can be optimized against.
- **Generative models** (BoltzGen, Proteina) *propose* structures directly. They
  cannot be optimized against — there is no question to differentiate.

So BoltzGen is used *before* the optimizer, to suggest starting points:

```bash
# 1. target structure (needs the CIF, not just sequence)
sbatch singularity/predict-target.sbatch
# 2. generate proposals — fast, ~2.5 s per design
sbatch singularity/boltzgen.sbatch
# 3. refine them
sbatch --array=0-3 --export=ALL,INIT_FASTA=/full/path/designs/.../boltzgen_designs.fasta,... \
       singularity/campaign.sbatch
```

**Honest status: this does not currently help.** BoltzGen's raw output looks
*more* protein-like than optimized designs, but refinement keeps only ~15% of the
seed sequence — the optimizer walks away and lands where it would have anyway.
Losses were not meaningfully better.

If you want to make this work, the lever is to stop the optimizer destroying the
proposal: fewer soft steps, or anchor most positions and let only the interface
move. Watch **sequence identity to the seed**, not loss — that is the number that
reveals what is happening.

---

## 6. When something goes wrong

| Symptom | Meaning | Fix |
|---|---|---|
| Job vanishes instantly | scheduler rejected it | `sacct -j <id>` — usually wrong partition/account |
| `AssocMaxSubmitJobLimit` | wrong account | must be `kempner_bsabatini_lab` |
| Only one model in the log | comma in `MODELS` | use `+` |
| `RESOURCE_EXHAUSTED` / OOM | too much on one GPU | lower `BATCH`, or fewer models |
| MSA says "only the query" | no homologs exist | expected for orphans; predictions weaker |
| All designs identical | collapsed run | more seeds, check the loss is not saturated |
| Sequences look like garbage | too few steps, or objective imbalance | raise `SOFT_STEPS`; check composition |

Every job writes a log to `logs/`. The first thing to read is always:

```bash
grep -E "structure backends|target MSA|seeded|best loss" logs/campaign-<jobid>_0.out
```

Those four lines tell you what the job actually did, which is not always what you
thought you asked for.

---

## 7. Working with Claude Code

This repo ships helper skills. In a Claude Code session, just say what you want:

- *"set up a new target from this UniProt ID"* → `new-target`
- *"launch a design campaign against X"* → `launch-campaign`
- *"how did my designs turn out?"* → `inspect-designs`

They handle path wiring, the `+` separator trap, and the post-launch checks that
catch silent misconfiguration.
