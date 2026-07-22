---
name: inspect-designs
description: Analyse the output of a design campaign — rank candidates, sanity-check for degenerate optimization, compare runs, and say honestly whether the results mean anything. Use when the user asks "how did my designs turn out", "show me the best binders", "did that campaign work", or wants two campaigns compared.
---

# Inspect design results

## Rank the candidates

```bash
python3 -c "
import json,glob,sys
rows=[]
for p in glob.glob(sys.argv[1]+'/*.json'):
    d=json.load(open(p))
    rows += [(r['loss'], r['sequence'], d['seed'], ','.join(d['models'])) for r in d['results']]
for loss,seq,seed,m in sorted(rows)[:15]:
    print(f'{loss:8.3f}  seed{seed:<3} {m:16s} {seq}')
print(f'\n{len(rows)} designs total')
" <OUT_DIR>
```

## Sanity checks — run these before calling anything a result

**1. Did it optimize the thing it claimed to?**
```bash
grep -hE "structure backends|target MSA|n_msa|seeded" logs/campaign-<jobid>_*.out | sort -u
```
`n_msa 1` with an MSA configured means the alignment was silently dropped.
Missing models means the `--export` comma trap.

**2. Composition.** A design that is 40% one residue has gamed the objective
rather than solved it. Compare to natural frequencies; expect de novo designs to
be enriched in E/K/L/A/D but not dominated by any single residue.

**3. Diversity.** Near-identical sequences across independent seeds mean the run
collapsed — you have one answer, not N. Check pairwise identity.

**4. Cysteines.** Should be zero unless `--no-no-cys` was passed. Any cysteine is
a bug in the decode path, not a design choice.

`singularity/make_report_plots.py` produces loss-distribution and composition
figures; run it inside the container.

## Comparing two campaigns

**Losses are only comparable when `MODELS` is identical.** Weights are split
evenly across backends, so a two-model run must satisfy two critics and will show
a higher number at equal design quality. Comparing across different `MODELS` is
meaningless — say so rather than reporting a spurious winner.

For seeded (BoltzGen-initialized) runs, the number that matters is **identity to
the seed**, not loss:

```bash
python3 -c "
import json,glob,sys,statistics as st
seeds=[l.strip() for l in open(sys.argv[2]) if not l.startswith('>')]
ids=[]
for p in sorted(glob.glob(sys.argv[1]+'/*.json')):
    d=json.load(open(p))
    for r in d['results']:
        s=seeds[(d['seed']*len(d['results'])+r['trajectory'])%len(seeds)]
        ids.append(sum(a==b for a,b in zip(s,r['sequence']))/len(s)*100)
print(f'mean identity to seed: {st.mean(ids):.1f}%  (random ~5%)')
" <OUT_DIR> <boltzgen_designs.fasta>
```

Low identity (~15%) means the optimizer discarded the proposal and the seeding
bought nothing. That is the observed behaviour with default settings — report it
plainly rather than presenting seeded runs as an improvement.

## What to tell the user

Be direct about what these numbers do and do not support:

- Loss is a weighted heuristic — contact geometry, compactness, sequence
  plausibility. **Not** an affinity, a Kd, or a probability of binding.
- Nothing here validates a binder. Candidates need re-folding at higher
  recycling settings and then a wet-lab assay.
- Designs are optimized at `recycling_steps=1` for speed; re-fold the top few at
  4–20 before drawing conclusions.

If the results look degenerate, say so and suggest the cause (too few steps,
objective imbalance, collapsed run) rather than presenting a ranked list as
though it were meaningful.

## Next steps worth offering

- Re-fold top candidates at high recycling and report pLDDT / interface PAE.
- Rerun the best sequence through a *different* single model to check it is not
  exploiting one predictor's blind spot.
- Raise the ESM-C weight if sequences look chemically implausible.
