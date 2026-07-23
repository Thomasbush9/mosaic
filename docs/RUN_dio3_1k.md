# DIO3 binder run — 1000 candidates, generate → screen → optimize

**Date:** 2026-07-22 · **Target:** human DIO3 extracellular domain (UniProt
P55073, residues 68–304, Sec170→Cys) · **Cluster:** Kempner, 4 GPUs per stage

A full pipeline run at scale, and an honest account of what worked.

## The funnel

| Stage | What ran | Result |
|---|---|---|
| **Generate** | 1000 BoltzGen backbones, hotspots on the 35–45 patch, 4 GPUs | ~10 min/GPU |
| **Screen** | fold → ProteinMPNN inverse-fold → AF2 refold → rank all 1000 | **best ipTM 0.821**, median 0.108 |
| **Optimize** | mid-tier (rank 2–25) refined with Boltz-2 + MPNN + ESM-C, judged by AF2 | mean 0.207 → 0.160 |

**The screen produced the binder, not the optimizer.** Scale is why: 8 candidates
had topped out at ipTM 0.22; 1000 reached 0.82.

## What the optimization taught us

Two runs. The first degraded the designs (pLDDT 49, aromatic sludge) by
optimizing a soft-sequence proxy at recycling 1 and walking away from the seed.
The corrected run fixed that — recycling 3, strong MPNN+ESM priors, gentle steps
near the seed, and optimize-with-Boltz-2 / judge-with-AF2:

| | ipTM mean / best | pLDDT |
|---|---|---|
| mid-tier seeds | 0.207 / 0.499 | 80.3 |
| v1 (broken) | 0.182 / 0.500 | 48.7 |
| v2 (corrected) | 0.160 / 0.443 | 78.3 |

The fixes worked on sequence quality (pLDDT held at 78, clean sequences) but
binding still did not improve. The reason is structural: **AF2's interface
confidence is set by the binding pose — the backbone geometry — which
sequence-space gradient descent cannot move.** Optimizing the sequence of a
mediocre BoltzGen backbone cannot make AF2 like a pose it dislikes. Cross-model
validation (optimize Boltz-2, judge AF2) earned its keep by revealing that the
improvement did not transfer — same-model scoring would have hidden it.

This is also why mosaic's own `boltzgen_pipeline.py` stops at screen-and-filter.

## The top candidate — design 358

**ipTM 0.821, pLDDT 87.7, interface PAE 16.4.** 80-residue binder.

```
SAEEIERELREEIERLLEETREQMKGLSVEEATELAQQTMQEIDRLVDEAIERGLPLDRAIELLLEAGERLGELLGEVVE
```

- **A charged amphipathic helical bundle** — 42% charged (D/E/K/R), 39%
  hydrophobic, E-rich (28%). Exactly the profile of a soluble helical mini-binder,
  and the topology BoltzGen was conditioned to make. Zero cysteines (as designed).
- **A large interface** — 49 of 80 binder residues contact the target, and 47
  target residues are engaged. A broad, extended interface is consistent with the
  high ipTM.
- **It binds the catalytic site.** The contacted epitope includes the
  **S100–C101–T102–C103–P104–P105–F106–M107–A108–R109** stretch — the
  CTU(Sec)PP catalytic motif of DIO3, with C103 being the Sec170→Cys catalytic
  residue itself. The binder sits directly over the deiodinase active-site motif,
  plus a second patch around Y187–Y190 and R208–E223.
- **Hotspot conditioning partly steered it** — 7 of the 11 intended 35–45 patch
  residues (N39–A45) are contacted, but the binder found a much larger surface
  than the patch alone, centred on the catalytic motif.

Biologically that is a striking place to land: a binder that engages the
deiodinase catalytic pocket is exactly what you would want as a starting point
for a functional inhibitor. **With the strong caveat that this is a single
computational hypothesis** — ipTM is a fold model's confidence, not an affinity,
and the Sec→Cys substitution changes the active-site chemistry. It needs
orthogonal folding, then wet-lab validation, before any of that is more than a
suggestion.

## Files

`mosaic_setup/dio3_candidates/`:
- `screen_ranked.csv` / `.fasta` — all 1000, ranked, with metrics
- `top20_complexes/` — binder+target CIFs (chain A binder, chain B target)
- `optimized_ranked.csv`, `README.md`

The one to open first is
`top20_complexes/rank01_design358_iptm0.821.cif`.
