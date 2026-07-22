---
name: new-target
description: Set up a new binder-design target from a sequence or UniProt ID — trim to the bindable domain, write the FASTA, launch the MSA search, and optionally predict the structure. Use when the user wants to "add a target", "set up X for design", or gives a UniProt accession / raw sequence to design against.
---

# Set up a new design target

Goal: turn "I want to bind protein X" into `targets/<name>.fasta` plus a
finished MSA, with the domain choice made deliberately rather than by accident.

## 1. Get the sequence and topology

If given a UniProt accession, fetch both the sequence and the topology — the
topology is what determines the trim, and guessing it is the most common way to
ruin a campaign before it starts.

```bash
curl -sS "https://rest.uniprot.org/uniprotkb/<ACC>.fasta"
curl -sS "https://rest.uniprot.org/uniprotkb/<ACC>.json?fields=ft_transmem,ft_topo_dom,ft_domain,ft_signal,cc_subcellular_location" \
  | python3 -c "
import json,sys
for f in json.load(sys.stdin).get('features',[]):
    l=f['location']; print(f\"{f['type']:22s} {l['start']['value']:>4}-{l['end']['value']:<4} {f.get('description','')}\")"
```

## 2. Trim to what a binder can reach

**This is the highest-leverage decision in the whole pipeline. Do not skip it,
and state your reasoning to the user.**

- Single-pass membrane protein → keep only the extracellular (or only the
  cytoplasmic) topological domain. Never include the transmembrane span: it is a
  hydrophobic stretch with no membrane in the model, so the predictor buries it
  and distorts the fold around it.
- Signal peptide → drop it, it is not in the mature protein.
- Secreted/soluble protein → usually keep the whole mature chain.
- Very large multi-domain target → consider the single domain of interest;
  cost grows steeply with length, and MSA search grows worse than linearly.

Report the trim explicitly: "residues A–B, which is the X domain; excluded Y
because Z."

## 3. Handle non-standard residues

`TOKENS` is the standard 20. Anything else has no column and must be substituted
**before** the MSA search, so the alignment query matches what the models see.

`U`→`C`, `O`→`K`, `B`→`D`, `Z`→`E`, `J`→`L`, `X`→`A`.

`U`→`C` (selenocysteine→cysteine) is chemically reasonable — Sec is the selenium
analog of Cys, and Sec→Cys mutants are a standard construct — but say plainly
that it changes active-site chemistry, so no catalytic conclusions from the
resulting models.

Write the FASTA with substitutions already applied, and put the trim in the
header, e.g. `>DIO3_ECD_68-304_U170C`.

## 4. Launch the MSA

```bash
cd <repo>
mkdir -p logs
sbatch --export=ALL,TARGET_FASTA=<abs path>,TARGET_NAME=<name> \
       singularity/msa-search.sbatch
```

Absolute paths only. Expect 20 min (short) to 2 h (200+ residues). When it
finishes, report the depth:

```bash
grep -E "unique sequences|WARNING" logs/msa-<jobid>.out
```

A few thousand is healthy. "Only the query" means no homologs — a real
biological result worth telling the user, not an error, but predictions will be
weaker.

## 5. Structure, only if BoltzGen is wanted

BoltzGen conditions on geometry, so it needs a CIF. Scoring-model campaigns do
not.

```bash
<repo>/singularity/mosaic-exec.sh python <repo>/predict_target.py \
  --target <fasta> --target-msa <a3m> --out targets/<name>.cif
```
(run via sbatch, not on the login node). Report mean pLDDT — note mosaic reports
it on **0–1**, so multiply by 100 when talking to the user. Below 70 means a
weak basis for structure-conditioned design.

## Rules

- Never run heavy work on the login node — always `sbatch`.
- Never modify anything under any `ProtForge` directory. Copy what you need.
- Always use absolute paths.
