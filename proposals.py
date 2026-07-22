"""Proposal sets: the handoff between generation and refinement.

A *proposal set* is what a generative model produces and what a campaign
consumes. Standardising it is the point — before this, generation wrote a FASTA
and the refinement stage read only the letters, discarding everything the
generator actually knew: the backbone, and crucially *where on the target it
chose to bind*. Refinement then re-derived a pose from scratch under a generic
"contact the target" term, which is why refined designs kept only ~15% of their
seed. The stages were concatenated, not connected.

Layout on disk::

    <designs>/<name>/
        manifest.json          what this is, how it was made, derived epitope
        proposals.fasta        sequences, one per design
        design_000.cif         binder + target complex, one per design
        ...

``manifest.json`` carries the epitope so the campaign can aim at the same
interface the generator chose, rather than wandering off to another surface.

Importable without mosaic (the webapp reads manifests host-side); gemmi is only
needed by the contact calculation, which runs inside the container.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MANIFEST = "manifest.json"
FASTA = "proposals.fasta"


@dataclass
class ProposalSet:
    name: str
    generator: str
    target_name: str
    target_fasta: str
    target_structure: str | None
    binder_length: int
    n_designs: int
    sequences: list[str] = field(default_factory=list)
    # 0-based indices into the TARGET chain — the same convention
    # BinderTargetContact.epitope_idx uses (structure_prediction.py:281).
    epitope_idx: list[int] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        d = dict(self.__dict__)
        return json.dumps(d, indent=2)


def write(out_dir: Path, ps: ProposalSet) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / MANIFEST).write_text(ps.to_json())
    with (out_dir / FASTA).open("w") as f:
        for i, s in enumerate(ps.sequences):
            f.write(f">{ps.name}_{i:03d}\n{s}\n")
    return out_dir / MANIFEST


def read(path: Path) -> ProposalSet | None:
    """Load a proposal set from its directory or manifest path."""
    p = Path(path)
    if p.is_dir():
        p = p / MANIFEST
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    known = ProposalSet.__dataclass_fields__.keys()
    return ProposalSet(**{k: v for k, v in d.items() if k in known})


def list_sets(designs_dir: Path) -> list[tuple[str, ProposalSet]]:
    out = []
    for d in sorted(Path(designs_dir).glob("*")):
        ps = read(d) if d.is_dir() else None
        if ps is not None:
            out.append((d.name, ps))
    return out


def epitope_from_complex(cif_path: Path, binder_length: int | None = None,
                         cutoff: float = 8.0) -> list[int]:
    """Target residues the generated binder actually touches.

    This is the connective tissue: a generative model places the binder
    somewhere specific, and without extracting that, refinement has no idea
    where it was meant to go.

    Chains are identified by LENGTH, not by name. BoltzGen's YAML declares the
    binder as chain B, but the structure writer emits the binder as chain A and
    the target as chain B — trusting the declared names silently reads the wrong
    chain and yields a meaningless epitope.

    Returns 0-based indices into the target chain, matching
    ``BinderTargetContact.epitope_idx``. Any-atom within ``cutoff`` counts as
    contact; the point is to aim the refinement, not to make a final call.
    """
    import gemmi

    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()
    st.remove_hydrogens()
    model = st[0]
    chains = [ch for ch in model if len(ch) > 0]
    if len(chains) < 2:
        return []

    if binder_length is not None:
        binder = min(chains, key=lambda c: abs(len(c) - binder_length))
        target = max((c for c in chains if c.name != binder.name), key=len)
    else:
        ordered = sorted(chains, key=len)
        binder, target = ordered[0], ordered[-1]

    ns = gemmi.NeighborSearch(st, cutoff).populate()
    hits: set[int] = set()
    tgt_pos = {res.seqid.num: i for i, res in enumerate(target)}
    for res in binder:
        for atom in res:
            for m in ns.find_atoms(atom.pos, "\0", radius=cutoff):
                cra = m.to_cra(model)
                if cra.chain.name != target.name:
                    continue
                i = tgt_pos.get(cra.residue.seqid.num)
                if i is not None:
                    hits.add(i)
    return sorted(hits)


def consensus_epitope(cif_paths: list[Path], min_fraction: float = 0.3,
                      **kw) -> tuple[list[int], list[str]]:
    """Epitope shared across a set of designs.

    A single design's contacts are noisy. Residues touched by at least
    ``min_fraction`` of designs are the interface the generator consistently
    chose, which is a better target for refinement than any one pose.
    """
    from collections import Counter

    counts: Counter[int] = Counter()
    used = 0
    notes: list[str] = []
    for p in cif_paths:
        try:
            idx = epitope_from_complex(Path(p), **kw)
        except Exception as e:  # a malformed CIF should not sink the run
            notes.append(f"{Path(p).name}: {e}")
            continue
        if idx:
            counts.update(idx)
            used += 1
    if not used:
        notes.append("no usable complexes — epitope not derived")
        return [], notes
    threshold = max(1, int(round(min_fraction * used)))
    ep = sorted(i for i, c in counts.items() if c >= threshold)
    notes.append(f"epitope from {used} complexes, residues contacted by "
                 f">={threshold} of them")
    return ep, notes
