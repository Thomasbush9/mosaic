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

#: ``ProposalSet.epitope_frame`` value meaning "0-based into the full target
#: sequence". Anything else must not be fed to a loss.
EPITOPE_FRAME_TARGET = "target"


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
    # 0-based indices into the FULL TARGET SEQUENCE — the same convention
    # BinderTargetContact.epitope_idx uses (structure_prediction.py:281).
    epitope_idx: list[int] = field(default_factory=list)
    # Which coordinate system epitope_idx is in. "target" is the only frame a
    # campaign can consume. "" means unstamped, i.e. written before the frame
    # was recorded: correct for a full target, but crop-relative (and therefore
    # wrong) for any set generated with hotspots. See EPITOPE_FRAME_TARGET.
    epitope_frame: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        d = dict(self.__dict__)
        return json.dumps(d, indent=2)


def write(out_dir: Path, ps: ProposalSet) -> Path:
    out_dir = Path(out_dir)
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


def _seqids(chain) -> list[int]:
    return [res.seqid.num for res in chain]


def _is_contiguous_from_one(chain, n: int) -> bool:
    """A chain the generator built itself: exactly ``n`` residues, numbered 1..n."""
    return len(chain) == n and _seqids(chain) == list(range(1, n + 1))


def _pick_binder_and_target(chains, binder_length: int | None):
    """Split the complex into (binder, target).

    Not by chain NAME: BoltzGen's YAML declares the binder as chain B, but the
    structure writer emits the binder as chain A and the target as chain B, so
    trusting the declared names reads the wrong chain.

    Not by LENGTH alone either, which is what the original heuristic did. A
    cropped target can be *shorter* than the binder — an 8 A pocket around 11
    hotspots on the 237-residue DIO3 ECD is 46 residues against an 80-residue
    binder — so "shortest chain is the binder" inverts on exactly the runs this
    function most needs to get right.

    The reliable tell is numbering. The binder is generated de novo and is always
    numbered 1..binder_length with no gaps; a cropped target carries the sparse
    seqids of the residues that survived the crop.
    """
    if binder_length is not None:
        exact = [c for c in chains if _is_contiguous_from_one(c, binder_length)]
        if len(exact) == 1:
            binder = exact[0]
        else:
            binder = min(chains, key=lambda c: (abs(len(c) - binder_length), c.name))
    else:
        gapped = [c for c in chains if _seqids(c) != list(range(1, len(c) + 1))]
        if len(gapped) == 1:
            target = gapped[0]
            return max((c for c in chains if c.name != target.name), key=len), target
        binder = min(chains, key=len)
    target = max((c for c in chains if c.name != binder.name), key=len)
    return binder, target


def epitope_from_complex(cif_path: Path, binder_length: int | None = None,
                         cutoff: float = 8.0,
                         target_length: int | None = None) -> list[int]:
    """Target residues the generated binder actually touches.

    This is the connective tissue: a generative model places the binder
    somewhere specific, and without extracting that, refinement has no idea
    where it was meant to go.

    Returns 0-based indices into the FULL target sequence, matching
    ``BinderTargetContact.epitope_idx``. Any-atom within ``cutoff`` counts as
    contact; the point is to aim the refinement, not to make a final call.

    Indices come from ``seqid``, not from the residue's position in the chain.
    That distinction is the whole correctness argument. BoltzGen is handed a
    *cropped* target when hotspots are used (there is no attractive hotspot
    term, so the crop is the mechanism), and its output complexes carry only the
    pocket residues — with their original, sparse seqids preserved. Enumerating
    that chain yields positions within the crop, which then get read as
    positions in the full target: on a real run, "pocket residue 7" was steered
    at "domain residue 7", ~100 residues away. Reading seqid is correct for both
    cases, since an uncropped target is numbered 1..N contiguously.

    Solvent and ligands are dropped before the search. They are numbered outside
    the polymer's range, so they would otherwise contribute indices past the end
    of the target sequence, and a JAX gather clamps those silently rather than
    raising. Pass ``target_length`` to enforce that bound explicitly.
    """
    import gemmi

    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()
    st.remove_ligands_and_waters()
    st.remove_hydrogens()
    model = st[0]
    chains = [ch for ch in model if len(ch) > 0]
    if len(chains) < 2:
        return []

    binder, target = _pick_binder_and_target(chains, binder_length)

    ns = gemmi.NeighborSearch(st, cutoff).populate()
    hits: set[int] = set()
    for res in binder:
        for atom in res:
            for m in ns.find_atoms(atom.pos, "\0", radius=cutoff):
                cra = m.to_cra(model)
                if cra.chain.name != target.name:
                    continue
                i = cra.residue.seqid.num - 1
                if i < 0 or (target_length is not None and i >= target_length):
                    continue
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


def epitope_crop_to_target(epitope_idx: list[int], pocket1: list[int]) -> list[int]:
    """Rebase a crop-relative epitope onto the full target.

    For repairing manifests written before ``epitope_from_complex`` indexed by
    seqid. The crop is emitted in sorted pocket order, so a stored index ``i``
    is the ``i``-th pocket residue: ``pocket1[i]`` 1-based, hence ``- 1``.

    Exact rather than approximate — it reproduces what the fixed contact
    calculation returns from the same CIFs — but only valid on a set that was
    actually generated against ``pocket1``. Raises rather than guessing if an
    index does not fit, since a silent mis-map is what caused this in the
    first place.
    """
    pocket = sorted(pocket1)
    bad = [i for i in epitope_idx if not 0 <= i < len(pocket)]
    if bad:
        raise ValueError(
            f"indices {bad} are outside the {len(pocket)}-residue crop — this "
            "epitope is not crop-relative, so rebasing it would corrupt it"
        )
    return sorted({pocket[i] - 1 for i in epitope_idx})


def target_chain_length(cif_path: Path, chain_id: str = "A") -> int:
    """Length of the target's polymer chain, for bounds-checking an epitope.

    The highest seqid rather than the residue count: the two agree for a target
    written by ``predict_target.py``, but only the former stays meaningful if a
    structure is missing residues.
    """
    import gemmi

    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()
    st.remove_ligands_and_waters()
    model = st[0]
    names = [c.name for c in model]
    chain = model[chain_id] if chain_id in names else max(model, key=len)
    return max(_seqids(chain), default=0)


def parse_positions(text: str) -> list[int]:
    """"95-110,143" -> sorted 1-based positions."""
    out: list[int] = []
    for chunk in (text or "").replace(" ", "").split(","):
        if not chunk:
            continue
        if "-" in chunk.lstrip("-"):
            a, b = chunk.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def as_res_index(positions: list[int]) -> str:
    """Format 1-based positions for BoltzGen's include/res_index.

    Its parser is 1-based and spells ranges with '..' rather than '-'
    (boltzgen/data/parse/schema.py:646-664).
    """
    if not positions:
        return ""
    parts, start, prev = [], positions[0], positions[0]
    for v in positions[1:] + [None]:
        if v is not None and v == prev + 1:
            prev = v
            continue
        parts.append(f"{start}" if start == prev else f"{start}..{prev}")
        if v is not None:
            start = prev = v
    return ",".join(parts)


def pocket_residues(cif_path, hotspots1: list[int], shell: float,
                    chain_id: str = "A") -> list[int]:
    """Target residues within `shell` of any hotspot, as 1-based positions.

    Cropping to bare hotspots would hand the model a handful of disconnected
    residues; a proximity shell gives it a coherent surface patch to design
    against.
    """
    import gemmi

    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()
    st.remove_hydrogens()
    model = st[0]
    names = [c.name for c in model]
    chain = model[chain_id] if chain_id in names else max(model, key=len)
    residues = list(chain)
    hot = [residues[i - 1] for i in hotspots1 if 1 <= i <= len(residues)]
    if not hot:
        return []
    keep = set(hotspots1)
    for i, res in enumerate(residues, start=1):
        if i in keep:
            continue
        for a1 in res:
            done = False
            for h in hot:
                for a2 in h:
                    if a1.pos.dist(a2.pos) <= shell:
                        keep.add(i)
                        done = True
                        break
                if done:
                    break
            if done:
                break
    return sorted(keep)


def merge_sets(part_dirs: list[Path], out_dir: Path, name: str) -> "ProposalSet":
    """Combine sharded proposal parts (part_0, part_1, ...) into one set.

    Sequences and CIFs are concatenated and renumbered; the epitope is the union
    of the parts (each part already consensus-filtered within its own shard).
    """
    import shutil

    parts = [read(d) for d in part_dirs]
    parts = [p for p in parts if p is not None]
    if not parts:
        raise SystemExit("no readable parts to merge")

    # Unioning epitopes only means anything if every part counts residues the
    # same way. A pool that mixes a cropped BoltzGen set with a full-target
    # Proteina set would otherwise merge two coordinate systems into one list
    # and look entirely normal doing it.
    frames = {p.epitope_frame for p in parts if p.epitope_idx}
    if len(frames) > 1:
        raise SystemExit(
            f"parts disagree on epitope_frame ({sorted(frames)}) — refusing to "
            "union epitopes across coordinate systems; repair the parts first "
            "(see repair_epitope.py)"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    seqs: list[str] = []
    epi: set[int] = set()
    k = 0
    for pdir, ps in zip(part_dirs, parts):
        for cif in sorted(Path(pdir).glob("design_*.cif")):
            shutil.copy(cif, out_dir / f"design_{k:03d}.cif")
            k += 1
        seqs.extend(ps.sequences)
        epi.update(ps.epitope_idx)

    base = parts[0]
    merged = ProposalSet(
        name=name, generator=base.generator, target_name=base.target_name,
        target_fasta=base.target_fasta, target_structure=base.target_structure,
        binder_length=base.binder_length, n_designs=len(seqs),
        sequences=seqs, epitope_idx=sorted(epi),
        epitope_frame=(frames.pop() if frames else base.epitope_frame),
        params={**base.params, "merged_from": [str(d) for d in part_dirs]},
        notes=[f"merged from {len(parts)} shards, {len(seqs)} designs total"],
    )
    write(out_dir, merged)
    return merged
