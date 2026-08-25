"""`proposals.epitope_from_complex` must count residues in the TARGET's frame.

The bug these guard against: the function used to index by the residue's
position in the chain it was handed. That is only the same thing as a target
position when the chain is the whole target. BoltzGen is handed a *cropped*
target whenever hotspots are used — the crop is how hotspots are implemented,
since BoltzGen has no attractive hotspot term — and its output complexes carry
only the pocket residues, with their original sparse seqids. Enumerating those
yields positions within the crop, which the campaign then read as positions in
the full target. Silent: no error, plausible numbers, wrong residues.

Everything here is synthetic and runs in a second; `test_archived_cropped_run`
is the real-data anchor and skips when the archive is not present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import proposals as P  # noqa: E402

gemmi = pytest.importorskip("gemmi")

BINDER_N = 12


def _residue(name: str, num: int, xyz, element: str = "C", atom: str = "CA"):
    r = gemmi.Residue()
    r.name = name
    r.seqid = gemmi.SeqId(num, " ")
    at = gemmi.Atom()
    at.name = atom
    at.element = gemmi.Element(element)
    at.pos = gemmi.Position(*xyz)
    r.add_atom(at)
    return r


def _complex(tmp_path: Path, target_seqids, contacts, waters=(), name="c.cif") -> Path:
    """A two-chain complex where exactly `contacts` are within 8 A of the binder.

    Chain A is the binder, numbered 1..BINDER_N like anything a generator builds.
    Chain B is the target, carrying whatever seqids it is given, so a crop can be
    expressed simply by leaving gaps in them.
    """
    st = gemmi.Structure()
    st.spacegroup_hm = "P 1"
    model = gemmi.Model("1")

    binder = gemmi.Chain("A")
    for i in range(1, BINDER_N + 1):
        binder.add_residue(_residue("ALA", i, (0.0, 3.8 * (i - 1), 0.0)))
    model.add_chain(binder)

    target = gemmi.Chain("B")
    near = far = 0
    for s in target_seqids:
        if s in contacts:
            target.add_residue(_residue("ALA", s, (4.0, 0.4 * near, 0.0)))
            near += 1
        else:
            target.add_residue(_residue("ALA", s, (200.0, 0.4 * far, 0.0)))
            far += 1
    # Solvent sits on the same auth chain and is numbered past the polymer, so
    # it is the classic source of indices that run off the end of the sequence.
    for j, w in enumerate(waters):
        target.add_residue(_residue("HOH", w, (4.0, 0.4 * j, 0.0), element="O",
                                    atom="O"))
    model.add_chain(target)

    st.add_model(model)
    st.setup_entities()
    path = tmp_path / name
    st.make_mmcif_document().write_file(str(path))
    return path


# ------------------------------------------------- the frame the indices are in


def test_full_target_is_unchanged_by_the_seqid_fix(tmp_path):
    """The uncropped case, where enumeration and seqid agree.

    Guards the other direction: the fix must be inert on every run that was
    already correct, which is all of them that did not use hotspots.
    """
    cif = _complex(tmp_path, target_seqids=list(range(1, 41)),
                   contacts={5, 6, 7, 20})
    assert P.epitope_from_complex(cif, binder_length=BINDER_N) == [4, 5, 6, 19]


def test_cropped_target_indices_are_positions_in_the_full_target(tmp_path):
    """The regression. A crop keeps original seqids, so 174 must stay 174.

    Enumerating this chain would call the last residue 5; it is residue 174 of
    the target, and steering a contact loss at 5 aims ~170 residues away.
    """
    crop = [1, 2, 3, 40, 41, 42, 100, 174]
    cif = _complex(tmp_path, target_seqids=crop, contacts={40, 41, 174})
    assert P.epitope_from_complex(cif, binder_length=BINDER_N) == [39, 40, 173]


def test_cropped_epitope_stays_inside_the_pocket_it_came_from(tmp_path):
    """The invariant generate_boltzgen.py now enforces at write time.

    A binder cannot touch a residue the model was never shown, so the epitope
    is always a subset of the crop. Under the old indexing it was not, which is
    what made the fault detectable after the fact.
    """
    crop = [1, 2, 3, 40, 41, 42, 100, 174]
    cif = _complex(tmp_path, target_seqids=crop, contacts={40, 100})
    ep = P.epitope_from_complex(cif, binder_length=BINDER_N)
    assert {e + 1 for e in ep} <= set(crop)


# ------------------------------------------------------ things that are not the
#                                                         target's own residues


def test_solvent_does_not_reach_the_epitope(tmp_path):
    """Waters are numbered past the polymer and would index off the end.

    A JAX gather clamps an out-of-range index instead of raising, so this used
    to optimise against the last target column and report an ordinary number.
    Generated CIFs carry no solvent, which is why it never bit the pipeline,
    but the function is also pointed at deposited structures.
    """
    cif = _complex(tmp_path, target_seqids=list(range(1, 41)),
                   contacts={5}, waters=[401, 402, 403])
    ep = P.epitope_from_complex(cif, binder_length=BINDER_N)
    assert ep == [4]
    assert all(i < 40 for i in ep)


def test_target_length_bounds_the_result(tmp_path):
    """The explicit belt to the solvent removal's braces."""
    cif = _complex(tmp_path, target_seqids=[1, 2, 3, 100],
                   contacts={1, 2, 3, 100})
    assert P.epitope_from_complex(cif, binder_length=BINDER_N,
                                  target_length=50) == [0, 1, 2]


def test_target_chain_length_ignores_solvent(tmp_path):
    cif = _complex(tmp_path, target_seqids=list(range(1, 41)), contacts={5},
                   waters=[401, 402])
    assert P.target_chain_length(cif, "B") == 40


# --------------------------------------------------- which chain is which


def test_binder_is_found_when_the_crop_is_shorter_than_the_binder(tmp_path):
    """"Shortest chain is the binder" inverts on exactly the cropped runs.

    An 8 A pocket around 11 hotspots on the DIO3 ECD is 46 residues against an
    80-residue binder. Numbering is the reliable tell: a generated binder is
    always 1..N with no gaps, a crop is not.
    """
    crop = [1, 2, 3, 40, 41, 100]           # 6 residues vs a 12-residue binder
    cif = _complex(tmp_path, target_seqids=crop, contacts={40, 41})
    assert P.epitope_from_complex(cif, binder_length=BINDER_N) == [39, 40]
    # and the gap heuristic gets there even without being told the length
    assert P.epitope_from_complex(cif) == [39, 40]


# ------------------------------------------------------------ repairing on disk


def test_rebase_maps_crop_positions_onto_the_target():
    pocket = [1, 2, 3, 40, 41, 42, 100, 174]
    assert P.epitope_crop_to_target([0, 3, 7], pocket) == [0, 39, 173]


def test_rebase_refuses_indices_that_are_not_crop_relative():
    """Rebasing an already-correct epitope would corrupt it, so it raises."""
    with pytest.raises(ValueError, match="outside the 3-residue crop"):
        P.epitope_crop_to_target([0, 173], [1, 2, 3])


def test_rebase_is_the_inverse_of_the_old_enumeration():
    pocket = [1, 2, 3, 40, 41, 42, 100, 174]
    true_target_idx = [39, 40, 173]
    old = [pocket.index(i + 1) for i in true_target_idx]      # what used to ship
    assert P.epitope_crop_to_target(old, pocket) == true_target_idx


# --------------------------------------------------------------- merging pools


def _write_set(d: Path, epitope, frame):
    d.mkdir(parents=True, exist_ok=True)
    P.write(d, P.ProposalSet(
        name=d.name, generator="boltzgen", target_name="t", target_fasta="",
        target_structure=None, binder_length=BINDER_N, n_designs=1,
        sequences=["A" * BINDER_N], epitope_idx=epitope, epitope_frame=frame))
    return d


def test_merge_refuses_to_union_two_coordinate_systems(tmp_path):
    """A pool mixing a cropped BoltzGen set with a full-target Proteina set.

    This is how the fault spread past the generator: the union looked entirely
    normal while holding indices counted two different ways.
    """
    parts = [_write_set(tmp_path / "part_0", [1, 2], P.EPITOPE_FRAME_TARGET),
             _write_set(tmp_path / "part_1", [1, 2], "")]
    with pytest.raises(SystemExit, match="epitope_frame"):
        P.merge_sets(parts, tmp_path / "pool", "pool")


def test_merge_carries_the_frame_through(tmp_path):
    parts = [_write_set(tmp_path / "p0", [1, 2], P.EPITOPE_FRAME_TARGET),
             _write_set(tmp_path / "p1", [7], P.EPITOPE_FRAME_TARGET)]
    merged = P.merge_sets(parts, tmp_path / "pool", "pool")
    assert merged.epitope_idx == [1, 2, 7]
    assert merged.epitope_frame == P.EPITOPE_FRAME_TARGET
    assert P.read(tmp_path / "pool").epitope_frame == P.EPITOPE_FRAME_TARGET


def test_read_treats_a_legacy_manifest_as_unstamped(tmp_path):
    d = tmp_path / "legacy"
    d.mkdir()
    (d / P.MANIFEST).write_text(json.dumps({
        "name": "legacy", "generator": "boltzgen", "target_name": "t",
        "target_fasta": "", "target_structure": None, "binder_length": 80,
        "n_designs": 1, "sequences": ["A"], "epitope_idx": [0, 1]}))
    assert P.read(d).epitope_frame == ""


# ------------------------------------------- the guard on the way into a loss


def _manifest(tmp_path: Path, **over) -> Path:
    d = tmp_path / over.pop("dirname", "src")
    d.mkdir(parents=True, exist_ok=True)
    m = {"name": "s", "generator": "boltzgen", "target_name": "t",
         "target_fasta": "", "target_structure": None, "binder_length": 80,
         "n_designs": 1, "sequences": ["A"], "epitope_idx": [0, 5, 9]}
    m.update(over)
    (d / P.MANIFEST).write_text(json.dumps(m))
    return d


@pytest.fixture(scope="module")
def node():
    return pytest.importorskip("pipeline_node")


def test_loss_guard_passes_a_stamped_epitope(tmp_path, node):
    src = _manifest(tmp_path, epitope_frame=P.EPITOPE_FRAME_TARGET)
    assert node._epitope_for_loss(src, 100) == [0, 5, 9]


def test_loss_guard_allows_an_unstamped_uncropped_manifest(tmp_path, node):
    """Every pre-fix run that did not use hotspots is correct and must keep working."""
    src = _manifest(tmp_path, params={"pocket": []})
    assert node._epitope_for_loss(src, 100) == [0, 5, 9]


def test_loss_guard_refuses_an_unstamped_cropped_manifest(tmp_path, node):
    """The exact combination that cost three campaign rounds."""
    src = _manifest(tmp_path, params={"pocket": [10, 20, 30, 40, 50]})
    with pytest.raises(SystemExit, match="crop-relative"):
        node._epitope_for_loss(src, 100)


def test_loss_guard_refuses_indices_past_the_target(tmp_path, node):
    """JAX would clamp these and report an ordinary-looking number."""
    src = _manifest(tmp_path, epitope_idx=[0, 500],
                    epitope_frame=P.EPITOPE_FRAME_TARGET)
    with pytest.raises(SystemExit, match="outside the 100-residue target"):
        node._epitope_for_loss(src, 100)


def test_loss_guard_is_quiet_when_there_is_nothing_to_carry(tmp_path, node):
    assert node._epitope_for_loss(tmp_path / "missing", 100) == []
    assert node._epitope_for_loss(_manifest(tmp_path, epitope_idx=[]), 100) == []


def test_target_length_reads_a_fasta_or_a_literal_sequence(tmp_path, node):
    f = tmp_path / "t.fasta"
    f.write_text(">t\nACDEFGHIKL\nMNPQRST\n")
    assert node._target_seq_length(str(f)) == 17
    assert node._target_seq_length("ACDEFG") == 6
    assert node._target_seq_length("") == 0


# ------------------------------------------------------------ real data anchor

ARCHIVE = (Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup")
           / "_archive_20260724/designs/dio3_run1k/gen/part_0")


@pytest.mark.skipif(not ARCHIVE.is_dir(), reason="archived cropped run not present")
def test_archived_cropped_run_is_recovered(tmp_path):
    """The run the bug was found on: 46-residue crop of the 237-residue ECD.

    Recomputing from its CIFs must put every epitope residue inside the pocket
    and recover all 11 hotspots, and the offline rebase of the shipped manifest
    must agree with that recomputation.
    """
    man = json.loads((ARCHIVE / "manifest.json").read_text())
    pocket, hotspots = man["params"]["pocket"], man["params"]["hotspots"]
    cifs = sorted(ARCHIVE.glob("design_*.cif"))[:25]

    ep, _ = P.consensus_epitope(cifs, binder_length=man["binder_length"],
                                cutoff=man["params"]["contact_cutoff"],
                                target_length=237)
    assert {e + 1 for e in ep} <= set(pocket)
    assert all(h - 1 in ep for h in hotspots)
    assert set(P.epitope_crop_to_target(man["epitope_idx"], pocket)) == set(ep)
