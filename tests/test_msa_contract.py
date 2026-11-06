"""Every MSA-capable backend must read the caller's `.a3m`, or refuse to run.

The bug these guard against was silent: `TargetChain.msa_path` was honoured by
AF2, Boltz-1 and Boltz-2 and ignored by OpenFold3, Protenix, ESMFold2 and
OpenDDE, which queried the public ColabFold server instead. Nothing failed and
nothing appeared in the output; the models simply folded against different
alignments, and a model-vs-model comparison silently measured that difference
too. So these tests assert on where the alignment came from, and patch
`socket.socket.connect` to turn any stray network call into a failure rather
than a slow test.
"""

import socket

import pytest

from mosaic.msa import (
    ALLOW_SERVER_ENV,
    as_of3_a3m,
    as_protenix_msa_dir,
    require_msa,
)
from mosaic.structure_prediction import TargetChain

TARGET = (
    "DDNRLCTLASLKAVWHGQKLDFFKQAHEGGPAPNSEVVLPDGFQSQHILDYAQGNRPLVLNFGSCTCPPFM"
)
BINDER = "M" + "A" * 39
DEPTH = 12


@pytest.fixture
def a3m(tmp_path):
    """A small unpaired a3m for TARGET, in the shape msa-search.sbatch emits:
    a ColabFold-style numeric first header, query on line 2."""
    lines = [">101", TARGET]
    for i in range(DEPTH - 1):
        lines += [f">hit{i}", TARGET[:-1] + "ACDEFGHIKLM"[i % 11]]
    p = tmp_path / "target.a3m"
    p.write_text("\n".join(lines) + "\n")
    return p


@pytest.fixture
def no_network(monkeypatch):
    def _fail(self, addr):
        raise AssertionError(f"network call attempted to {addr}")

    monkeypatch.setattr(socket.socket, "connect", _fail)


def depth_of(path):
    return sum(1 for line in open(path) if line.startswith(">"))


# --- the contract -----------------------------------------------------------


def test_no_msa_requested_returns_none():
    binder = TargetChain(sequence=BINDER, use_msa=False)
    assert require_msa(binder, model="test") is None


def test_msa_path_is_resolved(a3m):
    chain = TargetChain(sequence=TARGET, use_msa=True, msa_path=str(a3m))
    assert require_msa(chain, model="test") == a3m


def test_missing_msa_path_raises():
    """The whole point: no silent fallback to the server."""
    chain = TargetChain(sequence=TARGET, use_msa=True)
    with pytest.raises(ValueError, match="msa_path is not set"):
        require_msa(chain, model="test")


def test_missing_a3m_file_raises(tmp_path):
    chain = TargetChain(
        sequence=TARGET, use_msa=True, msa_path=str(tmp_path / "nope.a3m")
    )
    with pytest.raises(FileNotFoundError):
        require_msa(chain, model="test")


def test_server_opt_in_warns(monkeypatch):
    monkeypatch.setenv(ALLOW_SERVER_ENV, "1")
    chain = TargetChain(sequence=TARGET, use_msa=True)
    with pytest.warns(UserWarning, match="NOT comparable"):
        assert require_msa(chain, model="test") is None


# --- per-backend staging ----------------------------------------------------


def test_of3_stages_local_a3m(a3m, monkeypatch, tmp_path, no_network):
    """OF3 derives its representative id from the a3m's *parent directory*, so
    staging has to put the file somewhere unique per alignment — handing it the
    a3m in place would key every target under a dir called `msa/` to one id."""
    monkeypatch.setenv("MOSAIC_MSA_CACHE", str(tmp_path / "cache"))
    staged = as_of3_a3m(a3m)
    assert depth_of(staged) == DEPTH
    assert staged.name == "colabfold_main.a3m"
    assert staged.parent.name != a3m.parent.name


def test_protenix_stages_local_a3m(a3m, monkeypatch, tmp_path, no_network):
    """Protenix wants a directory of `non_pairing.a3m` + `pairing.a3m`. Our
    alignments are unpaired, which is what AF2 and Boltz consume here, so the
    paired slot gets a query-only alignment rather than a second alignment the
    other backends never saw."""
    monkeypatch.setenv("MOSAIC_MSA_CACHE", str(tmp_path / "cache"))
    d = as_protenix_msa_dir(a3m, TARGET)
    assert depth_of(d / "non_pairing.a3m") == DEPTH
    assert depth_of(d / "pairing.a3m") == 1


def test_staging_is_reused_not_recopied(a3m, monkeypatch, tmp_path, no_network):
    """Scoring runs re-featurize per design; staging must be paid once per
    target, not once per sequence."""
    monkeypatch.setenv("MOSAIC_MSA_CACHE", str(tmp_path / "cache"))
    first = as_of3_a3m(a3m)
    stamp = first.stat().st_mtime_ns
    assert as_of3_a3m(a3m) == first
    assert first.stat().st_mtime_ns == stamp


# --- backends that build their own input ------------------------------------


def test_boltz2_emits_msa_key_and_refuses_without_one(a3m, monkeypatch, tmp_path):
    monkeypatch.setenv("MOSAIC_MSA_CACHE", str(tmp_path / "cache"))
    from mosaic.models.boltz2 import chain_yaml

    chain = TargetChain(sequence=TARGET, use_msa=True, msa_path=str(a3m))
    yaml = chain_yaml("B", chain)
    assert "msa:" in yaml and "empty" not in yaml

    # Omitting the key is exactly what makes boltz query the server.
    with pytest.raises(ValueError, match="msa_path is not set"):
        chain_yaml("B", TargetChain(sequence=TARGET, use_msa=True))


def test_boltz2_binder_gets_empty_msa():
    from mosaic.models.boltz2 import chain_yaml

    assert "msa: empty" in chain_yaml("A", TargetChain(sequence=BINDER, use_msa=False))


@pytest.mark.slow
def test_of3_featurization_uses_local_a3m(a3m, monkeypatch, tmp_path, no_network):
    """End to end through the real OF3 query-set builder (imports jopenfold3)."""
    monkeypatch.setenv("MOSAIC_MSA_CACHE", str(tmp_path / "cache"))
    from mosaic.models.of3 import _build_query_set, _compute_msas

    binder = TargetChain(sequence=BINDER, use_msa=False)
    target = TargetChain(sequence=TARGET, use_msa=True, msa_path=str(a3m))

    qs = _compute_msas(
        _build_query_set([binder, target]),
        chains=[binder, target],
        msa_dir=tmp_path / "msas",
    )
    chains = list(qs.queries["design"].chains)

    (tgt,) = [c for c in chains if c.sequence == TARGET]
    assert [str(p) for p in tgt.main_msa_file_paths] == [str(as_of3_a3m(a3m))]
    assert depth_of(tgt.main_msa_file_paths[0]) == DEPTH

    # The binder must never be handed the target's alignment. It gets OF3's
    # own query-only placeholder, which lands under `dummy/` as an .npz —
    # MsaComputationSettings.msa_file_format defaults to npz, not a3m.
    (bnd,) = [c for c in chains if c.sequence == BINDER]
    (bnd_path,) = bnd.main_msa_file_paths
    assert bnd_path != as_of3_a3m(a3m)
    assert "dummy" in bnd_path.parts
