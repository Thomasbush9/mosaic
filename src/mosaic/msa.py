"""Local MSA support: use a precomputed ColabFold ``.a3m`` instead of the server.

Set ``TargetChain.msa_path`` to an ``.a3m`` produced by
``singularity/msa-search.sbatch`` (GPU mmseqs2 against the local ColabFold
database) and no network call is made at all. ``require_msa`` makes that the
only supported path: every MSA-capable backend routes through it, so all of them
read the same alignment or the run fails.

That used to be untrue in a way nothing surfaced. AF2, Boltz-1 and Boltz-2 read
``msa_path``; OpenFold3, Protenix, ESMFold2 and OpenDDE ignored it and queried
the public ColabFold server (``https://api.colabfold.com``) themselves. On the
DIO3 target that meant half the models folding against a 3,032-sequence local
alignment and ESMFold2 against a 3,643-sequence server one, with nothing in the
output to say so. Boltz and OpenFold3 additionally wrote to a
``TemporaryDirectory``, re-querying that public service on *every*
featurization — 416 identical queries for one 416-design scoring run.

Only the target chain ever needs this. Every ``binder_features()`` implementation
hardcodes ``use_msa=False`` for the binder, and the loss code depends on it —
a de novo binder has no homologs by construction. So this is a
precompute-once-per-target problem, not a per-iteration one.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import warnings
from pathlib import Path

__all__ = [
    "msa_cache_dir",
    "a3m_for_chain",
    "sequence_cache_path",
    "require_msa",
    "as_of3_a3m",
    "as_protenix_msa_dir",
    "ALLOW_SERVER_ENV",
]

# Opt back in to per-backend server fetches. Off by default: a run where some
# models read the local a3m and others query ColabFold produces a model-vs-model
# comparison confounded by a change of MSA, which is invisible in the output.
ALLOW_SERVER_ENV = "MOSAIC_ALLOW_MSA_SERVER"


def msa_cache_dir() -> Path:
    """Where derived MSA files are written.

    Matches the location ``models/esmfold2.py`` and ``models/opendde.py`` use for
    ``MOSAIC_MSA_CACHE``, so a single bind covers both them and this module.
    """
    return Path(
        os.environ.get("MOSAIC_MSA_CACHE", "~/.cache/mosaic/msa")
    ).expanduser()


def sequence_cache_path(sequence: str) -> Path:
    """The path ESMFold2/OpenDDE look up for ``sequence``.

    Those backends key their cache on ``sha256(sequence)[:16]``
    (``models/esmfold2.py:85``, ``models/opendde.py:72``). Writing an ``.a3m``
    here makes them use it with no code change and no network access.
    """
    digest = hashlib.sha256(sequence.encode()).hexdigest()[:16]
    return msa_cache_dir() / f"{digest}.a3m"


def a3m_for_chain(a3m_path: str | Path, chain_id: str) -> Path:
    """Return an ``.a3m`` whose first header is ``chain_id``.

    Boltz keys ``chain_to_msa`` off the first header line of the a3m, but
    ColabFold writes its own internal index there (e.g. ``>101``). If they
    disagree the MSA is silently dropped and the model runs single-sequence —
    which looks like a quality problem, not a plumbing bug, so it is worth
    getting right. (Learned from ProtForge's ``organize_msa_outputs.py``, which
    solves the same problem for its Snakemake pipeline.)

    The chain id is positional in mosaic — ``binder_features`` puts the binder at
    ``A``, pushing the first target to ``B`` — so the required header depends on
    how the chain is used, not on the file. Rather than mutate the stored a3m,
    write a per-chain copy into the MSA cache, keyed by content and chain id so
    it is derived once and reused.
    """
    a3m_path = Path(a3m_path).expanduser()
    if not a3m_path.is_file():
        raise FileNotFoundError(f"MSA not found: {a3m_path}")

    text = a3m_path.read_text()
    key = hashlib.sha256(f"{a3m_path.resolve()}:{chain_id}".encode()).hexdigest()[:16]
    out = msa_cache_dir() / f"chain_{chain_id}_{key}.a3m"
    if out.is_file():
        return out

    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith(">"):
            lines[i] = f">{chain_id}\n"
            break
    else:
        raise ValueError(f"No header line in {a3m_path} — not an a3m?")

    out.parent.mkdir(parents=True, exist_ok=True)
    # Write via a temporary file: concurrent array tasks derive the same path,
    # and a half-written a3m read by another task is worse than a slow one.
    tmp = out.with_suffix(f".a3m.tmp.{os.getpid()}")
    tmp.write_text("".join(lines))
    os.replace(tmp, out)
    return out


def _atomic_copy(src: Path, dest: Path) -> None:
    """Copy ``src`` to ``dest`` so concurrent array tasks never read a partial
    file. Derived MSA artefacts are content-addressed, so two tasks racing to
    build the same path are writing identical bytes; the rename makes the
    loser's work harmless rather than corrupting the winner's."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.tmp.{os.getpid()}")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)


def _staged_dir(a3m_path: Path, kind: str) -> Path:
    """A stable per-(a3m, backend) directory under the MSA cache.

    Keyed by the resolved source path so the staging cost is paid once per
    target rather than once per design — the OF3 matrix run re-featurizes for
    every sequence, and copying a 3k-sequence a3m 416 times is pure waste.
    """
    key = hashlib.sha256(str(a3m_path.resolve()).encode()).hexdigest()[:16]
    return msa_cache_dir() / kind / key


def require_msa(chain, *, model: str) -> Path | None:
    """The ``.a3m`` that ``model`` must read for ``chain``, or None for no MSA.

    Every MSA-capable backend routes through this, so "which alignment did this
    model actually see" has one answer instead of one per wrapper. Backends used
    to disagree silently: AF2, Boltz-1 and Boltz-2 read ``msa_path``, while
    OpenFold3, Protenix, ESMFold2 and OpenDDE ignored it and queried the
    ColabFold server. On one DIO3 run that meant AF2 folding against a
    3,032-sequence local alignment and ESMFold2 against a 3,643-sequence server
    one — a difference that lands in the numbers as if it were a model
    difference.

    Raises rather than falling back to the network, because the fallback is the
    bug: it succeeds, returns plausible structures, and leaves nothing in the
    output to say the alignment changed. Set ``MOSAIC_ALLOW_MSA_SERVER=1`` for
    exploratory work against a target with no precomputed alignment.
    """
    if not chain.use_msa:
        return None

    if chain.msa_path is None:
        if os.environ.get(ALLOW_SERVER_ENV) == "1":
            warnings.warn(
                f"{model}: no msa_path for sequence {chain.sequence[:24]!r}...; "
                f"falling back to the ColabFold server because "
                f"{ALLOW_SERVER_ENV}=1. Scores from this run are NOT comparable "
                "with runs that used a local .a3m.",
                stacklevel=3,
            )
            return None
        raise ValueError(
            f"{model} requested an MSA but TargetChain.msa_path is not set for "
            f"sequence {chain.sequence[:24]!r}... Every model must read the same "
            "alignment or a model-vs-model comparison measures the alignment as "
            "well as the model. Pass msa_path=<a3m> (see "
            "singularity/msa-search.sbatch), set use_msa=False, or set "
            f"{ALLOW_SERVER_ENV}=1 to accept a per-backend server fetch."
        )

    path = Path(chain.msa_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"MSA not found: {path}")
    return path


def as_of3_a3m(a3m_path: str | Path) -> Path:
    """Stage ``a3m_path`` where OpenFold3's MSA parser will accept it.

    OF3 takes ``Chain.main_msa_file_paths`` as a list, but for an ``.a3m`` it
    derives the representative id from the *parent directory* name
    (``core/data/io/sequence/msa.py``, ``msa_file_path.parent.stem``), not the
    filename. Handing it the a3m in place would key every target that happens to
    live under a directory called ``msa/`` to the same rep id. Staging into a
    content-addressed directory makes the id unique per alignment.
    """
    a3m_path = Path(a3m_path).expanduser()
    if not a3m_path.is_file():
        raise FileNotFoundError(f"MSA not found: {a3m_path}")
    out = _staged_dir(a3m_path, "of3") / "colabfold_main.a3m"
    if not out.is_file():
        _atomic_copy(a3m_path, out)
    return out


def as_protenix_msa_dir(a3m_path: str | Path, sequence: str) -> Path:
    """Stage ``a3m_path`` as a Protenix ``precomputed_msa_dir``.

    Protenix wants a directory holding ``non_pairing.a3m`` and ``pairing.a3m``
    (``protenix/data/template/template_featurizer.py``). Alignments produced by
    ``singularity/msa-search.sbatch`` are unpaired (``use_pairing=False``), which
    is also what AF2 and Boltz consume here, so the paired slot gets a
    query-only alignment — the same no-pairing behaviour the other backends
    have, rather than a second alignment they never saw.
    """
    a3m_path = Path(a3m_path).expanduser()
    if not a3m_path.is_file():
        raise FileNotFoundError(f"MSA not found: {a3m_path}")

    out = _staged_dir(a3m_path, "protenix")
    non_pairing, pairing = out / "non_pairing.a3m", out / "pairing.a3m"
    if not non_pairing.is_file():
        _atomic_copy(a3m_path, non_pairing)
    if not pairing.is_file():
        pairing.parent.mkdir(parents=True, exist_ok=True)
        tmp = pairing.with_name(f"pairing.a3m.tmp.{os.getpid()}")
        tmp.write_text(f">query\n{sequence}\n")
        os.replace(tmp, pairing)
    return out
