"""Local MSA support: use a precomputed ColabFold ``.a3m`` instead of the server.

By default every MSA-capable backend fetches from the public ColabFold server
(``https://api.colabfold.com``) during featurization, and only ESMFold2 and
OpenDDE cache the result — Boltz and OpenFold3 write to a ``TemporaryDirectory``
and therefore re-query on *every* featurization. A 64-trajectory design array
against one target hits that public service 64 times for an identical query.

Set ``TargetChain.msa_path`` to an ``.a3m`` produced by
``singularity/msa-search.sbatch`` (GPU mmseqs2 against the local ColabFold
database) and no network call is made at all.

Only the target chain ever needs this. Every ``binder_features()`` implementation
hardcodes ``use_msa=False`` for the binder, and the loss code depends on it —
a de novo binder has no homologs by construction. So this is a
precompute-once-per-target problem, not a per-iteration one.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

__all__ = ["msa_cache_dir", "a3m_for_chain", "sequence_cache_path"]


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
