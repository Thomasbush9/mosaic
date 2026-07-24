"""Filesystem reads: targets, MSAs, campaigns, designs. No Streamlit here.

Kept pure so it can be tested and reused from a CLI. Mirrors the split ProtForge
uses between `results.py` and `results_tab.py`.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path

# The working directory is per-session and set by app.py on every rerun, so
# switching projects switches the whole view (targets, MSAs, designs, configs).
# REPO stays fixed — it is where the job scripts live, not where data goes.
REPO = Path(__file__).resolve().parent.parent
_WORKDIR = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup")

# --------------------------------------------------------------------------
# Memoization
#
# Streamlit reruns this whole module's callers on every widget interaction, and
# list_targets() (which reads every MSA in full to count depth) is called at the
# top of three tabs. On a login node — where FASRC's arbiter kills CPU-heavy
# processes — re-reading multi-MB a3m files on every click is exactly what gets
# the app killed. So the file reads are memoized, content-addressed by
# (path, mtime, size): a file that has not changed is never re-read, and a job
# that writes a new result changes the signature so the next Refresh picks it up.
#
# Kept as a plain process-local dict rather than st.cache_data so store.py stays
# free of any Streamlit dependency and works unchanged from a CLI or a test.
# --------------------------------------------------------------------------
_CACHE: dict = {}


def _sig(path: Path):
    try:
        s = path.stat()
        return (s.st_mtime_ns, s.st_size)
    except OSError:
        return None


def _by_file(fn):
    """Memoize a single-path reader on that file's (mtime, size)."""
    @functools.wraps(fn)
    def wrap(path, *a, **k):
        path = Path(path)
        key = (fn.__name__, str(path), _sig(path), a, tuple(sorted(k.items())))
        if key not in _CACHE:
            _CACHE[key] = fn(path, *a, **k)
        return _CACHE[key]
    return wrap


def _glob_sig(root: Path, pattern: str):
    """A signature of a directory's matching files, for cache invalidation.

    Changes when a file is added, removed, or rewritten — so a cached scan is
    reused across reruns but a Refresh after a job finishes recomputes.
    """
    out = []
    for p in sorted(root.glob(pattern)):
        out.append((p.name, _sig(p)))
    return tuple(out)


def set_workdir(path) -> None:
    global _WORKDIR
    _WORKDIR = Path(path)


def workdir() -> Path:
    return _WORKDIR


def sub(name: str) -> Path:
    """A subdirectory of the working directory, created on demand."""
    d = _WORKDIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class Target:
    name: str
    fasta: Path
    sequence: str
    msa: Path | None
    msa_depth: int | None
    structure: Path | None

    @property
    def length(self) -> int:
        return len(self.sequence)


@_by_file
def read_fasta(path: Path) -> str:
    return "".join(
        ln.strip() for ln in path.read_text().splitlines()
        if ln.strip() and not ln.startswith(">")
    ).upper()


@_by_file
def msa_depth(path: Path) -> int:
    """Unique sequences in an a3m. Depth is the honest measure of MSA value —
    a file can be large and still contain only near-duplicates."""
    seen = set()
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith(">"):
                seen.add(ln)
    return len(seen)


def list_targets() -> list[Target]:
    out = []
    for fa in sorted(sub("targets").glob("*.fasta")):
        a3m = sub("msa") / f"{fa.stem}.a3m"
        cif = sub("targets") / f"{fa.stem}.cif"
        out.append(Target(
            name=fa.stem, fasta=fa, sequence=read_fasta(fa),
            msa=a3m if a3m.exists() else None,
            msa_depth=msa_depth(a3m) if a3m.exists() else None,
            structure=cif if cif.exists() else None,
        ))
    return out


def is_campaign_dir(d: Path) -> bool:
    """Scored designs, as opposed to BoltzGen proposals which share the tree
    with a different schema."""
    key = ("is_campaign_dir", str(d), _glob_sig(d, "*.json"))
    if key in _CACHE:
        return _CACHE[key]
    result = False
    for f in d.glob("*.json"):
        try:
            if "results" in json.loads(f.read_text()):
                result = True
                break
        except Exception:
            continue
    _CACHE[key] = result
    return result


def _campaign_path(name: str) -> Path:
    """Resolve a campaign name to its directory.

    Two roots feed the Results tab: single-stage campaigns live under
    designs/<name>, while a pipeline DAG writes each node under
    pipelines/<pipeline>/<node>. A name that carries a slash is the latter and
    is joined onto the working directory verbatim; a bare name is a designs/
    campaign. This is what lets an optimize/hallucinate node's output show up in
    Results without copying it out of the pipeline tree.
    """
    if "/" in name:
        return _WORKDIR / name
    return sub("designs") / name


def _has_design_json(d: Path) -> bool:
    """A design_set: at least one designs_seed*.json (loss + sequence rows).

    Deliberately narrower than is_campaign_dir — a screen node's screen.json
    also has a `results` key but a different schema, and belongs in the screens
    list, not here.
    """
    return next(iter(d.glob("designs_seed*.json")), None) is not None


def list_campaigns() -> list[str]:
    out: list[str] = []
    root = sub("designs")
    if root.exists():
        out += [p.name for p in root.glob("*")
                if p.is_dir() and is_campaign_dir(p)]
    # Pipeline node outputs that are design_sets (optimize / hallucinate nodes).
    proot = _WORKDIR / "pipelines"
    if proot.exists():
        for node in proot.glob("*/*"):
            if node.is_dir() and _has_design_json(node):
                out.append(f"pipelines/{node.parent.name}/{node.name}")
    return sorted(out)


def load_designs(campaign: str) -> list[dict]:
    """Flatten a campaign into one row per design, best first."""
    root = _campaign_path(campaign)
    key = ("load_designs", str(root), _glob_sig(root, "*.json"))
    if key in _CACHE:
        return _CACHE[key]
    rows = []
    for p in sorted(root.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if "results" not in d:
            continue
        models = "+".join(d.get("models", []))
        for r in d["results"]:
            rows.append({
                "loss": r["loss"],
                "seed": d.get("seed"),
                "trajectory": r.get("trajectory"),
                "models": models,
                "length": len(r["sequence"]),
                "sequence": r["sequence"],
            })
    rows.sort(key=lambda r: r["loss"])
    _CACHE[key] = rows
    return rows


def campaign_config(campaign: str) -> dict | None:
    """The resolved config a campaign ran with, if it recorded one."""
    root = _campaign_path(campaign)
    # A pipeline node writes its resolved objective to config.json.
    for p in sorted(root.glob("config_seed*.json")) + \
            ([root / "config.json"] if (root / "config.json").is_file() else []):
        try:
            return json.loads(p.read_text())
        except Exception:
            continue
    for p in sorted(root.glob("designs_seed*.json")):
        try:
            return json.loads(p.read_text()).get("config")
        except Exception:
            continue
    return None


def list_screens() -> list[str]:
    """Directories containing screen.json (refold-and-rank results)."""
    out: list[str] = []
    root = sub("designs")
    if root.exists():
        out += [p.name for p in root.glob("*")
                if p.is_dir() and (p / "screen.json").exists()]
    proot = _WORKDIR / "pipelines"
    if proot.exists():
        for node in proot.glob("*/*"):
            if (node / "screen.json").is_file():
                out.append(f"pipelines/{node.parent.name}/{node.name}")
    return sorted(out)


def load_screen(name: str) -> dict | None:
    f = _campaign_path(name) / "screen.json"
    if not f.is_file():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


# Natural amino-acid frequencies (SwissProt averages, %), for composition
# comparison. Reference only — de novo designs legitimately differ.
NATURAL_AA = {
    "A": 8.3, "R": 5.5, "N": 4.1, "D": 5.5, "C": 1.4, "Q": 3.9, "E": 6.7,
    "G": 7.1, "H": 2.3, "I": 5.9, "L": 9.7, "K": 5.8, "M": 2.4, "F": 3.9,
    "P": 4.7, "S": 6.6, "T": 5.4, "W": 1.1, "Y": 2.9, "V": 6.9,
}


def composition(seqs: list[str]) -> dict[str, float]:
    allseq = "".join(seqs)
    if not allseq:
        return {}
    return {a: allseq.count(a) / len(allseq) * 100 for a in NATURAL_AA}


def pairwise_identity(seqs: list[str], cap: int = 40) -> list[float]:
    """Identity between every pair, as percentages.

    Diversity is the check that a run has not collapsed: many seeds converging
    on one sequence means you have a single answer, not a population.
    """
    s = seqs[:cap]
    out = []
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            a, b = s[i], s[j]
            n = min(len(a), len(b))
            if n:
                out.append(sum(x == y for x, y in zip(a[:n], b[:n])) / n * 100)
    return out


def flags(rows: list[dict]) -> list[str]:
    """Problems worth surfacing before anyone reads a ranked list as a result."""
    msgs = []
    if not rows:
        return ["No designs found."]
    seqs = [r["sequence"] for r in rows]
    comp = composition(seqs)
    if comp:
        top, pct = max(comp.items(), key=lambda kv: kv[1])
        if pct > 25:
            msgs.append(
                f"Degenerate composition: {top} is {pct:.0f}% of all residues. "
                "The optimizer is gaming the objective rather than solving it."
            )
    cys = sum(s.count("C") for s in seqs)
    if cys:
        msgs.append(
            f"{cys} cysteines present — expected zero unless no_cys was "
            "disabled. Check the decode path."
        )
    ids = pairwise_identity(seqs)
    if ids and sum(ids) / len(ids) > 80:
        msgs.append(
            f"Mean pairwise identity {sum(ids)/len(ids):.0f}% — the run "
            "collapsed to one answer rather than a population."
        )
    if len({r["models"] for r in rows}) > 1:
        msgs.append(
            "Mixed model settings in one directory: losses are NOT comparable "
            "across different models, since each backend adds a critic."
        )
    return msgs
