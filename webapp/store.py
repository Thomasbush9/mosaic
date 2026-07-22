"""Filesystem reads: targets, MSAs, campaigns, designs. No Streamlit here.

Kept pure so it can be tested and reused from a CLI. Mirrors the split ProtForge
uses between `results.py` and `results_tab.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SETUP = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup")
REPO = SETUP / "mosaic"


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


def read_fasta(path: Path) -> str:
    return "".join(
        ln.strip() for ln in path.read_text().splitlines()
        if ln.strip() and not ln.startswith(">")
    ).upper()


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
    for fa in sorted((SETUP / "targets").glob("*.fasta")):
        a3m = SETUP / "msa" / f"{fa.stem}.a3m"
        cif = SETUP / "targets" / f"{fa.stem}.cif"
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
    for f in d.glob("*.json"):
        try:
            if "results" in json.loads(f.read_text()):
                return True
        except Exception:
            continue
    return False


def list_campaigns() -> list[str]:
    root = SETUP / "designs"
    if not root.exists():
        return []
    return sorted(p.name for p in root.glob("*") if p.is_dir() and is_campaign_dir(p))


def load_designs(campaign: str) -> list[dict]:
    """Flatten a campaign into one row per design, best first."""
    rows = []
    for p in sorted((SETUP / "designs" / campaign).glob("*.json")):
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
    return rows


def campaign_config(campaign: str) -> dict | None:
    """The resolved config a campaign ran with, if it recorded one."""
    for p in sorted((SETUP / "designs" / campaign).glob("config_seed*.json")):
        try:
            return json.loads(p.read_text())
        except Exception:
            continue
    for p in sorted((SETUP / "designs" / campaign).glob("designs_seed*.json")):
        try:
            return json.loads(p.read_text()).get("config")
        except Exception:
            continue
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
