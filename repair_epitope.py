#!/usr/bin/env python
"""Rebase crop-relative epitopes in manifests written before the seqid fix.

Any proposal set generated with ``--hotspots`` was generated against a *cropped*
target, and the epitope stored in its manifest counts positions within that crop
rather than in the target sequence. Refining against it aims the contact loss at
the wrong residues, quietly.

The rebase is exact, not a re-derivation: the crop is emitted in sorted pocket
order, so stored index ``i`` is ``params["pocket"][i]``. Nothing is re-read from
the CIFs, so this needs neither gemmi nor a GPU — only a python new enough for
the repo (the cluster's system python3 is 3.6, hence the wrapper below).

    ./singularity/mosaic-exec.sh python repair_epitope.py designs pipelines
    ./singularity/mosaic-exec.sh python repair_epitope.py designs --apply

Sets that were generated against the full target are stamped as already correct
and left otherwise untouched, so a whole tree can be passed in one go.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proposals as P  # noqa: E402


def classify(m: dict) -> tuple[str, str]:
    """(status, explanation) for one manifest dict."""
    ep = m.get("epitope_idx") or []
    pocket = (m.get("params") or {}).get("pocket") or []
    if m.get("epitope_frame") == P.EPITOPE_FRAME_TARGET:
        return "ok", "already stamped target-relative"
    if not ep:
        return "stamp", "no epitope to rebase"
    if not pocket:
        return "stamp", f"{len(ep)} residues, uncropped target — indices already correct"
    if max(ep) >= len(pocket):
        # Cannot be crop-relative: it names a residue the crop does not have.
        return "stamp", (f"{len(ep)} residues but max index {max(ep)} exceeds the "
                         f"{len(pocket)}-residue crop — already rebased")
    return "rebase", f"{len(ep)} crop-relative residues over a {len(pocket)}-residue pocket"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="directories to scan for manifests")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite manifests; without it nothing is modified")
    a = ap.parse_args()

    manifests = sorted({p for r in a.roots
                        for p in Path(r).rglob(P.MANIFEST)})
    if not manifests:
        print("no manifests found")
        return 0

    counts = {"ok": 0, "stamp": 0, "rebase": 0}
    for man in manifests:
        try:
            m = json.loads(man.read_text())
        except Exception as e:
            print(f"  SKIP  {man}: {e}")
            continue
        status, why = classify(m)
        counts[status] += 1
        print(f"  {status.upper():6s} {man}\n         {why}")

        if status == "ok" or not a.apply:
            continue

        if status == "rebase":
            pocket = m["params"]["pocket"]
            before = m["epitope_idx"]
            after = P.epitope_crop_to_target(before, pocket)
            m["epitope_idx"] = after
            m.setdefault("notes", []).append(
                f"epitope rebased from crop-relative to target-relative "
                f"({len(before)} indices over a {len(pocket)}-residue pocket)"
            )
            print(f"         {before[:6]}... -> {after[:6]}...")
        m["epitope_frame"] = P.EPITOPE_FRAME_TARGET
        man.write_text(json.dumps(m, indent=2))

    verb = "rewrote" if a.apply else "would rewrite"
    print(f"\n{len(manifests)} manifests: {counts['ok']} already stamped, "
          f"{verb} {counts['stamp']} stamp-only and {counts['rebase']} rebased")
    if not a.apply and (counts["stamp"] or counts["rebase"]):
        print("re-run with --apply to write the changes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
