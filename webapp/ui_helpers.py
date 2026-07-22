"""Reusable widgets and parsing shared by the tabs."""

from __future__ import annotations

import streamlit as st


def parse_ranges(text: str, lo: int, hi: int) -> tuple[list[int], list[str]]:
    """Parse "10-20, 35, 40-44" into 1-based positions, with problems reported.

    Returns (positions, errors). Positions are 1-based here and converted to
    0-based only at the point of use — see the note in binding_site().
    """
    text = (text or "").strip()
    if not text:
        return [], []
    out: list[int] = []
    errs: list[str] = []
    for chunk in text.replace(" ", "").split(","):
        if not chunk:
            continue
        try:
            if "-" in chunk.lstrip("-"):
                a, b = chunk.split("-", 1)
                a_i, b_i = int(a), int(b)
                if a_i > b_i:
                    a_i, b_i = b_i, a_i
                rng = list(range(a_i, b_i + 1))
            else:
                rng = [int(chunk)]
        except ValueError:
            errs.append(f"could not parse {chunk!r}")
            continue
        for v in rng:
            if not (lo <= v <= hi):
                errs.append(f"{v} is outside 1-{hi}")
            elif v not in out:
                out.append(v)
    return sorted(out), errs


def show_selection(sequence: str, positions: list[int], label: str) -> None:
    """Echo the actual residues chosen.

    This exists because the indices are easy to get wrong in a way nothing else
    would catch: a trimmed target renumbers everything (DIO3's ECD starts at
    global residue 68, so ECD position 1 is K68), and a silently wrong epitope
    produces a perfectly plausible campaign aimed at the wrong surface.
    """
    if not positions:
        return
    shown = " ".join(f"{sequence[p - 1]}{p}" for p in positions[:40])
    more = f"  (+{len(positions) - 40} more)" if len(positions) > 40 else ""
    st.caption(f"{label} — {len(positions)} residues: {shown}{more}")


def widget(key: str, name: str, spec: dict, current):
    """One control, driven by the catalog's declared type."""
    t = spec["type"]
    label = name.replace("_", " ")
    help_ = spec.get("help")
    if t == "int":
        return st.number_input(label, min_value=spec.get("min", 0),
                               max_value=spec.get("max", 10000),
                               value=int(current if current is not None
                                         else spec.get("default", 0)),
                               step=1, key=key, help=help_)
    if t == "float":
        return st.number_input(label, min_value=float(spec.get("min", 0.0)),
                               max_value=float(spec.get("max", 1e6)),
                               value=float(current if current is not None
                                           else spec.get("default", 0.0)),
                               step=0.05, format="%.3f", key=key, help=help_)
    if t == "bool":
        return st.checkbox(label, value=bool(current if current is not None
                                             else spec.get("default", False)),
                           key=key, help=help_)
    if t == "choice":
        opts = spec["options"]
        cur = current if current in opts else spec.get("default", opts[0])
        return st.selectbox(label, opts, index=opts.index(cur), key=key, help=help_)
    if t in ("optint", "optfloat"):
        # Optional numerics need an explicit "unset": None is meaningful, e.g.
        # target_radius=None means "no target radius", not zero.
        use = st.checkbox(f"set {label}", value=current is not None, key=key + "_on")
        if not use:
            return None
        if t == "optint":
            return st.number_input(label, min_value=spec.get("min", 0),
                                   max_value=spec.get("max", 10000),
                                   value=int(current or spec.get("min", 0)),
                                   step=1, key=key, help=help_)
        return st.number_input(label, min_value=float(spec.get("min", 0.0)),
                               max_value=float(spec.get("max", 1e6)),
                               value=float(current or spec.get("min", 0.0)),
                               step=0.5, format="%.2f", key=key, help=help_)
    if t == "idxlist":
        # Rendered by the Binding site section instead — see binding_site().
        return current
    return current


def params_block(prefix: str, params_spec: dict, current: dict,
                 skip: tuple[str, ...] = ()) -> dict:
    """Lay a parameter set out in columns."""
    out = dict(current)
    names = [n for n in params_spec if n not in skip]
    if not names:
        return out
    cols = st.columns(min(4, len(names)))
    for i, pname in enumerate(names):
        with cols[i % len(cols)]:
            out[pname] = widget(f"{prefix}_{pname}", pname, params_spec[pname],
                                current.get(pname, params_spec[pname].get("default")))
    return out
