"""3D structure views: target with hotspots, and binder+target complexes.

Uses py3Dmol, embedded through streamlit.components. Selection is read-only in
the sense that matters: py3Dmol renders inside an iframe with no channel back to
Python, so a click cannot directly populate a Streamlit widget. Clicking a
residue therefore *labels* it with its number, which you type or paste into the
hotspot box beside the viewer. That is honest about the boundary rather than
pretending at two-way binding.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# Validated categorical palette; hotspots take the warm slot so they read
# clearly against a cool cartoon.
C_TARGET = "#2a78d6"
C_BINDER = "#1baf7a"
C_HOTSPOT = "#eb6834"


def _view_html(pdb_blocks: list[tuple[str, str]], styles: list[dict],
               height: int = 480, clickable: bool = False,
               spin: bool = False) -> str:
    import py3Dmol

    v = py3Dmol.view(width="100%", height=height)
    for i, (fmt, data) in enumerate(pdb_blocks):
        v.addModel(data, fmt)
    v.setStyle({"model": -1}, {})
    for sel, sty in styles:
        v.setStyle(sel, sty)
    if clickable:
        # Label the residue on click. This is the only feedback path available:
        # the iframe cannot write back into Streamlit's session state.
        v.setClickable(
            {}, True,
            "function(atom,viewer){"
            "viewer.addLabel(atom.chain+' '+atom.resn+atom.resi,"
            "{position:atom,backgroundColor:'#eb6834',fontColor:'white',"
            "fontSize:12,backgroundOpacity:0.9});viewer.render();}")
    v.zoomTo()
    if spin:
        v.spin(True)
    return v._make_html()


def _embed(html: str, height: int) -> None:
    """Embed the viewer.

    py3Dmol needs JavaScript, so this needs st.html with
    unsafe_allow_javascript. Note st.iframe is NOT the replacement the
    deprecation notice implies — it takes a URL, not markup.
    """
    if hasattr(st, "html"):
        st.html(html, unsafe_allow_javascript=True)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=False)


def show(html: str, height: int = 480) -> None:
    _embed(html, height + 20)


def target_view(cif_path: Path, hotspots1: list[int] | None = None,
                chain: str = "A", height: int = 480,
                clickable: bool = True) -> str:
    """Target cartoon, hotspot residues highlighted as sticks."""
    data = Path(cif_path).read_text()
    styles = [({}, {"cartoon": {"color": C_TARGET, "opacity": 0.9}})]
    if hotspots1:
        # py3Dmol selects by author residue number; our positions are 1-based
        # indices into the sequence, which match seqid for these files.
        styles.append(({"resi": [str(i) for i in hotspots1]},
                       {"cartoon": {"color": C_HOTSPOT},
                        "stick": {"colorscheme": "orangeCarbon", "radius": 0.25}}))
    return _view_html([("cif", data)], styles, height=height, clickable=clickable)


def complex_view(cif_path: Path, binder_length: int | None = None,
                 epitope1: list[int] | None = None, height: int = 480) -> str:
    """Binder + target, with the contacted epitope highlighted.

    Chains are distinguished by LENGTH, not name: BoltzGen's writer emits the
    binder as chain A and the target as chain B even though its YAML declares
    the binder as B, so keying off names shows the wrong thing.
    """
    import gemmi

    path = Path(cif_path)
    st_ = gemmi.read_structure(str(path))
    st_.setup_entities()
    chains = [c for c in st_[0] if len(c) > 0]
    if len(chains) < 2:
        return target_view(path, epitope1, height=height, clickable=False)
    if binder_length is not None:
        binder = min(chains, key=lambda c: abs(len(c) - binder_length))
    else:
        binder = min(chains, key=len)
    target = max((c for c in chains if c.name != binder.name), key=len)

    styles = [
        ({"chain": target.name}, {"cartoon": {"color": C_TARGET, "opacity": 0.85}}),
        ({"chain": binder.name}, {"cartoon": {"color": C_BINDER}}),
    ]
    if epitope1:
        styles.append(({"chain": target.name, "resi": [str(i) for i in epitope1]},
                       {"cartoon": {"color": C_HOTSPOT},
                        "stick": {"colorscheme": "orangeCarbon", "radius": 0.2}}))
    return _view_html([("cif", path.read_text())], styles, height=height)


def legend(items: list[tuple[str, str]]) -> None:
    """Colour key — identity must never be carried by colour alone."""
    chips = " ".join(
        f"<span style='display:inline-flex;align-items:center;gap:.4em;"
        f"margin-right:1.2em;font-size:.85rem'>"
        f"<span style='width:.85em;height:.85em;border-radius:3px;"
        f"background:{c};display:inline-block'></span>{label}</span>"
        for label, c in items)
    st.markdown(chips, unsafe_allow_html=True)
