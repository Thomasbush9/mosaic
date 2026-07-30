"""Tab — the documentation, rendered in the browser.

Reads the markdown in docs/ and renders it as styled HTML, so the manual is
available where the work happens rather than only on disk. `build_html()` writes
standalone .html files as a side effect, which are also useful outside the app.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from store import REPO

DOCS = [
    ("Manual", "MANUAL.md", "How to run a campaign, start to finish"),
    ("Models", "MODELS.md", "What each model is, and every parameter"),
    ("Webapp", "WEBAPP.md", "This app, tab by tab"),
    ("Pipelines as files", "PIPELINE_FILE.md",
     "The same DAG, defined in a file and submitted from a shell"),
]

CSS = """
<style>
.mosaic-doc { max-width: 62rem; line-height: 1.6; font-size: 0.95rem; }
.mosaic-doc h1 { font-size: 1.7rem; margin: 0 0 .3em; }
.mosaic-doc h2 { font-size: 1.3rem; margin: 1.6em 0 .4em;
                 border-bottom: 1px solid rgba(128,128,128,.25);
                 padding-bottom: .25em; }
.mosaic-doc h3 { font-size: 1.05rem; margin: 1.2em 0 .3em; }
.mosaic-doc code { background: rgba(128,128,128,.14); padding: .12em .35em;
                   border-radius: 4px; font-size: .88em; }
.mosaic-doc pre { background: rgba(128,128,128,.12); padding: .8em 1em;
                  border-radius: 6px; overflow-x: auto; }
.mosaic-doc pre code { background: none; padding: 0; }
.mosaic-doc table { border-collapse: collapse; margin: 1em 0; width: 100%;
                    display: block; overflow-x: auto; }
.mosaic-doc th, .mosaic-doc td { border: 1px solid rgba(128,128,128,.3);
                                 padding: .45em .7em; text-align: left; }
.mosaic-doc th { background: rgba(128,128,128,.12); }
.mosaic-doc blockquote { border-left: 3px solid rgba(128,128,128,.45);
                         margin: 1em 0; padding: .2em 0 .2em 1em; opacity: .9; }
.mosaic-doc img { max-width: 100%; }
</style>
"""


def to_html(md_text: str, title: str) -> str:
    import markdown

    body = markdown.markdown(
        md_text, extensions=["tables", "fenced_code", "toc", "sane_lists"])
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{title}</title>{CSS}</head>"
            f"<body><div class='mosaic-doc'>{body}</div></body></html>")


def build_html(out_dir: Path | None = None) -> list[Path]:
    """Write standalone .html next to the .md sources."""
    out_dir = Path(out_dir or (REPO / "docs"))
    written = []
    for title, fname, _ in DOCS:
        src = REPO / "docs" / fname
        if not src.is_file():
            continue
        dst = out_dir / (Path(fname).stem + ".html")
        dst.write_text(to_html(src.read_text(), f"mosaic — {title}"))
        written.append(dst)
    return written


def render() -> None:
    names = [d[0] for d in DOCS]
    pick = st.radio("Document", names, horizontal=True, key="docs_pick")
    title, fname, blurb = next(d for d in DOCS if d[0] == pick)
    src = REPO / "docs" / fname
    if not src.is_file():
        st.error(f"{src} not found.")
        return

    st.caption(blurb)
    md = src.read_text()

    c1, c2 = st.columns([1, 4])
    with c1:
        st.download_button("Download HTML", data=to_html(md, f"mosaic — {title}"),
                           file_name=f"{Path(fname).stem}.html",
                           mime="text/html", key=f"dl_{fname}")
    with c2:
        if st.button("Write .html files to docs/", key="build_docs"):
            written = build_html()
            st.success("Wrote " + ", ".join(p.name for p in written))

    st.divider()
    # Rendered natively rather than in an iframe: Streamlit's markdown handles
    # tables and code fences, inherits the app's theme, and scrolls with the
    # page. The HTML build above is for reading outside the app.
    st.markdown(md)
