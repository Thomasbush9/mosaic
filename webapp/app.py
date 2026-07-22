"""mosaic design webapp — sessions, launch, generate, monitor, results.

Run on a cluster login node:

    webapp/.venv/bin/streamlit run webapp/app.py --server.port 8502 \
        --server.address 127.0.0.1

Then from your laptop:

    ssh -L 8502:localhost:8502 <user>@holylogin06.rc.fas.harvard.edu

Nothing heavy runs in this process. The app reads files and queues SLURM jobs;
every model evaluation happens in a batch job on a GPU node. That is what makes
it safe on a login node, where FASRC's arbiter kills CPU-heavy processes.

Layout follows ProtForge's: pure modules (store, cluster, session) hold the
logic, tab modules hold the UI, this file is only bootstrap and dispatch.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import design_config as dc  # noqa: E402
import docs_tab  # noqa: E402
import generate_tab  # noqa: E402
import launch_tab  # noqa: E402
import monitor_tab  # noqa: E402
import results_tab  # noqa: E402
import session as sess  # noqa: E402
import store  # noqa: E402

st.set_page_config(page_title="mosaic design", page_icon="🧬", layout="wide")

# ---------------------------------------------------------------- session
active = sess.active()
sess.ensure_dirs(active)
# Every rerun re-points the store, so switching sessions switches the whole
# view: targets, MSAs, designs and configs all come from the session workdir.
store.set_workdir(active.path)

# Config is per session, so switching projects does not carry settings across.
key = f"cfg::{active.id}"
if key not in st.session_state:
    st.session_state[key] = dc.default_config()

with st.sidebar:
    st.markdown("### mosaic design")
    st.caption(f"{os.environ.get('USER', '?')} @ {socket.gethostname()}")

    sessions = sess.list_sessions()
    labels = {s.id: s.name for s in sessions}
    ids = [s.id for s in sessions]
    chosen = st.selectbox("Project", ids, index=ids.index(active.id),
                          format_func=lambda i: labels.get(i, i))
    if chosen != active.id:
        sess.set_active(chosen)
        st.rerun()

    st.caption(f"Working directory\n\n`{active.path}`")

    with st.expander("Manage projects"):
        st.markdown("**New project**")
        nname = st.text_input("Name", key="sess_new_name")
        nwd = st.text_input("Working directory", key="sess_new_wd",
                            value=str(active.path.parent / "new_project"),
                            help="targets/, msa/, designs/, configs/ are created "
                                 "here. Point it at an existing tree to adopt it.")
        if st.button("Create", disabled=not nname.strip()):
            try:
                s = sess.create(nname, nwd)
                st.success(f"Created {s.name}")
                st.rerun()
            except OSError as e:
                st.error(f"Could not create {nwd}: {e}")

        st.divider()
        st.markdown("**Rename current**")
        rn = st.text_input("New name", value=active.name, key="sess_rename")
        if st.button("Rename") and rn.strip():
            sess.rename(active.id, rn)
            st.rerun()

        st.divider()
        if st.button("Remove current from list", type="secondary"):
            # Forgets the session only. The working directory and every campaign
            # in it are left untouched — the data is the point.
            sess.delete(active.id)
            st.rerun()
        st.caption("Removing a project forgets it here. Files on disk are kept.")

    st.divider()
    if st.button("Reset config to defaults"):
        st.session_state[key] = dc.default_config()
        st.rerun()
    up = st.file_uploader("Load a config", type="json")
    if up is not None:
        try:
            st.session_state[key] = json.loads(up.read().decode())
            st.success("Config loaded")
        except Exception as e:
            st.error(f"Could not read config: {e}")
    st.download_button("Save config",
                       data=json.dumps(st.session_state[key], indent=2),
                       file_name="campaign_config.json", mime="application/json")

    st.divider()
    st.caption("**Nothing heavy runs here.** The app queues SLURM jobs and reads "
               "their output.")
    st.caption("Docs: `docs/MANUAL.md` · `docs/MODELS.md` · `docs/WEBAPP.md`")

# Generate first: it is the first stage of the pipeline (propose candidates),
# and Launch consumes what it produces. Note Launch still RENDERS first below —
# tab order is presentation, but a proposal shipped from Generate must be
# applied before Launch draws its widgets, or the prefill lands a rerun late.
generate, launch, monitor, results, docs = st.tabs(
    ["Generate", "Launch", "Monitor", "Results", "Docs"])

with launch:
    st.session_state[key] = launch_tab.render(st.session_state[key])

with generate:
    generate_tab.render(st.session_state[key])

with monitor:
    monitor_tab.render()

with results:
    results_tab.render()

with docs:
    docs_tab.render()
