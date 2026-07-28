"""mosaic design webapp — sessions, launch, generate, monitor, results.

Preferred way to run it, from the repo root:

    sbatch singularity/webapp.sbatch      # then read logs/webapp-<jobid>.out

Nothing heavy runs in this process. The app reads files and queues SLURM jobs;
every model evaluation happens in a batch job on a GPU node. It is light enough
for a login node, but a login node holds every user to a shared 8 GiB / 1-core
cgroup and cannot keep a process alive past an SSH disconnect, so the batch job
above is the route that actually stays up. See docs/WEBAPP.md.

Layout follows ProtForge's: pure modules (store, cluster, session) hold the
logic, tab modules hold the UI, this file is only bootstrap and dispatch.

**One page renders per run.** Streamlit's st.tabs is client-side only: every
`with tab:` body executes on every rerun regardless of which tab is showing, so
six tabs meant six file scans and eight SLURM subprocess calls per keystroke —
and an exception in any one of them blanked all six. Sidebar navigation renders
exactly the page you are looking at, and the try/except below keeps a failure
inside the page that caused it.
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

import cluster  # noqa: E402
import design_config as dc  # noqa: E402
import docs_tab  # noqa: E402
import generate_tab  # noqa: E402
import launch_tab  # noqa: E402
import monitor_tab  # noqa: E402
import pipeline_tab  # noqa: E402
import results_tab  # noqa: E402
import session as sess  # noqa: E402
import store  # noqa: E402

st.set_page_config(page_title="mosaic design", page_icon="🧬", layout="wide")

PAGES = ["Launch", "Generate", "Pipeline", "Monitor", "Results", "Docs"]

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

# A proposal shipped from Generate is applied by Launch, so follow the handoff.
# Setting the nav key before the radio is constructed is what makes it stick;
# afterwards Streamlit owns the key. This replaces the old ordering hack, in
# which Launch had to be *rendered* before Generate even though it was shown
# second, or a shipped proposal landed a rerun late.
if (st.session_state.get("pending_proposal")
        and st.session_state.get("nav") != "Launch"):
    st.session_state["nav"] = "Launch"

with st.sidebar:
    st.markdown("### mosaic design")
    st.caption(f"{os.environ.get('USER', '?')} @ {socket.gethostname()}")

    page = st.radio("View", PAGES, key="nav", label_visibility="collapsed")
    st.divider()

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
    if st.button("Refresh cluster info"):
        # accounts/partitions are memoized for an hour; this is the escape hatch
        # for the day an association or partition actually changes.
        cluster.forget_cluster_cache()
        st.rerun()
    st.caption("**Nothing heavy runs here.** The app queues SLURM jobs and reads "
               "their output.")
    st.caption("Docs: `docs/MANUAL.md` · `docs/MODELS.md` · `docs/WEBAPP.md`")

# Dispatch. Each page is rendered inside its own error boundary: an exception
# used to escape to Streamlit's script runner and replace the entire app with a
# traceback, including the five pages that were fine. Now a broken page costs
# you that page.
try:
    if page == "Launch":
        st.session_state[key] = launch_tab.render(st.session_state[key])
    elif page == "Generate":
        generate_tab.render(st.session_state[key])
    elif page == "Pipeline":
        pipeline_tab.render(st.session_state[key])
    elif page == "Monitor":
        monitor_tab.render()
    elif page == "Results":
        results_tab.render()
    elif page == "Docs":
        docs_tab.render()
except Exception as exc:  # noqa: BLE001 — the boundary is the point
    st.error(f"The **{page}** page failed to render: {exc}")
    st.caption("The other pages still work. If this followed loading a saved "
               "pipeline or config, its stored values may disagree with what "
               "the forms accept — the raw JSON editor on each pipeline node "
               "lets you correct it.")
    with st.expander("Traceback"):
        st.exception(exc)
