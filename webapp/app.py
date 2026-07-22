"""mosaic design webapp — configure, launch, monitor and explore campaigns.

Run on a cluster login node:

    streamlit run webapp/app.py --server.port 8502 --server.address 127.0.0.1

Then from your laptop:

    ssh -L 8502:localhost:8502 <user>@holylogin06.rc.fas.harvard.edu

and open http://localhost:8502

Nothing heavy runs in this process. The app reads files and queues SLURM jobs;
every model evaluation happens in a batch job on a GPU node. That is what makes
it safe on a login node, where FASRC's arbiter kills CPU-heavy processes.

Layout follows ProtForge's: pure modules (store.py, cluster.py) hold the logic,
tab modules hold the UI, this file is only bootstrap and dispatch.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import design_config as dc  # noqa: E402
import launch_tab  # noqa: E402
import monitor_tab  # noqa: E402
import results_tab  # noqa: E402
from store import SETUP  # noqa: E402

st.set_page_config(page_title="mosaic design", page_icon="🧬", layout="wide")

if "cfg" not in st.session_state:
    st.session_state["cfg"] = dc.default_config()

with st.sidebar:
    st.markdown("### mosaic design")
    st.caption(f"{os.environ.get('USER', '?')} @ {socket.gethostname()}")
    st.caption(f"`{SETUP}`")
    st.divider()
    st.markdown(
        "**Nothing heavy runs here.** This app queues SLURM jobs and reads "
        "their output. Every model evaluation happens on a GPU node."
    )
    st.divider()
    if st.button("Reset config to defaults"):
        st.session_state["cfg"] = dc.default_config()
        st.rerun()
    up = st.file_uploader("Load a config", type="json")
    if up is not None:
        import json
        try:
            st.session_state["cfg"] = json.loads(up.read().decode())
            st.success("Config loaded")
        except Exception as e:
            st.error(f"Could not read config: {e}")
    st.divider()
    st.caption("Docs: `docs/MANUAL.md` · `docs/MODELS.md`")

launch, monitor, results = st.tabs(["Launch", "Monitor", "Results"])

with launch:
    st.session_state["cfg"] = launch_tab.render(st.session_state["cfg"])

with monitor:
    monitor_tab.render()

with results:
    results_tab.render()
