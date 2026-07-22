"""Tab 2 — monitor running campaigns.

The important part is not the queue table, it is the verification block. Every
failure mode encountered in this pipeline was silent: the job completes, writes
plausible output, and only the log shows it ran one model instead of two, or
single-sequence instead of MSA-backed.
"""

from __future__ import annotations

import os

import streamlit as st

import cluster


def render() -> None:
    user = os.environ.get("USER", "")

    st.subheader("Queue")
    c1, c2 = st.columns([1, 3])
    if c1.button("Refresh"):
        st.rerun()
    mine = c2.checkbox("Only my jobs", value=True)

    jobs = cluster.queue(user if mine else None)
    if jobs:
        st.dataframe(
            [{"job": j.job_id, "name": j.name, "state": j.state,
              "elapsed": j.elapsed, "reason": j.reason} for j in jobs],
            width="stretch", hide_index=True,
        )
    else:
        st.info("Nothing queued or running.")

    st.divider()
    st.subheader("Verify a job")
    st.caption(
        "Confirm what the job **says** it did. The submit command is not "
        "evidence: SLURM's --export splits on commas, so a model list can be "
        "silently truncated and the run still succeeds."
    )

    last = st.session_state.get("last_job") or {}
    job_id = st.text_input("Job ID", value=last.get("id", ""))
    if not job_id:
        st.stop()

    expect_models = last.get("models") if last.get("id") == job_id else None
    expect_msa = last.get("expect_msa", True) if last.get("id") == job_id else True

    rows = cluster.verify(job_id, expect_models, expect_msa)
    for label, value, status in rows:
        if status == "ok":
            st.success(f"**{label}** — {value}")
        elif status == "fail":
            st.error(f"**{label}** — {value}   ← not what was requested")
        else:
            st.info(f"**{label}** — {value}")

    if expect_models:
        st.caption(f"Expected models: {', '.join(expect_models)}")

    st.divider()
    st.subheader("Progress")
    n_tasks = int(last.get("n_tasks", 0)) if last.get("id") == job_id else 0
    done, failed, running = cluster.progress(job_id, n_tasks)
    p1, p2, p3 = st.columns(3)
    p1.metric("Completed", done)
    p2.metric("Failed", failed)
    p3.metric("Running / pending", running)
    total = n_tasks or (done + failed + running)
    if total:
        st.progress(min(1.0, done / total), text=f"{done}/{total} tasks complete")
    if failed:
        st.warning(f"{failed} task(s) failed — check the log below for the "
                   "traceback before rerunning.")

    st.divider()
    st.subheader("Log")
    logs = cluster.log_paths(job_id)
    if not logs:
        st.info("No log yet — the first task has not started.")
    else:
        names = [p.name for p in logs]
        pick = st.selectbox("Task log", names, index=0)
        n_lines = st.slider("Lines", 20, 400, 80, 20)
        st.code(cluster.tail(logs[names.index(pick)], n_lines))

    st.divider()
    if st.button("Cancel this job", type="secondary"):
        ok, msg = cluster.cancel(job_id)
        (st.success if ok else st.error)(msg or ("cancelled" if ok else "failed"))
