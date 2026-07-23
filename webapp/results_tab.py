"""Tab 3 — results, filtering and statistics.

Presents designs honestly: a ranked list is easy, but the checks that decide
whether the list means anything (composition, diversity, comparability) come
first.
"""

from __future__ import annotations

import streamlit as st

import store

# Validated categorical palette, fixed order, never cycled.
C_DESIGN, C_NATURAL = "#2a78d6", "#1baf7a"


def _hist(values: list[float], nbins: int = 24):
    import plotly.graph_objects as go
    fig = go.Figure(go.Histogram(x=values, nbinsx=nbins, marker_color=C_DESIGN))
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="loss (lower is better)", yaxis_title="designs",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.05,
    )
    return fig


def _composition_fig(comp: dict[str, float]):
    import plotly.graph_objects as go
    aas = list(store.NATURAL_AA)
    fig = go.Figure([
        go.Bar(x=aas, y=[comp.get(a, 0.0) for a in aas], name="designed",
               marker_color=C_DESIGN),
        go.Bar(x=aas, y=[store.NATURAL_AA[a] for a in aas], name="natural",
               marker_color=C_NATURAL),
    ])
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10), barmode="group",
        yaxis_title="% of residues", bargap=0.25, bargroupgap=0.05,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.15),
    )
    return fig


def _per_seed_fig(rows: list[dict]):
    import plotly.graph_objects as go
    seeds = sorted({r["seed"] for r in rows if r["seed"] is not None})
    fig = go.Figure(go.Scatter(
        x=[r["seed"] for r in rows], y=[r["loss"] for r in rows],
        mode="markers", marker=dict(size=10, color=C_DESIGN,
                                    line=dict(width=2, color="white")),
    ))
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="seed", yaxis_title="loss",
        xaxis=dict(tickmode="array", tickvals=seeds),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render() -> None:
    mode = st.radio("Show", ["Optimized campaigns", "Screened proposals"],
                    horizontal=True, key="res_kind")
    if mode == "Screened proposals":
        _render_screens()
        return

    campaigns = store.list_campaigns()
    if not campaigns:
        st.info("No finished campaigns yet.")
        return

    c1, c2 = st.columns([2, 1])
    campaign = c1.selectbox("Campaign", campaigns, index=len(campaigns) - 1)
    if c2.button("Refresh", key="res_refresh"):
        st.rerun()

    rows = store.load_designs(campaign)
    if not rows:
        st.warning("No scored designs in this directory.")
        return

    # ---- checks before results, deliberately -----------------------------
    for msg in store.flags(rows):
        st.error(msg)

    losses = [r["loss"] for r in rows]
    n = len(losses)
    srt = sorted(losses)
    median = srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2

    m = st.columns(5)
    m[0].metric("Designs", n)
    m[1].metric("Best", f"{min(losses):.3f}")
    m[2].metric("Median", f"{median:.3f}")
    m[3].metric("Worst", f"{max(losses):.3f}")
    m[4].metric("Models", rows[0]["models"] or "?")

    st.caption(
        "Loss is a weighted heuristic — contact geometry, compactness, sequence "
        "plausibility. It is **not** an affinity, a Kd, or a probability of "
        "binding. It is only comparable within one model setting: each extra "
        "backend adds a critic, so a two-model run reads higher at equal quality."
    )

    # ---- filtering -------------------------------------------------------
    st.divider()
    st.subheader("Filter")
    f1, f2 = st.columns(2)
    mode = f1.radio("By", ["Top N", "Loss threshold"], horizontal=True)
    if mode == "Top N":
        k = f2.slider("N", 1, min(200, n), min(20, n))
        shown = rows[:k]
    else:
        thr = f2.slider("Max loss", float(min(losses)), float(max(losses)),
                        float(median), 0.01)
        shown = [r for r in rows if r["loss"] <= thr]
    st.caption(f"{len(shown)} of {n} designs shown.")

    st.dataframe(
        [{"loss": round(r["loss"], 3), "seed": r["seed"], "traj": r["trajectory"],
          "len": r["length"], "sequence": r["sequence"]} for r in shown],
        width="stretch", hide_index=True,
    )
    st.download_button(
        "Download FASTA", key="res_fasta",
        data="".join(
            f">{campaign}_seed{r['seed']}_traj{r['trajectory']}_loss{r['loss']:.4f}\n"
            f"{r['sequence']}\n" for r in shown),
        file_name=f"{campaign}_filtered.fasta", mime="text/plain",
    )

    # ---- statistics ------------------------------------------------------
    st.divider()
    st.subheader("Statistics")
    try:
        s1, s2 = st.columns(2)
        with s1:
            st.caption("Loss distribution")
            st.plotly_chart(_hist(losses), width="stretch")
        with s2:
            st.caption("Spread across seeds — clustering by seed means the "
                       "starting point dominates, not the objective")
            st.plotly_chart(_per_seed_fig(rows), width="stretch")
        st.caption("Composition vs natural frequencies")
        st.plotly_chart(_composition_fig(
            store.composition([r["sequence"] for r in shown])), width="stretch")
    except ImportError:
        st.info("Install `plotly` for charts — the tables above work without it.")

    ids = store.pairwise_identity([r["sequence"] for r in shown])
    if ids:
        mean_id = sum(ids) / len(ids)
        st.metric("Mean pairwise identity", f"{mean_id:.0f}%",
                  help="High identity means the run converged on one answer "
                       "rather than exploring. Random is roughly 5%.")

    # ---- provenance ------------------------------------------------------
    cfg = store.campaign_config(campaign)
    if cfg:
        with st.expander("Config this campaign ran with"):
            st.json(cfg)
    else:
        st.caption("No config recorded — this campaign predates config-based runs.")

    st.divider()
    st.caption(
        "**Before believing any candidate:** re-fold the top few at higher "
        "recycling (design uses 1 for speed) and look at pLDDT and interface "
        "PAE; rerun the winner through a model that was not in the objective; "
        "then the wet lab."
    )


def _render_screens() -> None:
    screens = store.list_screens()
    if not screens:
        st.info("No screened proposal sets yet. Run **Generate -> Screen & rank**.")
        return
    pick = st.selectbox("Screen", screens, index=len(screens) - 1,
                        key="res_screen")
    data = store.load_screen(pick)
    if not data or not data.get("results"):
        st.warning("No results in this screen.")
        return
    rows = data["results"]
    st.caption(f"Refolded with **{data['screen_model']}** from "
               f"**{data['generator']}** proposals. Ranked by "
               f"`{data['score_formula']}`.")
    m = st.columns(4)
    m[0].metric("Candidates", len(rows))
    m[1].metric("Best ipTM", f"{max(r['iptm'] for r in rows):.3f}")
    m[2].metric("Best pLDDT", f"{max(r['binder_plddt'] for r in rows):.1f}")
    m[3].metric("Best iface PAE", f"{min(r['interface_pae'] for r in rows):.2f}")
    st.dataframe(
        [{"rank": i + 1, "design": r["design"], "ipTM": r["iptm"],
          "pLDDT": r["binder_plddt"], "iface PAE": r["interface_pae"],
          "score": r["score"], "sequence": r["sequence"]}
         for i, r in enumerate(rows)],
        hide_index=True, width="stretch")
    st.download_button(
        "Download FASTA (ranked)", key="dl_screen",
        data="".join(f">{pick}_{r['design']:03d}_iptm{r['iptm']:.3f}\n"
                     f"{r['sequence']}\n" for r in rows),
        file_name=f"{pick}_ranked.fasta", mime="text/plain")
    st.caption("ipTM/pLDDT are the folding model's confidence, not an affinity. "
               "Screen with a model other than the generator, and validate the "
               "top few before trusting them.")
