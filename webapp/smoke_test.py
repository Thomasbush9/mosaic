#!/usr/bin/env python
"""Headless checks for the webapp. Run before touching the UI:

    webapp/.venv/bin/python webapp/smoke_test.py

Plain script rather than pytest: pytest is not in webapp/.venv, and the app's
dependencies deliberately do not overlap the main uv environment (the webapp
never imports mosaic). Exit code is 0 only if every check passes.

Streamlit's AppTest runs the real app in-process, so these exercise the actual
render path — which is where the failures have been. Two of the three checks
here correspond to bugs that reached the user: a saved pipeline whose stored
num_designs exceeded the form's cap took the whole app down, and the node-add
form kept the id you had just used and greeted you with "already exists".

Note the checks read the live working directory, so the reopen test needs at
least one submitted pipeline under <workdir>/pipelines. It skips if there is
none rather than failing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
APP = str(REPO / "webapp" / "app.py")

sys.path.insert(0, str(REPO / "webapp"))


def _fresh():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    return at


def _errs(at):
    return [str(e.value) for e in at.exception]


def check_every_page_renders() -> list[str]:
    """Each page renders clean, and a failure in one cannot reach the others."""
    bad = []
    at = _fresh()
    if _errs(at):
        return [f"cold start raised {_errs(at)}"]
    for page in at.sidebar.radio[0].options:
        at.sidebar.radio[0].set_value(page).run()
        if _errs(at):
            bad.append(f"page {page} raised {_errs(at)}")
    return bad


def check_reopen_saved_pipeline() -> list[str]:
    """A submitted pipeline must reopen even if its params outgrew the catalog.

    The regression: pipelines/dio3_2way_500b stores num_designs=500 against a
    form that capped the field at 200, so Load into editor raised
    StreamlitValueAboveMaxError — and kept raising on every rerun afterwards,
    because the loaded DAG lives in session_state.
    """
    at = _fresh()
    at.sidebar.radio[0].set_value("Pipeline").run()
    try:
        button = at.button(key="load_past")
    except KeyError:
        print("    (skipped: no submitted pipelines on disk)")
        return []
    button.click().run()
    if _errs(at):
        return [f"reopening a saved pipeline raised {_errs(at)}"]
    at.run()
    if _errs(at):
        return [f"rerun after reopening raised {_errs(at)}"]
    if not at.session_state["pipeline"].nodes:
        return ["reopened pipeline has no nodes"]
    return []


def check_build_and_edit() -> list[str]:
    """Define nodes, rewire edges, set params both ways, and reach a launchable
    graph — the tab's whole job."""
    bad = []
    at = _fresh()
    at.sidebar.radio[0].set_value("Pipeline").run()
    at.button(key="pipe_clear").click().run()

    def add(ntype, nid, inputs=None):
        at.selectbox(key="add_type").set_value(ntype).run()
        at.text_input(key="add_id").set_value(nid)
        if inputs:
            at.multiselect(key="add_inputs").set_value(inputs)
        at.button(key="add_go").click().run()

    add("generate", "gen1")
    add("screen", "scr1", ["gen1"])
    add("optimize", "opt1", ["scr1"])
    if _errs(at):
        return [f"adding nodes raised {_errs(at)}"]

    got = [(n.id, n.type, n.inputs) for n in at.session_state["pipeline"].nodes]
    want = [("gen1", "generate", []), ("scr1", "screen", ["gen1"]),
            ("opt1", "optimize", ["scr1"])]
    if got != want:
        bad.append(f"graph is {got}, expected {want}")

    # The add form must not still be holding the id just used.
    if any("already exists" in str(e.value) for e in at.error):
        bad.append("add-node form reports a clash right after a successful add")

    # Edges editable after the fact.
    at.multiselect(key="pn::opt1::edges").set_value([]).run()
    if at.session_state["pipeline"].node("opt1").inputs:
        bad.append("detaching an edge did not take")
    at.multiselect(key="pn::opt1::edges").set_value(["scr1"]).run()
    if at.session_state["pipeline"].node("opt1").inputs != ["scr1"]:
        bad.append("reattaching an edge did not take")

    # Raw JSON is the parity guarantee: it must set values no form draws and
    # keys no catalog knows.
    raw = dict(at.session_state["pipeline"].node("gen1").params)
    raw["num_designs"] = 4096
    raw["a_key_no_form_draws"] = {"nested": [1, 2, 3]}
    area = [t for t in at.text_area if t.label == "gen1 params (JSON)"]
    if not area:
        return bad + ["no raw params editor on gen1"]
    area[0].set_value(json.dumps(raw)).run()
    at.button(key="pn::gen1::rawapply").click().run()
    if _errs(at):
        return bad + [f"applying raw JSON raised {_errs(at)}"]
    p = at.session_state["pipeline"].node("gen1").params
    if p.get("num_designs") != 4096:
        bad.append(f"raw JSON num_designs became {p.get('num_designs')}, not 4096")
    if p.get("a_key_no_form_draws") != {"nested": [1, 2, 3]}:
        bad.append("raw JSON dropped a key the forms do not draw")

    # Per-node resource override.
    at.text_input(key="pn::opt1::rtime").set_value("23:59:00").run()
    if (at.session_state["pipeline"].node("opt1").params
            .get("resources", {}).get("time_limit") != "23:59:00"):
        bad.append("per-node resource override did not take")

    if at.button(key="pipe_submit").disabled:
        bad.append("a complete graph left Submit disabled: "
                   + str([str(e.value) for e in at.error]))
    return bad


def check_out_of_range_is_clamped_and_reported() -> list[str]:
    """An out-of-range stored value must clamp, say so, and never raise."""
    at = _fresh()
    at.sidebar.radio[0].set_value("Pipeline").run()
    at.button(key="pipe_clear").click().run()
    at.selectbox(key="add_type").set_value("hallucinate").run()
    at.text_input(key="add_id").set_value("h1")
    at.button(key="add_go").click().run()

    raw = dict(at.session_state["pipeline"].node("h1").params)
    raw["num_designs"] = 9999          # hallucinate caps at 64 (GPU memory)
    [t for t in at.text_area if t.label == "h1 params (JSON)"][0] \
        .set_value(json.dumps(raw)).run()
    at.button(key="pn::h1::rawapply").click().run()

    if _errs(at):
        return [f"an out-of-range value raised {_errs(at)}"]
    got = at.session_state["pipeline"].node("h1").params.get("num_designs")
    if got != 64:
        return [f"clamped to {got}, expected 64"]
    if not [c for c in at.caption if "outside this field" in c.value]:
        return ["clamped silently — the user was not told"]
    return []


CHECKS = [
    ("every page renders", check_every_page_renders),
    ("reopen a saved pipeline", check_reopen_saved_pipeline),
    ("build, rewire and edit a graph", check_build_and_edit),
    ("out-of-range value clamps and reports", check_out_of_range_is_clamped_and_reported),
]


def main() -> int:
    failures = 0
    for name, fn in CHECKS:
        print(f"==> {name}")
        problems = fn()
        for p in problems:
            print(f"    FAIL: {p}")
        failures += len(problems)
        if not problems:
            print("    ok")
    print()
    print("FAILED" if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
