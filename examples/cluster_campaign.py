import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    import subprocess
    import glob
    from pathlib import Path
    return Path, glob, json, mo, subprocess


@app.cell
def _(mo):
    mo.md(
        r"""
    # Launching a binder design campaign on the cluster

    The other notebooks in `examples/` run models **here, in this process** —
    good for exploring a loss, bad for a real campaign, because one GPU designs
    one thing at a time and JIT warmup is minutes.

    This notebook does the opposite. Nothing heavy runs here. You configure a
    campaign, this builds the `sbatch` command, submits it, and then reads the
    results back once the cluster is done.

    **Why a cluster array rather than more GPUs in one process:** mosaic is
    single-device throughout — there is no `pmap`, `shard_map`, `jax.sharding`
    or `jax.distributed` anywhere in `src/`. A second GPU in this process would
    sit idle. Scaling has two axes that multiply:

    - **across GPUs** — a SLURM array, one GPU per task, tasks differing by seed
    - **within a GPU** — `batch` independent trajectories, which also share one
      JIT compile

    So `tasks x batch` designs. See `docs/MANUAL.md` for the non-computational
    version of all of this.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## 1. Where things live""")
    return


@app.cell
def _(Path):
    SETUP = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup")
    REPO = SETUP / "mosaic"
    return REPO, SETUP


@app.cell
def _(SETUP, mo):
    # Only targets that already have an MSA can be run — the search is a
    # one-off per target and takes 20 min to 2 h, so it is not done from here.
    _targets = sorted(p.stem for p in (SETUP / "targets").glob("*.fasta"))
    _with_msa = [t for t in _targets if (SETUP / "msa" / f"{t}.a3m").exists()]

    target = mo.ui.dropdown(
        options=_with_msa or _targets,
        value=(_with_msa or _targets)[0] if (_with_msa or _targets) else None,
        label="Target",
    )
    mo.vstack([
        mo.md(f"**{len(_targets)}** targets, **{len(_with_msa)}** with an MSA."),
        target,
    ])
    return (target,)


@app.cell
def _(SETUP, mo, target):
    _fa = SETUP / "targets" / f"{target.value}.fasta"
    _a3m = SETUP / "msa" / f"{target.value}.a3m"
    _seq = "".join(
        ln.strip() for ln in _fa.read_text().splitlines() if not ln.startswith(">")
    )
    _depth = "none"
    if _a3m.exists():
        _uniq = {
            ln.strip()
            for ln in _a3m.read_text().splitlines()
            if ln.strip() and not ln.startswith(">")
        }
        _depth = f"{len(_uniq):,} unique sequences"

    mo.md(
        f"""
    **{target.value}** — {len(_seq)} residues
    MSA: {_depth}

    ```
    {_seq[:80]}{"..." if len(_seq) > 80 else ""}
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. Configure

    Defaults are a reasonable standard campaign. The two settings that change
    results most are the binder length and which models must agree.
    """
    )
    return


@app.cell
def _(mo):
    binder_length = mo.ui.slider(
        40, 140, value=80, step=5, label="Binder length", show_value=True
    )
    n_tasks = mo.ui.slider(
        1, 64, value=16, step=1, label="GPUs (array tasks)", show_value=True
    )
    batch = mo.ui.slider(
        1, 4, value=2, step=1, label="Designs per GPU", show_value=True
    )
    soft_steps = mo.ui.slider(
        20, 300, value=100, step=10, label="Soft steps", show_value=True
    )
    sharp_steps = mo.ui.slider(
        5, 100, value=25, step=5, label="Sharp steps", show_value=True
    )
    models = mo.ui.multiselect(
        options=["boltz2", "boltz1", "af2", "of3", "protenix"],
        value=["boltz2"],
        label="Structure models (all must agree)",
    )
    mo.vstack([
        mo.hstack([binder_length, batch], justify="start"),
        mo.hstack([soft_steps, sharp_steps], justify="start"),
        n_tasks,
        models,
    ])
    return batch, binder_length, models, n_tasks, sharp_steps, soft_steps


@app.cell
def _(batch, mo, models, n_tasks, soft_steps):
    # Per-model cost for 5 soft steps at batch 2, measured on an H100 with a
    # 237-residue target and an 80-residue binder.
    _COST5 = {"protenix": 39.4, "boltz2": 91.0, "of3": 92.4, "af2": 109.2,
              "boltz1": 123.8}
    _per_step = sum(_COST5.get(m, 90.0) for m in models.value) / 5.0
    _mins = (_per_step * soft_steps.value * 1.25) / 60.0 + 2.0

    _warn = ""
    if not models.value:
        _warn = "\n\n> **Pick at least one model.**"
    elif len(models.value) > 2:
        _warn = (
            "\n\n> Three or more models roughly triples cost for a diminishing "
            "return — usually better spent on more seeds."
        )
    if batch.value > 2:
        _warn += (
            "\n\n> `batch` above 2 is untested here: GPU memory and a host-side "
            "simplex projection that scales with batch are both plausible "
            "ceilings."
        )
    if "af2" in models.value:
        _warn += (
            "\n\n> **AF2 never receives an MSA** — its wrapper rejects them "
            "(`models/af2.py:391`), so in a mixed run its view of the target is "
            "single-sequence while the others get the full alignment."
        )

    mo.md(
        f"""
    **{n_tasks.value * batch.value} designs** from {n_tasks.value} GPUs
    ≈ **{_mins:.0f} min** per task (all tasks run in parallel){_warn}
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## 3. The command""")
    return


@app.cell
def _(REPO, SETUP, batch, binder_length, models, n_tasks, sharp_steps, soft_steps, target):
    out_dir = SETUP / "designs" / f"{target.value}_nb"

    # '+' between model names, NOT a comma. SLURM's --export is itself a
    # comma-separated KEY=VALUE list, so MODELS=boltz2,af2 is silently truncated
    # to MODELS=boltz2 and the stray 'af2' is discarded with no error. That
    # mistake once invalidated an entire 8-GPU comparison.
    _exports = ",".join([
        "ALL",
        f"TARGET_FASTA={SETUP / 'targets' / f'{target.value}.fasta'}",
        f"TARGET_MSA={SETUP / 'msa' / f'{target.value}.a3m'}",
        f"BINDER_LENGTH={binder_length.value}",
        f"MODELS={'+'.join(models.value)}",
        f"BATCH={batch.value}",
        f"SOFT_STEPS={soft_steps.value}",
        f"SHARP_STEPS={sharp_steps.value}",
        f"OUT_DIR={out_dir}",
    ])
    submit_cmd = [
        "sbatch",
        f"--array=0-{n_tasks.value - 1}",
        f"--export={_exports}",
        "singularity/campaign.sbatch",
    ]
    return out_dir, submit_cmd


@app.cell
def _(REPO, mo, submit_cmd):
    # Built outside the f-string: a backslash inside an f-string expression is
    # only legal from Python 3.12, and this file should stay readable by older
    # tooling even though the container ships 3.12.
    _pretty = " \\\n  ".join(submit_cmd)
    mo.md(
        f"""
    Run from `{REPO}`:

    ```bash
    {_pretty}
    ```
    """
    )
    return


@app.cell
def _(mo):
    submit = mo.ui.run_button(label="Submit to the cluster")
    mo.vstack([
        mo.md(
            "Submitting is safe from here — it only queues the job. Nothing "
            "heavy runs in this notebook or on the login node."
        ),
        submit,
    ])
    return (submit,)


@app.cell
def _(REPO, mo, submit, submit_cmd, subprocess):
    mo.stop(not submit.value, mo.md("*Not submitted yet.*"))

    _p = subprocess.run(submit_cmd, cwd=REPO, capture_output=True, text=True)
    _ok = _p.returncode == 0
    job_id = _p.stdout.strip().split()[-1] if _ok else None

    mo.md(
        f"""
    {"**Submitted** — job `" + str(job_id) + "`" if _ok else "**Failed**"}

    ```
    {(_p.stdout + _p.stderr).strip()}
    ```

    {"Now verify it is doing what you asked — see the next cell." if _ok else ""}
    """
    )
    return (job_id,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4. Verify — do not skip this

    The failure modes here are **silent**: a job that runs one model instead of
    two, or single-sequence instead of MSA-backed, completes successfully and
    writes plausible output. The log is the only honest record of what ran.

    ```bash
    grep -E "structure backends|target MSA|n_msa" logs/campaign-<jobid>_0.out
    ```

    - `structure backends:` must list **every** model you selected
    - `target MSA:` must show a path, not `(none - single sequence)`
    - `n_msa` should be in the thousands, not `1`
    """
    )
    return


@app.cell
def _(REPO, job_id, mo, subprocess):
    mo.stop(job_id is None, mo.md("*Submit first.*"))
    _log = REPO / "logs" / f"campaign-{job_id}_0.out"
    mo.stop(
        not _log.exists(),
        mo.md(f"*Waiting for `{_log.name}` — the task has not started yet.*"),
    )
    _hits = subprocess.run(
        ["grep", "-E", "structure backends|target MSA|n_msa|best loss", str(_log)],
        capture_output=True, text=True,
    ).stdout
    mo.md(f"```\n{_hits or '(nothing yet)'}\n```")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5. Explore the results

    Point this at any campaign directory — it does not have to be one submitted
    from this notebook.
    """
    )
    return


@app.cell
def _(SETUP, mo):
    def _is_campaign(d):
        # A campaign dir holds scored designs. BoltzGen proposal dirs live in the
        # same tree with a different schema, so check rather than assume.
        import json as _j
        for f in d.glob("*.json"):
            try:
                if "results" in _j.load(open(f)):
                    return True
            except Exception:
                pass
        return False

    _dirs = sorted(
        p.name for p in (SETUP / "designs").glob("*") if p.is_dir() and _is_campaign(p)
    )
    results_dir = mo.ui.dropdown(
        options=_dirs, value=_dirs[-1] if _dirs else None, label="Campaign"
    )
    results_dir
    return (results_dir,)


@app.cell
def _(SETUP, glob, json, mo, results_dir):
    mo.stop(results_dir.value is None, mo.md("*No finished campaigns yet.*"))

    rows = []
    for _p in sorted(glob.glob(str(SETUP / "designs" / results_dir.value / "*.json"))):
        _d = json.load(open(_p))
        # BoltzGen output lives in the same tree but has a different schema
        # (proposals, not scored designs) — skip it rather than crashing.
        if "results" not in _d:
            continue
        for _r in _d["results"]:
            rows.append({
                "loss": round(_r["loss"], 3),
                "seed": _d["seed"],
                "traj": _r["trajectory"],
                "models": "+".join(_d["models"]),
                "sequence": _r["sequence"],
            })
    rows.sort(key=lambda r: r["loss"])
    return (rows,)


@app.cell
def _(mo, rows):
    mo.stop(not rows, mo.md("*No designs found.*"))
    mo.vstack([
        mo.md(
            f"**{len(rows)} designs.** Lower loss is better — but it is a weighted "
            "heuristic (contact geometry, compactness, sequence plausibility), "
            "**not** an affinity or a probability of binding.\n\n"
            "Losses are only comparable within one `models` setting: a two-model "
            "run must satisfy two critics and will read higher at equal quality."
        ),
        mo.ui.table(rows[:25], selection=None),
    ])
    return


@app.cell
def _(mo, rows):
    # Composition is the cheapest check that the optimizer has not collapsed:
    # a design dominated by one or two residues has gamed the objective.
    mo.stop(not rows, mo.md(""))
    _all = "".join(r["sequence"] for r in rows)
    _counts = sorted(
        ((c, _all.count(c) / len(_all) * 100) for c in set(_all)),
        key=lambda t: -t[1],
    )
    _top = ", ".join(f"{c} {p:.0f}%" for c, p in _counts[:6])
    _cys = _all.count("C")

    _flag = ""
    if _counts[0][1] > 25:
        _flag = (
            f"\n\n> **{_counts[0][0]} is {_counts[0][1]:.0f}% of all residues.** "
            "That is degenerate — the optimizer is gaming the objective rather "
            "than solving it."
        )
    if _cys:
        _flag += (
            f"\n\n> **{_cys} cysteines present.** Expected zero unless "
            "`--no-no-cys` was passed — investigate the decode path."
        )

    mo.md(
        f"""
    **Composition** ({len(set(_all))}/20 residue types): {_top}
    Cysteines: {_cys}{_flag}
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 6. What this does not tell you

    Nothing here says a design binds. A low loss means several structure
    predictors, evaluated at **1 recycling step for speed**, believe the binder
    contacts the target.

    Before believing any candidate:

    1. **Re-fold the top few properly** — 4–20 recycling steps, and look at
       pLDDT and interface PAE rather than the design loss.
    2. **Check it is not model-specific** — rerun the winner through a model that
       was *not* in the objective.
    3. **Then the wet lab.**

    See `docs/MANUAL.md` §4 and `docs/MODELS.md` for the reasoning behind each
    of these.
    """
    )
    return


if __name__ == "__main__":
    app.run()
