"""SLURM interaction: submit, query, cancel, read logs. No Streamlit here.

Everything shells out. Nothing here runs a model — the app only ever queues work,
which is what keeps it safe to run on a login node.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from store import REPO

# --------------------------------------------------------------------------
# Time-based memoization for the two SLURM lookups that are drawn on every
# rerun. accounts() and partitions() feed a selectbox in the resource picker,
# which appears several times per page, so an uncached call meant half a dozen
# subprocess spawns per keystroke — latency for the user and needless load on
# slurmctld from an idle browser tab. Neither answer changes on the order of an
# hour. Kept here as a plain dict rather than st.cache_data so this module stays
# free of any Streamlit dependency, as the docstring promises.
# --------------------------------------------------------------------------
_TTL = 3600.0
_MEMO: dict = {}


def _memoized(key: str, fn, ttl: float = _TTL):
    hit = _MEMO.get(key)
    if hit is not None and (time.monotonic() - hit[0]) < ttl:
        return hit[1]
    val = fn()
    _MEMO[key] = (time.monotonic(), val)
    return val


def forget_cluster_cache() -> None:
    """Drop the memoized lookups, so a Refresh re-queries SLURM."""
    _MEMO.clear()

# bsabatini_lab has MaxJobs=0/MaxSubmit=0 and cannot submit anything, so this is
# the default rather than a suggestion. Overridable per campaign in the UI.
DEFAULT_ACCOUNT = "kempner_bsabatini_lab"
GPU_PARTITIONS = ["kempner_h100", "kempner_h200", "kempner_requeue"]
CPU_PARTITION = "kempner_interactive"  # GPU partitions reject jobs with no GPU


@dataclass
class Job:
    job_id: str
    name: str
    state: str
    elapsed: str
    reason: str = ""


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def submit_campaign(config_path: Path, out_dir: Path, n_tasks: int,
                    throttle: int | None = None, partition: str = "kempner_h100",
                    account: str = DEFAULT_ACCOUNT,
                    time_limit: str = "08:00:00", mem: str = "128G",
                    cpus: int = 8) -> tuple[bool, str, str | None]:
    """Queue a campaign array. Returns (ok, output, job_id).

    The config travels as a FILE PATH, not as exported variables. That is not a
    style choice: SLURM's --export is a comma-separated KEY=VALUE list, so any
    value containing a comma is silently truncated. A whole 8-GPU comparison was
    once invalidated by `MODELS=boltz2,af2` quietly becoming `MODELS=boltz2`.
    A path has no commas and the config carries everything else.
    """
    array = f"0-{n_tasks - 1}" + (f"%{throttle}" if throttle else "")
    exports = ",".join([
        "ALL",
        f"DESIGN_CONFIG={config_path}",
        f"OUT_DIR={out_dir}",
    ])
    cmd = [
        "sbatch", "--parsable",
        f"--array={array}",
        f"--partition={partition}",
        f"--account={account}",
        f"--time={time_limit}",
        f"--mem={mem}",
        f"--cpus-per-task={cpus}",
        f"--export={exports}",
        "singularity/campaign.sbatch",
    ]
    p = _run(cmd, cwd=REPO)
    ok = p.returncode == 0
    job_id = p.stdout.strip().split(";")[0] if ok else None
    return ok, (p.stdout + p.stderr).strip(), job_id


def submit_script(script: str, exports: dict[str, str],
                  account: str = DEFAULT_ACCOUNT,
                  partition: str | None = None,
                  time_limit: str | None = None, mem: str | None = None,
                  cpus: int | None = None, gres: str | None = None
                  ) -> tuple[bool, str, str | None]:
    """Queue a helper job (MSA search, structure prediction, generate, screen).

    Resource flags passed here OVERRIDE the #SBATCH defaults baked into the
    script, so the UI can size a job without editing the sbatch file.
    """
    ex = ",".join(["ALL"] + [f"{k}={v}" for k, v in exports.items()])
    cmd = ["sbatch", "--parsable", f"--account={account}"]
    if partition:
        cmd.append(f"--partition={partition}")
    if time_limit:
        cmd.append(f"--time={time_limit}")
    if mem:
        cmd.append(f"--mem={mem}")
    if cpus:
        cmd.append(f"--cpus-per-task={cpus}")
    if gres:
        cmd.append(f"--gres={gres}")
    cmd += [f"--export={ex}", script]
    p = _run(cmd, cwd=REPO)
    ok = p.returncode == 0
    return ok, (p.stdout + p.stderr).strip(), (p.stdout.strip().split(";")[0] if ok else None)


def queue(user: str | None = None) -> list[Job]:
    fmt = "%i|%j|%T|%M|%R"
    cmd = ["squeue", "-h", "-o", fmt] + (["-u", user] if user else [])
    p = _run(cmd)
    jobs = []
    for line in p.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) >= 5:
            jobs.append(Job(parts[0], parts[1], parts[2], parts[3], parts[4]))
    return jobs


def history(job_id: str) -> list[Job]:
    p = _run(["sacct", "-j", str(job_id), "-n", "-P",
              "--format=JobID,JobName,State,Elapsed"])
    jobs = []
    for line in p.stdout.strip().splitlines():
        parts = line.split("|")
        # Skip .batch / .extern steps — they duplicate the parent's state.
        if len(parts) >= 4 and "." not in parts[0]:
            jobs.append(Job(parts[0], parts[1], parts[2], parts[3]))
    return jobs


def cancel(job_id: str) -> tuple[bool, str]:
    p = _run(["scancel", str(job_id)])
    return p.returncode == 0, (p.stdout + p.stderr).strip()


def log_paths(job_id: str) -> list[Path]:
    return sorted((REPO / "logs").glob(f"campaign-{job_id}_*.out"))


def tail(path: Path, n: int = 60) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError as e:
        return f"(cannot read {path}: {e})"
    return "\n".join(lines[-n:])


# The verification checks. These exist because every failure mode we hit was
# SILENT: the job completes, writes plausible output, and only the log reveals
# that it ran one model instead of two, or single-sequence instead of MSA-backed.
CHECKS = [
    ("Models loaded", r"structure backends:\s*(.+)"),
    ("Target MSA", r"target MSA:\s*(.+)"),
    ("MSA rows ingested", r"n_msa\s+(\d+)"),
    ("Loss terms", r"loss terms:\s*(.+)"),
    ("Seeded from", r"seeded \d+ trajectories from (.+)"),
]


def verify(job_id: str, expect_models: list[str] | None = None,
           expect_msa: bool = True) -> list[tuple[str, str, str]]:
    """Read what a job SAYS it did. Returns (label, value, status)."""
    logs = log_paths(job_id)
    if not logs:
        return [("Log", "not written yet", "pending")]
    text = logs[0].read_text(errors="replace")
    out = []
    for label, pattern in CHECKS:
        m = re.search(pattern, text)
        if not m:
            continue
        val = m.group(1).strip()
        status = "ok"
        if label == "Models loaded" and expect_models:
            got = {x.strip() for x in val.split(",")}
            if got != set(expect_models):
                status = "fail"
        if label == "Target MSA" and expect_msa and "none" in val.lower():
            status = "fail"
        if label == "MSA rows ingested" and expect_msa and val.strip() == "1":
            status = "fail"
        out.append((label, val, status))
    if not out:
        out = [("Log", "job has not reached the reporting stage", "pending")]
    return out


def progress(job_id: str, n_tasks: int) -> tuple[int, int, int]:
    """(completed, failed, running) across an array."""
    done = failed = running = 0
    for j in history(job_id):
        st = j.state.split()[0]
        if st == "COMPLETED":
            done += 1
        elif st in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"):
            failed += 1
        elif st in ("RUNNING", "PENDING"):
            running += 1
    return done, failed, running


def accounts(user: str | None = None) -> list[str]:
    """Accounts this user can actually submit under.

    Worth querying rather than hardcoding: an association with MaxSubmit=0
    accepts nothing, and the rejection (AssocMaxSubmitJobLimit) arrives at
    submit time with no hint about which account to use instead.
    """
    import os
    u = user or os.environ.get("USER", "")
    return _memoized(f"accounts:{u}", lambda: _accounts_uncached(u))


def _accounts_uncached(u: str) -> list[str]:
    p = _run(["sacctmgr", "-nP", "show", "assoc", f"user={u}",
              "format=Account,MaxSubmit"])
    out = []
    for line in p.stdout.strip().splitlines():
        parts = line.split("|")
        if not parts or not parts[0]:
            continue
        limit = parts[1] if len(parts) > 1 else ""
        if limit.strip() == "0":
            continue          # cannot submit under this one
        if parts[0] not in out:
            out.append(parts[0])
    return out or [DEFAULT_ACCOUNT]


def partitions() -> list[str]:
    return _memoized("partitions", _partitions_uncached)


def _partitions_uncached() -> list[str]:
    p = _run(["sinfo", "-h", "-o", "%P"])
    names = sorted({x.strip().rstrip("*") for x in p.stdout.split() if x.strip()})
    gpu = [n for n in names if "gpu" in n or "kempner" in n]
    return gpu or GPU_PARTITIONS
