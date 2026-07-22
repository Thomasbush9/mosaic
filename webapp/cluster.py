"""SLURM interaction: submit, query, cancel, read logs. No Streamlit here.

Everything shells out. Nothing here runs a model — the app only ever queues work,
which is what keeps it safe to run on a login node.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from store import REPO, SETUP

ACCOUNT = "kempner_bsabatini_lab"   # bsabatini_lab has MaxSubmit=0 and cannot submit
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
        f"--account={ACCOUNT}",
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


def submit_script(script: str, exports: dict[str, str]) -> tuple[bool, str, str | None]:
    """Queue one of the helper jobs (MSA search, structure prediction, BoltzGen)."""
    ex = ",".join(["ALL"] + [f"{k}={v}" for k, v in exports.items()])
    p = _run(["sbatch", "--parsable", f"--export={ex}", script], cwd=REPO)
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
