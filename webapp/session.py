"""Project sessions: a named working directory with its own targets and results.

The registry lives in the user's home, not the repo, so sessions survive a repo
move and are not shared or committed by accident.

A session is just a name plus a working directory. Everything else — targets,
msa, designs, configs — lives under that directory, so switching sessions
switches the whole view.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REGISTRY = Path.home() / ".mosaic_webapp" / "sessions.json"

# Where the existing data already lives — used as the default session's workdir
# so a first run shows the work already on disk rather than an empty app.
DEFAULT_WORKDIR = Path(
    "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/mosaic_setup"
)

SUBDIRS = ("targets", "msa", "designs", "configs", "structures")


@dataclass
class Session:
    id: str
    name: str
    workdir: str
    created: float

    @property
    def path(self) -> Path:
        return Path(self.workdir)


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return s or "session"


def _load() -> dict:
    if not REGISTRY.exists():
        return {"sessions": [], "active": None}
    try:
        return json.loads(REGISTRY.read_text())
    except Exception:
        # A corrupt registry should not lock the user out of the app.
        return {"sessions": [], "active": None}


def _save(reg: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=2))


def list_sessions() -> list[Session]:
    return [Session(**s) for s in _load()["sessions"]]


def create(name: str, workdir: str | Path) -> Session:
    reg = _load()
    base = _slug(name)
    existing = {s["id"] for s in reg["sessions"]}
    sid, n = base, 2
    while sid in existing:
        sid, n = f"{base}-{n}", n + 1

    wd = Path(workdir).expanduser()
    wd.mkdir(parents=True, exist_ok=True)
    for sub in SUBDIRS:
        (wd / sub).mkdir(exist_ok=True)

    s = Session(id=sid, name=name.strip() or sid, workdir=str(wd),
                created=time.time())
    reg["sessions"].append(asdict(s))
    reg["active"] = sid
    _save(reg)
    return s


def delete(session_id: str) -> None:
    """Forget a session. Never touches the working directory itself — the data
    is the point, and an accidental click should not destroy a campaign."""
    reg = _load()
    reg["sessions"] = [s for s in reg["sessions"] if s["id"] != session_id]
    if reg.get("active") == session_id:
        reg["active"] = reg["sessions"][0]["id"] if reg["sessions"] else None
    _save(reg)


def rename(session_id: str, name: str) -> None:
    reg = _load()
    for s in reg["sessions"]:
        if s["id"] == session_id:
            s["name"] = name.strip() or s["name"]
    _save(reg)


def set_active(session_id: str) -> None:
    reg = _load()
    if any(s["id"] == session_id for s in reg["sessions"]):
        reg["active"] = session_id
        _save(reg)


def active() -> Session:
    """The current session, bootstrapping a default one on first run."""
    reg = _load()
    if not reg["sessions"]:
        return create("Default", DEFAULT_WORKDIR)
    aid = reg.get("active") or reg["sessions"][0]["id"]
    for s in reg["sessions"]:
        if s["id"] == aid:
            return Session(**s)
    return Session(**reg["sessions"][0])


def ensure_dirs(s: Session) -> None:
    for sub in SUBDIRS:
        (s.path / sub).mkdir(parents=True, exist_ok=True)
