#!/usr/bin/env python
"""Environment check for the mosaic container.

Runs in two contexts:

  build time  — `singularity build` executes this from %test. No GPU is visible
                and no weights are mounted, so only the interpreter and the
                installed packages are checked. Failures here are fatal.

  run time    — `singularity exec --nv ... selftest.py` on a compute node.
                Adds GPU visibility, mounted weights and cache writability.
                Any failure exits non-zero so design.sbatch preflight can
                abort the array before the remaining tasks burn GPU hours.

Every failure mode this catches otherwise surfaces as an error that does not
name its own cause: a missing --nv looks like a very slow job, an unbound
cache looks like a home-quota error from an unrelated program, and a truncated
BoltzGen checkpoint looks like a pickle error deep inside torch.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REQUIRED_MODULES = ("jax", "equinox", "optax", "torch", "gemmi", "numpy", "mosaic")

# Weight caches, at the paths mosaic itself looks for. mosaic-exec.sh binds the
# shared read-only tree directly onto these, so they are ordinary directories —
# not symlinks, and not redirected by any environment variable.
WEIGHT_LOCATIONS = (
    ("~/.cache/huggingface", "HuggingFace cache (ESM-C, ESMFold2, ESM2)"),
    ("~/.boltz", "Boltz-1 / Boltz-2 / BoltzGen checkpoints + CCD"),
    ("~/.alphafold/params", "AlphaFold2 parameters"),
    ("~/.protenix", "Protenix weights + reference data"),
)

WRITABLE_PATHS = ("/jax_cache", "/work", "~/.cache/mosaic")

_ok = "  ok   "
_bad = " FAIL  "
_warn = " warn  "

failures: list[str] = []
warnings: list[str] = []


def report(status: str, label: str, detail: str = "") -> None:
    line = f"[{status}] {label}"
    if detail:
        line += f"  — {detail}"
    print(line)


def check_interpreter() -> None:
    major_minor = sys.version_info[:2]
    if major_minor == (3, 12):
        report(_ok, "python 3.12", sys.version.split()[0])
    else:
        failures.append("python version")
        report(_bad, "python 3.12", f"got {major_minor[0]}.{major_minor[1]}")


def check_modules() -> None:
    missing = [m for m in REQUIRED_MODULES if importlib.util.find_spec(m) is None]
    if missing:
        failures.append("imports")
        report(_bad, "packages importable", f"missing: {', '.join(missing)}")
    else:
        report(_ok, "packages importable", f"{len(REQUIRED_MODULES)} checked")


def check_wget() -> None:
    # models/boltzgen.py shells out to the wget binary and does not check the
    # return code, so its absence fails silently and late.
    from shutil import which

    if which("wget"):
        report(_ok, "wget present", "needed by models/boltzgen.py")
    else:
        failures.append("wget")
        report(_bad, "wget present", "boltzgen downloads will fail silently")


def check_cuda_version() -> None:
    """Refuse CUDA-13 JAX wheels — cluster driver max is 12.9.1."""
    from importlib.metadata import PackageNotFoundError, distributions, version

    for name in ("jax-cuda13-plugin", "jax-cuda13-pjrt"):
        try:
            v = version(name)
        except PackageNotFoundError:
            continue
        failures.append("cuda version")
        report(_bad, "JAX CUDA <13", f"{name}=={v} installed — need jax[cuda12]")
        return

    try:
        v = version("jax-cuda12-plugin")
        report(_ok, "JAX CUDA <13", f"jax-cuda12-plugin {v}")
        return
    except PackageNotFoundError:
        pass

    # Fallback: any nvidia-cuda-runtime* package major version.
    cuda_major: int | None = None
    runtime_name = ""
    runtime_ver = ""
    for dist in distributions():
        name = (dist.metadata["Name"] or "").lower()
        if "cuda-runtime" not in name:
            continue
        runtime_name = name
        runtime_ver = dist.version
        try:
            cuda_major = int(runtime_ver.split(".", 1)[0])
        except ValueError:
            continue
        break

    if cuda_major is None:
        failures.append("cuda version")
        report(_bad, "JAX CUDA <13", "no jax-cuda12-plugin / nvidia-cuda-runtime found")
    elif cuda_major >= 13:
        failures.append("cuda version")
        report(_bad, "JAX CUDA <13", f"{runtime_name}=={runtime_ver}")
    else:
        report(_ok, "JAX CUDA <13", f"{runtime_name}=={runtime_ver}")


def check_gpu(build_time: bool) -> None:
    try:
        import jax
    except Exception as exc:  # pragma: no cover - import already checked above
        failures.append("jax import")
        report(_bad, "jax imports", str(exc))
        return

    backend = jax.default_backend()
    devices = jax.devices()

    if backend == "gpu":
        names = ", ".join(sorted({d.device_kind for d in devices}))
        report(_ok, f"GPU visible ({len(devices)})", names)
        if len(devices) > 1:
            warnings.append("multiple GPUs")
            report(
                _warn,
                "more than one GPU allocated",
                "mosaic is single-device; the extras will sit idle",
            )
    elif build_time:
        report(_ok, "jax backend", f"{backend} — expected during build")
    else:
        failures.append("gpu")
        report(_bad, "GPU visible", "backend is CPU — did you forget --nv?")


def _broken_link_in(path: Path) -> Path | None:
    """First component of `path` that is a symlink which does not resolve.

    Walks root -> leaf so the outermost break is reported, which is the one the
    user has to fix. Returns None if nothing along the path is a broken link.
    """
    chain: list[Path] = []
    cur = path
    while True:
        chain.append(cur)
        if cur.parent == cur:
            break
        cur = cur.parent
    for component in reversed(chain):
        if component.is_symlink() and not component.exists():
            return component
    return None


def check_weights(build_time: bool) -> None:
    if build_time:
        return
    for raw, label in WEIGHT_LOCATIONS:
        path = Path(os.path.expanduser(raw))
        # Defensive: the current design uses bind mounts, not symlinks, so this
        # should never fire. It catches a hand-rolled setup that linked these to
        # a host path — which resolves on the login node and dangles in the
        # container, where it is the only place that matters. Checks every
        # component, not just the leaf: ~/.alphafold/params is reached *through*
        # ~/.alphafold, so the break would be a parent.
        broken = _broken_link_in(path)
        if broken is not None:
            failures.append(f"{label} (dangling)")
            report(_bad, label, f"{broken} -> {os.readlink(broken)} does not resolve here")
            continue
        if not path.exists():
            warnings.append(label)
            report(_warn, label, f"{raw} not found (fine if unused)")
            continue
        try:
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        except OSError as exc:
            warnings.append(label)
            report(_warn, label, f"unreadable: {exc}")
            continue
        # The failure mode of explicit bind mounts: mosaic-exec.sh creates the
        # destination before mounting, so a bind that was skipped or pointed at
        # an empty source leaves a real but empty directory. mosaic would then
        # silently re-download tens of GB into scratch instead of using the
        # shared copy.
        if size == 0:
            failures.append(f"{label} (empty)")
            report(_bad, label, f"{raw} exists but is empty — bind did not happen")
            continue
        report(_ok, label, f"{size / 2**30:.1f} GiB at {raw}")


def check_truncated_checkpoints(build_time: bool) -> None:
    """Catch the BoltzGen failure mode before it reaches torch.load.

    models/boltzgen.py fetches checkpoints with `subprocess.run(["wget", ...])`
    and inspects neither the return code nor the result, so a failed download
    leaves an HTML error page where a multi-GB checkpoint belongs.
    """
    if build_time:
        return
    boltz = Path(os.path.expanduser("~/.boltz"))
    if not boltz.exists():
        return
    for ckpt in sorted(boltz.glob("*.ckpt")):
        size_mb = ckpt.stat().st_size / 2**20
        if size_mb < 100:
            failures.append(f"truncated {ckpt.name}")
            report(_bad, f"{ckpt.name}", f"only {size_mb:.1f} MiB — re-download it")
        else:
            report(_ok, f"{ckpt.name}", f"{size_mb / 1024:.1f} GiB")


def check_writable(build_time: bool) -> None:
    if build_time:
        return
    for raw in WRITABLE_PATHS:
        path = Path(os.path.expanduser(raw))
        probe = path / ".mosaic-write-probe"
        try:
            probe.touch()
            probe.unlink()
        except OSError as exc:
            failures.append(f"{raw} writable")
            report(_bad, f"{raw} writable", str(exc))
        else:
            report(_ok, f"{raw} writable")


def check_jax_cache(build_time: bool) -> None:
    if build_time:
        return
    # Not fatal, but without it every array task recompiles from scratch and
    # the first iteration can cost minutes.
    entries = list(Path("/jax_cache").glob("*")) if Path("/jax_cache").exists() else []
    if entries:
        report(_ok, "JAX compilation cache", f"{len(entries)} entries — warm")
    else:
        warnings.append("cold jax cache")
        report(_warn, "JAX compilation cache", "empty — first run will be slow")


def main() -> int:
    build_time = "--build" in sys.argv or os.environ.get("SINGULARITY_NAME") is None

    print(f"mosaic environment check  ({'build' if build_time else 'runtime'})")
    print("-" * 64)

    check_interpreter()
    check_modules()
    check_wget()
    check_cuda_version()
    check_gpu(build_time)
    check_weights(build_time)
    check_truncated_checkpoints(build_time)
    check_writable(build_time)
    check_jax_cache(build_time)

    print("-" * 64)
    if failures:
        # Always non-zero: build %test must fail the image, and design.sbatch
        # preflight must stop the array. Warnings alone do not fail.
        print(f"{len(failures)} problem(s): {', '.join(failures)}")
        return 1
    if warnings:
        print(f"no failures, {len(warnings)} warning(s)")
    else:
        print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
