"""Acquire and convert the official Base checkpoint using the frozen Mosaic runtime."""
import datetime
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket

import requests
import torch
import equinox as eqx
import jax
import numpy as np
from ml_collections import ConfigDict

import protenix.backend as backend
import protenix.protenij  # Register the installed torch-to-Equinox converters.
from protenix.config import parse_configs
from protenix.configs.configs_base import configs as base_configs
from protenix.configs.configs_data import data_configs
from protenix.configs.configs_inference import inference_configs
from protenix.configs.configs_model_type import model_configs
from protenix.model.protenix import Protenix
from protenix.web_service.dependency_url import URL

NAME = "protenix_base_default_v1.0.0"
CACHE = Path.home() / ".protenix"
REVISION = "d48a0b0456b582cef10f1bacfb19941a04f7e8c8"
direct_url = json.loads(importlib.metadata.distribution("protenix").read_text("direct_url.json"))
assert direct_url["vcs_info"]["commit_id"] == REVISION, direct_url
assert jax.default_backend() == "cpu"
assert not torch.cuda.is_available()
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path, **metadata):
    return {"filename": path.name, "size_bytes": path.stat().st_size,
            "sha256": digest(path), **metadata}


for suffix in (".eqx", ".skeleton.pkl", ".manifest.json"):
    if (CACHE / (NAME + suffix)).exists():
        raise FileExistsError(f"Refusing to replace {NAME + suffix}")
for filename in backend.DATA_FILES:
    if not (CACHE / filename).is_file():
        raise FileNotFoundError(f"Required existing shared data missing: {filename}")

checkpoint = CACHE / (NAME + ".pt")
url = URL[NAME]
if not checkpoint.exists():
    partial = checkpoint.with_suffix(".pt.partial")
    print(f"Downloading {url}", flush=True)
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        expected = response.headers.get("Content-Length")
        with partial.open("xb") as output:
            for chunk in response.iter_content(8 * 1024 * 1024):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if expected is not None and partial.stat().st_size != int(expected):
            raise RuntimeError("Incomplete checkpoint download")
    os.link(partial, checkpoint)  # Atomic publication, never replace an existing file.
    partial.unlink()
source_record = record(checkpoint, source_url=url)
print(json.dumps(source_record), flush=True)

configs = parse_configs(
    {**base_configs, "data": data_configs, **inference_configs},
    arg_str=f"--model_name {NAME}", fill_required_with_null=True,
)
configs.update(ConfigDict(model_configs[NAME]))
model = Protenix(configs).cpu()
checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
state = checkpoint_data["model"]
if next(iter(state)).startswith("module."):
    state = {key.removeprefix("module."): value for key, value in state.items()}
model.load_state_dict(state, strict=True)
model.eval()
parameter_count = sum(parameter.numel() for parameter in model.parameters())
print(f"Strict checkpoint load passed: {parameter_count} torch parameters", flush=True)
del state, checkpoint_data
gc.collect()
converted = backend.from_torch(model)
del model
gc.collect()
staging = CACHE / (NAME + ".conversion-partial")
backend.save_model(converted, str(staging))
expected_leaves = jax.tree.leaves(converted)
# Exercise the real loader offline, including all shared-data existence checks.
def forbid_download(*args, **kwargs):
    raise RuntimeError("Unexpected network attempt during offline load")
backend._hf_download = forbid_download
socket.socket.connect = forbid_download
loaded = backend.load_model(str(staging))
actual_leaves = jax.tree.leaves(loaded)
assert len(expected_leaves) == len(actual_leaves)
for expected, actual in zip(expected_leaves, actual_leaves, strict=True):
    if eqx.is_array(expected):
        np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
    else:
        assert expected == actual
array_elements = sum(leaf.size for leaf in actual_leaves if eqx.is_array(leaf))
print(f"Offline round-trip passed: {array_elements} array elements", flush=True)
# Skeleton first; .eqx is the backend's cache-ready sentinel.
for suffix in (".skeleton.pkl", ".eqx"):
    partial = Path(str(staging) + suffix)
    os.link(partial, CACHE / (NAME + suffix))
    partial.unlink()
root = Path(backend.__file__).parent
sources = ["backend.py", "protenij.py", "configs/configs_base.py",
           "configs/configs_model_type.py", "web_service/dependency_url.py"]
data_keys = ["ccd_components_file", "ccd_components_rdkit_mol_file", "pdb_cluster_file"]
manifest = {
    "model_name": NAME,
    "acquired_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "backend": {"distribution": "protenix", "version": importlib.metadata.version("protenix"),
                "source": direct_url, "source_files": [record(root / p, relative_path=p) for p in sources]},
    "versions": {name: importlib.metadata.version(name) for name in ["torch", "jax", "equinox", "numpy"]},
    "checkpoint": source_record,
    "runtime_assets": [record(CACHE / (NAME + suffix), derived_from=checkpoint.name)
                       for suffix in [".eqx", ".skeleton.pkl"]],
    "shared_existing_assets": [record(CACHE / filename, upstream_source_url=URL[key],
                                      acquisition="pre-existing; not downloaded or changed by this job")
                               for filename, key in zip(backend.DATA_FILES, data_keys, strict=True)],
    "conversion": {"method": "installed Protenix(configs), load_state_dict(strict=True), backend.from_torch, backend.save_model",
                   "model_config_overrides": model_configs[NAME], "torch_parameter_count": parameter_count,
                   "equinox_array_elements": array_elements, "device": "cpu",
                   "offline_roundtrip": "backend.load_model with network disabled; every array equals converted source",
                   "gpu_inference_verified": False},
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "acquisition_script": record(Path(__file__)),
    "container_path": os.environ.get("ACQUISITION_SIF"),
}
manifest_path = CACHE / (NAME + ".manifest.json")
partial_manifest = manifest_path.with_suffix(".json.partial")
partial_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
os.link(partial_manifest, manifest_path)
partial_manifest.unlink()
print(f"ACQUISITION COMPLETE: {manifest_path}", flush=True)
