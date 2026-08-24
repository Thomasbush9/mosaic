"""Which backends are actually usable for the scoring benchmark?

Reports IMPORT and WEIGHTS separately per model -- see model-audit.sbatch for
why conflating them hides real problems.
"""

import importlib
import os
from pathlib import Path

H = os.path.expanduser("~")

# name -> (module, attribute the pipeline calls, weight paths that must exist)
MODELS = {
    "AF2 (multimer)":   ("mosaic.models.af2", "AF2",
                         [f"{H}/.alphafold/params/params_model_1_multimer_v3.npz"]),
    "AF2 + MSA":        ("mosaic.models.af2_msa", None, []),
    "Boltz-1":          ("mosaic.models.boltz1", None, [f"{H}/.boltz/boltz1_conf.ckpt"]),
    "Boltz-2":          ("mosaic.models.boltz2", None, [f"{H}/.boltz/boltz2_conf.ckpt"]),
    "OpenFold3":        ("mosaic.models.of3", None,
                         [f"{H}/.openfold3/jax/of3.eqx"]),
    "Protenix (mini)":  ("mosaic.models.protenix", None,
                         [f"{H}/.protenix/protenix_mini_default_v0.5.0.eqx"]),
    "ESMFold2":         ("mosaic.models.esmfold2", "ESMFold2Fast",
                         [f"{H}/.cache/huggingface/hub/models--biohub--ESMFold2-Fast"]),
    "BoltzGen":         ("mosaic.models.boltzgen", None,
                         [f"{H}/.boltz/boltzgen1_diverse.ckpt"]),
    "Proteina":         ("mosaic.models.proteina", None,
                         [f"{H}/.cache/jproteina_complexa/weights_v2/denoiser.eqx"]),
    "ProMERA":          ("mosaic.models.promera", None, []),
    "OpenDDE":          ("mosaic.models.opendde", None, []),
    "ProteinMPNN":      ("mosaic.losses.protein_mpnn", "ProteinMPNN", []),
    "ESM-C":            ("mosaic.losses.esmc", None,
                         [f"{H}/.cache/huggingface/hub/models--biohub--ESMC-300M"]),
    "ESM2":             ("mosaic.losses.esm", None, []),
    "AbLang":           ("mosaic.losses.ablang", None, []),
    "AbLang2":          ("mosaic.losses.ablang2", None, []),
    "stability head":   ("mosaic.losses.stability", None, []),
    "trigram":          ("mosaic.losses.trigram", None, []),
}

# Named in the benchmark plan but with no wrapper in mosaic at all -- prove the
# absence rather than asserting it, so the list stays honest if one gets added.
ABSENT = {
    "Chai-1":               "chai_lab",
    "ESM-IF1":              "esm.inverse_folding",
    "PyRosetta / Rosetta":  "pyrosetta",
}

print(f"{'model':22s} {'import':10s} {'weights':10s}  detail")
print("-" * 84)


def weight_state(paths):
    if not paths:
        return "n/a", ""
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        return "MISSING", missing[0].replace(H, "~")
    total = 0
    for p in paths:
        q = Path(p)
        total += (sum(f.stat().st_size for f in q.rglob("*") if f.is_file())
                  if q.is_dir() else q.stat().st_size)
    return "ok", f"{total / 2**30:.1f} GiB"


for name, (mod, attr, paths) in MODELS.items():
    try:
        m = importlib.import_module(mod)
        imp, detail = "ok", ""
        if attr and not hasattr(m, attr):
            imp, detail = "PARTIAL", f"no attribute {attr}"
    except Exception as e:
        imp, detail = "FAIL", f"{type(e).__name__}: {str(e)[:150]}"
    w, wdetail = weight_state(paths)
    print(f"{name:22s} {imp:10s} {w:10s}  {detail or wdetail}")

print()
print("named in the benchmark plan, not installed:")
for name, mod in ABSENT.items():
    try:
        importlib.import_module(mod)
        print(f"  {name:22s} PRESENT after all ({mod})")
    except Exception:
        print(f"  {name:22s} absent  (no module {mod})")
